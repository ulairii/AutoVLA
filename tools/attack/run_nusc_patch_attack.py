"""Generate and evaluate visual patch attacks on AutoVLA nuScenes samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
from tools.attack.patch_attack import PatchAttackConfig, iter_scene_paths, load_scene, make_attacked_scene
from tools.attack.visualization import visualize_attack_sample


def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml")
    parser.add_argument("--scene_dir", type=Path, required=True, help="Preprocessed AutoVLA nuScenes JSON directory.")
    parser.add_argument("--sensor_data_path", type=str, required=True, help="Root used to resolve clean scene image paths.")
    parser.add_argument("--work_dir", type=Path, default=Path("/mnt/indigo/tigersec/runw/workdirs/autovla_attack"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional AutoVLA checkpoint for clean/adv prediction.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--pattern", choices=["checkerboard", "black", "white", "red"], default="checkerboard")
    parser.add_argument("--patch_ratio", type=float, default=0.18)
    parser.add_argument("--position", choices=["bottom_center", "center", "top_center", "bottom_left", "bottom_right"], default="bottom_center")
    parser.add_argument("--frames", default="all", help="'all', 'first', 'last', or comma-separated frame indices.")
    parser.add_argument("--opacity", type=float, default=1.0)
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--skip_visualization", action="store_true")
    return parser.parse_args()


def build_model(config: Dict, checkpoint: Path, device: str):
    import torch

    from models.autovla import SFTAutoVLA

    model = SFTAutoVLA(config)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state.get("state_dict", state)
    state_dict = {
        key.replace("autovla.", "").replace("drivevla.", ""): value
        for key, value in state_dict.items()
    }
    load_result = model.autovla.load_state_dict(state_dict, strict=False)
    print(
        "Loaded checkpoint with "
        f"{len(load_result.missing_keys)} missing and "
        f"{len(load_result.unexpected_keys)} unexpected keys"
    )
    del state, state_dict
    model.to(device)
    model.autovla.device = device
    model.eval()
    return model


def build_features(dataset: SFTDataset, scene_path: Path, sensor_data_path: Optional[str]) -> Tuple[Dict, Dict, Dict]:
    scene_data = load_scene(scene_path)
    input_features: Dict = {}
    target_trajectory: Dict = {}
    for builder in dataset._agent.get_feature_builders():
        input_features.update(builder.compute_features(scene_data))
    for builder in dataset._agent.get_target_builders():
        target_trajectory.update(builder.compute_targets(scene_data))
    input_features["sensor_data_path"] = sensor_data_path
    return scene_data, input_features, target_trajectory


def predict_scene(model, dataset, scene_path: Path, sensor_data_path: Optional[str]):
    import torch

    _, input_features, _ = build_features(dataset, scene_path, sensor_data_path)
    result = {
        "trajectory": None,
        "text": None,
        "num_action_tokens": 0,
        "action_tokens": [],
        "error": None,
    }

    with torch.no_grad():
        inputs = model.autovla.get_prompt(input_features)
        model_inputs = {k: v.to(model.autovla.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        outputs = model.autovla.vlm.generate(
            **model_inputs,
            max_length=model.autovla.gen_conf["max_length"],
            do_sample=True,
            temperature=model.autovla.gen_conf["temperature"],
            top_k=model.autovla.gen_conf["top_k"],
            top_p=model.autovla.gen_conf["top_p"],
        )

    outputs_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ][0]
    if len(outputs_trimmed) and outputs_trimmed[-1].item() == model.autovla.processor.tokenizer.eos_token_id:
        outputs_trimmed = outputs_trimmed[:-1]

    result["text"] = model.autovla.processor.decode(outputs_trimmed)
    action_tokens = outputs_trimmed[outputs_trimmed >= model.autovla.action_start_id].cpu()
    result["num_action_tokens"] = int(len(action_tokens))
    result["action_tokens"] = [int(token) for token in action_tokens.tolist()]

    if len(action_tokens) == 0:
        result["error"] = "no_action_tokens"
        return result

    trajectory = model.autovla.action_tokenizer.decode_token_ids_to_trajectory(action_tokens)
    if isinstance(trajectory, list):
        result["error"] = "decode_failed"
        return result

    trajectory = trajectory[0, 1:]
    if hasattr(trajectory, "detach"):
        trajectory = trajectory.detach().cpu().tolist()
    result["trajectory"] = trajectory
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_dir = args.work_dir / "nusc_patch_attack"
    adv_scene_dir = run_dir / "adv_scenes"
    adv_image_root = run_dir / "adv_images"
    vis_dir = run_dir / "visualizations"
    output_jsonl = args.output_jsonl or run_dir / "attack_results.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    attack_config = PatchAttackConfig(
        pattern=args.pattern,
        patch_ratio=args.patch_ratio,
        position=args.position,
        opacity=args.opacity,
        frames=args.frames,
    )

    model = None
    clean_dataset = None
    adv_dataset = None
    if args.checkpoint is not None:
        from transformers import AutoProcessor

        from dataset_utils.sft_dataset import SFTDataset

        model = build_model(config, args.checkpoint, args.device)
        processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
        clean_dataset = SFTDataset(
            {"json_dataset_path": str(args.scene_dir), "sensor_data_path": args.sensor_data_path},
            config["model"],
            processor,
            using_cot=config["model"].get("use_cot", False),
        )
        adv_dataset = SFTDataset(
            {"json_dataset_path": str(adv_scene_dir), "sensor_data_path": str(adv_image_root)},
            config["model"],
            processor,
            using_cot=config["model"].get("use_cot", False),
        )

    with output_jsonl.open("w") as result_file:
        for scene_path in iter_scene_paths(args.scene_dir, args.num_samples):
            adv_scene_path, adv_scene = make_attacked_scene(
                scene_path=scene_path,
                output_scene_dir=adv_scene_dir,
                output_image_root=adv_image_root,
                sensor_data_path=args.sensor_data_path,
                config=attack_config,
            )
            clean_scene = load_scene(scene_path)

            clean_result = None
            adv_result = None
            clean_prediction = None
            adv_prediction = None

            if model is not None and clean_dataset is not None and adv_dataset is not None:
                clean_result = predict_scene(model, clean_dataset, scene_path, args.sensor_data_path)
                adv_result = predict_scene(model, adv_dataset, adv_scene_path, str(adv_image_root))
                clean_prediction = clean_result["trajectory"]
                adv_prediction = adv_result["trajectory"]

            if not args.skip_visualization:
                token = str(clean_scene.get("token", scene_path.stem))
                visualize_attack_sample(
                    clean_scene=clean_scene,
                    adv_scene=adv_scene,
                    output_path=vis_dir / f"{token}.png",
                    clean_sensor_root=args.sensor_data_path,
                    adv_sensor_root=str(adv_image_root),
                    clean_prediction=clean_prediction,
                    adv_prediction=adv_prediction,
                    title="AutoVLA nuScenes patch attack",
                )

            record = {
                "token": clean_scene.get("token", scene_path.stem),
                "clean_scene": str(scene_path),
                "adv_scene": str(adv_scene_path),
                "attack": adv_scene.get("attack", {}),
                "visualization": str(vis_dir / f"{clean_scene.get('token', scene_path.stem)}.png"),
                "clean_prediction": clean_prediction,
                "adv_prediction": adv_prediction,
                "clean_text": clean_result["text"] if clean_result else None,
                "adv_text": adv_result["text"] if adv_result else None,
                "clean_result": clean_result,
                "adv_result": adv_result,
            }
            result_file.write(json.dumps(record) + "\n")
            print(f"Wrote attack sample: {record['token']}")

    print(f"Results: {output_jsonl}")
    print(f"Adversarial scenes: {adv_scene_dir}")
    if not args.skip_visualization:
        print(f"Visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
