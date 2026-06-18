"""White-box gradient-based patch attack for AutoVLA on nuScenes.

This adapts recent white-box VLA patch-attack ideas to AutoVLA by optimizing a
shared visible image-space patch against the action-token loss. The optimized
patch is then pasted onto the raw camera frames, saved, and evaluated through
the normal AutoVLA inference path.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoProcessor

from dataset_utils.sft_dataset import DataCollator, SFTDataset
from tools.attack.patch_attack import CAMERA_KEYS, load_scene, save_scene
from tools.attack.run_nusc_patch_attack import build_features, build_model, predict_scene
from tools.attack.visualization import visualize_attack_sample


@dataclass(frozen=True)
class WhiteBoxPatchConfig:
    patch_ratio: float = 0.18
    position: str = "bottom_center"
    steps: int = 30
    lr: float = 8.0 / 255.0
    max_delta: float = 32.0 / 255.0
    action_loss_weight: float = 1.0
    cameras: Tuple[str, ...] = ("front_camera_paths",)
    frames: Tuple[int, ...] = (2, 3)


def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--sensor_data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, default=Path("/mnt/indigo/tigersec/runw/workdirs/autovla_attack"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--patch_ratio", type=float, default=0.18)
    parser.add_argument("--position", choices=["bottom_center", "center", "top_center", "bottom_left", "bottom_right"], default="bottom_center")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=8.0 / 255.0)
    parser.add_argument("--max_delta", type=float, default=32.0 / 255.0)
    parser.add_argument("--cameras", type=str, default="front_camera_paths")
    parser.add_argument("--frames", type=str, default="2,3")
    return parser.parse_args()


def _patch_box(width: int, height: int, ratio: float, position: str) -> Tuple[int, int, int, int]:
    patch_w = max(1, int(width * ratio))
    patch_h = max(1, int(height * ratio))
    margin = max(4, int(min(width, height) * 0.04))
    if position == "bottom_center":
        x0 = (width - patch_w) // 2
        y0 = height - patch_h - margin
    elif position == "center":
        x0 = (width - patch_w) // 2
        y0 = (height - patch_h) // 2
    elif position == "top_center":
        x0 = (width - patch_w) // 2
        y0 = margin
    elif position == "bottom_left":
        x0 = margin
        y0 = height - patch_h - margin
    elif position == "bottom_right":
        x0 = width - patch_w - margin
        y0 = height - patch_h - margin
    else:
        raise ValueError(f"Unknown patch position: {position}")
    return x0, y0, x0 + patch_w, y0 + patch_h


def _to_unit_tensor(image: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def _to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def _scene_videos_from_paths(scene: Mapping, sensor_root: str, device: str) -> Tuple[List[List[torch.Tensor]], Dict[Tuple[str, int], torch.Tensor]]:
    videos: List[List[torch.Tensor]] = []
    frame_map: Dict[Tuple[str, int], torch.Tensor] = {}
    for camera_key in CAMERA_KEYS:
        frames: List[torch.Tensor] = []
        for frame_idx, rel_path in enumerate(scene.get(camera_key) or []):
            with Image.open(Path(sensor_root) / rel_path) as image:
                tensor = _to_unit_tensor(image, device)
            frames.append(tensor)
            frame_map[(camera_key, frame_idx)] = tensor
        videos.append(frames)
    return videos, frame_map


def _apply_patch_to_videos(
    videos: Sequence[Sequence[torch.Tensor]],
    config: WhiteBoxPatchConfig,
    patch_param: torch.Tensor,
) -> Tuple[List[List[torch.Tensor]], Dict[Tuple[str, int], torch.Tensor]]:
    patched_videos: List[List[torch.Tensor]] = []
    patched_frames: Dict[Tuple[str, int], torch.Tensor] = {}
    for cam_idx, camera_key in enumerate(CAMERA_KEYS):
        patched_camera: List[torch.Tensor] = []
        for frame_idx, frame in enumerate(videos[cam_idx]):
            patched = frame.clone()
            if camera_key in config.cameras and frame_idx in config.frames:
                _, h, w = patched.shape
                x0, y0, x1, y1 = _patch_box(w, h, config.patch_ratio, config.position)
                patch_resized = F.interpolate(
                    patch_param.unsqueeze(0),
                    size=(y1 - y0, x1 - x0),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                patched[:, y0:y1, x0:x1] = torch.clamp(
                    patched[:, y0:y1, x0:x1] + patch_resized,
                    0.0,
                    1.0,
                )
            patched_camera.append(patched)
            patched_frames[(camera_key, frame_idx)] = patched
        patched_videos.append(patched_camera)
    return patched_videos, patched_frames


def _prepare_model_batch(
    processor,
    prompt_text: str,
    videos: Sequence[Sequence[torch.Tensor]],
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gt_trajectory: torch.Tensor,
    gt_action: torch.Tensor,
    has_cot: torch.Tensor,
    device: str,
) -> Dict[str, torch.Tensor]:
    processed = processor(
        text=[prompt_text],
        videos=list(videos),
        images=None,
        padding=True,
        return_tensors="pt",
    )
    batch = {
        "input_ids": processed["input_ids"].to(device),
        "attention_mask": processed["attention_mask"].to(device),
        "pixel_values_videos": processed["pixel_values_videos"].to(device),
        "video_grid_thw": processed["video_grid_thw"].to(device),
        "labels": labels.to(device),
        "gt_trajectory": gt_trajectory.to(device),
        "gt_action": gt_action.to(device),
        "has_cot": has_cot.to(device),
    }
    # Keep the original tokenization for loss alignment.
    batch["input_ids"] = input_ids.to(device)
    batch["attention_mask"] = attention_mask.to(device)
    return batch


def _action_token_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model.autovla(batch)
    logits = outputs.logits
    labels = batch["labels"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    logits_flat = shift_logits.view(-1, vocab_size)
    labels_flat = shift_labels.view(-1)
    action_mask = labels_flat >= model.autovla.action_start_id
    ce = F.cross_entropy(logits_flat, labels_flat, reduction="none")
    if action_mask.any():
        return ce[action_mask].mean()
    return ce.mean()


def optimize_patch(
    model,
    processor,
    prompt_text: str,
    clean_videos: Sequence[Sequence[torch.Tensor]],
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gt_trajectory: torch.Tensor,
    gt_action: torch.Tensor,
    has_cot: torch.Tensor,
    config: WhiteBoxPatchConfig,
    device: str,
) -> torch.Tensor:
    patch_param = torch.nn.Parameter(torch.zeros(3, 48, 48, device=device))
    optimizer = torch.optim.Adam([patch_param], lr=config.lr)
    for _ in range(config.steps):
        optimizer.zero_grad()
        patched_videos, _ = _apply_patch_to_videos(clean_videos, config, patch_param)
        batch = _prepare_model_batch(
            processor=processor,
            prompt_text=prompt_text,
            videos=patched_videos,
            labels=labels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gt_trajectory=gt_trajectory,
            gt_action=gt_action,
            has_cot=has_cot,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action_loss = _action_token_loss(model, batch)
            loss = -config.action_loss_weight * action_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            patch_param.clamp_(-config.max_delta, config.max_delta)
    return patch_param.detach()


def _write_adv_scene_and_images(
    clean_scene: Mapping,
    sensor_root: str,
    patched_frames: Mapping[Tuple[str, int], torch.Tensor],
    output_scene_dir: Path,
    output_image_root: Path,
    attack_meta: Dict,
) -> Tuple[Path, Dict]:
    token = str(clean_scene.get("token", "scene"))
    adv_scene = copy.deepcopy(clean_scene)
    attacked_images = []
    for camera_key in CAMERA_KEYS:
        new_paths = []
        for frame_idx, rel_path in enumerate(clean_scene.get(camera_key) or []):
            src = Path(sensor_root) / rel_path
            dst = output_image_root / token / camera_key / f"{frame_idx:02d}{src.suffix or '.jpg'}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            tensor = patched_frames[(camera_key, frame_idx)]
            _to_uint8_image(tensor).save(dst)
            new_paths.append(str(dst.relative_to(output_image_root)))
            if frame_idx in attack_meta["frames"]:
                attacked_images.append({"camera": camera_key, "frame": frame_idx, "adversarial": str(dst)})
        adv_scene[camera_key] = new_paths
    adv_scene["attack"] = {
        "name": "whitebox_patch",
        "patch_ratio": attack_meta["patch_ratio"],
        "position": attack_meta["position"],
        "steps": attack_meta["steps"],
        "lr": attack_meta["lr"],
        "max_delta": attack_meta["max_delta"],
        "frames": list(attack_meta["frames"]),
        "attacked_images": attacked_images,
    }
    adv_scene_path = output_scene_dir / f"{token}.json"
    save_scene(adv_scene, adv_scene_path)
    return adv_scene_path, adv_scene


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    attack_cfg = WhiteBoxPatchConfig(
        patch_ratio=args.patch_ratio,
        position=args.position,
        steps=args.steps,
        lr=args.lr,
        max_delta=args.max_delta,
        cameras=tuple(part.strip() for part in args.cameras.split(",") if part.strip()),
        frames=tuple(int(part.strip()) for part in args.frames.split(",") if part.strip()),
    )

    run_dir = args.work_dir / "nusc_whitebox_patch_attack"
    adv_scene_dir = run_dir / "adv_scenes"
    adv_image_root = run_dir / "adv_images"
    vis_dir = run_dir / "visualizations"
    output_jsonl = run_dir / "attack_results.jsonl"
    for path in (run_dir, adv_scene_dir, adv_image_root, vis_dir):
        path.mkdir(parents=True, exist_ok=True)

    model = build_model(config, args.checkpoint, args.device)
    for param in model.autovla.vlm.parameters():
        param.requires_grad_(False)
    model.autovla.vlm.eval()
    model.autovla.vlm.model.gradient_checkpointing_enable()
    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    dataset = SFTDataset(
        {"json_dataset_path": str(args.scene_dir), "sensor_data_path": args.sensor_data_path},
        config["model"],
        processor,
        using_cot=config["model"].get("use_cot", False),
    )
    collator = DataCollator(
        processor=processor,
        ignore_index=config["model"]["tokens"]["ignore_index"],
        assistant_id=config["model"]["tokens"]["assistant_id"],
    )
    with output_jsonl.open("w") as result_file:
        for idx, (scene_path, _) in enumerate(dataset.scenes[: args.num_samples]):
            clean_scene = load_scene(scene_path)
            token = str(clean_scene.get("token", scene_path.stem))
            sample = dataset[idx]
            collated = collator([sample])

            clean_result = predict_scene(model, dataset, scene_path, args.sensor_data_path)
            clean_videos, _ = _scene_videos_from_paths(clean_scene, args.sensor_data_path, args.device)
            patch = optimize_patch(
                model=model,
                processor=processor,
                prompt_text=sample["text"],
                clean_videos=clean_videos,
                labels=collated["labels"],
                input_ids=collated["input_ids"],
                attention_mask=collated["attention_mask"],
                gt_trajectory=collated["gt_trajectory"],
                gt_action=collated["gt_action"],
                has_cot=collated["has_cot"],
                config=attack_cfg,
                device=args.device,
            )
            patched_videos, patched_frames = _apply_patch_to_videos(clean_videos, attack_cfg, patch)
            adv_scene_path, adv_scene = _write_adv_scene_and_images(
                clean_scene=clean_scene,
                sensor_root=args.sensor_data_path,
                patched_frames=patched_frames,
                output_scene_dir=adv_scene_dir,
                output_image_root=adv_image_root,
                attack_meta={
                    "patch_ratio": attack_cfg.patch_ratio,
                    "position": attack_cfg.position,
                    "steps": attack_cfg.steps,
                    "lr": attack_cfg.lr,
                    "max_delta": attack_cfg.max_delta,
                    "frames": attack_cfg.frames,
                },
            )

            adv_dataset = SFTDataset(
                {"json_dataset_path": str(adv_scene_dir), "sensor_data_path": str(adv_image_root)},
                config["model"],
                processor,
                using_cot=config["model"].get("use_cot", False),
            )
            adv_result = predict_scene(model, adv_dataset, adv_scene_path, str(adv_image_root))
            visualize_attack_sample(
                clean_scene=clean_scene,
                adv_scene=adv_scene,
                output_path=vis_dir / f"{token}.png",
                clean_sensor_root=args.sensor_data_path,
                adv_sensor_root=str(adv_image_root),
                clean_prediction=clean_result["trajectory"],
                adv_prediction=adv_result["trajectory"],
                title="AutoVLA nuScenes white-box patch attack",
            )
            record = {
                "token": token,
                "clean_scene": str(scene_path),
                "adv_scene": str(adv_scene_path),
                "attack": adv_scene["attack"],
                "visualization": str(vis_dir / f"{token}.png"),
                "clean_prediction": clean_result["trajectory"],
                "adv_prediction": adv_result["trajectory"],
                "clean_result": clean_result,
                "adv_result": adv_result,
            }
            result_file.write(json.dumps(record) + "\n")
            print(f"Wrote white-box attack sample: {token}")

    print(f"Results: {output_jsonl}")


if __name__ == "__main__":
    main()
