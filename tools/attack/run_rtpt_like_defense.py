"""RTPT-like test-time defense for AutoVLA attacked samples.

This is not a literal port of the CLIP R-TPT code. Instead, it adapts the
same high-level idea to AutoVLA:
1. generate several light test-time augmentations of attacked images
2. run AutoVLA on each view
3. aggregate trajectories by consistency
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoProcessor

from dataset_utils.sft_dataset import SFTDataset
from tools.attack.patch_attack import CAMERA_KEYS, load_scene, resolve_image_path
from tools.attack.run_nusc_patch_attack import build_model, load_config, predict_scene
from tools.attack.visualization import visualize_attack_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attack_results_jsonl", type=Path, required=True)
    parser.add_argument("--clean_sensor_root", type=str, required=True)
    parser.add_argument("--adv_sensor_root", type=str, required=True)
    parser.add_argument("--work_dir", type=Path, default=Path("/mnt/indigo/tigersec/runw/workdirs/autovla_attack"))
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def select_top_shift_samples(results_jsonl: Path, num_samples: int) -> List[Dict]:
    rows = []
    for line in results_jsonl.open("r"):
        record = json.loads(line)
        clean = record["clean_prediction"]
        adv = record["adv_prediction"]
        distances = [
            math.dist(clean[i][:2], adv[i][:2]) for i in range(min(len(clean), len(adv)))
        ]
        record["mean_xy_shift_m"] = float(sum(distances) / len(distances))
        record["final_xy_shift_m"] = float(distances[-1])
        rows.append(record)
    rows.sort(key=lambda row: row["final_xy_shift_m"], reverse=True)
    return rows[:num_samples]


def _apply_variant(image: Image.Image, variant: str) -> Image.Image:
    if variant == "identity":
        return image.copy()
    if variant == "median3":
        return image.filter(ImageFilter.MedianFilter(size=3))
    if variant == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    if variant == "brightness_down":
        return ImageEnhance.Brightness(image).enhance(0.9)
    if variant == "contrast_up":
        return ImageEnhance.Contrast(image).enhance(1.1)
    raise ValueError(f"Unknown defense variant: {variant}")


def create_defended_scene(
    adv_scene_path: Path,
    adv_sensor_root: str,
    defense_image_root: Path,
    defense_scene_dir: Path,
    variant: str,
) -> Path:
    scene = load_scene(adv_scene_path)
    token = str(scene["token"])
    defended_scene = dict(scene)

    for camera_key in CAMERA_KEYS:
        new_paths = []
        for idx, rel_path in enumerate(scene[camera_key]):
            src = resolve_image_path(rel_path, adv_sensor_root)
            dst = defense_image_root / variant / token / camera_key / f"{idx:02d}{src.suffix or '.jpg'}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as image:
                defended = _apply_variant(image.convert("RGB"), variant)
                defended.save(dst)
            new_paths.append(str(dst.relative_to(defense_image_root / variant)))
        defended_scene[camera_key] = new_paths

    defended_scene["defense"] = {"name": "rtpt_like", "variant": variant}
    out_path = defense_scene_dir / variant / f"{token}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(defended_scene, indent=2))
    return out_path


def aggregate_predictions(predictions: Sequence[Dict]) -> Dict:
    valid = [pred for pred in predictions if pred["trajectory"] is not None]
    if not valid:
        return {
            "trajectory": None,
            "text": None,
            "error": "no_valid_defense_prediction",
            "selected_variant": None,
            "all_predictions": predictions,
        }

    trajs = [np.asarray(pred["trajectory"], dtype=float) for pred in valid]
    scores = []
    for i, traj in enumerate(trajs):
        others = [np.linalg.norm(traj[:, :2] - other[:, :2], axis=1).mean() for j, other in enumerate(trajs) if i != j]
        scores.append(float(np.mean(others)) if others else 0.0)
    best_idx = int(np.argmin(scores))
    best = valid[best_idx]
    return {
        "trajectory": best["trajectory"],
        "text": best["text"],
        "error": best["error"],
        "selected_variant": best["variant"],
        "all_predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = build_model(config, args.checkpoint, args.device)
    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)

    selected_records = select_top_shift_samples(args.attack_results_jsonl, args.num_samples)
    run_dir = args.work_dir / "rtpt_like_defense"
    defense_image_root = run_dir / "defended_images"
    defense_scene_dir = run_dir / "defended_scenes"
    vis_dir = run_dir / "visualizations"
    output_jsonl = run_dir / "defense_results.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    defense_image_root.mkdir(parents=True, exist_ok=True)
    defense_scene_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    variants = ["identity", "median3", "blur", "brightness_down", "contrast_up"]

    with output_jsonl.open("w") as out:
        for record in selected_records:
            token = record["token"]
            clean_scene = load_scene(Path(record["clean_scene"]))
            adv_scene = load_scene(Path(record["adv_scene"]))
            predictions = []

            for variant in variants:
                defended_scene_path = create_defended_scene(
                    adv_scene_path=Path(record["adv_scene"]),
                    adv_sensor_root=args.adv_sensor_root,
                    defense_image_root=defense_image_root,
                    defense_scene_dir=defense_scene_dir,
                    variant=variant,
                )
                dataset = SFTDataset(
                    {
                        "json_dataset_path": str(defended_scene_path.parent),
                        "sensor_data_path": str(defense_image_root / variant),
                    },
                    config["model"],
                    processor,
                    using_cot=config["model"].get("use_cot", False),
                )
                pred = predict_scene(model, dataset, defended_scene_path, str(defense_image_root / variant))
                pred["variant"] = variant
                predictions.append(pred)

            defended_result = aggregate_predictions(predictions)
            record["defended_result"] = defended_result
            record["defended_prediction"] = defended_result["trajectory"]
            record["defense_name"] = "rtpt_like"

            visualize_attack_sample(
                clean_scene=clean_scene,
                adv_scene=adv_scene,
                output_path=vis_dir / f"{token}.png",
                clean_sensor_root=args.clean_sensor_root,
                adv_sensor_root=args.adv_sensor_root,
                clean_prediction=record["clean_prediction"],
                adv_prediction=record["adv_prediction"],
                defended_prediction=record["defended_prediction"],
                title="AutoVLA nuScenes attack + RTPT-like defense",
            )
            record["defense_visualization"] = str(vis_dir / f"{token}.png")
            out.write(json.dumps(record) + "\n")
            print(f"Defended {token} with {defended_result['selected_variant']}")

    print(f"Defense results: {output_jsonl}")


if __name__ == "__main__":
    main()
