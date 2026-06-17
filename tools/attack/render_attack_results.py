"""Render AutoVLA attack visualizations from a saved JSONL result file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.attack.patch_attack import load_scene
from tools.attack.visualization import visualize_attack_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_jsonl", type=Path, required=True)
    parser.add_argument("--clean_sensor_root", type=str, required=True)
    parser.add_argument("--adv_sensor_root", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.results_jsonl.open("r") as f:
        for line in f:
            record = json.loads(line)
            clean_scene = load_scene(Path(record["clean_scene"]))
            adv_scene = load_scene(Path(record["adv_scene"]))
            token = str(record["token"])
            visualize_attack_sample(
                clean_scene=clean_scene,
                adv_scene=adv_scene,
                output_path=args.output_dir / f"{token}.png",
                clean_sensor_root=args.clean_sensor_root,
                adv_sensor_root=args.adv_sensor_root,
                clean_prediction=record.get("clean_prediction"),
                adv_prediction=record.get("adv_prediction"),
                defended_prediction=record.get("defended_prediction"),
                title="AutoVLA nuScenes patch attack",
            )
            print(f"Rendered {token}")


if __name__ == "__main__":
    main()
