"""Black-box visual patch attacks for AutoVLA nuScenes samples.

The implementation is intentionally model-agnostic: it edits the camera frames
referenced by a preprocessed AutoVLA scene JSON and returns a new scene JSON
that points to adversarial copies of those frames.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from PIL import Image, ImageDraw


CAMERA_KEYS = (
    "front_camera_paths",
    "front_left_camera_paths",
    "front_right_camera_paths",
)


@dataclass(frozen=True)
class PatchAttackConfig:
    pattern: str = "checkerboard"
    patch_ratio: float = 0.18
    position: str = "bottom_center"
    opacity: float = 1.0
    cameras: Tuple[str, ...] = CAMERA_KEYS
    frames: str = "all"


def load_scene(scene_path: Path) -> Dict:
    with scene_path.open("r") as f:
        return json.load(f)


def save_scene(scene: Mapping, scene_path: Path) -> None:
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    with scene_path.open("w") as f:
        json.dump(scene, f, indent=2)


def resolve_image_path(path_value: str, sensor_data_path: str | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute() or not sensor_data_path:
        return path
    return Path(sensor_data_path) / path


def frame_indices(num_frames: int, frames: str) -> List[int]:
    if frames == "all":
        return list(range(num_frames))
    if frames == "last":
        return [num_frames - 1]
    if frames == "first":
        return [0]
    return [int(item) for item in frames.split(",") if item.strip()]


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


def _draw_patch(image: Image.Image, config: PatchAttackConfig) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = _patch_box(base.width, base.height, config.patch_ratio, config.position)
    alpha = max(0, min(255, int(config.opacity * 255)))

    if config.pattern == "black":
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, alpha))
    elif config.pattern == "white":
        draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, alpha))
    elif config.pattern == "red":
        draw.rectangle((x0, y0, x1, y1), fill=(255, 0, 0, alpha))
    elif config.pattern == "checkerboard":
        cell = max(4, min(x1 - x0, y1 - y0) // 8)
        for yy in range(y0, y1, cell):
            for xx in range(x0, x1, cell):
                color = (255, 255, 255, alpha) if ((xx - x0) // cell + (yy - y0) // cell) % 2 else (0, 0, 0, alpha)
                draw.rectangle((xx, yy, min(xx + cell, x1), min(yy + cell, y1)), fill=color)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0, alpha), width=max(2, cell // 4))
    else:
        raise ValueError(f"Unknown patch pattern: {config.pattern}")

    return Image.alpha_composite(base, overlay).convert(image.mode)


def _relative_to_output(path: Path, output_image_root: Path) -> Path:
    try:
        return path.relative_to(output_image_root)
    except ValueError:
        return Path(path.name)


def make_attacked_scene(
    scene_path: Path,
    output_scene_dir: Path,
    output_image_root: Path,
    sensor_data_path: str | None,
    config: PatchAttackConfig,
) -> Tuple[Path, Dict]:
    """Create an adversarial copy of a scene JSON and its referenced images."""

    scene = load_scene(scene_path)
    adv_scene: MutableMapping = copy.deepcopy(scene)
    token = str(scene.get("token", scene_path.stem))
    attacked_images: List[Dict[str, str]] = []

    for camera_key in config.cameras:
        image_paths = list(scene.get(camera_key) or [])
        if not image_paths:
            continue
        indices = set(frame_indices(len(image_paths), config.frames))
        new_paths: List[str] = []

        for idx, image_path_value in enumerate(image_paths):
            source_path = resolve_image_path(image_path_value, sensor_data_path)
            suffix = source_path.suffix or ".jpg"
            adv_path = output_image_root / token / camera_key / f"{idx:02d}{suffix}"
            adv_path.parent.mkdir(parents=True, exist_ok=True)

            if idx in indices:
                with Image.open(source_path) as image:
                    attacked = _draw_patch(image, config)
                    attacked.save(adv_path)
                attacked_images.append(
                    {
                        "camera": camera_key,
                        "frame": str(idx),
                        "source": str(source_path),
                        "adversarial": str(adv_path),
                    }
                )
            else:
                shutil.copy2(source_path, adv_path)

            new_paths.append(str(_relative_to_output(adv_path, output_image_root)))

        adv_scene[camera_key] = new_paths

    adv_scene["attack"] = {
        "name": "visual_patch",
        "pattern": config.pattern,
        "patch_ratio": config.patch_ratio,
        "position": config.position,
        "opacity": config.opacity,
        "frames": config.frames,
        "attacked_images": attacked_images,
    }

    output_scene_path = output_scene_dir / f"{token}.json"
    save_scene(adv_scene, output_scene_path)
    return output_scene_path, dict(adv_scene)


def iter_scene_paths(scene_dir: Path, limit: int | None = None) -> Iterable[Path]:
    count = 0
    for scene_path in sorted(scene_dir.glob("*.json")):
        if limit is not None and count >= limit:
            break
        yield scene_path
        count += 1
