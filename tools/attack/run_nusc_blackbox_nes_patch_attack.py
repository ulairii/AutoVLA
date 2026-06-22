"""Black-box NES patch attack for AutoVLA on nuScenes.

This ports the query-based NES idea from VLM patch-attack work to AutoVLA while
keeping AutoVLA's default multi-camera, four-frame prompt. The model is queried
only through normal inference; no gradients flow through AutoVLA.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoProcessor

from dataset_utils.sft_dataset import SFTDataset
from tools.attack.patch_attack import CAMERA_KEYS, frame_indices, load_scene, resolve_image_path, save_scene
from tools.attack.run_nusc_patch_attack import build_features, build_model
from tools.attack.visualization import visualize_attack_sample


@dataclass(frozen=True)
class NESPatchConfig:
    patch_ratio: float = 0.18
    position: str = "bottom_center"
    cameras: Tuple[str, ...] = ("front_camera_paths",)
    frames: str = "all"
    steps: int = 20
    directions: int = 8
    sigma: float = 0.10
    lr: float = 0.08
    eot_samples: int = 1
    jitter_px: int = 4
    tv_lambda: float = 0.001
    objective: str = "trajectory_shift"
    target_behavior: str = "right_shift"
    target_lateral_offset: float = 3.5
    target_speed_scale: float = 0.35
    contrastive_tau: float = 0.20
    contrastive_weight: float = 1.0
    target_l2_weight: float = 0.25
    shift_loss_weight: float = 0.25
    md_forward_min: float = 0.0
    md_forward_max: float = 40.0
    md_lateral_min: float = -15.0
    md_lateral_max: float = 15.0
    md_heading_min: float = -3.141592653589793
    md_heading_max: float = 3.141592653589793
    md_forward_weight: float = 1.0
    md_lateral_weight: float = 2.0
    md_heading_weight: float = 0.25
    mean_shift_weight: float = 1.0
    final_shift_weight: float = 2.0
    max_shift_weight: float = 0.5
    seed: int = 0


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
    parser.add_argument("--cameras", type=str, default="front_camera_paths")
    parser.add_argument("--frames", default="all", help="'all', 'first', 'last', or comma-separated frame indices.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--eot_samples", type=int, default=1)
    parser.add_argument("--jitter_px", type=int, default=4)
    parser.add_argument("--tv_lambda", type=float, default=0.001)
    parser.add_argument(
        "--objective",
        choices=["trajectory_shift", "lateral_shift", "max_discrepancy", "contrastive_target", "hybrid_contrastive"],
        default="trajectory_shift",
    )
    parser.add_argument("--target_behavior", choices=["stop", "slowdown", "left_shift", "right_shift"], default="right_shift")
    parser.add_argument("--target_lateral_offset", type=float, default=3.5)
    parser.add_argument("--target_speed_scale", type=float, default=0.35)
    parser.add_argument("--contrastive_tau", type=float, default=0.20)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--target_l2_weight", type=float, default=0.25)
    parser.add_argument("--shift_loss_weight", type=float, default=0.25)
    parser.add_argument("--md_forward_min", type=float, default=0.0)
    parser.add_argument("--md_forward_max", type=float, default=40.0)
    parser.add_argument("--md_lateral_min", type=float, default=-15.0)
    parser.add_argument("--md_lateral_max", type=float, default=15.0)
    parser.add_argument("--md_heading_min", type=float, default=-3.141592653589793)
    parser.add_argument("--md_heading_max", type=float, default=3.141592653589793)
    parser.add_argument("--md_forward_weight", type=float, default=1.0)
    parser.add_argument("--md_lateral_weight", type=float, default=2.0)
    parser.add_argument("--md_heading_weight", type=float, default=0.25)
    parser.add_argument("--mean_shift_weight", type=float, default=1.0)
    parser.add_argument("--final_shift_weight", type=float, default=2.0)
    parser.add_argument("--max_shift_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_name", type=str, default="nusc_blackbox_nes_patch_attack")
    return parser.parse_args()


def _patch_box(width: int, height: int, ratio: float, position: str, jitter_px: int = 0) -> Tuple[int, int, int, int]:
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

    if jitter_px > 0:
        x0 += random.randint(-jitter_px, jitter_px)
        y0 += random.randint(-jitter_px, jitter_px)
    x0 = max(0, min(width - patch_w, x0))
    y0 = max(0, min(height - patch_h, y0))
    return x0, y0, x0 + patch_w, y0 + patch_h


def _pil_to_unit_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _unit_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def _apply_patch_to_image(image: Image.Image, patch: torch.Tensor, config: NESPatchConfig, jitter: bool) -> Image.Image:
    base = _pil_to_unit_tensor(image)
    _, height, width = base.shape
    x0, y0, x1, y1 = _patch_box(
        width,
        height,
        config.patch_ratio,
        config.position,
        config.jitter_px if jitter else 0,
    )
    resized_patch = F.interpolate(
        patch.detach().cpu().unsqueeze(0),
        size=(y1 - y0, x1 - x0),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    base[:, y0:y1, x0:x1] = resized_patch
    return _unit_tensor_to_pil(base)


def _write_scene_with_patch(
    clean_scene: Mapping,
    sensor_root: str,
    patch: torch.Tensor,
    output_scene_dir: Path,
    output_image_root: Path,
    config: NESPatchConfig,
    jitter: bool = False,
) -> Tuple[Path, Dict]:
    token = str(clean_scene.get("token", "scene"))
    adv_scene: MutableMapping = copy.deepcopy(clean_scene)
    attacked_images: List[Dict[str, str]] = []
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    output_image_root.mkdir(parents=True, exist_ok=True)

    for camera_key in CAMERA_KEYS:
        image_paths = list(clean_scene.get(camera_key) or [])
        selected = set(frame_indices(len(image_paths), config.frames))
        new_paths: List[str] = []
        for frame_idx, image_path_value in enumerate(image_paths):
            src = resolve_image_path(image_path_value, sensor_root)
            suffix = src.suffix or ".jpg"
            dst = output_image_root / token / camera_key / f"{frame_idx:02d}{suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if camera_key in config.cameras and frame_idx in selected:
                with Image.open(src) as image:
                    patched = _apply_patch_to_image(image, patch, config, jitter=jitter)
                    patched.save(dst)
                attacked_images.append(
                    {
                        "camera": camera_key,
                        "frame": str(frame_idx),
                        "source": str(src),
                        "adversarial": str(dst),
                    }
                )
            else:
                shutil.copy2(src, dst)
            new_paths.append(str(dst.relative_to(output_image_root)))
        adv_scene[camera_key] = new_paths

    adv_scene["attack"] = {
        "name": "blackbox_nes_patch",
        "patch_ratio": config.patch_ratio,
        "position": config.position,
        "cameras": list(config.cameras),
        "frames": config.frames,
        "steps": config.steps,
        "directions": config.directions,
        "sigma": config.sigma,
        "lr": config.lr,
        "eot_samples": config.eot_samples,
        "jitter_px": config.jitter_px,
        "objective": config.objective,
        "target_behavior": config.target_behavior,
        "target_lateral_offset": config.target_lateral_offset,
        "target_speed_scale": config.target_speed_scale,
        "contrastive_tau": config.contrastive_tau,
        "attacked_images": attacked_images,
    }
    scene_path = output_scene_dir / f"{token}.json"
    save_scene(adv_scene, scene_path)
    return scene_path, dict(adv_scene)


def _trajectory_shift_loss(clean_trajectory, adv_trajectory, config: NESPatchConfig) -> float:
    if clean_trajectory is None or adv_trajectory is None:
        return 0.0
    clean = np.asarray(clean_trajectory, dtype=np.float32)
    adv = np.asarray(adv_trajectory, dtype=np.float32)
    if clean.ndim != 2 or adv.ndim != 2 or clean.shape[1] < 2 or adv.shape[1] < 2:
        return 0.0
    n = min(len(clean), len(adv))
    if n == 0:
        return 0.0
    displacement = np.linalg.norm(clean[:n, :2] - adv[:n, :2], axis=1)
    objective = (
        config.mean_shift_weight * float(displacement.mean())
        + config.final_shift_weight * float(displacement[-1])
        + config.max_shift_weight * float(displacement.max())
    )
    return -objective


def _lateral_shift_loss(clean_trajectory, adv_trajectory, config: NESPatchConfig) -> float:
    if clean_trajectory is None or adv_trajectory is None:
        return 0.0
    clean = np.asarray(clean_trajectory, dtype=np.float32)
    adv = np.asarray(adv_trajectory, dtype=np.float32)
    if clean.ndim != 2 or adv.ndim != 2 or clean.shape[1] < 2 or adv.shape[1] < 2:
        return 0.0
    n = min(len(clean), len(adv))
    if n == 0:
        return 0.0
    internal_left_delta = adv[:n, 1] - clean[:n, 1]
    if config.target_behavior == "right_shift":
        signed_lateral = -internal_left_delta
    elif config.target_behavior == "left_shift":
        signed_lateral = internal_left_delta
    else:
        signed_lateral = np.abs(internal_left_delta)
    objective = (
        config.mean_shift_weight * float(signed_lateral.mean())
        + config.final_shift_weight * float(signed_lateral[-1])
        + config.max_shift_weight * float(signed_lateral.max())
    )
    return -objective


def _heading_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _max_discrepancy_loss(clean_trajectory, adv_trajectory, config: NESPatchConfig) -> float:
    if clean_trajectory is None or adv_trajectory is None:
        return 0.0
    clean = np.asarray(clean_trajectory, dtype=np.float32)
    adv = np.asarray(adv_trajectory, dtype=np.float32)
    if clean.ndim != 2 or adv.ndim != 2 or clean.shape[1] < 2 or adv.shape[1] < 2:
        return 0.0
    n = min(len(clean), len(adv))
    if n == 0:
        return 0.0
    clean = clean[:n]
    adv = adv[:n]

    f_min, f_max = float(config.md_forward_min), float(config.md_forward_max)
    l_min, l_max = float(config.md_lateral_min), float(config.md_lateral_max)
    f_ref = clean[:, 0]
    l_ref = clean[:, 1]
    f_target = np.where(np.abs(f_ref - f_min) >= np.abs(f_ref - f_max), f_min, f_max)
    l_target = np.where(np.abs(l_ref - l_min) >= np.abs(l_ref - l_max), l_min, l_max)

    f_den = np.maximum(np.abs(f_target - f_ref), 1e-3)
    l_den = np.maximum(np.abs(l_target - l_ref), 1e-3)
    f_score = ((adv[:, 0] - f_ref) * np.sign(f_target - f_ref)) / f_den
    l_score = ((adv[:, 1] - l_ref) * np.sign(l_target - l_ref)) / l_den
    score = (
        config.md_forward_weight * f_score
        + config.md_lateral_weight * l_score
    )

    if clean.shape[1] >= 3 and adv.shape[1] >= 3 and config.md_heading_weight:
        h_min, h_max = float(config.md_heading_min), float(config.md_heading_max)
        h_ref = clean[:, 2]
        h_target = np.where(np.abs(_heading_delta(h_ref, h_min)) >= np.abs(_heading_delta(h_ref, h_max)), h_min, h_max)
        h_direction = np.sign(_heading_delta(h_target, h_ref))
        h_score = (_heading_delta(adv[:, 2], h_ref) * h_direction) / np.maximum(np.abs(_heading_delta(h_target, h_ref)), 1e-3)
        score = score + config.md_heading_weight * h_score

    objective = (
        config.mean_shift_weight * float(score.mean())
        + config.final_shift_weight * float(score[-1])
        + config.max_shift_weight * float(score.max())
    )
    return -objective


def _valid_trajectory(trajectory) -> bool:
    arr = np.asarray(trajectory, dtype=np.float32)
    return arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) > 0


def _target_trajectory(clean_trajectory, config: NESPatchConfig):
    if not _valid_trajectory(clean_trajectory):
        return None
    clean = np.asarray(clean_trajectory, dtype=np.float32)
    target = clean.copy()
    ramp = np.linspace(0.0, 1.0, len(target), dtype=np.float32)

    if config.target_behavior == "stop":
        target[:, :2] = 0.0
    elif config.target_behavior == "slowdown":
        target[:, :2] *= float(config.target_speed_scale)
    elif config.target_behavior == "left_shift":
        target[:, 1] += float(config.target_lateral_offset) * ramp
    elif config.target_behavior == "right_shift":
        target[:, 1] -= float(config.target_lateral_offset) * ramp
    else:
        raise ValueError(f"Unknown target behavior: {config.target_behavior}")
    return target


def _trajectory_embedding(trajectory) -> np.ndarray:
    arr = np.asarray(trajectory, dtype=np.float32)
    xy = arr[:, :2]
    if len(xy) > 1:
        delta = np.diff(xy, axis=0, prepend=xy[:1])
        feat = np.concatenate([xy.reshape(-1), delta.reshape(-1)], axis=0)
    else:
        feat = xy.reshape(-1)
    norm = float(np.linalg.norm(feat))
    if norm < 1e-6:
        return feat
    return feat / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def _target_distance_loss(target_trajectory, adv_trajectory) -> float:
    if target_trajectory is None or not _valid_trajectory(adv_trajectory):
        return 0.0
    target = np.asarray(target_trajectory, dtype=np.float32)
    adv = np.asarray(adv_trajectory, dtype=np.float32)
    n = min(len(target), len(adv))
    if n == 0:
        return 0.0
    displacement = np.linalg.norm(target[:n, :2] - adv[:n, :2], axis=1)
    return float(displacement.mean() + displacement[-1])


def _contrastive_target_loss(clean_trajectory, adv_trajectory, config: NESPatchConfig) -> float:
    target = _target_trajectory(clean_trajectory, config)
    if target is None or not _valid_trajectory(adv_trajectory):
        return 0.0
    adv_emb = _trajectory_embedding(adv_trajectory)
    clean_emb = _trajectory_embedding(clean_trajectory)
    target_emb = _trajectory_embedding(target)
    tau = max(float(config.contrastive_tau), 1e-6)
    target_logit = _cosine(adv_emb, target_emb) / tau
    clean_logit = _cosine(adv_emb, clean_emb) / tau
    max_logit = max(target_logit, clean_logit)
    log_denom = max_logit + np.log(np.exp(target_logit - max_logit) + np.exp(clean_logit - max_logit))
    nce = -target_logit + float(log_denom)
    return nce + config.target_l2_weight * _target_distance_loss(target, adv_trajectory)


def _attack_loss(clean_trajectory, adv_trajectory, config: NESPatchConfig) -> float:
    shift_loss = _trajectory_shift_loss(clean_trajectory, adv_trajectory, config)
    if config.objective == "trajectory_shift":
        return shift_loss
    if config.objective == "lateral_shift":
        return _lateral_shift_loss(clean_trajectory, adv_trajectory, config)
    if config.objective == "max_discrepancy":
        return _max_discrepancy_loss(clean_trajectory, adv_trajectory, config)
    contrastive_loss = _contrastive_target_loss(clean_trajectory, adv_trajectory, config)
    if config.objective == "contrastive_target":
        return config.contrastive_weight * contrastive_loss
    if config.objective == "hybrid_contrastive":
        return config.contrastive_weight * contrastive_loss + config.shift_loss_weight * shift_loss
    raise ValueError(f"Unknown objective: {config.objective}")


def _trajectory_shift_metrics(clean_trajectory, adv_trajectory) -> Dict[str, float]:
    if clean_trajectory is None or adv_trajectory is None:
        return {"mean_shift": 0.0, "final_shift": 0.0, "max_shift": 0.0}
    clean = np.asarray(clean_trajectory, dtype=np.float32)
    adv = np.asarray(adv_trajectory, dtype=np.float32)
    if clean.ndim != 2 or adv.ndim != 2 or clean.shape[1] < 2 or adv.shape[1] < 2:
        return {"mean_shift": 0.0, "final_shift": 0.0, "max_shift": 0.0}
    n = min(len(clean), len(adv))
    if n == 0:
        return {"mean_shift": 0.0, "final_shift": 0.0, "max_shift": 0.0}
    displacement = np.linalg.norm(clean[:n, :2] - adv[:n, :2], axis=1)
    return {
        "mean_shift": float(displacement.mean()),
        "final_shift": float(displacement[-1]),
        "max_shift": float(displacement.max()),
    }


def _target_metrics(clean_trajectory, adv_trajectory, config: NESPatchConfig) -> Dict[str, float]:
    target = _target_trajectory(clean_trajectory, config)
    if target is None or not _valid_trajectory(adv_trajectory):
        return {"target_mean_dist": 0.0, "target_final_dist": 0.0, "target_cosine": 0.0, "clean_cosine": 0.0}
    adv = np.asarray(adv_trajectory, dtype=np.float32)
    n = min(len(target), len(adv))
    if n == 0:
        return {"target_mean_dist": 0.0, "target_final_dist": 0.0, "target_cosine": 0.0, "clean_cosine": 0.0}
    displacement = np.linalg.norm(target[:n, :2] - adv[:n, :2], axis=1)
    adv_emb = _trajectory_embedding(adv[:n])
    target_emb = _trajectory_embedding(target[:n])
    clean_emb = _trajectory_embedding(np.asarray(clean_trajectory, dtype=np.float32)[:n])
    return {
        "target_mean_dist": float(displacement.mean()),
        "target_final_dist": float(displacement[-1]),
        "target_cosine": _cosine(adv_emb, target_emb),
        "clean_cosine": _cosine(adv_emb, clean_emb),
    }


def _component_shift_metrics(clean_trajectory, adv_trajectory) -> Dict[str, float]:
    if clean_trajectory is None or adv_trajectory is None:
        return {
            "final_forward_delta": 0.0,
            "final_right_delta": 0.0,
            "final_abs_lateral_delta": 0.0,
            "mean_abs_lateral_delta": 0.0,
            "max_abs_lateral_delta": 0.0,
        }
    clean = np.asarray(clean_trajectory, dtype=np.float32)
    adv = np.asarray(adv_trajectory, dtype=np.float32)
    if clean.ndim != 2 or adv.ndim != 2 or clean.shape[1] < 2 or adv.shape[1] < 2:
        return {
            "final_forward_delta": 0.0,
            "final_right_delta": 0.0,
            "final_abs_lateral_delta": 0.0,
            "mean_abs_lateral_delta": 0.0,
            "max_abs_lateral_delta": 0.0,
        }
    n = min(len(clean), len(adv))
    if n == 0:
        return {
            "final_forward_delta": 0.0,
            "final_right_delta": 0.0,
            "final_abs_lateral_delta": 0.0,
            "mean_abs_lateral_delta": 0.0,
            "max_abs_lateral_delta": 0.0,
        }
    delta = adv[:n, :2] - clean[:n, :2]
    right_delta = -delta[:, 1]
    abs_lateral = np.abs(right_delta)
    return {
        "final_forward_delta": float(delta[-1, 0]),
        "final_right_delta": float(right_delta[-1]),
        "final_abs_lateral_delta": float(abs_lateral[-1]),
        "mean_abs_lateral_delta": float(abs_lateral.mean()),
        "max_abs_lateral_delta": float(abs_lateral.max()),
    }


def _all_metrics(clean_trajectory, adv_trajectory, config: NESPatchConfig) -> Dict[str, float]:
    return {
        **_trajectory_shift_metrics(clean_trajectory, adv_trajectory),
        **_target_metrics(clean_trajectory, adv_trajectory, config),
        **_component_shift_metrics(clean_trajectory, adv_trajectory),
    }


def _tv_grad(patch: torch.Tensor) -> torch.Tensor:
    p = patch.detach().clone().requires_grad_(True)
    dh = p[:, :, 1:] - p[:, :, :-1]
    dw = p[:, 1:, :] - p[:, :-1, :]
    tv = dh.abs().mean() + dw.abs().mean()
    tv.backward()
    return p.grad.detach()


def _query_loss(
    model,
    dataset: SFTDataset,
    clean_scene: Mapping,
    clean_trajectory,
    sensor_root: str,
    patch: torch.Tensor,
    query_scene_dir: Path,
    query_image_root: Path,
    config: NESPatchConfig,
    jitter: bool,
) -> Tuple[float, Dict, Path, Dict]:
    query_scene_path, query_scene = _write_scene_with_patch(
        clean_scene=clean_scene,
        sensor_root=sensor_root,
        patch=patch,
        output_scene_dir=query_scene_dir,
        output_image_root=query_image_root,
        config=config,
        jitter=jitter,
    )
    result = predict_scene_deterministic(model, dataset, query_scene_path, str(query_image_root))
    loss = _attack_loss(clean_trajectory, result["trajectory"], config)
    return loss, result, query_scene_path, query_scene


def predict_scene_deterministic(model, dataset: SFTDataset, scene_path: Path, sensor_data_path: str | None) -> Dict:
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
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
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


def optimize_patch_nes(
    model,
    dataset: SFTDataset,
    clean_scene: Mapping,
    clean_trajectory,
    sensor_root: str,
    query_scene_dir: Path,
    query_image_root: Path,
    config: NESPatchConfig,
    device: str,
) -> Tuple[torch.Tensor, Dict]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    patch = torch.clamp(torch.randn(3, 64, 64, device=device) * 0.25 + 0.5, 0.0, 1.0)
    total_queries = 0
    best_loss = float("inf")
    best_patch = patch.detach().clone()
    history = []
    current_lr = config.lr
    current_loss, current_result, _, _ = _query_loss(
        model,
        dataset,
        clean_scene,
        clean_trajectory,
        sensor_root,
        patch,
        query_scene_dir,
        query_image_root,
        config,
        jitter=False,
    )
    total_queries += 1
    best_loss = float(current_loss)
    best_patch = patch.detach().clone()
    best_metrics = _all_metrics(clean_trajectory, current_result["trajectory"], config)
    print(
        "NES init: "
        f"loss={best_loss:.4f}, mean_shift={best_metrics['mean_shift']:.3f}, "
        f"final_shift={best_metrics['final_shift']:.3f}, max_shift={best_metrics['max_shift']:.3f}, "
        f"target_final_dist={best_metrics['target_final_dist']:.3f}, "
        f"queries={total_queries}",
        flush=True,
    )

    for step in range(1, config.steps + 1):
        grad_est = torch.zeros_like(patch)
        step_loss = 0.0
        for _ in range(config.directions):
            noise = torch.randn_like(patch)
            p_plus = torch.clamp(patch + config.sigma * noise, 0.0, 1.0)
            p_minus = torch.clamp(patch - config.sigma * noise, 0.0, 1.0)

            plus_loss = 0.0
            minus_loss = 0.0
            for _ in range(config.eot_samples):
                lp, _, _, _ = _query_loss(
                    model,
                    dataset,
                    clean_scene,
                    clean_trajectory,
                    sensor_root,
                    p_plus,
                    query_scene_dir,
                    query_image_root,
                    config,
                    jitter=True,
                )
                lm, _, _, _ = _query_loss(
                    model,
                    dataset,
                    clean_scene,
                    clean_trajectory,
                    sensor_root,
                    p_minus,
                    query_scene_dir,
                    query_image_root,
                    config,
                    jitter=True,
                )
                plus_loss += lp
                minus_loss += lm
                total_queries += 2

            plus_loss /= float(config.eot_samples)
            minus_loss /= float(config.eot_samples)
            grad_est += (plus_loss - minus_loss) * noise
            step_loss += (plus_loss + minus_loss) / 2.0

        grad_est /= float(config.directions * config.sigma)
        if config.tv_lambda > 0:
            grad_est = grad_est + config.tv_lambda * _tv_grad(patch)
        patch = torch.clamp(patch - current_lr * grad_est, 0.0, 1.0)

        avg_loss = step_loss / float(config.directions)
        current_loss, current_result, _, _ = _query_loss(
            model,
            dataset,
            clean_scene,
            clean_trajectory,
            sensor_root,
            patch,
            query_scene_dir,
            query_image_root,
            config,
            jitter=False,
        )
        total_queries += 1
        current_metrics = _all_metrics(clean_trajectory, current_result["trajectory"], config)
        history.append(
            {
                "step": step,
                "nes_loss_estimate": float(avg_loss),
                "evaluated_loss": float(current_loss),
                **current_metrics,
                "queries": total_queries,
            }
        )
        if current_loss < best_loss:
            best_loss = float(current_loss)
            best_patch = patch.detach().clone()
            best_metrics = current_metrics
        if step in {max(1, int(0.5 * config.steps)), max(1, int(0.8 * config.steps))}:
            current_lr *= 0.5
        print(
            f"NES step {step}/{config.steps}: estimate={avg_loss:.4f}, "
            f"eval={current_loss:.4f}, best={best_loss:.4f}, "
            f"mean={current_metrics['mean_shift']:.3f}, final={current_metrics['final_shift']:.3f}, "
            f"max={current_metrics['max_shift']:.3f}, "
            f"target_final_dist={current_metrics['target_final_dist']:.3f}, "
            f"lr={current_lr:.5f}, queries={total_queries}",
            flush=True,
        )

    return best_patch.detach(), {
        "history": history,
        "best_loss": best_loss,
        "best_metrics": best_metrics,
        "queries": total_queries,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    attack_cfg = NESPatchConfig(
        patch_ratio=args.patch_ratio,
        position=args.position,
        cameras=tuple(part.strip() for part in args.cameras.split(",") if part.strip()),
        frames=args.frames,
        steps=args.steps,
        directions=args.directions,
        sigma=args.sigma,
        lr=args.lr,
        eot_samples=args.eot_samples,
        jitter_px=args.jitter_px,
        tv_lambda=args.tv_lambda,
        objective=args.objective,
        target_behavior=args.target_behavior,
        target_lateral_offset=args.target_lateral_offset,
        target_speed_scale=args.target_speed_scale,
        contrastive_tau=args.contrastive_tau,
        contrastive_weight=args.contrastive_weight,
        target_l2_weight=args.target_l2_weight,
        shift_loss_weight=args.shift_loss_weight,
        mean_shift_weight=args.mean_shift_weight,
        final_shift_weight=args.final_shift_weight,
        max_shift_weight=args.max_shift_weight,
        seed=args.seed,
    )

    run_dir = args.work_dir / args.output_name
    adv_scene_dir = run_dir / "adv_scenes"
    adv_image_root = run_dir / "adv_images"
    vis_dir = run_dir / "visualizations"
    patch_dir = run_dir / "patches"
    output_jsonl = run_dir / "attack_results.jsonl"
    for path in (run_dir, adv_scene_dir, adv_image_root, vis_dir, patch_dir):
        path.mkdir(parents=True, exist_ok=True)

    model = build_model(config, args.checkpoint, args.device)
    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    clean_dataset = SFTDataset(
        {"json_dataset_path": str(args.scene_dir), "sensor_data_path": args.sensor_data_path},
        config["model"],
        processor,
        using_cot=config["model"].get("use_cot", False),
    )

    with output_jsonl.open("w") as result_file, tempfile.TemporaryDirectory(prefix="autovla_nes_queries_") as tmp:
        tmp_root = Path(tmp)
        for sample_idx, (scene_path, _) in enumerate(clean_dataset.scenes[: args.num_samples]):
            clean_scene = load_scene(scene_path)
            token = str(clean_scene.get("token", scene_path.stem))
            print(f"=== NES black-box sample {sample_idx + 1}/{args.num_samples}: {token} ===", flush=True)
            clean_result = predict_scene_deterministic(model, clean_dataset, scene_path, args.sensor_data_path)

            sample_query_scene_dir = tmp_root / token / "scenes"
            sample_query_image_root = tmp_root / token / "images"
            patch, opt_meta = optimize_patch_nes(
                model=model,
                dataset=clean_dataset,
                clean_scene=clean_scene,
                clean_trajectory=clean_result["trajectory"],
                sensor_root=args.sensor_data_path,
                query_scene_dir=sample_query_scene_dir,
                query_image_root=sample_query_image_root,
                config=attack_cfg,
                device=args.device,
            )
            patch_path = patch_dir / f"{token}.pt"
            torch.save(patch.detach().cpu(), patch_path)
            _unit_tensor_to_pil(patch.detach().cpu()).save(patch_dir / f"{token}.png")

            adv_scene_path, adv_scene = _write_scene_with_patch(
                clean_scene=clean_scene,
                sensor_root=args.sensor_data_path,
                patch=patch,
                output_scene_dir=adv_scene_dir,
                output_image_root=adv_image_root,
                config=attack_cfg,
                jitter=False,
            )
            adv_dataset = SFTDataset(
                {"json_dataset_path": str(adv_scene_dir), "sensor_data_path": str(adv_image_root)},
                config["model"],
                processor,
                using_cot=config["model"].get("use_cot", False),
            )
            adv_result = predict_scene_deterministic(model, adv_dataset, adv_scene_path, str(adv_image_root))
            shift_loss = _attack_loss(clean_result["trajectory"], adv_result["trajectory"], attack_cfg)
            shift_metrics = _all_metrics(clean_result["trajectory"], adv_result["trajectory"], attack_cfg)

            visualize_attack_sample(
                clean_scene=clean_scene,
                adv_scene=adv_scene,
                output_path=vis_dir / f"{token}.png",
                clean_sensor_root=args.sensor_data_path,
                adv_sensor_root=str(adv_image_root),
                clean_prediction=clean_result["trajectory"],
                adv_prediction=adv_result["trajectory"],
                title="AutoVLA nuScenes black-box NES patch attack",
            )
            record = {
                "token": token,
                "clean_scene": str(scene_path),
                "adv_scene": str(adv_scene_path),
                "patch_tensor": str(patch_path),
                "patch_png": str(patch_dir / f"{token}.png"),
                "visualization": str(vis_dir / f"{token}.png"),
                "attack": adv_scene["attack"],
                "optimization": opt_meta,
                "final_shift_objective_loss": shift_loss,
                "shift_metrics": shift_metrics,
                "clean_prediction": clean_result["trajectory"],
                "adv_prediction": adv_result["trajectory"],
                "clean_result": clean_result,
                "adv_result": adv_result,
            }
            result_file.write(json.dumps(record) + "\n")
            result_file.flush()
            print(f"Wrote black-box NES attack sample: {token}", flush=True)

    print(f"Results: {output_jsonl}")
    print(f"Visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
