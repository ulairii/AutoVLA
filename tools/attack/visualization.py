"""Visualization helpers for AutoVLA attack samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tools.attack.patch_attack import CAMERA_KEYS, resolve_image_path


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _trajectory_plot_xy(trajectory):
    """Return display coordinates with forward up and right positive."""
    arr = np.asarray(trajectory, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
        return None
    right = -arr[:, 1]
    forward = arr[:, 0]
    return right, forward


def _draw_trajectory(ax, trajectory, label: str, color: str, linestyle: str = "-") -> None:
    if trajectory is None:
        return
    plot_xy = _trajectory_plot_xy(trajectory)
    if plot_xy is None:
        return
    right, forward = plot_xy
    ax.plot(right, forward, linestyle=linestyle, marker="o", label=label, color=color, linewidth=2)


def _set_trajectory_limits(ax, trajectories: Sequence[Sequence[Sequence[float]] | np.ndarray | None]) -> None:
    points = []
    for trajectory in trajectories:
        if trajectory is None:
            continue
        arr = np.asarray(trajectory, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
            continue
        points.append(np.stack([-arr[:, 1], arr[:, 0]], axis=1))
    if not points:
        return
    all_points = np.concatenate(points, axis=0)
    xs = all_points[:, 0]
    ys = all_points[:, 1]
    x_pad = max(1.0, float(xs.max() - xs.min()) * 0.15)
    y_pad = max(1.0, float(ys.max() - ys.min()) * 0.15)
    ax.set_xlim(float(xs.min() - x_pad), float(xs.max() + x_pad))
    ax.set_ylim(float(ys.min() - y_pad), float(ys.max() + y_pad))


def _set_fixed_trajectory_limits(ax, xlim=(-15.0, 15.0), ylim=(0.0, 40.0)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def visualize_attack_sample(
    clean_scene: Mapping,
    adv_scene: Mapping,
    output_path: Path,
    clean_sensor_root: str | None,
    adv_sensor_root: str | None,
    clean_prediction=None,
    adv_prediction=None,
    defended_prediction=None,
    title: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), gridspec_kw={"height_ratios": [1, 1, 0.55, 1.15]})

    camera_labels = {
        "front_left_camera_paths": "front-left",
        "front_camera_paths": "front",
        "front_right_camera_paths": "front-right",
    }
    ordered_cameras = ("front_left_camera_paths", "front_camera_paths", "front_right_camera_paths")

    for col, camera_key in enumerate(ordered_cameras):
        clean_paths = clean_scene.get(camera_key) or []
        adv_paths = adv_scene.get(camera_key) or []
        frame_idx = min(3, len(clean_paths) - 1, len(adv_paths) - 1)

        for row, scene, paths, sensor_root, prefix in (
            (0, clean_scene, clean_paths, clean_sensor_root, "clean"),
            (1, adv_scene, adv_paths, adv_sensor_root, "adv"),
        ):
            ax = axes[row, col]
            ax.axis("off")
            if frame_idx < 0:
                ax.text(0.5, 0.5, "missing image", ha="center", va="center")
                continue
            image_path = resolve_image_path(paths[frame_idx], sensor_root)
            ax.imshow(_load_image(image_path))
            ax.set_title(f"{prefix} {camera_labels[camera_key]}")

    text_ax = axes[2, 0]
    text_ax.axis("off")
    meta = [
        f"token: {clean_scene.get('token', '')}",
        f"instruction: {clean_scene.get('instruction', '')}",
        f"velocity: {clean_scene.get('velocity', '')}",
        f"attack: {adv_scene.get('attack', {}).get('pattern', '')}",
        f"patch ratio: {adv_scene.get('attack', {}).get('patch_ratio', '')}",
    ]
    text_ax.text(0, 1, "\n".join(meta), va="top", fontsize=11)

    diff_ax = axes[2, 1]
    diff_ax.axis("off")
    diff_text = "No decoded prediction trajectory."
    if clean_prediction is not None and adv_prediction is not None:
        clean_arr = np.asarray(clean_prediction, dtype=float)
        adv_arr = np.asarray(adv_prediction, dtype=float)
        n = min(len(clean_arr), len(adv_arr))
        if n:
            displacement = np.linalg.norm(clean_arr[:n, :2] - adv_arr[:n, :2], axis=1)
            final_delta = adv_arr[n - 1, :2] - clean_arr[n - 1, :2]
            diff_text = (
                f"prediction shift\nmean: {float(displacement.mean()):.3f} m\n"
                f"final: {float(displacement[-1]):.3f} m\n"
                f"final forward: {float(final_delta[0]):+.3f} m\n"
                f"final right: {float(-final_delta[1]):+.3f} m\n"
            ) + "\n".join(
                f"t{i + 1}: {value:.3f} m" for i, value in enumerate(displacement[:6])
            )
            if defended_prediction is not None:
                def_arr = np.asarray(defended_prediction, dtype=float)
                m = min(len(clean_arr), len(def_arr))
                if m:
                    defended_disp = np.linalg.norm(clean_arr[:m, :2] - def_arr[:m, :2], axis=1)
                    diff_text += (
                        f"\ndef mean: {float(defended_disp.mean()):.3f} m"
                        f"\ndef final: {float(defended_disp[-1]):.3f} m"
                    )
    diff_ax.text(0, 1, diff_text, va="top", fontsize=11)

    legend_ax = axes[2, 2]
    legend_ax.axis("off")
    legend_ax.text(
        0,
        1,
        "trajectory panels\nshow GT + one prediction each",
        va="top",
        fontsize=11,
    )

    trajectory_specs = [
        ("clean trajectory", clean_prediction, "#1f77b4", "-"),
        ("attack trajectory", adv_prediction, "#d62728", "--"),
        ("defended trajectory", defended_prediction, "#2ca02c", "-."),
    ]
    for col, (panel_title, prediction, color, linestyle) in enumerate(trajectory_specs):
        traj_ax = axes[3, col]
        traj_ax.set_title(panel_title)
        traj_ax.set_aspect("equal", adjustable="box")
        traj_ax.grid(True, linewidth=0.4)
        traj_ax.axhline(0, color="0.8", linewidth=1)
        traj_ax.axvline(0, color="0.8", linewidth=1)
        _draw_trajectory(traj_ax, clean_scene.get("gt_trajectory"), "gt", "#222222")
        _draw_trajectory(traj_ax, prediction, "pred", color, linestyle)
        _set_fixed_trajectory_limits(traj_ax)
        traj_ax.set_xlabel("right (m)")
        if col == 0:
            traj_ax.set_ylabel("forward (m)")
        traj_ax.legend(loc="best")

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
