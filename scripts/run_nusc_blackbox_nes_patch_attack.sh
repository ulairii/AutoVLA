#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCENE_DIR="${SCENE_DIR:-/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nuscenes_val}"
SENSOR_DATA_PATH="${SENSOR_DATA_PATH:-/mnt/indigo/tigersec/runw/datasets/nuscenes/data_full/nuscenes}"
WORK_DIR="${WORK_DIR:-/mnt/indigo/tigersec/runw/workdirs/autovla_attack}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/eval/qwen2.5-vl-3B-nusc-sft-autovla-attack.yaml}"
CHECKPOINT="${CHECKPOINT:-${WORK_DIR}/models/autovla_ckpt/AutoVLA_PDMS_89.ckpt}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
DEVICE="${DEVICE:-cuda:0}"

ARGS=(
  --config "${CONFIG}"
  --scene_dir "${SCENE_DIR}"
  --sensor_data_path "${SENSOR_DATA_PATH}"
  --checkpoint "${CHECKPOINT}"
  --work_dir "${WORK_DIR}"
  --num_samples "${NUM_SAMPLES}"
  --device "${DEVICE}"
)

python "${PROJECT_ROOT}/tools/attack/run_nusc_blackbox_nes_patch_attack.py" "${ARGS[@]}" "$@"
