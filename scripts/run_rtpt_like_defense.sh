#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/eval/qwen2.5-vl-3B-nusc-sft-autovla-attack.yaml}"
CHECKPOINT="${CHECKPOINT:-/mnt/indigo/tigersec/runw/workdirs/autovla_attack/models/autovla_ckpt/AutoVLA_PDMS_89.ckpt}"
ATTACK_RESULTS_JSONL="${ATTACK_RESULTS_JSONL:-/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nusc_patch_attack/attack_results.jsonl}"
CLEAN_SENSOR_ROOT="${CLEAN_SENSOR_ROOT:-/mnt/indigo/tigersec/runw/datasets/nuscenes/data_full/nuscenes}"
ADV_SENSOR_ROOT="${ADV_SENSOR_ROOT:-/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nusc_patch_attack/adv_images}"
WORK_DIR="${WORK_DIR:-/mnt/indigo/tigersec/runw/workdirs/autovla_attack}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
DEVICE="${DEVICE:-cuda:0}"

python "${PROJECT_ROOT}/tools/attack/run_rtpt_like_defense.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --attack_results_jsonl "${ATTACK_RESULTS_JSONL}" \
  --clean_sensor_root "${CLEAN_SENSOR_ROOT}" \
  --adv_sensor_root "${ADV_SENSOR_ROOT}" \
  --work_dir "${WORK_DIR}" \
  --num_samples "${NUM_SAMPLES}" \
  --device "${DEVICE}" \
  "$@"
