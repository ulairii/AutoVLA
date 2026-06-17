#!/bin/bash
#SBATCH -J autovla_wb_patch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be launched with sbatch."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; are you on a GPU node?"
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Project/AutoVLA}"
if [[ -z "${SCRATCH_ROOT:-}" ]]; then
  if [[ "${USER:-}" == "coder" ]]; then
    SCRATCH_ROOT="/scratch/runw"
  else
    SCRATCH_ROOT="/scratch/$USER"
  fi
fi
AUTOVLA_ROOT="${AUTOVLA_ROOT:-${SCRATCH_ROOT}/autovla_attack}"
ENV_PREFIX="${ENV_PREFIX:-${SCRATCH_ROOT}/envs/autovla_codeclean}"
HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/.hf_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/.pip_cache}"
TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp/${SLURM_JOB_ID}}"

mkdir -p "${AUTOVLA_ROOT}/logs" "${HF_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}"
mkdir -p "${PROJECT_ROOT}/logs"

export HF_HOME
export PIP_CACHE_DIR
export TMPDIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDA_BIN="${CONDA_BIN:-/home/runw/miniconda/bin/conda}"
if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "conda not found at ${CONDA_BIN}"
  exit 1
fi

"${CONDA_BIN}" shell.bash hook >/dev/null 2>&1
eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${ENV_PREFIX}"

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/eval/qwen2.5-vl-3B-nusc-sft-autovla-attack.yaml}"
SCENE_DIR="${SCENE_DIR:-${AUTOVLA_ROOT}/nuscenes_val}"
SENSOR_DATA_PATH="${SENSOR_DATA_PATH:-/mnt/indigo/tigersec/runw/datasets/nuscenes/data_full/nuscenes}"
CHECKPOINT="${CHECKPOINT:-${AUTOVLA_ROOT}/models/autovla_ckpt/AutoVLA_PDMS_89.ckpt}"

for required in "${CONFIG}" "${SCENE_DIR}" "${CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}"
    exit 1
  fi
done

echo "=== AutoVLA white-box patch attack ==="
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "env_prefix=${ENV_PREFIX}"
echo "work_dir=${AUTOVLA_ROOT}"
echo "scene_dir=${SCENE_DIR}"
echo "checkpoint=${CHECKPOINT}"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

MONITOR_INTERVAL=120
monitor_loop() {
  while true; do
    echo "--- monitor $(date '+%F %T') ---"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    ps -u "$USER" -o pid,pcpu,pmem,etime,cmd --sort=-pmem | head -n 8
    sleep "${MONITOR_INTERVAL}"
  done
}

monitor_loop &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd "${PROJECT_ROOT}"

python - <<'PY'
import os
from pathlib import Path
from transformers import AutoProcessor
import yaml

config = yaml.safe_load(Path(os.environ["CONFIG"]).read_text())
AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
print("HF cache warm check passed.")
PY

python tools/attack/run_nusc_whitebox_patch_attack.py \
  --config "${CONFIG}" \
  --scene_dir "${SCENE_DIR}" \
  --sensor_data_path "${SENSOR_DATA_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --work_dir "${AUTOVLA_ROOT}" \
  --device cuda:0 \
  --num_samples 10 \
  --steps 10 \
  --lr 0.01 \
  --max_delta 0.08 \
  --downscale_short_edge 160 \
  --cameras front_camera_paths \
  --frames 3

echo "Done. Results under ${AUTOVLA_ROOT}/nusc_whitebox_patch_attack"
