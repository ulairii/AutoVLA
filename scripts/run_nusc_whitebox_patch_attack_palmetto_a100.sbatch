#!/bin/bash
#SBATCH -J autovla_wb_patch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mri2020
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH --output=/project/mpese/tigersec/runw/workdirs/autovla_attack/palmetto_log/autovla_wb_patch_%j.out
#SBATCH --error=/project/mpese/tigersec/runw/workdirs/autovla_attack/palmetto_log/autovla_wb_patch_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be launched with sbatch."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; are you on a GPU node?"
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/runw/project/AutoVLA}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/runw}"
AUTOVLA_ROOT="${AUTOVLA_ROOT:-/project/mpese/tigersec/runw/workdirs/autovla_attack}"
ENV_PREFIX="${ENV_PREFIX:-${SCRATCH_ROOT}/envs/autovla_codeclean}"
HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/.pip_cache}"
TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp/${SLURM_JOB_ID}}"
PALMETTO_LOG_DIR="${PALMETTO_LOG_DIR:-${AUTOVLA_ROOT}/palmetto_log}"

mkdir -p "${PALMETTO_LOG_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}"

export HF_HOME
export PIP_CACHE_DIR
export TMPDIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONPATH="${PROJECT_ROOT}"

CONDA_BIN="${CONDA_BIN:-/home/runw/miniconda/bin/conda}"
if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "conda not found at ${CONDA_BIN}"
  exit 1
fi

set +u
if [[ -f /home/runw/miniconda/etc/profile.d/conda.sh ]]; then
  source /home/runw/miniconda/etc/profile.d/conda.sh
elif [[ -f /home/runw/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/runw/miniconda3/etc/profile.d/conda.sh
else
  echo "Could not locate conda.sh"
  exit 1
fi
conda activate "${ENV_PREFIX}"
set -u

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/eval/qwen2.5-vl-3B-nusc-sft-autovla-attack.yaml}"
SCENE_DIR="${SCENE_DIR:-${AUTOVLA_ROOT}/nuscenes_val}"
SENSOR_DATA_PATH="${SENSOR_DATA_PATH:-/project/mpese/tigersec/runw/datasets/nuscenes/data_full/nuscenes}"
CHECKPOINT="${CHECKPOINT:-${AUTOVLA_ROOT}/models/autovla_ckpt/AutoVLA_PDMS_89.ckpt}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${AUTOVLA_ROOT}/models/qwen/Qwen2.5-VL-3B-Instruct}"
SOURCE_PREFIX_OLD="${SOURCE_PREFIX_OLD:-/mnt/indigo/tigersec/runw/datasets/nuscenes/data_full/nuscenes}"
SOURCE_PREFIX_NEW="${SOURCE_PREFIX_NEW:-${SENSOR_DATA_PATH}}"
export CONFIG SCENE_DIR SENSOR_DATA_PATH CHECKPOINT QWEN_MODEL_PATH SOURCE_PREFIX_OLD SOURCE_PREFIX_NEW

for required in "${CONFIG}" "${SCENE_DIR}" "${CHECKPOINT}" "${QWEN_MODEL_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}"
    exit 1
  fi
done

RUNTIME_CONFIG="${TMPDIR}/qwen2.5-vl-3B-nusc-sft-autovla-attack.palmetto.yaml"
export RUNTIME_CONFIG
RUNTIME_SCENE_DIR="${TMPDIR}/nuscenes_val_palmetto"
export RUNTIME_SCENE_DIR
python - <<'PY'
import os
import json
from pathlib import Path
import yaml

src = Path(os.environ["CONFIG"])
dst = Path(os.environ["RUNTIME_CONFIG"])
cfg = yaml.safe_load(src.read_text())
cfg["model"]["pretrained_model_path"] = os.environ["QWEN_MODEL_PATH"]
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"Wrote runtime config: {dst}")

scene_src = Path(os.environ["SCENE_DIR"])
scene_dst = Path(os.environ["RUNTIME_SCENE_DIR"])
scene_dst.mkdir(parents=True, exist_ok=True)
old_prefix = os.environ["SOURCE_PREFIX_OLD"]
new_prefix = os.environ["SOURCE_PREFIX_NEW"]

def rewrite_value(value):
    if isinstance(value, str):
        return value.replace(old_prefix, new_prefix)
    if isinstance(value, list):
        return [rewrite_value(v) for v in value]
    if isinstance(value, dict):
        return {k: rewrite_value(v) for k, v in value.items()}
    return value

count = 0
for scene_file in scene_src.glob("*.json"):
    data = json.loads(scene_file.read_text())
    data = rewrite_value(data)
    (scene_dst / scene_file.name).write_text(json.dumps(data, indent=2))
    count += 1
print(f"Rewrote {count} scene json files into {scene_dst}")
PY

echo "=== AutoVLA white-box patch attack ==="
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "env_prefix=${ENV_PREFIX}"
echo "work_dir=${AUTOVLA_ROOT}"
echo "scene_dir=${SCENE_DIR}"
echo "runtime_scene_dir=${RUNTIME_SCENE_DIR}"
echo "checkpoint=${CHECKPOINT}"
echo "qwen_model_path=${QWEN_MODEL_PATH}"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

MONITOR_INTERVAL=120
monitor_loop() {
  while true; do
    echo "--- monitor $(date '+%F %T') ---"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    ps -u "$(id -un)" -o pid,pcpu,pmem,etime,cmd --sort=-pmem | head -n 8
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

config = yaml.safe_load(Path(os.environ["RUNTIME_CONFIG"]).read_text())
AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
print("HF cache warm check passed.")
PY

python tools/attack/run_nusc_whitebox_patch_attack.py \
  --config "${RUNTIME_CONFIG}" \
  --scene_dir "${RUNTIME_SCENE_DIR}" \
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
