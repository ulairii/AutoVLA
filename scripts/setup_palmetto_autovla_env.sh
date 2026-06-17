#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Project/AutoVLA}"
if [[ -z "${SCRATCH_ROOT:-}" ]]; then
  if [[ "${USER:-}" == "coder" ]]; then
    SCRATCH_ROOT="/scratch/runw"
  else
    SCRATCH_ROOT="/scratch/$USER"
  fi
fi
ENV_ROOT="${ENV_ROOT:-${SCRATCH_ROOT}/envs}"
ENV_NAME="${ENV_NAME:-autovla_codeclean}"
ENV_PREFIX="${ENV_ROOT}/${ENV_NAME}"
HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/.hf_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/.pip_cache}"

mkdir -p "${ENV_ROOT}" "${HF_HOME}" "${PIP_CACHE_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load anaconda3 || true
fi

if [[ -f /etc/profile ]]; then
  source /etc/profile || true
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found in the current shell."
  exit 1
fi

eval "$(conda shell.bash hook)"

export HF_HOME
export PIP_CACHE_DIR
export TOKENIZERS_PARALLELISM=false

if [[ ! -d "${ENV_PREFIX}" ]]; then
  conda env create -p "${ENV_PREFIX}" -f "${PROJECT_ROOT}/environment.yml"
else
  echo "Conda env already exists at ${ENV_PREFIX}"
fi

conda activate "${ENV_PREFIX}"
cd "${PROJECT_ROOT}"
pip install -e . --no-warn-conflicts

if [[ -x "${PROJECT_ROOT}/install.sh" ]]; then
  bash "${PROJECT_ROOT}/install.sh"
fi

cd "${PROJECT_ROOT}/navsim"
pip install -e . --no-warn-conflicts

cd "${PROJECT_ROOT}"

python - <<'PY'
import sys
print("Python:", sys.version)
print("Palmetto AutoVLA env setup complete.")
PY
