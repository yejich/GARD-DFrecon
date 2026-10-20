#!/usr/bin/env bash
# Portable NVS launcher wrapper.
# Prepare local_nvs.yaml separately with prepare_handoff_config.py.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/hypersim_synthetic_full}"
NVS_CONFIG="${NVS_CONFIG:-$ROOT/run_configs/handoff/local_nvs.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$ROOT/ckpts/gard_denoiser.pt}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HYPERSIM_CAMERA_ROOT="${HYPERSIM_CAMERA_ROOT:-$DATA_ROOT/camera_reference/hypersim}"
export HYPERSIM_CAMERA_PARAMETERS="${HYPERSIM_CAMERA_PARAMETERS:-$DATA_ROOT/camera_reference/metadata_camera_parameters.csv}"
export HYPERSIM_GEOMETRY_ROOT="${HYPERSIM_GEOMETRY_ROOT:-$HYPERSIM_CAMERA_ROOT}"

if [[ ! -f "$NVS_CONFIG" ]]; then
  echo "NVS_CONFIG not found: $NVS_CONFIG" >&2
  echo "Run scripts/train/nvs/prepare_handoff_config.py first." >&2
  exit 1
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Initial checkpoint not found: $INIT_CHECKPOINT" >&2
  echo "Run download_scripts/gard_ckpt/download_ckpts.sh first." >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "DATA_ROOT not found: $DATA_ROOT" >&2
  exit 1
fi

export NVS_CONFIG INIT_CHECKPOINT
exec bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash "$@"
