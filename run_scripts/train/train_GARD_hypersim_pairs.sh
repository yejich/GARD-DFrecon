#!/usr/bin/env bash
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$GARD_ROOT:$GARD_ROOT/Depth-Anything-3/src:$GARD_ROOT/RAE/src:$GARD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gard_dfrecon_mpl}"
PYTHON_BIN="${GARD_PYTHON:-$GARD_ROOT/.venv/bin/python}"
CONFIG="${GARD_CONFIG:-$GARD_ROOT/run_configs/train/train_GARD_hypersim_pairs.yaml}"
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#GPU_IDS[@]}"
if [[ "$NUM_GPUS" -lt 1 || -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo 'Set CUDA_VISIBLE_DEVICES to NVIDIA GPU indices or UUIDs.' >&2
  exit 1
fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  RAE/src/train_hypersim_pairs.py --config "$CONFIG" "$@"
