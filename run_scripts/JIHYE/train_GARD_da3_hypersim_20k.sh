#!/usr/bin/env bash
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$GARD_ROOT:$GARD_ROOT/Depth-Anything-3/src:$GARD_ROOT/RAE/src:$GARD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gard_dfrecon_mpl}"
PYTHON_BIN="${GARD_PYTHON:-python}"
CONFIG="${GARD_CONFIG:-$GARD_ROOT/run_configs/JIHYE/train_GARD_da3_hypersim_20k.yaml}"
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#GPU_IDS[@]}"
if [[ "$NUM_GPUS" -lt 1 || -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo 'Set CUDA_VISIBLE_DEVICES to NVIDIA GPU indices or UUIDs.' >&2
  exit 1
fi
# Default to the fixed HF release. --manifest explicitly selects another snapshot.
EXPLICIT_MANIFEST=0
for ARG in "$@"; do
  case "$ARG" in
    --manifest|--manifest=*) EXPLICIT_MANIFEST=1 ;;
  esac
done
MANIFEST_ARGS=()
if [[ "$EXPLICIT_MANIFEST" == 0 ]]; then
  : "${HYPERSIM_PAIRS_ROOT:?Set HYPERSIM_PAIRS_ROOT to the downloaded pair directory}"
  MANIFEST="$GARD_ROOT/manifests/hypersim_hf_a935b4e251fa.json"
  MANIFEST_ARGS=(--manifest "$MANIFEST")
fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  RAE/src/JIHYE_train_GARD_da3_hypersim_20k.py --config "$CONFIG" "${MANIFEST_ARGS[@]}" "$@"
