#!/usr/bin/env bash
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA:-0}}"
export PYTHONPATH="$GARD_ROOT:$GARD_ROOT/RAE/src:$GARD_ROOT/src:$GARD_ROOT/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
exec "${GARD_PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="${#GPU_IDS[@]}" \
  RAE/src/train.py --config "${GARD_CONFIG:-run_configs/train/train_GARD.yaml}" "$@"
