#!/usr/bin/env bash
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA:-0}}"
export PYTHONPATH="$GARD_ROOT:$GARD_ROOT/RAE/src:$GARD_ROOT/src:$GARD_ROOT/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${GARD_PYTHON:-python}" -m depth_anything_3.bench.evaluator --config "${GARD_CONFIG:-run_configs/val/val_GARD_real_bench.yaml}" "$@"
