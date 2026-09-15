#!/usr/bin/env bash
# GARD-restored VGGT-Omega inference.
#   CKPT=.../checkpoints/latest.pt bash run_scripts/val/infer_GARD_omega.bash                       # all 80 held-out groups
#   CKPT=... bash run_scripts/val/infer_GARD_omega.bash --eval-groups 0 26 53
#   CKPT=... OUT=... bash run_scripts/val/infer_GARD_omega.bash --lq-images a.png b.png c.png
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA:-0}}"
export PYTHONPATH="$GARD_ROOT:$GARD_ROOT/Depth-Anything-3/src:$GARD_ROOT/RAE/src:$GARD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${GARD_PYTHON:-python}"
CONFIG="${GARD_CONFIG:-${CONFIG:-$GARD_ROOT/run_configs/train/train_GARD_omega_hypersim_pairs.yaml}}"
: "${CKPT:?set CKPT=/path/to/checkpoints/latest.pt}"
OUT="${OUT:-$GARD_ROOT/result_eval/omega_hypersim_pairs/$(basename "$(dirname "$(dirname "$CKPT")")")}"

"$PYTHON_BIN" RAE/src/infer_omega_gard.py --config "$CONFIG" --ckpt "$CKPT" --out "$OUT" "$@"
