#!/usr/bin/env bash
# Resume the original 10-epoch DA3 experiment to 20 total epochs.
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
CKPT="${CKPT:-$GARD_ROOT/result_train/hypersim_pairs_ai001_eval/hypersim_pairs/TRAIN__bf16__hypersim_pairs__near_camera-near_random__da3-GIANT-extractfeat17-mvrm__bs16-maxview4-accum8__lr-2e-05__msg-distractor_p07_ai001eval_group2/checkpoints/latest.pt}"
export GARD_CONFIG="${GARD_CONFIG:-$GARD_ROOT/run_configs/train/train_GARD_hypersim_pairs_ext20ep.yaml}"
exec bash "$GARD_ROOT/run_scripts/train/train_GARD_hypersim_pairs.sh" \
  --resume "$CKPT" "$@"
