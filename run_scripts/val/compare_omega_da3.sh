#!/usr/bin/env bash
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
export CKPT="${CKPT:-$GARD_ROOT/result_train/omega_hypersim_pairs_ai001_eval/hypersim_pairs/TRAIN__bf16__hypersim_pairs__near_camera-near_random__vggt_omega-1B-512-extractfeat3-mvrm__bs16-maxview4-accum8__lr-1e-04__msg-omega_inter3_p07_group2/checkpoints/latest.pt}"
export OUT="${OUT:-$GARD_ROOT/result_eval/omega_vs_da3_hypersim_10ep/omega}"
bash "$GARD_ROOT/run_scripts/val/infer_GARD_omega.bash" --eval-groups 0 26 53 --seed 42 "$@"
bash "$GARD_ROOT/run_scripts/val/run_experiment.sh" omega_vs_da3_hypersim_10ep/build_report.py
