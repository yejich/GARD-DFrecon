#!/usr/bin/env bash
# Usage: bash run_scripts/val/run_experiment.sh <experiment/script.py> [arguments]
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
if [[ $# -eq 0 ]]; then
  echo 'Usage: run_experiment.sh <path relative to scripts/eval> [arguments]' >&2
  exit 2
fi
SCRIPT="$GARD_ROOT/scripts/eval/$1"
shift
[[ -f "$SCRIPT" ]] || { echo "Missing experiment: $SCRIPT" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA:-0}}"
export PYTHONPATH="$GARD_ROOT:$GARD_ROOT/RAE/src:$GARD_ROOT/src:$GARD_ROOT/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gard_eval_mpl}"
exec "${GARD_PYTHON:-python}" -u "$SCRIPT" "$@"
