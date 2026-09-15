#!/usr/bin/env bash
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
RUN="$GARD_ROOT/run_scripts/val/run_experiment.sh"
BASE=hypersim_10ep_ai001/five_methods
bash "$RUN" "$BASE/infer_ours_da3.py"
GARD_PYTHON="${VGGT_PYTHON:-${GARD_PYTHON:-python}}" bash "$RUN" "$BASE/infer_baselines.py" --method vggt
GARD_PYTHON="${OMEGA_PYTHON:-${GARD_PYTHON:-python}}" bash "$RUN" "$BASE/infer_baselines.py" --method vggt_omega
GARD_PYTHON="${VGTW_PYTHON:-${GARD_PYTHON:-python}}" bash "$RUN" "$BASE/infer_baselines.py" --method vgtw
bash "$RUN" "$BASE/build_report.py"
