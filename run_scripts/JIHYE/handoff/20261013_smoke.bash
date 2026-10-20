#!/usr/bin/env bash
# Uses real data, tests all nine forward/backward variants, then one eval group.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export GLOBAL_BATCH_SIZE=${#GPUS[@]} MICROBATCH_SIZE=1
SMOKE_ROOT="${SMOKE_ROOT:-$ROOT/result_eval/$(date +%Y%m%d_%H%M%S)_handoff_smoke}"
for KIND in baseline_transient_off nvs_legacy_target; do
 bash "$ROOT/run_scripts/JIHYE/handoff/20261013_train.bash" --experiment "$KIND" --out "$SMOKE_ROOT/$KIND" --workers 0 --eval-limit 1 --smoke-matrix --eval-now "$@"
done
