#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Override these values at invocation time for each ablation.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export NVS_BRANCH="${NVS_BRANCH:-true}"
export FLOWMAP_FROM="${FLOWMAP_FROM:-noise_lq}"
export DISTRACTOR_AWARE_ATTN_ALIGN="${DISTRACTOR_AWARE_ATTN_ALIGN:-false}"
export ATTENTION_ALIGNMENT="${ATTENTION_ALIGNMENT:-true}"
export NVS_CONFIG="${NVS_CONFIG:-$ROOT/run_configs/handoff/local_nvs.yaml}"

exec bash "$ROOT/run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash" "$@"
