#!/usr/bin/env bash
# 2026-10-28: mixed Hypersim + negative NVS training with geometry SupCon.
# Soft self-view distractor attention is kept enabled by the config.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash
#
# Preparation/smoke test:
#   bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --prepare-only
#   CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --max-steps 1
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

export NVS_CONFIG="${NVS_CONFIG:-$ROOT/run_configs/train/train_GARD_da3_nvs_v4_n2_3_mixed_supcon_softself.yaml}"
export EXP_NAME="${EXP_NAME:-nvs_v4_n2-3_mixed_geometry_supcon_softself}"
export NVS_BRANCH="${NVS_BRANCH:-false}"
export FLOWMAP_FROM="${FLOWMAP_FROM:-noise_lq}"
export ATTENTION_ALIGNMENT="${ATTENTION_ALIGNMENT:-true}"
export DISTRACTOR_AWARE_ATTN_ALIGN="${DISTRACTOR_AWARE_ATTN_ALIGN:-true}"
export DISTRACTOR_LOSS="${DISTRACTOR_LOSS:-none}"
export MASK_AWARE_ATTENTION="${MASK_AWARE_ATTENTION:-false}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"

exec bash "$ROOT/run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash" "$@"
