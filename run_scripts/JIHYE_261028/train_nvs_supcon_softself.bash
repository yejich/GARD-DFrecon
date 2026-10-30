#!/usr/bin/env bash
# JIHYE 2026-10-28: mixed Hypersim + negative NVS with geometry SupCon.
# All experiment switches are visible in the command line; env vars remain
# available for scripting and batch sweeps.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash [options]

Main ablations:
  --nvs on|off                       NVS branch (default: off)
  --flow noise|noise_lq              flow-map source (default: noise_lq)
  --supcon on|off                    geometry SupCon (default: on)
  --attention-alignment on|off       attention alignment (default: on)
  --distractor-aware on|off          distractor-aware alignment (default: on)
  --camera-pose-mode MODE            da3_aligned|first_relative|source_target_relative (default: da3_aligned)

SupCon options:
  --supcon-lambda FLOAT              loss coefficient (default: 0.05)
  --supcon-temperature FLOAT         temperature (default: 0.07)
  --supcon-max-negatives INT|none    negatives per anchor (default: 64)
  --supcon-time-weight one_minus_t|none

Training options:
  --global-batch INT                 default: 16
  --per-gpu-batch INT                default: 2
  --distractor-loss none|push_only|infoNCE
  --wandb on|off                     default: off
  --help

The remaining generic options are forwarded, including --prepare-only,
--max-steps, --resume PATH, --out PATH, and --novel-weight FLOAT.

Examples:
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash \
    --nvs on --flow noise --supcon off --max-steps 1
EOF
}

as_bool() {
  case "${1,,}" in
    on|true|1) echo true ;;
    off|false|0) echo false ;;
    *) echo "Expected on/off, got: $1" >&2; exit 2 ;;
  esac
}

NVS_BRANCH=false
FLOWMAP_FROM=noise_lq
GEOMETRY_CONTRASTIVE_USE=true
ATTENTION_ALIGNMENT=true
DISTRACTOR_AWARE_ATTN_ALIGN=true
CAMERA_POSE_MODE=da3_aligned
GEOMETRY_CONTRASTIVE_LAMBDA=0.05
GEOMETRY_CONTRASTIVE_TEMPERATURE=0.07
GEOMETRY_CONTRASTIVE_MAX_NEGATIVE_TOKENS=64
GEOMETRY_CONTRASTIVE_TIME_WEIGHT=one_minus_t
DISTRACTOR_LOSS=none
GLOBAL_BATCH_SIZE=16
PER_GPU_BATCH_SIZE=2
WANDB_ENABLED=false
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --nvs|--nvs-branch) NVS_BRANCH="$(as_bool "${2:?$1 requires on/off}")"; shift 2 ;;
    --flow|--flowmap-from) FLOWMAP_FROM="${2:?$1 requires noise/noise_lq}"; shift 2 ;;
    --supcon) GEOMETRY_CONTRASTIVE_USE="$(as_bool "${2:?$1 requires on/off}")"; shift 2 ;;
    --attention-alignment) ATTENTION_ALIGNMENT="$(as_bool "${2:?$1 requires on/off}")"; shift 2 ;;
    --distractor-aware) DISTRACTOR_AWARE_ATTN_ALIGN="$(as_bool "${2:?$1 requires on/off}")"; shift 2 ;;
    --camera-pose-mode) CAMERA_POSE_MODE="${2:?$1 requires a camera pose mode}"; shift 2 ;;
    --supcon-lambda) GEOMETRY_CONTRASTIVE_LAMBDA="${2:?$1 requires a value}"; shift 2 ;;
    --supcon-temperature) GEOMETRY_CONTRASTIVE_TEMPERATURE="${2:?$1 requires a value}"; shift 2 ;;
    --supcon-max-negatives) GEOMETRY_CONTRASTIVE_MAX_NEGATIVE_TOKENS="${2:?$1 requires INT or none}"; shift 2 ;;
    --supcon-time-weight) GEOMETRY_CONTRASTIVE_TIME_WEIGHT="${2:?$1 requires one_minus_t or none}"; shift 2 ;;
    --distractor-loss) DISTRACTOR_LOSS="${2:?$1 requires a mode}"; shift 2 ;;
    --global-batch) GLOBAL_BATCH_SIZE="${2:?$1 requires an integer}"; shift 2 ;;
    --per-gpu-batch) PER_GPU_BATCH_SIZE="${2:?$1 requires an integer}"; shift 2 ;;
    --wandb) WANDB_ENABLED="$(as_bool "${2:?$1 requires on/off}")"; shift 2 ;;
    --prepare-only|--max-steps) PASSTHROUGH+=("$1"); shift ;;
    --resume|--out|--novel-weight) PASSTHROUGH+=("$1" "${2:?$1 requires a value}"); shift 2 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

case "$FLOWMAP_FROM" in noise|noise_lq) ;; *) echo '--flow must be noise or noise_lq' >&2; exit 2 ;; esac
case "$CAMERA_POSE_MODE" in da3_aligned|first_relative|source_target_relative) ;; *) echo '--camera-pose-mode must be da3_aligned, first_relative, or source_target_relative' >&2; exit 2 ;; esac
if [[ -z "${EXP_NAME:-}" ]]; then
  EXP_NAME="nvs_v4_n2-3"
fi

export NVS_CONFIG="${NVS_CONFIG:-$ROOT/run_configs/train/train_GARD_da3_nvs_v4_n2_3_mixed_supcon_softself.yaml}"
export EXP_NAME NVS_BRANCH FLOWMAP_FROM ATTENTION_ALIGNMENT DISTRACTOR_AWARE_ATTN_ALIGN CAMERA_POSE_MODE
export DISTRACTOR_LOSS GLOBAL_BATCH_SIZE PER_GPU_BATCH_SIZE WANDB_ENABLED
export SCENE_SPLIT_POLICY="${SCENE_SPLIT_POLICY:-}"
export HYPERSIM_CAMERA_ROOT="${HYPERSIM_CAMERA_ROOT:-}"
export HYPERSIM_CAMERA_PARAMETERS="${HYPERSIM_CAMERA_PARAMETERS:-}"
export GEOMETRY_CONTRASTIVE_USE GEOMETRY_CONTRASTIVE_LAMBDA
export GEOMETRY_CONTRASTIVE_TEMPERATURE GEOMETRY_CONTRASTIVE_MAX_NEGATIVE_TOKENS
export GEOMETRY_CONTRASTIVE_TIME_WEIGHT
# Optional fixed-eval code is not required for this handoff; train only.
export EVAL_EPOCHS="${EVAL_EPOCHS:-}"

echo "SupCon=${GEOMETRY_CONTRASTIVE_USE} NVS=${NVS_BRANCH} flow=${FLOWMAP_FROM} " \
     "attn=${ATTENTION_ALIGNMENT} distractor-aware=${DISTRACTOR_AWARE_ATTN_ALIGN} " \
     "camera-pose=${CAMERA_POSE_MODE} " \
     "global-batch=${GLOBAL_BATCH_SIZE} per-gpu=${PER_GPU_BATCH_SIZE}"

exec bash "$ROOT/run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash" "${PASSTHROUGH[@]}"
