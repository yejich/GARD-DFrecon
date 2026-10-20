#!/usr/bin/env bash
# One experiment per node. No W&B or internet access required during training.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
EXPERIMENT="${EXPERIMENT:-baseline_transient_off}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/handoff}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$ROOT/ckpts/gard_denoiser.pt}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/RAE/src:$ROOT/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
GPUS=${#DEVICES[@]}
PREPARE_ONLY=false
RESUME=""
TRAIN_ARGS=()
PREPARE_ARGS=()
while [[ $# -gt 0 ]]; do
 case "$1" in
  --help|-h)
   printf 'Usage: bash %s --experiment NAME --data DATA_ROOT [--prepare-only] [--resume latest.pt]\n' "$0"
   printf 'Environment: CUDA_VISIBLE_DEVICES (default 0-7), GLOBAL_BATCH_SIZE=16, MICROBATCH_SIZE=1, PYTHON, INIT_CHECKPOINT\n'
   printf 'See run_scripts/JIHYE/handoff/README.md and 20261013_list.bash.\n'
   exit 0 ;;
  --experiment) EXPERIMENT="$2"; shift 2 ;;
  --data) DATA_ROOT="$2"; shift 2 ;;
  --out) OUT_DIR="$2"; shift 2 ;;
  --resume) RESUME="$2"; shift 2 ;;
  --prepare-only) PREPARE_ONLY=true; shift ;;
  --eval-manifest|--epochs|--steps-per-epoch|--workers|--eval-epochs|--save-epochs|--eval-limit)
   PREPARE_ARGS+=("$1" "$2"); shift 2 ;;
  *) TRAIN_ARGS+=("$1"); shift ;;
 esac
done
if [[ -n "$RESUME" && -z "${OUT_DIR:-}" ]]; then OUT_DIR="$(dirname -- "$RESUME")"; fi
OUT_DIR="${OUT_DIR:-$ROOT/result_train/$(date +%Y%m%d_%H%M%S)_${EXPERIMENT}_bs${GLOBAL_BATCH_SIZE}}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
if [[ -n "$RESUME" ]]; then
 # Preserve the frozen config on resume, including schedule and eval selection.
 [[ -f "$OUT_DIR/config.yaml" ]] || { echo 'Resume config.yaml missing.' >&2; exit 1; }
else
 "$PYTHON" scripts/train/handoff/prepare.py --experiment "$EXPERIMENT" --data "$DATA_ROOT" --checkpoint "$INIT_CHECKPOINT" --out "$OUT_DIR" --gpus "$GPUS" --global-batch "$GLOBAL_BATCH_SIZE" --microbatch "$MICROBATCH_SIZE" "${PREPARE_ARGS[@]}" | tee "$OUT_DIR/prepare.log"
fi
"$PYTHON" scripts/train/handoff/validate.py --config "$OUT_DIR/config.yaml" | tee "$OUT_DIR/data_validation.log"
if [[ "$PREPARE_ONLY" == true ]]; then exit 0; fi
RESUME_ARGS=()
[[ -z "$RESUME" ]] || RESUME_ARGS=(--resume "$RESUME")
printf 'Experiment: %s\nOutput: %s\nGPUs: %s\n' "$EXPERIMENT" "$OUT_DIR" "$CUDA_VISIBLE_DEVICES"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" --max_restarts=0 \
 scripts/train/handoff/train.py --config "$OUT_DIR/config.yaml" --out "$OUT_DIR" "${RESUME_ARGS[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$OUT_DIR/console.log"
