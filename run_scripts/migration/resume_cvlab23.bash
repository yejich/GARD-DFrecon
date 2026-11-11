#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE="$(realpath "${1:?Usage: bash resume_cvlab23.bash BUNDLE [--prepare-only]}")"
shift
PREPARE_ONLY=false
if [[ "${1:-}" == --prepare-only ]]; then PREPARE_ONLY=true; shift; fi
if (( $# )); then echo 'Unexpected arguments' >&2; exit 2; fi
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/RAE/src:$ROOT/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
OUT="${RESUME_OUT:-$BUNDLE/resumed}"
mkdir -p "$OUT"
"$PYTHON" scripts/train/migration/prepare_resume.py "$BUNDLE" --output "$OUT/resume_config.yaml" --gpu-count "${#GPU_IDS[@]}"
if [[ "$PREPARE_ONLY" == true ]]; then exit 0; fi
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${#GPU_IDS[@]}" \
  scripts/train/nvs/train.py --config "$OUT/resume_config.yaml" \
  --out "$OUT/training" --checkpoint-dir "$OUT/checkpoints" \
  --resume "$BUNDLE/checkpoints/resume.pt" 2>&1 | tee -a "$OUT/train.log"
