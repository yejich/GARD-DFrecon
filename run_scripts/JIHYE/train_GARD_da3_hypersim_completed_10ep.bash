#!/usr/bin/env bash
set -euo pipefail

# 1. 사용자 설정: 저장소 경로, 데이터 경로, GPU
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/datasets/hypersim_pairs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cd "$PROJECT_ROOT"
PYTHON=python  # conda 또는 venv를 먼저 활성화하세요.
CONFIG="run_configs/JIHYE/train_GARD_da3_hypersim_completed_10ep.yaml"
export HYPERSIM_PAIRS_ROOT="$DATASET_ROOT"
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#GPU_IDS[@]}"
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo "CUDA_VISIBLE_DEVICES에 사용할 GPU 번호를 지정하세요." >&2
  exit 1
fi

# 2. 이번 실행의 이름과 데이터 목록 저장 위치
export DA3_RUN_ID="$(date +%Y%m%d_%H%M%S_%N)"
export DA3_COMPLETED_MANIFEST="$PWD/manifests/hypersim_hf_a935b4e251fa.json"
export PYTHONPATH="$PWD:$PWD/RAE/src:$PWD/src:$PWD/Depth-Anything-3/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/gard_dfrecon_mpl

# 3. HF 배포본의 고정 pair 목록 확인 (로컬 추가 생성분 제외)
"$PYTHON" scripts/check_hypersim_pairs.py --manifest "$DA3_COMPLETED_MANIFEST"

# --prepare-only: 데이터 확인까지만 수행
if [[ "${1:-}" == "--prepare-only" ]]; then
  exit 0
fi

# 4. 학습 시작 (추가 인자는 Python 학습 코드에 전달)
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  RAE/src/JIHYE_train_GARD_da3_hypersim_20k.py \
  --config "$CONFIG" --manifest "$DA3_COMPLETED_MANIFEST" "$@"
