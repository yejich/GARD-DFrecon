#!/usr/bin/env bash
set -euo pipefail

# 1. DA3와 동일한 저장소 경로, 데이터 경로, GPU 설정
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/datasets/hypersim_pairs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# Omega 소스는 저장소 내부에 포함; 공식 backbone 가중치는 별도 준비
export VGGT_OMEGA_REPO="${VGGT_OMEGA_REPO:-$PROJECT_ROOT/vggt-omega}"
export VGGT_OMEGA_CKPT="${VGGT_OMEGA_CKPT:-$PROJECT_ROOT/ckpts/vggt_omega_1b_512.pt}"

cd "$PROJECT_ROOT"
PYTHON=python  # conda 또는 venv를 먼저 활성화하세요.
CONFIG="run_configs/JIHYE/train_GARD_omega_hypersim_completed_260915.yaml"
export HYPERSIM_PAIRS_ROOT="$DATASET_ROOT"
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#GPU_IDS[@]}"
# 전체 배치 8 = GPU 수 × GPU당 1그룹 × 누적 횟수
case "$NUM_GPUS" in
  2|4|8) ;;
  *) echo "GPU는 2개, 4개 또는 8개를 지정하세요." >&2; exit 1 ;;
esac
if [[ "$CUDA_VISIBLE_DEVICES" == ,* || "$CUDA_VISIBLE_DEVICES" == *, || "$CUDA_VISIBLE_DEVICES" == *,,* ]]; then
  echo "CUDA_VISIBLE_DEVICES의 GPU 목록에 빈 항목이 있습니다." >&2
  exit 1
fi
export GARD_GRAD_ACCUM_STEPS="$((8 / NUM_GPUS))"
echo "[batch] GPUs=$NUM_GPUS, microbatch=1/GPU, accum=$GARD_GRAD_ACCUM_STEPS, global_batch=8"
export OMEGA_RUN_ID="$(date +%Y%m%d_%H%M%S_%N)"
export PYTHONPATH="$PWD:$PWD/RAE/src:$PWD/src:$PWD/Depth-Anything-3/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/gard_omega_mpl

# 2. DA3와 동일한 HF 배포본의 고정 pair 목록 사용
export OMEGA_PAIRS_MANIFEST="$PWD/manifests/hypersim_hf_a935b4e251fa.json"
"$PYTHON" scripts/check_hypersim_pairs.py --manifest "$OMEGA_PAIRS_MANIFEST"
if [[ "${1:-}" == "--prepare-only" ]]; then
  exit 0
fi

# 3. 이번 데이터로 Omega 특징 정규화 통계 계산 (첫 번째 GPU 사용)
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$PYTHON" \
  RAE/src/compute_omega_latent_stats.py --config "$CONFIG"

# 4. 공식 가중치에서 config에 지정된 epochs만큼 학습
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  RAE/src/train_omega_hypersim_pairs.py --config "$CONFIG" "$@"
