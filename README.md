# GARD-DFrecon


## 1. 환경 설정

<!-- 현재 서버에서는 복사·검증한 환경이 있으므로 아래 명령으로 바로 활성화할 수 있습니다. 새 환경을 만드는 경우에는 이어지는 conda 설치 절차를 사용하세요.

```bash
source .venv/bin/activate
export GARD_PYTHON="$PWD/.venv/bin/python"
``` -->

[MVR 설치 방법](https://github.com/jinlovespho/MVR#-installation-walkthrough)을 기반으로 합니다. 

```bash
git clone https://github.com/yejich/GARD-DFrecon.git
cd GARD-DFrecon

conda create -n gard-dfrecon python=3.10 -y
conda activate gard-dfrecon

python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e ./Depth-Anything-3 --no-deps

# 공식 GARD(https://github.com/cvlab-kaist/GARD)의 다운로드 스크립트 사용.
# GARD checkpoint는 Hugging Face에서 받고, DA3/LPIPS weight는 아래 cache 단계에서 미리 준비합니다.
bash download_scripts/gard_ckpt/download_ckpts.sh

# 학습에 사용할 초기 GARD denoiser checkpoint 확인
test -f ckpts/gard_denoiser.pt

# 오프라인 학습/평가 전에 DA3와 LPIPS weight를 미리 cache합니다.
# HF_HOME/HF_HUB_CACHE는 서버에 맞는 경로로 지정할 수 있습니다.
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
bash run_scripts/JIHYE/handoff/20261013_cache_models.bash
```

이후 학습 launcher는 `HF_HUB_OFFLINE=1`로 실행되므로, 새 서버에서는 위의 checkpoint 다운로드와 model cache 준비를 먼저 완료해야 합니다. cache 준비 결과는 `handoff_model_cache.json`에 기록됩니다.

## 2. 데이터 다운로드

[Hypersim-Distractor](https://huggingface.co/datasets/cyjcyj91/Hypersim-Distractor)의 TAR와 메타데이터를 내려받습니다. 

저장소 루트에서 실행합니다. 데이터는 저장소 내부의 `datasets` 폴더에 저장합니다.

```bash
export DATA_ROOT="$PWD/datasets"
mkdir -p "$DATA_ROOT/hypersim_release"
hf download cyjcyj91/Hypersim-Distractor --repo-type dataset \
  --revision a935b4e251faeaf1e96723a28463d44e8913c834 \
  --local-dir "$DATA_ROOT/hypersim_release"
python "$DATA_ROOT/hypersim_release/tools/extract_release.py" \
  --destination "$DATA_ROOT"

export HYPERSIM_PAIRS_ROOT="$DATA_ROOT/hypersim_pairs"
python scripts/check_hypersim_pairs.py
```


## 3. 학습 bash 실행 (260916)

두 백본 모두 global batch size는 **8**, 학습 기간은 **30 epochs**입니다. GPU는 **2개, 4개 또는 8개**를 사용할 수 있습니다. 기본 GPU당 batch size는 1이며, `PER_GPU_BATCH_SIZE`로 조절할 수 있습니다. gradient accumulation은 `global batch / (GPU 수 × GPU당 batch)`로 자동 계산됩니다.

체크포인트는 실험 폴더의 `checkpoints/`에 저장됩니다. `latest.pt`는 매 epoch 완료 시 학습 재개용 전체 상태로 갱신하며, 10·20·30 epoch 완료 시 `epoch_010.pt`, `epoch_020.pt`, `epoch_030.pt`에 추론용 EMA 가중치를 별도로 저장합니다. Omega의 추론용 파일에는 특징 정규화 통계도 포함됩니다.

### 3.1 GARD-DA3 backbone

저장소 루트에서 실행합니다. 
아래 예시는 GPU 2개를 사용합니다.

```bash
conda activate gard-dfrecon
export DATASET_ROOT="$PWD/datasets/hypersim_pairs"

CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/JIHYE/train_GARD_da3_hypersim_completed_260915.bash
```



### 3.2 GARD-VGGT-Omega backbone

[VGGT-Omega 소스](https://github.com/facebookresearch/vggt-omega)에서 `vggt_omega_1b_512.pt`를 `./ckpts`에 다운받으세요.

```bash
conda activate gard-dfrecon
export GARD_PYTHON="$CONDA_PREFIX/bin/python"
export HYPERSIM_PAIRS_ROOT="$PWD/datasets/hypersim_pairs"
export VGGT_OMEGA_CKPT="$PWD/ckpts/vggt_omega_1b_512.pt"
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/JIHYE/train_GARD_omega_hypersim_completed_260915.bash
```

<!-- 기존 상세 설명: [readme_detail_260914.md](readme_detail_260914.md) -->


---

## NVS 실험안내 (2026-10-20)

JIHYE님에게 전달할 최신 synthetic dataset 다운로드 및 NVS 실험 명령은 아래 문서에서 관리합니다.

[`run_scripts/JIHYE_261020/README.md`](run_scripts/JIHYE_261020/README.md)
