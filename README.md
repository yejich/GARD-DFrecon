# GARD-DFrecon

## 1. 환경 설정

[MVR 설치 방법](https://github.com/jinlovespho/MVR#-installation-walkthrough)을 기반으로 합니다. Python 3.10, PyTorch 2.4.1, CUDA 12.1을 사용하며 이 feature 학습 경로에는 gsplat을 설치하지 않습니다.

```bash
git clone https://github.com/yejich/GARD-DFrecon.git
cd GARD-DFrecon

conda create -n gard-dfrecon python=3.10 -y
conda activate gard-dfrecon

python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e ./Depth-Anything-3 --no-deps

# 사전학습 GARD 가중치. DA3-GIANT-1.1은 최초 실행 시 자동 다운로드됩니다.
hf download jinlovespho/GARD gard_denoiser.pt --local-dir ckpts
```

## 2. 데이터 다운로드

[Hypersim-Distractor](https://huggingface.co/datasets/cyjcyj91/Hypersim-Distractor)의 TAR와 메타데이터를 내려받습니다. 아래 `/datasets`는 저장할 경로로 변경하세요.

```bash
hf download cyjcyj91/Hypersim-Distractor --repo-type dataset \
  --local-dir /datasets/hypersim_release
python /datasets/hypersim_release/tools/extract_release.py --destination /datasets

export HYPERSIM_PAIRS_ROOT=/datasets/hypersim_pairs
python scripts/check_hypersim_pairs.py
```

압축 해제 도구가 체크섬을 검증합니다. `scenes_v2`, `scenes_002` 등의 다운로드된 pair를 통합하며 `ai_001_001`은 평가용으로 유지합니다.

## 3. 학습 bash 실행

저장소 루트에서 실행합니다. 기본 설정은 GPU 2개, 10 epochs, Group2 loss입니다.

```bash
conda activate gard-dfrecon
export GARD_PYTHON="$CONDA_PREFIX/bin/python"
export HYPERSIM_PAIRS_ROOT=/datasets/hypersim_pairs

CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh
```

W&B를 사용하는 경우:

```bash
wandb login
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh \
  --wandb --wandb-project cross-view-feature-completion \
  --wandb-run-name Hypersim_group2_p07_10ep
```

기존 상세 설명: [readme_detail_260914.md](readme_detail_260914.md)
