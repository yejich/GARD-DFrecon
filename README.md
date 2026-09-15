# GARD-DFrecon

## 1. 환경 설정

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
# 가중치는 Hugging Face에서 받습니다. DA3-GIANT-1.1은 최초 실행 시 자동 다운로드됩니다.
bash download_scripts/gard_ckpt/download_ckpts.sh
```

## 2. 데이터 다운로드

[Hypersim-Distractor](https://huggingface.co/datasets/cyjcyj91/Hypersim-Distractor)의 TAR와 메타데이터를 내려받습니다. 


```bash
mkdir ./datasets/hypersim_release
mkdir -p "$DATA_ROOT"
hf download cyjcyj91/Hypersim-Distractor --repo-type dataset \
  --local-dir ./datasets/hypersim_release
python /datasets/hypersim_release/tools/extract_release.py --destination /datasets

export HYPERSIM_PAIRS_ROOT=/datasets/hypersim_pairs
python scripts/check_hypersim_pairs.py
```

압축 해제 도구가 체크섬을 검증합니다. 

## 3. 학습 bash 실행

저장소 루트에서 실행합니다. 기본 설정은 GPU 2개, 10 epochs, weighted flow matching loss, distractor-aware alignment loss입니다.

```bash
conda activate gard-dfrecon
export GARD_PYTHON="$CONDA_PREFIX/bin/python"
export HYPERSIM_PAIRS_ROOT=/datasets/hypersim_pairs

CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/JIHYE_train_GARD_da3_hypersim_20k.sh
```


기존 상세 설명: [readme_detail_260914.md](readme_detail_260914.md)
