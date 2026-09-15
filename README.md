# GARD-DFrecon

GARD에서 진행하던 학습·추론·평가 실험의 기준 저장소입니다. **전체 실행 명령과 이전 내역은 [실험 안내](docs/experiments.md)**를 참고하세요.

JIHYE님께 요청할 실험은 [`run_scripts/JIHYE`](run_scripts/JIHYE/README.md)와 [`run_configs/JIHYE`](run_configs/JIHYE/README.md)에서 관리합니다. 완료된 전체 Hypersim 데이터로 DA3를 10 epochs 재학습하는 전용 bash/config를 포함합니다.

기존 결과와 체크포인트는 이 저장소에 복사했습니다. 가중치 4개는 SHA-256 검증을 마친 독립 파일이며, 대용량 원본 데이터는 외부 경로에 연결되어 있습니다.

## 1. 환경 설정

현재 서버에서는 복사·검증한 환경이 있으므로 아래 명령으로 바로 활성화할 수 있습니다. 새 환경을 만드는 경우에는 이어지는 conda 설치 절차를 사용하세요.

```bash
source .venv/bin/activate
export GARD_PYTHON="$PWD/.venv/bin/python"
```

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
압축 해제 도구가 체크섬을 검증합니다. 

## 3. 학습 bash 실행
### 3.1 GARD-DA3 backbone

저장소 루트에서 실행합니다. 
기본 설정은 GPU 2개, 10 epochs 입니다.

```bash
conda activate gard-dfrecon
export GARD_PYTHON="$CONDA_PREFIX/bin/python"
export HYPERSIM_PAIRS_ROOT="$PWD/datasets/hypersim_pairs"

CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/JIHYE/train_GARD_da3_hypersim_20k.sh
```



### 3.2 GARD-VGGT-Omega backbone

[VGGT-Omega 소스](https://github.com/facebookresearch/vggt-omega)에서 `vggt_omega_1b_512.pt`를 `./ckpts`에 다운받으세요.

```bash
conda activate gard-dfrecon
export GARD_PYTHON="$CONDA_PREFIX/bin/python"
export HYPERSIM_PAIRS_ROOT="$PWD/datasets/hypersim_pairs"
export VGGT_OMEGA_CKPT="$PWD/ckpts/vggt_omega_1b_512.pt"
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_omega_hypersim_pairs.bash
```

기존 상세 설명: [readme_detail_260914.md](readme_detail_260914.md)
