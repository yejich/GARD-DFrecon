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


## 3. 학습 bash 실행 (260916)

두 백본 모두 global batch size는 **8**, 학습 기간은 **30 epochs**입니다. GPU는 **2개, 4개 또는 8개**를 사용할 수 있으며, `CUDA_VISIBLE_DEVICES`에 지정한 GPU 수에 따라 gradient accumulation이 각각 **4, 2, 1**로 자동 조절됩니다. GPU당 batch size는 1입니다.

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

## 2026-10-14 업데이트: 전체 synthetic 데이터 및 NVS 학습

위의 기존 안내는 2026년 9월 snapshot과 학습 설정입니다. 아래는 확장 데이터 릴리스와 NVS 학습을 위한 추가 안내입니다.

### 데이터 다운로드 및 구분

HF 저장소는 **데이터만** 배포하고, 학습 코드와 설정은 이 Git 저장소에서 관리합니다.

| 학습 데이터 | 다운로드 후 경로 | Train / Eval / 전체 pair |
|---|---|---|
| transient-only | `transient_only/` | 42,006 / 1,057 / 43,063 |
| transient + static | `transient_static/pairs/` | 35,928 / 728 / 36,656 |
| real (NVS용) | `real_nvs/` 예정 | 아직 업로드되지 않음 |

Original 43,063쌍은 **기존 20,965쌍 + 추가 22,098쌍**입니다. `scenes_v2` 6,823쌍은 기존 20,965쌍의 일부이며, 별도 추가분이 아닙니다. Negative는 transient+static에 해당하는 별도 36,656쌍입니다. Pair 수는 multiview group 수와 다릅니다.

```bash
python -m pip install huggingface_hub requests
python scripts/release/download_synthetic.py --destination /data/hypersim_synthetic_full
```

추출기는 기존 TAR를 새 폴더 구조로 변환하고 체크섬을 검증하며 중단 후 재개할 수 있습니다. 데이터 snapshot, held-out scene, 파일 구성은 [데이터 안내](docs/dataset_handoff.md)와 [HF 데이터 카드](https://huggingface.co/datasets/cyjcyj91/Hypersim-Distractor)를 확인하세요.

### NVS 학습 실행

NVS는 전체 view `V=4`로 고정되고, 입력 source view `N=2~3`, novel view `M=4-N`을 사용합니다. 데이터 다운로드와 config 경로 정렬은 별도 단계로 수행한 뒤 아래 launcher로 학습합니다.

GPU 4개와 global batch 16이면 GPU당 batch 1, gradient accumulation 4가 자동으로 설정됩니다. launcher는 실행 옵션을 experiment name과 `launch_config.yaml`에 기록합니다.

```bash
python scripts/release/download_synthetic.py \
  --destination /data/hypersim_synthetic_full

python scripts/train/nvs/prepare_handoff_config.py \
  --data-root /data/hypersim_synthetic_full \
  --output run_configs/handoff/local_nvs.yaml \
  --wandb-entity yejichoi_kaist-kaist \
  --init-checkpoint ckpts/gard_denoiser.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash
```

학습 실행 옵션은 자동으로 experiment name과 `launch_config.yaml`에 기록됩니다.

| 옵션 | 값 | 의미 |
|---|---|---|
| `NVS_BRANCH` | `true` / `false` | novel-view branch/loss 사용 여부 |
| `FLOWMAP_FROM` | `noise_lq` / `noise` | source flow 시작점을 `LQ+noise` 또는 pure noise로 설정 |
| `DISTRACTOR_AWARE_ATTN_ALIGN` | `true` / `false` | distractor-aware attention alignment 사용 여부 |
| `ATTENTION_ALIGNMENT` | `true` / `false` | attention alignment loss 전체 사용 여부 |

모델 구조 5가지 실행:

```bash
# NVS branch ON
# noise+lq -> HQ flow
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=true FLOWMAP_FROM=noise_lq \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash

# NVS branch ON
# noise -> HQ flow
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=true FLOWMAP_FROM=noise \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash

# NVS branch OFF
# noise+lq -> HQ flow
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=false FLOWMAP_FROM=noise_lq \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash

# NVS branch OFF
# noise -> HQ flow
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=false FLOWMAP_FROM=noise \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash

# NVS branch OFF
# noise+lq -> HQ flow
# distractor-aware attention alignment: true
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=false FLOWMAP_FROM=noise_lq \
  DISTRACTOR_AWARE_ATTN_ALIGN=true ATTENTION_ALIGNMENT=true \
  NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash

```

초기 checkpoint 경로가 다르면 `INIT_CHECKPOINT=/path/to/gard_denoiser.pt`를 지정하고, 재시작 시에는 `--resume /path/to/latest.pt`를 추가합니다.
