# JIHYE 2026-10-20 NVS 실험

이 폴더는 JIHYE님에게 전달할 NVS 실험용 Bash 진입점입니다. 실제 학습 구현은 기존 NVS trainer와 공유하지만, 실행 명령은 이 폴더의 파일만 사용하면 됩니다.

아래 명령은 저장소 루트에서 실행합니다. `DATA_ROOT`는 HF 데이터가 설치된 경로로 바꾸세요.

## 준비

```bash
bash run_scripts/JIHYE_261020/20261013_cache_models.bash

DATA_ROOT=/data/gard \
bash run_scripts/JIHYE_261020/20261013_download_data.bash

DATA_ROOT=/data/gard \
bash run_scripts/JIHYE_261020/20261013_validate.bash --create-eval

DATA_ROOT=/data/gard \
bash run_scripts/JIHYE_261020/20261013_validate.bash --all-files
```

## 이번에 실행할 5개 실험

모든 실험은 하나의 `train_nvs.bash`를 사용하고, 아래 환경변수만 바꿉니다.

모든 실험은 다음 공통 설정을 사용합니다.

- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `GLOBAL_BATCH_SIZE=16`
- `ATTENTION_ALIGNMENT=true`
- `NVS_CONFIG=run_configs/handoff/local_nvs.yaml`

`ATTENTION_ALIGNMENT=true`는 다섯 실험 모두 공통입니다.

```bash
# NVS branch ON / noise+lq -> HQ flow
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=true FLOWMAP_FROM=noise_lq \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/JIHYE_261020/train_nvs.bash

# NVS branch ON / noise -> HQ flow
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=true FLOWMAP_FROM=noise \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/JIHYE_261020/train_nvs.bash

# NVS branch OFF / noise+lq -> HQ flow
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=false FLOWMAP_FROM=noise_lq \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/JIHYE_261020/train_nvs.bash

# NVS branch OFF / noise -> HQ flow
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=false FLOWMAP_FROM=noise \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/JIHYE_261020/train_nvs.bash

# NVS branch OFF / noise+lq -> HQ flow / distractor-aware attention ON
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=false FLOWMAP_FROM=noise_lq \
  DISTRACTOR_AWARE_ATTN_ALIGN=true ATTENTION_ALIGNMENT=true \
  NVS_CONFIG=run_configs/handoff/local_nvs.yaml \
  bash run_scripts/JIHYE_261020/train_nvs.bash
```

GPU, batch, config 경로를 바꾸려면 실행 시 환경변수로 덮어쓸 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=4,5 GLOBAL_BATCH_SIZE=8 \
  NVS_CONFIG=/data/gard/local_nvs.yaml \
  bash run_scripts/JIHYE_261020/train_nvs.bash
```

실제 학습 전에는 `--prepare-only`를 추가해 config와 데이터 manifest를 확인할 수 있습니다.

```bash
bash run_scripts/JIHYE_261020/train_nvs.bash --prepare-only
```

## 재개

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DATA_ROOT=/data/gard \
bash run_scripts/JIHYE_261020/20261013_train.bash \
  --resume /absolute/path/to/run/latest.pt
```

학습 launcher는 기본 설정상 W&B 인증이 필요합니다. `WANDB_ENV_FILE` 또는 `WANDB_API_KEY`가 준비되어 있어야 합니다. `local_nvs.yaml`은 먼저 `scripts/train/nvs/prepare_handoff_config.py`로 생성해야 합니다.
