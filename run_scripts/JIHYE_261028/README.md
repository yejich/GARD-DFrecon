# JIHYE 2026-10-28 — geometry SupCon NVS

실행 파일은 `train_nvs_supcon_softself.bash`입니다. 기본 설정은 다음과 같습니다.

- mixed Hypersim + negative 데이터
- `geometry_contrastive_loss.use: true`
- geometry-positive / distractor-token global-negative SupCon
- distractor-aware attention alignment ON
- soft self-view behavior
- mask-aware attention OFF
- NVS branch OFF, `flowmap_from=noise_lq`
- global batch 16, per-GPU batch 2, 4 GPUs 기준 gradient accumulation 2
- W&B OFF

저장소 루트에서 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash
```

위 기본값은 `GLOBAL_BATCH_SIZE=16`, `PER_GPU_BATCH_SIZE=2`이므로 GPU 4개에서 accumulation이 자동으로 2가 됩니다. GPU 수나 batch를 바꿀 때도 `GLOBAL_BATCH_SIZE = GPU 수 × PER_GPU_BATCH_SIZE × accumulation`이 되도록 맞춥니다.

이 launcher는 학습만 수행하며 optional fixed-eval pipeline은 자동 실행하지 않습니다. 학습에 필요한 pair/negative 이미지와 `ckpts/gard_denoiser.pt`는 서버에 별도로 준비되어 있어야 합니다.

설정 ablation은 실행 옵션으로 바로 바꿀 수 있습니다.

```bash
# NVS ON + noise flow + SupCon OFF
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash \
  --nvs on --flow noise --supcon off

# SupCon coefficient / negative pool 변경
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash \
  --supcon on --supcon-lambda 0.1 --supcon-temperature 0.1 \
  --supcon-max-negatives 128

# 전체 옵션 확인
bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --help
```

## 전체 ablation 명령어

비교할 네 가지 축은 다음과 같습니다.

- `--distractor-aware on|off`: distractor-aware attention alignment
- `--flow noise|noise_lq`: flow-map 초기화 방식
- `--nvs on|off`: NVS branch 사용 여부
- `--supcon on|off`: geometry SupCon loss 사용 여부

아래 명령은 네 축의 전체 조합 중, 이미 실행한 5개 조합을 제외한 11개입니다. 각 명령은 독립적인 run directory를 생성합니다. GPU를 바꾸려면 `CUDA_VISIBLE_DEVICES`만 수정하면 됩니다.

이미 실행한 조합:

```text
20261026_nvs_v4_n2-3_nvsfalse_flownoise_dattnfalse_aligntrue_15
20261026_nvs_v4_n2-3_nvsfalse_flownoise_lq_dattnfalse_aligntrue_15
20261026_nvs_v4_n2-3_nvsfalse_flownoise_lq_dattntrue_aligntrue_15
20261026_nvs_v4_n2-3_nvstrue_flownoise_dattnfalse_aligntrue_10
20261026_nvs_v4_n2-3_nvstrue_flownoise_lq_dattnfalse_aligntrue_10
```

```bash
# 1차
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware off --flow noise --nvs on --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware off --flow noise_lq --nvs on --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise --nvs on --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise --nvs off --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise_lq --nvs on --supcon on



# 2차
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise_lq --nvs on --supcon off
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise_lq --nvs off --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise --nvs on --supcon off

```




위 명령 중 하나를 그대로 복사해서 실행하면 됩니다. 여러 실험을 동시에 실행할 때는 각 명령의 `CUDA_VISIBLE_DEVICES=0,1,2,3`를 서로 다른 GPU 조합으로 바꿔야 합니다. `--nvs off`에서도 flow 옵션은 config에 기록되므로 실험 조건을 일관되게 관리할 수 있지만, 실제 NVS branch가 꺼져 있어 flow 차이는 학습 loss에 적용되지 않습니다.

실행 전 설정 생성과 파일 검사는 다음처럼 할 수 있습니다.

```bash
bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --prepare-only
CUDA_VISIBLE_DEVICES=0,1 \
  bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --max-steps 1
```

기본 config는 `run_configs/train/train_GARD_da3_nvs_v4_n2_3_mixed_supcon_softself.yaml`입니다. 다른 경로를 사용할 때는 `NVS_CONFIG=/absolute/path/to/config.yaml`로 지정합니다.

## Stage 1: NVS-only 사전학습

Hypersim, TartanAir, DL3DV의 clean multi-view 데이터만 사용해 NVS branch를 먼저 학습할 수 있습니다. 이 단계에서는 distractor-aware attention, attention alignment, distractor loss, SupCon을 모두 끄고 `nvs_branch=true`, `full_view_loss=true`로 실행합니다. Variable-N loader가 현재 per-GPU microbatch 1을 요구하므로, GPU 4개에서 global batch 16은 accumulation 4로 맞춰집니다.

세 데이터셋 모두 로컬에 없다면 아래 다운로드 명령으로 준비할 수 있습니다. Hypersim과 TartanAir는 저장소에 포함된 upstream 다운로드 스크립트를 사용하고, DL3DV는 접근 승인 후 공식 downloader를 사용합니다. 데이터 root 구조는 다음과 같아야 합니다.

```text
$HYPERSIM_ROOT/data/ai_*/...
$TARTANAIR_ROOT/data/<environment>/Easy/<sequence>/image_left/
$DL3DV_ROOT/<split>/<scene>/transforms.json
```

Hypersim 전체 다운로드는 용량이 크므로 필요한 scene만 `download_scripts/hypersim/hypersim.py`의 URL 목록에 남겨서 실행하는 것도 가능합니다. 기본 스크립트는 Phase-1 manifest가 기대하는 `$HYPERSIM_ROOT/data/ai_*` 구조로 압축을 풉니다.

```bash
export HYPERSIM_ROOT="$PWD/data/train/hypersim"
mkdir -p "$HYPERSIM_ROOT"

# Apple Hypersim scene archives: download + extract.
bash download_scripts/hypersim/download_hypersim.sh
```

TartanAir는 NVS manifest에 필요한 left RGB와 pose를 포함하도록 Easy trajectory를 받습니다. 기본 스크립트는 Hugging Face mirror를 사용해 left RGB/depth를 내려받습니다. 전체 데이터는 매우 크므로, 우선 RGB만 받으려면 아래처럼 실행할 수 있습니다.

```bash
export TARTANAIR_ROOT="$PWD/data/train/tartanair"
mkdir -p "$TARTANAIR_ROOT"
python -m pip install -U huggingface_hub colorama

# Recommended: left RGB + depth, Easy trajectories, Hugging Face mirror.
bash download_scripts/tartanair_tools/download_tartanair.sh

# Smaller first pass: left RGB only.
python download_scripts/tartanair_tools/download_training.py \
  --output-dir "$TARTANAIR_ROOT" --rgb --only-left --only-easy \
  --unzip --huggingface
```

DL3DV는 접근 승인 후 공식 downloader에서 필요한 subset/resolution을 선택합니다.

```bash
export DL3DV_ROOT="$PWD/data/train/dl3dv"
mkdir -p "$DL3DV_ROOT"

# DL3DV-10K at 480P: download the complete 10K split.
# Run after accepting the dataset terms and installing the official downloader.
python /path/to/DL3DV-10K/Dataset/scripts/download.py \
  --odir "$DL3DV_ROOT" --subset 10K --resolution 480P \
  --file_type images+poses --clean_cache
```

각 dataset은 원본 라이선스와 다운로드 약관을 확인해야 하며, Hypersim/TartanAir 전체 다운로드는 수백 GB 이상이 될 수 있습니다. 이미 다운로드한 root가 있다면 다시 받을 필요 없이 아래 manifest 생성 단계에서 그 root만 지정합니다.

다운로드가 끝나면 Hypersim, TartanAir, DL3DV의 실제 경로를 positional argument로 입력해 manifest를 생성합니다.
지혜님께 전달하는 공식 Stage-1 manifest는 **전체 scene 사용**이 기본입니다. `MAX_SCENES=0`은 scene 수 제한이 없다는 뜻이며, 기존 dataset별 100-scene pilot manifest는 재사용하지 않습니다.

```bash
HYPERSIM_CAMERA_CSV=/absolute/path/to/metadata_camera_parameters.csv \
PHASE1_MANIFEST=/absolute/path/to/phase1_nvs_manifest.json \
MAX_SCENES=0 \
bash run_scripts/JIHYE_261028/prepare_nvs_stage1_data.bash \
  /absolute/path/to/hypersim \
  /absolute/path/to/tartanair \
  /absolute/path/to/dl3dv
```

Stage 1 실행 명령은 다음과 같습니다.

```bash
# Stage 1: pure noise flow-map
FLOWMAP_FROM=noise \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PHASE1_MANIFEST=/absolute/path/to/phase1_nvs_manifest.json \
bash run_scripts/JIHYE_261028/train_nvs_stage1.bash

# Stage 1: noise + low-quality latent flow-map
FLOWMAP_FROM=noise_lq \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PHASE1_MANIFEST=/absolute/path/to/phase1_nvs_manifest.json \
bash run_scripts/JIHYE_261028/train_nvs_stage1.bash
```

기본 Stage 1 출력은 `result_train/YYYYMMDD_nvs_stage1/`이며, `EPOCHS`, `STEPS_PER_EPOCH`, `OUT`, `INIT_CHECKPOINT`, `WANDB_ENABLED`, `FLOWMAP_FROM` 환경변수로 변경할 수 있습니다. 이어서 학습할 때는 다음처럼 `latest.pt`를 지정합니다.

```bash
RESUME=/absolute/path/to/result_train/YYYYMMDD_nvs_stage1/checkpoints/latest.pt \
bash run_scripts/JIHYE_261028/train_nvs_stage1.bash
```
