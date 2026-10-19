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

### NVS 학습: V=4 고정, N=2~3, M=4-N

현재 NVS 학습의 전체 view 수 `V`는 4로 고정되어 있습니다. 입력 source view `N`은 매 microstep마다 2 또는 3으로 선택되고, novel view `M=4-N`입니다. 따라서 가능한 조합은 `(N,M)=(2,2)`와 `(3,1)`입니다. 입력 N개만 DA3 LQ 추출에 사용하며, M RGB는 all-clean teacher 감독에만 사용합니다. M 포즈는 Plucker 조건으로 주어집니다. 현재 아래 설정은 synthetic transient-only와 transient+static을 함께 사용하며, 아직 real 학습 경로를 구현한 설정은 아닙니다.

학습 launcher는 다음 Git 경로에 있습니다.

```text
run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash
```

GPU 수와 global batch를 지정하면 launcher가 `accumulation = GLOBAL_BATCH_SIZE / GPU 수`를 자동 계산합니다. 예를 들어 GPU 4개에서 global batch 16이면 GPU당 batch 1, gradient accumulation 4입니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash
```

실행 전에 config와 경로만 검증하려면:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash --prepare-only
```

이 bash launcher가 데이터 경로와 학습 config를 생성하고, GPU 수에 맞춰 gradient accumulation을 계산한 뒤 `scripts/train/nvs/train.py`를 DDP로 실행합니다. 직접 `prepare_handoff_config.py`와 `torch.distributed.run`을 호출하는 방식은 이식용 저수준 경로이며, 현재 서버의 표준 실행 경로는 위 bash입니다.

동일 launcher에서 주요 flow/NVS ablation을 바꿀 수 있습니다. 옵션은 자동으로 experiment name과 frozen `launch_config.yaml`에 기록됩니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
NVS_BRANCH=true FLOWMAP_FROM=noise_lq \
DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash
```

주요 ablation 실행 예시는 다음과 같습니다.

```bash
# 기본: NVS branch ON, flow map은 noise+LQ -> HQ
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=true FLOWMAP_FROM=noise_lq \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash

# flow map만 noise -> HQ로 변경
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=true FLOWMAP_FROM=noise \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash

# NVS branch를 끄고 source restoration만 학습
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=false FLOWMAP_FROM=noise \
  DISTRACTOR_AWARE_ATTN_ALIGN=false ATTENTION_ALIGNMENT=true \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash

# distractor-aware attention alignment 추가
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  NVS_BRANCH=true FLOWMAP_FROM=noise \
  DISTRACTOR_AWARE_ATTN_ALIGN=true ATTENTION_ALIGNMENT=true \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash
```

`FLOWMAP_FROM=noise_lq`는 source flow를 `LQ+0.3ε→HQ`로 시작하고 LQ condition을 유지합니다. `FLOWMAP_FROM=noise`는 source와 novel의 flow state를 `ε→HQ`로 시작하지만, LQ는 model condition으로 계속 제공합니다. 즉 두 옵션은 flow target/path만 비교하며 conditioning 유무를 섞지 않습니다. `NVS_BRANCH=false`는 novel view token/loss와 camera branch를 제외하고 source restoration만 학습합니다. `DISTRACTOR_AWARE_ATTN_ALIGN`은 attention target에서 distractor key를 제외하고 clean/distractor query group을 분리해 계산합니다.

초기 체크포인트 `ckpts/gard_denoiser.pt`가 필요합니다. 경로가 다르면 실행 전에 `INIT_CHECKPOINT=/path/to/gard_denoiser.pt`를 지정하세요. W&B 인증은 본인 환경에서 설정하며 저장소에는 API key를 포함하지 않습니다.

설정은 [nvs_v4_n2_3.yaml](run_configs/handoff/nvs_v4_n2_3.yaml): GARD FP32, 504 해상도, GPU당 1그룹, GPU 4개 × accumulation 4 = global batch 16, 30 epochs, LR 2e-4, EMA 0.9995입니다. GPU 수를 바꾸면 bash가 global batch에 맞춰 accumulation을 조정합니다. Mask head OFF, attention alignment ON, mask-aware attention OFF입니다. Frozen DA3는 기존 내부 mixed precision을 유지합니다.

Loss = N clean group MSE + N distractor group MSE + M 전체 token MSE + attention alignment이며 가중치는 각각 1입니다. N camera token은 clean group에 포함되고 M camera token은 M의 전체 token 평균에 포함됩니다. Camera token에 별도 loss 가중치를 부여하지 않습니다. M에만 Plucker pose 조건을 주고 N에는 LQ feature를 addition으로 제공합니다.

매 epoch transient-only에서 10,000그룹을 샘플링하고 transient+static의 train variant를 모두 한 번씩 사용합니다. EMA checkpoint는 1·3·10·20·30 epoch에 저장하며 `latest.pt`에는 model/EMA/optimizer/RNG 등 재시작 상태를 저장합니다. 재개 시 동일 명령에 `--resume result_train/handoff_nvs/checkpoints/latest.pt`를 추가하세요. 이 이식용 명령은 producer의 fixed eval을 자동 실행하지 않습니다. YAML의 `eval_epochs` 목록은 producer용 기록이며, 수신 환경의 평가 루틴을 별도로 실행하세요.


## 2026-10-14 추가 업데이트: V=4, N=2~3 가변 학습

전체 V=4와 all-clean teacher 4뷰는 고정하고, 매 microstep마다 입력 N을 2 또는 3에서 균등하게 선택합니다. Novel M은 4−N으로, `(N,M)=(2,2)` 또는 `(3,1)`입니다. 같은 microstep의 모든 DDP rank는 같은 N을 사용합니다. Transient-only와 transient+static 모두 이 규칙을 적용하며 negative variant의 epoch당 1회 방문은 유지합니다.

Source DA3에는 N뷰 RGB만 들어가고, teacher DA3에는 항상 clean N+M=4뷰가 들어갑니다. 모든 M뷰에 각각 Plucker 조건을 제공합니다. N camera token은 clean loss에 포함하며 M loss는 M의 모든 camera/patch token에 대한 평균입니다. 여러 M뷰를 추가했다고 M loss를 합산하지 않습니다. 포즈는 기존 source baseline 기반 DA3 scale 정렬을 유지하므로 이 설정에서는 N≥2가 필요합니다.

새 이식용 기본 설정은 [nvs_v4_n2_3.yaml](run_configs/handoff/nvs_v4_n2_3.yaml)입니다. 위 `prepare_handoff_config.py` 명령은 이제 이 가변 설정을 기본으로 사용합니다. 기존 고정 N=3 설정이 필요하면 `--template run_configs/handoff/nvs_v4_n3_m1.yaml`을 명시하세요. `nvs` 항목이 없는 기존 run config는 계속 N=3, M=1로 처리됩니다.

현재 서버에서 새 학습을 실행하려면:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 \
  bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash
```

기본 GPU는 0,1,2,3이며 FP32, attention alignment ON, mask-aware OFF입니다. 기존 1·3·10·20·30 EMA 저장, `latest.pt` 재시작, fixed eval 일정을 유지합니다. Fixed eval은 기존 평가 계획의 N=4, M=1 프로토콜을 사용하므로 학습의 V=4와 구분해야 합니다. 실행 중인 기존 학습을 자동으로 중단하거나 재시작하지 않습니다.
