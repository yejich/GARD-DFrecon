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

## 2. 데이터 다운로드 및 구분

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

## 3. NVS 학습: V=4, N=3, M=1

전체 뷰 V = 입력 N + novel M입니다. 입력 N개만 DA3 LQ 추출에 사용하며 M RGB는 all-clean teacher 감독에만 사용합니다. M 포즈는 Plucker 조건으로 주어집니다. 현재 아래 설정은 synthetic transient-only와 transient+static을 함께 사용하며, 아직 real 학습 경로를 구현한 설정은 아닙니다.

```bash
python scripts/train/nvs/prepare_handoff_config.py \
  --data-root /data/hypersim_synthetic_full \
  --output run_configs/handoff/local_nvs.yaml \
  --wandb-entity YOUR_ENTITY
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/train/nvs/train.py \
  --config run_configs/handoff/local_nvs.yaml \
  --out result_train/handoff_nvs \
  --checkpoint-dir result_train/handoff_nvs/checkpoints
```

초기 체크포인트 `ckpts/gard_denoiser.pt`가 필요합니다. 경로가 다르면 config 준비 명령에 `--init-checkpoint /path/to/gard_denoiser.pt`를 추가하세요. W&B 인증은 본인 환경에서 설정하며 저장소에는 API key를 포함하지 않습니다.

설정은 [nvs_v4_n3_m1.yaml](run_configs/handoff/nvs_v4_n3_m1.yaml): GARD FP32, 504 해상도, GPU당 1그룹, GPU 2개 × accumulation 4 = global batch 8, 30 epochs, LR 2e-4, EMA 0.9995입니다. 위 명령은 GPU 2개 기준이며 다른 GPU 수에서는 global batch에 맞춰 accumulation을 조정해야 합니다. Mask head OFF, attention alignment ON, mask-aware attention OFF입니다. Frozen DA3는 기존 내부 mixed precision을 유지합니다.

Loss = N clean group MSE + N distractor group MSE + M 전체 token MSE + attention alignment이며 가중치는 각각 1입니다. N camera token은 clean group에 포함되고 M camera token은 M의 전체 token 평균에 포함됩니다. Camera token에 별도 loss 가중치를 부여하지 않습니다. M에만 Plucker pose 조건을 주고 N에는 LQ feature를 addition으로 제공합니다.

매 epoch transient-only에서 10,000그룹을 샘플링하고 transient+static의 train variant를 모두 한 번씩 사용합니다. EMA checkpoint는 1·3·10·20·30 epoch에 저장하며 `latest.pt`에는 model/EMA/optimizer/RNG 등 재시작 상태를 저장합니다. 재개 시 동일 명령에 `--resume result_train/handoff_nvs/checkpoints/latest.pt`를 추가하세요. 이 이식용 명령은 producer의 fixed eval을 자동 실행하지 않습니다. YAML의 `eval_epochs` 목록은 producer용 기록이며, 수신 환경의 평가 루틴을 별도로 실행하세요.

기존 DA3/Omega 학습 방법은 [이전 README](https://github.com/yejich/GARD-DFrecon/blob/7293b67a76e6d1c3cece49af813e5578800ce9d7/README.md)에 보존되어 있습니다.
