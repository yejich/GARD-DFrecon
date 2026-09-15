# JIHYE님께 요청할 실험

Bash는 `run_scripts/JIHYE/`, 같은 이름의 설정은 `run_configs/JIHYE/`에서 관리합니다. 아래 명령은 GARD-DFrecon 루트에서 실행합니다.

## HF 배포본 Hypersim pairs로 DA3 10 epochs 재학습

- Bash: `train_GARD_da3_hypersim_completed_260915.bash`
- Config: `../../run_configs/JIHYE/train_GARD_da3_hypersim_completed_260915.yaml`
- 초기 가중치: `ckpts/gard_denoiser.pt`. 기존 10-epoch 학습 checkpoint에서 resume하지 않습니다.
- DA3-GIANT-1.1 backbone은 고정하고 GARD denoiser를 학습합니다.
- GPU 2/4/8개, global batch 8, GPU당 microbatch 1, gradient accumulation 자동 4/2/1, fp32.
- 10 epochs, bf16, LR 2e-5 → 2e-6, warmup 1 epoch.
- view 수 1~4, distractor 확률 0.7, 기존 group2 flow matching 및 attention alignment 설정 유지.
- `ai_001_001`은 평가 전용: 80 pairs / 고정 80개 평가 그룹.
- HF 배포 버전 `a935b4e251faeaf1e96723a28463d44e8913c834`의 pair만 사용합니다. 로컬에 추가 생성된 pair는 제외합니다.
- 고정 manifest: `manifests/hypersim_hf_a935b4e251fa.json`. DA3와 Omega가 같은 목록을 사용합니다.
- 학습 20,885 pairs / 평가 80 pairs. 학습 도중이나 다음 실행에서도 추가 생성분을 자동 포함하지 않습니다.

## 설치 → 다운로드 → 실행

처음 사용하는 서버에서는 저장소 [README](../../README.md)의 1~2절대로 환경·공식 가중치·HF 데이터셋을 준비하세요. 데이터 다운로드 후 `datasets/hypersim_pairs/` 아래에 scene collection들이 있어야 합니다.

```bash
conda activate gard-dfrecon
# GARD-DFrecon 루트에서 실행
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/JIHYE/train_GARD_da3_hypersim_completed_260915.bash
```

기존 venv를 사용하는 경우에는 `source .venv/bin/activate`로 대신 활성화하면 됩니다. Bash는 활성화된 환경의 `python`을 사용합니다.

### 수정할 값은 세 가지

| 설정 | 기본값 | 의미 |
|---|---|---|
| `PROJECT_ROOT` | 현재 디렉터리 | Git으로 clone한 GARD-DFrecon 경로 |
| `DATASET_ROOT` | `$PROJECT_ROOT/datasets/hypersim_pairs` | HF 데이터를 압축 해제한 상위 폴더 |
| `CUDA_VISIBLE_DEVICES` | `0,1` | 사용할 GPU 번호 |

Bash 상단을 수정하거나 실행 시 환경변수로 전달할 수 있습니다.

```bash
PROJECT_ROOT=/work/GARD-DFrecon \
DATASET_ROOT=/data/hypersim_pairs \
CUDA_VISIBLE_DEVICES=2,3 \
bash /work/GARD-DFrecon/run_scripts/JIHYE/train_GARD_da3_hypersim_completed_260915.bash
```

경로에 공백이 있다면 값을 따옴표로 감싸세요. `DATASET_ROOT`에는 절대 경로를 사용하세요. 예시의 `/work`, `/data`는 사용자 서버 경로로 바꿉니다.

DA3·Omega의 `completed_260915.bash`는 지정한 GPU 목록에서 개수를 계산합니다. **GPU 2/4/8개**를 지원하며, 전체 배치 8과 GPU당 microbatch 1을 유지하도록 gradient accumulation을 각각 **4/2/1**로 자동 설정합니다. 다른 GPU 개수는 실행 전에 오류로 안내합니다.

```bash
# GPU 학습 없이 데이터 목록 생성·필수 파일 검사
bash run_scripts/JIHYE/train_GARD_da3_hypersim_completed_260915.bash --prepare-only

# 짧은 학습 확인
bash run_scripts/JIHYE/train_GARD_da3_hypersim_completed_260915.bash --max-steps 1
```

출력은 `result_train/da3_hypersim_completed_10ep/<실행 시각>/` 아래의 실험 폴더에 저장됩니다. 설정, `pairs_manifest.json`, 로그와 `checkpoints/latest.pt`를 포함합니다. 개인 서버의 원래 데이터 폴더나 원래 GARD 저장소는 필요하지 않습니다.

각자가 공식 `gard_denoiser.pt`에서 새로 시작하므로 학습 중간 체크포인트 전달이나 `--resume`은 필요하지 않습니다. 같은 데이터로 비교하려면 같은 HF 데이터셋 버전을 사용하세요. W&B는 기본 비활성화이며 인증 후 `--wandb`를 추가할 수 있습니다. `--prepare-only`는 첫 번째 인자로 전달하세요.

## 기존 DA3 진입점

`train_GARD_da3_hypersim_20k.sh`와 동명 YAML도 이 폴더들로 이동했습니다. Python trainer는 기존 `RAE/src/JIHYE_train_GARD_da3_hypersim_20k.py`를 공유합니다. 과거 `run_scripts/train/JIHYE_...sh` 및 config 경로는 호환용으로 유지하고, 새 실험 안내에는 JIHYE 폴더 경로를 사용합니다.
