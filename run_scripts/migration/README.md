# cvlab23로 현재 camera-separate 학습 재개

대상: `20261110_192202_jihye_ep10_camera_separate_nvs_on_supcon_on_gb16_gpu01`.
이전 cross-positive observed/global-batch-8 실험과 다릅니다.
현재 설정은 NVS ON, 기존 geometry SupCon, source camera loss 별도,
GPU 2개 × per-GPU batch 1 × accumulation 8 = global batch 16입니다.

## 전송

완료된 `latest.pt`를 같은 파일시스템에서 hard link하여 `checkpoints/resume.pt`로
고정한 migration bundle을 전송합니다. 학습의 atomic checkpoint 교체와 무관하게
고정된 파일을 보낼 수 있습니다. `migration.json`에 step/epoch/hash가 있습니다.
모델, EMA, optimizer, rank별 RNG, epoch, next_batch가 모두 포함됩니다.

```bash
# cvlab20: 원격 계정과 저장 경로를 지정 (SSH 키 인증 필요)
rsync -a --partial --append-verify --protect-args --info=progress2 \
  /absolute/local/migration_bundle/ USER@cvlab23:/absolute/remote/migration_bundle/
```

원격 bundle의 부모 폴더는 미리 생성하세요. 원본 학습은 전송만으로 종료되지
않습니다. 이전 checkpoint로 재개하면 snapshot 이후 원본의 진행분은 포함되지
않으므로, 완전히 전환할 때에는 마지막 완료 checkpoint를 다시 고정·전송하세요.
전송 완료 후 원격에서:

```bash
cd /absolute/remote/migration_bundle
sha256sum -c SHA256SUMS
```

코드는 이 README를 포함하는 commit까지 가져와야 합니다. 원격 Git 접근 대신
bundle에 `migration-code.bundle`이 있으면 기존 clone에서 아래처럼 적용할 수 있습니다.
코드 bundle은 기준 commit `4bf8521`이 존재하는 clone용 incremental bundle입니다.

```bash
cd /absolute/path/to/GARD-DFrecon
git bundle verify /absolute/remote/migration_bundle/migration-code.bundle
git fetch /absolute/remote/migration_bundle/migration-code.bundle HEAD
git merge --ff-only FETCH_HEAD
```

## cvlab23 실행

기존 학습 환경과 호환되는 Python/PyTorch/DA3 의존성을 준비합니다. 데이터와
DA3 encoder weights는 별도이며 checkpoint 전송에 포함되지 않습니다.
Manifest는 bundle의 동일 파일을 사용하고 데이터 내용/프레임 선택은 변경하지 않습니다.
카메라 경로는 cvlab23에 실제 HDF5 scene 폴더가 있는 위치를 지정합니다.

```bash
cd /absolute/path/to/GARD-DFrecon
export HYPERSIM_PAIRS_ROOT=/absolute/path/to/hypersim_pairs
export HYPERSIM_NEGATIVE_ROOT=/absolute/path/to/hypersim_negative
export HYPERSIM_CAMERA_ROOT=/absolute/path/to/hypersim/data
export HYPERSIM_CAMERA_PARAMETERS=/absolute/path/to/metadata_camera_parameters.csv
export CUDA_VISIBLE_DEVICES=0,1
# 필요할 때 PYTHON=/absolute/path/to/python 지정; 기본 .venv/bin/python

bash run_scripts/migration/resume_cvlab23.bash /absolute/remote/migration_bundle --prepare-only
bash run_scripts/migration/resume_cvlab23.bash /absolute/remote/migration_bundle
```

`--prepare-only`는 checkpoint 구조, GPU 수, manifest 해시, 데이터/카메라 root를
검사하고 `resumed/resume_config.yaml`을 만듭니다. 모든 이미지 존재나 GPU 메모리를
검증하는 것은 아닙니다. 기본 출력은 bundle의 `resumed/`; `RESUME_OUT`으로 변경할 수 있습니다.

재개는 `--resume`만 사용합니다. `--warm-start-ema`를 사용하면 optimizer/RNG가
초기화되므로 migration resume에 사용하지 않습니다. 원래 world size 2와 batch/
accumulation을 유지합니다. 하드웨어·라이브러리 차이에 따른 수치 차이는 있을 수 있습니다.
