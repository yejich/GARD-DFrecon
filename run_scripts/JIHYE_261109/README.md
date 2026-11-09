# JIHYE 2026-11-09 — clean-anchor 확장 / cross-view positive 출처 비교

기존 `run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash`를 재사용합니다.
실제 전달했던 261028 설정을 기준으로 SupCon만 변경합니다. 261030의
`first_relative` pose로 자동 변경하지 않습니다. 기존 실험에서 지정했던
`--nvs`, `--flow`, `--distractor-aware`, `--camera-pose-mode` 옵션을 그대로 붙이세요.

## 두 실험의 공통 loss

- Anchor: source 입력뷰의 예측 clean latent x0-hat. Distractor와 clean 패치 모두 포함.
- Self-positive: 두 버전 모두 같은 위치의 detached clean GT teacher feature.
- Cross-positive: 다른 source view마다 visibility를 만족하는 geometry 대응을 고른 뒤,
  그 위치가 입력의 distractor mask상 clean인 경우에만 사용. View당 최대 1개.
- Negative: 같은 sample의 모든 source view에서 관측된 distractor feature. 최대 64개.
  Distractor anchor는 자기 입력 negative를 유지하고, clean anchor는 pool에서 샘플링.
- CLS와 novel view는 anchor/key에서 제외. Teacher source feature 자체는 기존처럼
  clean source+novel forward에서 추출될 수 있음.
- Positive/negative key는 모두 detach. Geometry와 mask에 의한 correspondence 선택은
  두 버전 모두 기존 teacher geometry와 학습 mask를 사용.
- 공통 softmax 분모는 self + 유효 cross positives + sampled negatives.
  Anchor loss = self의 -log probability + cross들의 -log probability 평균.
  Cross가 없으면 self 항만 사용. Sample에 distractor negative가 없으면 SupCon 제외.
- 유효 anchor들을 기존 time weighting으로 평균. Clean/distractor anchor를 별도로
  균형화하지 않음. 기존 positive 전체 평균과 달리 두 항을 합산하므로 cross가 있는
  anchor의 계수 합은 2. SupCon lambda 0.05, temperature 0.07은 기존 값 유지.

`clean_gt`: cross-positive도 clean GT teacher에서 가져옴.
`observed`: cross-positive만 관측 입력 feature에서 가져옴. Self는 여전히 clean GT.
따라서 두 버전 간 차이는 cross-positive feature 출처 하나뿐입니다.

## 실행

먼저 `git pull --ff-only origin main`으로 업데이트하세요. GPU당 batch 2의
초기 메모리 검사 수정 commit `22d0dac`이 포함되어야 합니다.
`git merge-base --is-ancestor 22d0dac HEAD`가 성공하는지 확인할 수 있습니다.

기존 전달 때와 동일하게 `HYPERSIM_CAMERA_ROOT`, `HYPERSIM_CAMERA_PARAMETERS`
환경변수에 서버의 카메라 데이터 폴더와 metadata CSV 경로를 설정하세요.
기존 launcher의 경로 검사를 그대로 사용합니다.

저장소 루트에서, 각 실험에 할당된 GPU로 독립 실행합니다.
총 4개 실험은 cross-positive 출처 2종 × NVS ON/OFF입니다.
Flow=noise_lq, D-attn ON, attention alignment ON, pose=da3_aligned는 동일합니다.
각 wrapper는 NVS 조건을 고정하며 나머지 기존 옵션은 그대로 전달합니다.

```bash
# 1. GT cross-positive / NVS ON
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261109/train_clean_gt_nvs_on.bash

# 2. Observed cross-positive / NVS ON
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261109/train_observed_nvs_on.bash

# 3. GT cross-positive / NVS OFF
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261109/train_clean_gt_nvs_off.bash

# 4. Observed cross-positive / NVS OFF
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261109/train_observed_nvs_off.bash
```

같은 GPU로 위 네 명령을 동시에 실행하지 마세요. 기본 global batch 16,
per-GPU batch 2는 기존 launcher와 동일합니다. Batch를 변경할 때는 두 실험에
동일한 `--global-batch` / `--per-gpu-batch` 옵션을 적용하세요.

기본 출력 폴더는 다음 prefix 뒤에 NVS/flow/D-attn/pose 등의 조건이 붙습니다.
NVS ON/OFF는 각각 `_nvstrue_` / `_nvsfalse_`로 구분되어 총 4개 폴더가 생성됩니다.

- `result_train/YYYYMMDD_crossview_supcon_allanchors_selfgt_cross_clean_gt_.../`
- `result_train/YYYYMMDD_crossview_supcon_allanchors_selfgt_cross_observed_.../`

`launch_config.yaml`, 소스 snapshot, train.log 및 기존 구조의 checkpoints가 저장됩니다.
재실행 시 별도 위치를 원하면 `--out /absolute/new/run`을 지정하세요.
설정 파일은 `run_configs/train/crossview_supcon_20261109/{clean_gt,observed}.yaml`입니다.
기존 초기 checkpoint, dataset, optimizer, epoch, 저장 주기는 그대로 사용합니다.
자동 fixed evaluation은 기본 OFF입니다. `EVAL_EPOCHS=1,3,10,20,30`으로
명시적으로 요청한 경우에만 기존 평가 pipeline을 실행합니다.

```bash
# GPU 학습 없이 설정 준비 및 기존 파일 경로 검사
bash run_scripts/JIHYE_261109/train_crossview_supcon.bash clean_gt --prepare-only
bash run_scripts/JIHYE_261109/train_crossview_supcon.bash observed --prepare-only

# GPU가 할당된 뒤 1-step smoke test (별도 출력 폴더)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261109/train_crossview_supcon.bash clean_gt \
  --nvs on --per-gpu-batch 1 --max-steps 1 --out /absolute/smoke/clean_gt
```

이 변경에 대한 검증: 기존 NVS loss 테스트 + 두 key 출처의 수식/gradient,
self GT 유지, clean anchor gradient, CLS/novel 제외, cross/negative 부재,
legacy 기본값, config 차이 검사. 실제 장시간 GPU 학습은 지혜님 환경에서 실행합니다.

## 전달 전 검증 범위

- `22d0dac` 별도 checkout에서 train.py import 및 테스트 18개를 확인했습니다.
  네 wrapper 모두 `--prepare-only` 성공, global batch 16, NVS 설정,
  cross-positive 출처, self GT 유지, 자동 평가 OFF를 확인했습니다.
- `--prepare-only`는 설정과 주요 경로 검사이며 모든 이미지 존재나 GPU 학습 성공을
  보장하지 않습니다. 기존에 학습했던 서버의 데이터 root/manifest 구성을 유지하세요.
- 현재 개발 서버는 데이터 폴더명이 변경되어, 원격 manifest를 그대로 적용하면
  `hypersim_negative/scenes_002/.../clean.png`가 없는 경우가 확인되었습니다.
  이는 loss 변경과 별개인 데이터 경로 차이입니다. 지혜님 서버에서도 실제 경로가
  다르다면 서버의 데이터와 일치하는 manifest를 사용해야 합니다.
- 개발 서버의 관측 cross-positive / NVS ON은 GPU 0,1, global batch 8
  (per-GPU batch 1 × 2 GPUs × accumulation 4)로 실제 첫 optimizer step까지
  확인했습니다. 서버에 맞게 조정된 기존 데이터 경로와 로컬 camera resolver를
  사용했습니다. 지혜님 서버의 파일 존재나 나머지 세 설정의 실제 GPU 학습을
  검증한 것은 아닙니다.
- GPU당 batch 2에서도 실제 forward/backward 메모리 검사는 통과했습니다
  (peak allocated 약 41.4 GiB). 다른 평가와 GPU를 공유하면 OOM이 발생했으므로
  학습 GPU는 독점 사용하세요. 필요하면 `--per-gpu-batch 1`로 microbatch를 줄여도
  global batch는 유지되고 accumulation이 자동 증가합니다.
- 각 wrapper에 `--max-steps 1 --out /absolute/unique/smoke_directory`를 붙여
  서버의 실제 데이터·모델로 먼저 확인할 수 있습니다. Smoke 출력은 본 학습과
  다른 폴더를 사용하세요.
