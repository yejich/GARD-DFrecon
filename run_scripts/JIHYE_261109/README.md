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

기존 전달 때와 동일하게 `HYPERSIM_CAMERA_ROOT`, `HYPERSIM_CAMERA_PARAMETERS`
환경변수에 서버의 카메라 데이터 폴더와 metadata CSV 경로를 설정하세요.
기존 launcher의 경로 검사를 그대로 사용합니다.

저장소 루트에서, 각 실험에 할당된 GPU로 독립 실행합니다.
아래는 NVS ON / noise_lq / D-attn ON 조합이며 이전에 실행한 다른 조합은
동일한 옵션을 두 명령에 똑같이 적용하면 됩니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261109/train_crossview_supcon.bash clean_gt \
  --nvs on --flow noise_lq --distractor-aware on

CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261109/train_crossview_supcon.bash observed \
  --nvs on --flow noise_lq --distractor-aware on
```

같은 GPU로 위 두 명령을 동시에 실행하지 마세요. 기본 global batch 16,
per-GPU batch 2는 기존 launcher와 동일합니다. Batch를 변경할 때는 두 실험에
동일한 `--global-batch` / `--per-gpu-batch` 옵션을 적용하세요.

기본 출력 폴더는 다음 prefix 뒤에 NVS/flow/D-attn/pose 등의 조건이 붙습니다.

- `result_train/YYYYMMDD_crossview_supcon_allanchors_selfgt_cross_clean_gt_.../`
- `result_train/YYYYMMDD_crossview_supcon_allanchors_selfgt_cross_observed_.../`

`launch_config.yaml`, 소스 snapshot, train.log 및 기존 구조의 checkpoints가 저장됩니다.
재실행 시 별도 위치를 원하면 `--out /absolute/new/run`을 지정하세요.
설정 파일은 `run_configs/train/crossview_supcon_20261109/{clean_gt,observed}.yaml`입니다.
기존 초기 checkpoint, dataset, optimizer, epoch, 저장 주기는 그대로 사용합니다.

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
