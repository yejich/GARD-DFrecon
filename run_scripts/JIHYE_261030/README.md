# JIHYE 2026-10-30 — relative-pose NVS

이 폴더는 mixed Hypersim + negative NVS 학습에서 첫 번째 source view 기준
relative camera pose를 사용하는 전달용 launcher입니다.

기본 설정:

- NVS branch ON 여부는 실행 인자에서 지정
- geometry SupCon은 기본 ON
- camera pose mode는 `first_relative`
- 첫 번째 source camera를 identity로 변환
- 전체 camera translation을 최대 camera distance 기준으로 정규화
- 기존 방식은 `--camera-pose-mode da3_aligned`로 선택 가능

저장소 루트에서 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash run_scripts/JIHYE_261030/train_nvs_supcon_softself.bash \
  --nvs on --supcon on
```

기존 DA3-aligned pose 방식과 비교하려면:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash run_scripts/JIHYE_261030/train_nvs_supcon_softself.bash \
  --nvs on --supcon on --camera-pose-mode da3_aligned
```

새 pose 방식으로 생성된 checkpoint에는 `camera_pose_mode: first_relative`가
저장됩니다. Fixed eval은 checkpoint config를 읽어 학습과 동일한 pose convention을
자동으로 사용합니다. 기존 checkpoint처럼 해당 항목이 없으면 `da3_aligned`로
해석됩니다.

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash run_scripts/JIHYE_261030/train_nvs_supcon_softself.bash \
  --nvs on --supcon on --max-steps 1
```

전체 옵션은 다음으로 확인할 수 있습니다.

```bash
bash run_scripts/JIHYE_261030/train_nvs_supcon_softself.bash --help
```
