# JIHYE 2026-10-28 — geometry SupCon NVS

실행 파일은 `train_nvs_supcon_softself.bash`입니다. 기본 설정은 다음과 같습니다.

- mixed Hypersim + negative 데이터
- `geometry_contrastive_loss.use: true`
- geometry-positive / distractor-token global-negative SupCon
- distractor-aware attention alignment ON
- soft self-view behavior
- mask-aware attention OFF
- NVS branch OFF, `flowmap_from=noise_lq`
- W&B OFF

저장소 루트에서 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash
```

실행 전 설정 생성과 파일 검사는 다음처럼 할 수 있습니다.

```bash
bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --prepare-only
CUDA_VISIBLE_DEVICES=0,1 \
  bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --max-steps 1
```

기본 config는 `run_configs/train/train_GARD_da3_nvs_v4_n2_3_mixed_supcon_softself.yaml`입니다. 다른 경로를 사용할 때는 `NVS_CONFIG=/absolute/path/to/config.yaml`로 지정합니다.
