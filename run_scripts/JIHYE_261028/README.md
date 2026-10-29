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

실행 전 설정 생성과 파일 검사는 다음처럼 할 수 있습니다.

```bash
bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --prepare-only
CUDA_VISIBLE_DEVICES=0,1 \
  bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --max-steps 1
```

기본 config는 `run_configs/train/train_GARD_da3_nvs_v4_n2_3_mixed_supcon_softself.yaml`입니다. 다른 경로를 사용할 때는 `NVS_CONFIG=/absolute/path/to/config.yaml`로 지정합니다.
