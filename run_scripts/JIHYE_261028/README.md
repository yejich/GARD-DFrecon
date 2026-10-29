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

아래 명령은 네 축의 전체 조합 중, 2026-10-20에 이미 실행한 조합을 제외한 15개입니다. 각 명령은 독립적인 run directory를 생성합니다. GPU를 바꾸려면 `CUDA_VISIBLE_DEVICES`만 수정하면 됩니다.

이미 실행한 조합:

```text
distractor-aware=off, flow=noise_lq, nvs=on, supcon=off
result_train/20261020_nvs_push_only_nvstrue_flownoise_lq_dattnfalse_aligntrue_000457_414860899
```

```bash
# distractor-aware ON
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise --nvs on --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise --nvs on --supcon off
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise --nvs off --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise --nvs off --supcon off
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise_lq --nvs on --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise_lq --nvs on --supcon off
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise_lq --nvs off --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware on --flow noise_lq --nvs off --supcon off

# distractor-aware OFF
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware off --flow noise --nvs on --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware off --flow noise --nvs on --supcon off
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware off --flow noise --nvs off --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware off --flow noise --nvs off --supcon off
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware off --flow noise_lq --nvs on --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware off --flow noise_lq --nvs off --supcon on
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --distractor-aware off --flow noise_lq --nvs off --supcon off
```

위 명령 중 하나를 그대로 복사해서 실행하면 됩니다. 여러 실험을 동시에 실행할 때는 각 명령의 `CUDA_VISIBLE_DEVICES=0,1,2,3`를 서로 다른 GPU 조합으로 바꿔야 합니다. `--nvs off`에서도 flow 옵션은 config에 기록되므로 실험 조건을 일관되게 관리할 수 있지만, 실제 NVS branch가 꺼져 있어 flow 차이는 학습 loss에 적용되지 않습니다.

실행 전 설정 생성과 파일 검사는 다음처럼 할 수 있습니다.

```bash
bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --prepare-only
CUDA_VISIBLE_DEVICES=0,1 \
  bash run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash --max-steps 1
```

기본 config는 `run_configs/train/train_GARD_da3_nvs_v4_n2_3_mixed_supcon_softself.yaml`입니다. 다른 경로를 사용할 때는 `NVS_CONFIG=/absolute/path/to/config.yaml`로 지정합니다.
