# GARD-DFrecon

Hypersim clean–distractor multi-view pair로 [GARD](https://github.com/cvlab-kaist/GARD)를 10 epochs fine-tuning한 실험 코드입니다. 입력의 distractor를 제거하도록 **feature flow의 Group2 MSE**와 기존 GARD attention alignment를 함께 학습합니다.

이 저장소는 upstream GARD 커밋 `de483e0`에 Hypersim pair 학습 코드를 추가한 연구용 확장입니다. 원본 저자·라이선스·코드 이력을 유지합니다. 원본 프로젝트 설명은 [README_GARD.md](README_GARD.md)에 있습니다.

**공유 범위:** 학습 코드, 10-epoch 설정, 당시의 고정 multi-view manifest, loss 기록, 재현 안내. Fine-tuned 체크포인트와 데이터 이미지는 포함하지 않습니다. 사전학습 GARD/DA3 가중치는 원 배포처에서 별도로 받습니다.

## 재현할 실험

| 항목 | 설정 |
|---|---|
| Backbone | DA3-GIANT-1.1, frozen |
| 학습 대상 | 사전학습 GARD denoiser를 초기값으로 fine-tuning |
| 데이터 | Hypersim `*_001`의 기존 `scenes_v2` clean–distractor pairs |
| Train | 41개 씬, 6,743 pairs |
| Holdout | `ai_001_001` 전체, 80 pairs |
| Train views | microbatch마다 N=1–4 균등 선택, DDP rank 간 N 동기화 |
| 그룹 샘플링 | 적격 씬 균등 → anchor 랜덤 → 충분히 겹치는 N−1뷰 랜덤 |
| 겹침 기준 | 같은 카메라 trajectory, clean anchor 기준 GT 가시성 ≥25% |
| Distractor 확률 | 각 입력 뷰에서 독립적으로 p=0.7; HQ는 항상 clean |
| 해상도 | 전체 이미지 378×504, patch size 14 |
| Loss | Group2 feature-flow MSE + 1.0 × attention alignment |
| 학습 | 10 epochs, BF16, LR 2e-5 → 2e-6, warmup 1 epoch |
| Batch | 2 GPUs × 1 group/GPU × accumulation 8 = effective 16 groups |
| EMA | 0.9995 |
| Eval | 고정 4뷰 그룹 80개, 입력 clean/distractor 선택·noise seed 고정 |

Anchor는 다시 선택될 수 있고, 후보끼리 서로 겹치거나 최소 baseline을 만족해야 하는 조건은 없습니다. 모든 뷰가 clean 또는 distractor인 그룹도 허용합니다. 이 holdout은 fine-tuning split이며 backbone/GARD 사전학습에서 보지 않은 씬임을 보증하는 split은 아닙니다.

## 설치

원본의 Python 3.10 환경과 `uv.lock`을 보존했습니다. NVIDIA/CUDA 환경에서 저장소 루트 기준으로 실행합니다. upstream 의존성에 `gsplat` 빌드가 포함되어 있어 CUDA toolkit/nvcc와 C++ 빌드 도구가 필요할 수 있습니다. 실제 학습은 GPU 2개, 각 48GB에서 검증했습니다.

```bash
git clone https://github.com/yejich/GARD-DFrecon.git
cd GARD-DFrecon
uv sync --frozen
source .venv/bin/activate

# 원본 GARD 초기 가중치 (이 저장소의 fine-tuned checkpoint가 아님)
hf download jinlovespho/GARD gard_denoiser.pt --local-dir ckpts
# DA3-GIANT-1.1은 최초 모델 로딩 때 공식 Hugging Face에서 다운로드됩니다.
```

RGB decoder는 이 학습 경로에서 사용하지 않습니다. 원본 GARD RGB 추론까지 사용할 경우 원본 checkpoint 다운로드 스크립트로 decoder도 받으세요.

## 데이터 준비

당시 학습한 그룹은 [manifests/hypersim_001_groups.json](manifests/hypersim_001_groups.json)에 들어 있습니다. **이미지 파일이 없는 상태에서는 학습할 수 없습니다.** 기존에 생성한 `scenes_v2`를 별도 전달받아 아래처럼 지정합니다.

```bash
export HYPERSIM_PAIRS_ROOT=/datasets/hypersim_pairs/scenes_v2
python scripts/check_hypersim_pairs.py
```

필요한 구조:

```text
$HYPERSIM_PAIRS_ROOT/
  ai_001_001/cam_00/frame.0000/
    clean.png
    distractor.png
    mask_object.png
    mask_shadow.png
  ai_002_001/...
```

총 6,823 pairs의 위 네 파일 **27,292개**를 사용합니다. 기존 snapshot에 대응하는 데이터 크기는 약 **14.45GB**입니다. 이 고정 manifest로 학습할 때 원본 Hypersim HDF5, GLB, `review.jpg`, `shadow_strength_u16.png`는 필요하지 않습니다.

데이터가 있는 서버에서 전달용 archive를 만들 수 있습니다. Git에는 넣지 않습니다.

```bash
python scripts/export_hypersim_pairs.py \
  --root /datasets/hypersim_pairs/scenes_v2 \
  --output /exports/gard_hypersim_001_pairs.tar
# 받는 서버에서:
tar -xf gard_hypersim_001_pairs.tar -C /datasets/hypersim_pairs
```

새 합성 데이터 생성 코드는 [hypersim-synthesis](https://github.com/yejich/hypersim-synthesis)에 있습니다. 현재 공개 synthesis의 작은 asset 샘플링 설정은 기존 `scenes_v2`와 다르므로, **새로 생성했다고 당시 이미지와 동일해지는 것은 아닙니다.** 정확히 같은 실험을 재현하려면 기존 pair snapshot을 사용하세요.

새 pair 집합으로 새 실험을 할 때는 각 카메라의 `clean_multiview.json`을 포함한 합성 출력으로 manifest를 다시 생성합니다. 기본 split/probability는 동일하지만 그룹 목록·프레임 수가 달라집니다.

```bash
python -m mvr.dataset.hypersim_pairs \
  --root /datasets/new_pairs/scenes_v2 \
  --output manifests/new_groups.json
# 학습 명령에 --manifest manifests/new_groups.json 전달
```

## 학습 실행

```bash
export HYPERSIM_PAIRS_ROOT=/datasets/hypersim_pairs/scenes_v2
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh
```

GPU 6,7을 쓰려면 `CUDA_VISIBLE_DEVICES=6,7`로 바꾸세요. 스크립트가 선택한 GPU 수에 맞춰 DDP 프로세스 수를 설정합니다. 기본 batch 설정은 **2 GPU용**입니다. GPU 수를 바꿀 때 effective batch와 accumulation도 확인해야 합니다.

경로를 YAML 편집 없이 지정할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh \
  --manifest manifests/hypersim_001_groups.json \
  --init-checkpoint ckpts/gard_denoiser.pt \
  --result-root result_train/hypersim_10ep
```

- `GARD_CONFIG`: 다른 YAML 경로
- `GARD_PYTHON`: 다른 Python 환경 실행 파일
- `--global-batch-size`, `--grad-accum-steps`: effective batch / accumulation 변경
- 실행은 foreground입니다. SSH 연결 이후에도 유지하려면 `tmux` 등에서 실행하세요.

짧은 GPU 검증은 아래처럼 실행합니다. 8개 N=4 microbatch로 optimizer update 1회와 고정 eval 2그룹을 수행하고 종료하며 checkpoint를 쓰지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh \
  --max-steps 1 --fixed-views 4
```

## Loss와 고정 evaluation

입력의 object/shadow mask 합집합을 해상도 조정한 뒤 14×14 patch coverage ≥0.5로 token mask를 만듭니다. clean 입력은 mask가 비어 있고 CLS token은 clean 그룹입니다. DA3가 reference view를 앞으로 옮기는 순서를 sample별로 적용합니다.

```text
Group2 = 1.0 × mean(MSE over distractor tokens)
       + 1.0 × mean(MSE over all remaining tokens)
Total  = Group2 + 1.0 × attention alignment
```

MSE는 RGB가 아니라 **feature flow velocity**의 예측 오차입니다. 각 그룹을 별도로 평균하므로 token 수가 적은 distractor 영역의 loss가 희석되지 않습니다. sample별 그룹 평균 후 batch 평균을 사용하며 빈 그룹의 기여는 0입니다.

Attention target은 clean HQ의 **DA3 예측 depth·pose·intrinsics**로 계산한 3D 대응입니다. Hypersim GT 대응을 직접 attention loss로 사용하지 않습니다. 기존 GARD layer 9 attention, temperature 0.01, cycle-consistency threshold 1.5 설정을 유지합니다.

Epoch마다 EMA 모델로 고정된 80개 4뷰 그룹의 Group2 flow loss를 평가합니다. 평가에는 attention loss를 더하지 않습니다. 당시 기록은 [docs/hypersim_10ep_flow_loss.json](docs/hypersim_10ep_flow_loss.json)에 있으며 epoch 1 **1.681975 → epoch 10 0.575686**입니다. 이 값은 RGB/3D 복원 성능 지표가 아닙니다.

## W&B와 checkpoint

W&B는 로그인 또는 환경변수로 인증합니다. 저장소에 인증값을 넣지 않습니다.

```bash
wandb login
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh \
  --wandb --wandb-project cross-view-feature-completion \
  --wandb-run-name Hypersim_group2_p07_10ep
```

선택 사항: `--wandb-entity USER_OR_TEAM`. Rank 0만 기록하며 새 실행마다 별도 W&B run을 만듭니다. `train_group/distractor`, `train_group/clean`, `train/loss_transport`, `train_attn/loss_attn`, `eval/group_mse`를 확인하세요.

Checkpoint는 epoch 종료마다 `result_train/.../checkpoints/latest.pt`로 저장합니다. model/EMA/optimizer/scheduler/각 rank RNG를 포함하며, 동일 world size·manifest·설정으로 epoch 경계에서 재개합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh \
  --resume /checkpoints/latest.pt
```

완료된 10-epoch 가중치는 이 공개 저장소에서 배포하지 않습니다. default `epochs: 10`에서 이미 epoch 10인 checkpoint를 resume하면 추가 학습은 하지 않습니다. 20-epoch 연장 실험은 이 공유본에 포함하지 않습니다.

## 파일과 검증

- [학습 설정](run_configs/train/train_GARD_hypersim_pairs.yaml)
- [학습 진입점](RAE/src/train_hypersim_pairs.py)
- [데이터·그룹 샘플러](mvr/dataset/hypersim_pairs.py)
- [Group2/Group3 loss](mvr/grouped_mse.py)
- [고정 평가·resume](mvr/pair_training.py)
- [상세 학습 안내](run_scripts/train/README_hypersim_pairs.md)
- [검증 기록](docs/validation.md)

```bash
python -m unittest discover -s tests -v
```

데이터가 없으면 실제 이미지 integration test는 skip하고 sampler/loss 검사를 실행합니다. `HYPERSIM_PAIRS_ROOT`를 설정하면 실제 이미지까지 검사합니다.

원본 GARD의 저자, 인용 방법 및 라이선스는 [원본 안내](README_GARD.md)와 [LICENSE](LICENSE)를 확인하세요. 이 저장소는 원본 GARD benchmark 학습 recipe와 구분되는 distractor fine-tuning 실험입니다.
