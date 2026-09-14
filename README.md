# GARD-DFrecon

Hypersim clean–distractor multi-view pair로 [GARD](https://github.com/cvlab-kaist/GARD)를 10 epochs fine-tuning한 실험 코드입니다. 입력의 distractor를 제거하도록 **feature flow의 Group2 MSE**와 기존 GARD attention alignment를 함께 학습합니다.

이 저장소는 upstream GARD 커밋 `de483e0`에 Hypersim pair 학습 코드를 추가한 연구용 확장입니다. 원본 저자·라이선스·코드 이력을 유지합니다. 원본 프로젝트 설명은 [README_GARD.md](README_GARD.md)에 있습니다.

**공유 범위:** 학습 코드, 10-epoch 설정, 당시의 고정 multi-view manifest, loss 기록, 재현 안내. Fine-tuned 체크포인트와 데이터 이미지는 포함하지 않습니다. 사전학습 GARD/DA3 가중치는 원 배포처에서 별도로 받습니다.

## 기본 설정과 기존 10-epoch 실험

| 항목 | 설정 |
|---|---|
| Backbone | DA3-GIANT-1.1, frozen |
| 학습 대상 | 사전학습 GARD denoiser를 초기값으로 fine-tuning |
| 데이터 | 다운로드한 Hypersim synthetic clean–distractor pair collections |
| Train | 다운로드된 pair로 자동 구성. 기존 실험은 41개 씬, 6,743 pairs |
| Holdout | `ai_001_001` 전체. 기존 다운로드에서는 80 pairs |
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

## 다운로드한 데이터로 학습

**기본 학습 목록은 다운로드된 데이터에서 새로 만듭니다.** 기존 6,743쌍에 제한하지 않습니다. pair 폴더를 내려받은 뒤 `HYPERSIM_PAIRS_ROOT`만 지정합니다. 원본 Hypersim RGB만으로는 distractor 학습 데이터가 되지 않습니다.

다음 두 구조를 지원합니다.

```text
# 여러 suffix를 함께 학습하는 경우 (권장)
/datasets/hypersim_pairs/
  scenes_v2/ai_001_001/cam_00/...
  scenes_002/ai_001_002/cam_00/...
  scenes_003/...
  scenes_004/...
  scenes_005/...
  scenes_006/...

# 하나의 collection만 사용하는 경우
/datasets/scenes_v2/ai_001_001/cam_00/...
```

각 카메라 폴더에는 `clean_multiview.json`, 각 `frame.NNNN` 폴더에는 `clean.png`, `distractor.png`, `mask_object.png`, `mask_shadow.png`가 필요합니다. **멀티뷰 후보를 새로 계산하려면 `clean_multiview.json`도 반드시 함께 다운로드/전달하세요.** 네 이미지 중 하나라도 없는 pair는 목록에서 제외합니다.

```bash
export HYPERSIM_PAIRS_ROOT=/datasets/hypersim_pairs
python scripts/check_hypersim_pairs.py
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh
```

학습 bash가 DDP 실행 전에 한 번 스캔하여 다음을 수행합니다.

- 지정 폴더가 collection이면 그 안을, 상위 폴더이면 `scenes_v2`와 `scenes_숫자3자리`를 합쳐 사용합니다.
- 구버전 중복 `scenes` 폴더는 상위 폴더 자동 탐색에서 제외합니다. 인식된 collection끼리 같은 `(scene, camera, frame)`이 중복되면 오류로 알려줍니다.
- `ai_001_001`은 항상 평가에만 사용하고, 나머지는 학습에 넣습니다. `_002–006`만 내려받고 holdout이 없으면 오류가 나므로 `scenes_v2`의 평가 씬도 포함하세요.
- 실제 다운로드된 이웃 뷰 중 clean GT 가시성 ≥0.25인 후보만 남깁니다. 가시성 파일이 없다고 무작위 이웃으로 대체하지 않습니다.
- 새 학습 데이터가 추가되어도 같은 평가 씬 파일·seed·설정을 쓰면 평가 그룹과 clean/distractor 선택은 유지됩니다.
- `data/hypersim_pairs/manifests/groups_<fingerprint>.json`으로 현재 목록을 고정합니다. 한 학습 실행 중에는 파일 추가를 자동 반영하지 않고, **다음 새 실행 때** 다시 스캔합니다.

수동으로 생성할 수도 있습니다.

```bash
python -m mvr.dataset.hypersim_manifest \
  --root "$HYPERSIM_PAIRS_ROOT" --output data/hypersim_pairs/groups.json
# 이 snapshot만 쓰려면 학습 bash에 --manifest data/hypersim_pairs/groups.json 전달
```

생성 코드는 [hypersim-synthesis](https://github.com/yejich/hypersim-synthesis)에 있습니다. `review.jpg`, 원본 HDF5/GLB 및 그림자 강도 이미지는 이 학습 데이터로더에 필요하지 않습니다.

전달용 archive도 다운로드된 전체 목록을 기준으로 만들 수 있습니다. 카메라별 가시성 파일을 함께 포함하며 Git에 넣지 않습니다.

```bash
python scripts/export_hypersim_pairs.py \
  --root "$HYPERSIM_PAIRS_ROOT" --output /exports/hypersim_pairs.tar
# 받는 서버에서: tar -xf hypersim_pairs.tar -C /datasets
# HYPERSIM_PAIRS_ROOT=/datasets/hypersim_pairs 로 지정
```

### 예전 10-epoch 실험만 재현하는 경우

당시 목록 [manifests/hypersim_001_groups.json](manifests/hypersim_001_groups.json)은 참고용으로 보존합니다. 기본 학습에서는 사용하지 않습니다. 동일 이미지가 필요하며, 현재 공개 synthesis 설정으로 새로 만든 이미지가 과거 pair와 동일하다는 뜻은 아닙니다.

```bash
HYPERSIM_PAIRS_ROOT=/datasets/hypersim_pairs/scenes_v2 \
  CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh \
  --manifest manifests/hypersim_001_groups.json
```

## 학습 실행

```bash
export HYPERSIM_PAIRS_ROOT=/datasets/hypersim_pairs
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh
```

GPU 6,7을 쓰려면 `CUDA_VISIBLE_DEVICES=6,7`로 바꾸세요. 스크립트가 선택한 GPU 수에 맞춰 DDP 프로세스 수를 설정합니다. 기본 batch 설정은 **2 GPU용**입니다. GPU 수를 바꿀 때 effective batch와 accumulation도 확인해야 합니다.

경로를 YAML 편집 없이 지정할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh \
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

Epoch마다 EMA 모델로 현재 snapshot에 고정된 4뷰 그룹의 Group2 flow loss를 평가합니다. 기존 holdout 파일을 모두 사용하면 80그룹입니다. 평가에는 attention loss를 더하지 않습니다. 당시 기록은 [docs/hypersim_10ep_flow_loss.json](docs/hypersim_10ep_flow_loss.json)에 있으며 epoch 1 **1.681975 → epoch 10 0.575686**입니다. 이 값은 RGB/3D 복원 성능 지표가 아닙니다.

## W&B와 checkpoint

W&B는 로그인 또는 환경변수로 인증합니다. 저장소에 인증값을 넣지 않습니다.

```bash
wandb login
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/train/train_GARD_hypersim_pairs.sh \
  --wandb --wandb-project cross-view-feature-completion \
  --wandb-run-name Hypersim_group2_p07_10ep
```

선택 사항: `--wandb-entity USER_OR_TEAM`. Rank 0만 기록하며 새 실행마다 별도 W&B run을 만듭니다. `train_group/distractor`, `train_group/clean`, `train/loss_transport`, `train_attn/loss_attn`, `eval/group_mse`를 확인하세요.

현재 사용한 목록은 결과 폴더의 `pairs_manifest.json`에도 보존합니다. 새 checkpoint에는 데이터 fingerprint를 저장하므로 데이터가 바뀐 상태에서 단순 `--resume`하면 오류를 냅니다. 데이터가 늘었다면 새 학습 실행으로 시작하거나, 재개하려면 기존 manifest와 데이터를 유지하세요. 예전 fingerprint 없는 checkpoint에는 이 검증이 적용되지 않습니다.

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
