# GARD 실험 전달 및 로컬 검증

기존 GARD baseline 3개와 NVS flow/camera 조합 6개를 실행한다. 코드는 Git 저장소에서 받고 데이터는 별도 HF release에서 받는다. 아래 명령은 저장소 최상위에서 실행한다. Push는 사용자가 수행한다.

## 환경과 필요한 가중치

저장소의 Python 3.10 환경을 준비한다. 기본 Python은 `.venv/bin/python`이며 다른 환경은 `PYTHON=/path/to/python`으로 지정한다. 검증 환경의 라이브러리 버전은 `run_configs/JIHYE/handoff/environment_reference.json`에 기록한다. H100에서는 해당 노드의 CUDA에 맞는 PyTorch를 사용하고 smoke 테스트로 호환성을 확인한다.

`ckpts/gard_denoiser.pt`, `ckpts/mae_adapter_giant.pt`가 필요하다. 기존 저장소의 `bash download_scripts/gard_ckpt/download_ckpts.sh`로 준비할 수 있다. 위치가 다른 초기 denoiser는 `INIT_CHECKPOINT`로 지정한다. RGB decoder adapter는 현재 저장소의 `ckpts/mae_adapter_giant.pt`를 사용한다.

학습 launcher는 기본 `HF_HUB_OFFLINE=1`이다. 온라인 연결이 가능한 준비 단계에서 DA3와 LPIPS 가중치를 캐시한다. HF 캐시 위치는 표준 `HF_HOME`/`HF_HUB_CACHE` 환경변수로 지정한다.

```bash
bash run_scripts/JIHYE/handoff/20261013_cache_models.bash
```

## 데이터 준비와 검증

HF 업로드가 완료된 이후 실행한다. 다운로드 스크립트는 release의 완료 상태, pinned commit 및 archive SHA256을 확인한다. 다른 대화에서 진행 중인 업로드에는 관여하지 않는다.

```bash
DATA_ROOT=/data/gard bash run_scripts/JIHYE/handoff/20261013_download_data.bash
```

수신 데이터 루트는 다음 구조를 사용한다.

```text
/data/gard/
  manifests/original_full.json
  manifests/negative_full.json
  manifests/scene_split_policy.json
  hypersim_pairs/
  hypersim_negative/
  camera_reference/hypersim/
  camera_reference/metadata_camera_parameters.csv
  eval_visualization/manifest.json
```

`eval_visualization/manifest.json`은 동결된 평가 입력을 지정한다. 입력 이미지 경로, clean RGB 경로, 4개 camera_records(scene/camera/frame), transient object mask 경로, SHA256을 포함한다. NVS는 앞 3뷰를 관측하고 마지막 뷰를 novel target으로 사용한다. Baseline도 같은 앞 3뷰만 관측한다. Transient mask는 static A를 포함하지 않는다.

별도 평가 manifest를 전달받으면 `--eval-manifest /path/to/manifest.json`으로 지정한다. HF release에 평가 manifest가 아직 없을 때, 합성 데이터의 heldout 씬에서 임시 검증용 그룹을 명시적으로 생성할 수 있다. 기존 평가 manifest는 덮어쓰지 않는다. 이 자동 생성 샘플을 기존 65그룹 전체 eval과 혼동하지 않는다.

```bash
DATA_ROOT=/data/gard bash run_scripts/JIHYE/handoff/20261013_validate.bash --create-eval
DATA_ROOT=/data/gard bash run_scripts/JIHYE/handoff/20261013_validate.bash --all-files
```

로컬 원본을 이미지 복사 없이 수신 환경과 같은 구조로 연결하려면 `scripts/train/handoff/link_local.py --help`를 참고한다.

## 실험 선택 및 실행

```bash
bash run_scripts/JIHYE/handoff/20261013_list.bash
```

| 실험 이름 | 데이터 | Flow | Camera | Distractor aware |
|---|---|---|---|---|
| baseline_transient_off | transient-only | 기존 restoration | 없음 | OFF |
| baseline_mixed_off | transient+static negative | 기존 restoration | 없음 | OFF |
| baseline_mixed_on | transient+static negative | 기존 restoration | 없음 | ON |
| nvs_legacy_target | mixed | 기존 source/novel 경로 | target-only | ON |
| nvs_legacy_source_target | mixed | 기존 source/novel 경로 | source+target | ON |
| nvs_pure_noise_target | mixed | 모두 pure noise | target-only | ON |
| nvs_pure_noise_source_target | mixed | 모두 pure noise | source+target | ON |
| nvs_pseudo_lq_target | mixed | pseudo-LQ restoration | target-only | ON |
| nvs_pseudo_lq_source_target | mixed | pseudo-LQ restoration | source+target | ON |

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DATA_ROOT=/data/gard \
  bash run_scripts/JIHYE/handoff/20261013_train.bash \
  --experiment nvs_legacy_target --prepare-only

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DATA_ROOT=/data/gard \
  bash run_scripts/JIHYE/handoff/20261013_train.bash \
  --experiment nvs_legacy_target
```

기본 global batch는 16이다. GPU 8개와 microbatch 1이면 accumulation 2, GPU 2개이면 accumulation 8이다. `MICROBATCH_SIZE=2`를 지정하면 accumulation을 다시 계산한다. 메모리를 자동 탐색하는 기능은 없으며 나누어떨어지지 않는 조합은 실행 전에 오류를 낸다. Microbatch는 이미지 한 장이 아닌 multiview 그룹 하나다. 로컬 48GB 공유 GPU의 메모리를 줄이려면 `HANDOFF_ATTENTION_INNER_CHECKPOINT=true HANDOFF_ATTENTION_HEAD_CHUNK_SIZE=4`를 지정할 수 있다. Attention 재계산으로 속도가 느려지고 head 평균의 부동소수점 합산 순서가 달라질 수 있으나 supervision 규칙은 동일하다. 유효 옵션은 `runtime.json`에 기록된다.

한 실험당 단일 노드의 8 GPU를 사용하는 launcher다. 두 노드에서는 다른 실험을 하나씩 실행한다. 다중 노드 단일 실험 launcher는 포함하지 않는다.

## Flow와 학습량

Source 3뷰는 모든 모델에 동일하게 제공한다. NVS는 clean novel 1뷰까지 teacher로 사용한다. L17 encoder는 frozen DA3-Giant다. 픽셀별 `[direction, origin × direction, novel_role]` 7채널 map을 `Conv2d(7,1536,14,14)`로 patch embedding하고 encoder의 patch에 더한다. Camera/CLS token에는 추가하지 않는다. Projection은 zero initialization이다.

- Legacy: source 시작 `LQ + 0.3ε`, novel 시작 `ε`. Source LQ만 매 forward에 더한다.
- Pure noise: 모든 시작 `ε`. Source LQ는 매 forward에 더하고 novel LQ 조건은 없다.
- Pseudo LQ: source는 관측 LQ, novel은 source-only depth/pose를 이용한 RGB warp를 DA3에 encode한 pseudo latent. 모든 시작 `condition + 0.3ε`, 매 forward에 condition을 더한다. RGB hole은 ImageNet mean. 실제 target RGB는 warp나 조건 생성에 사용하지 않는다.

모든 실험은 epoch당 500 optimizer updates, 30 epochs로 고정한다. Global batch 16이면 epoch당 8000그룹이다. Mixed는 original/negative 각각 0.5 확률, scene-balanced replacement sampling이다. 기존 exactly-once negative epoch sampler와 다르며 실험간 update 수를 동일하게 유지하기 위한 전달용 선택이다. 이 비율과 budget은 config에서 명시적으로 바꿀 수 있다.

일반 geometry alignment는 모두 ON이다. Distractor-aware ON에서는 transient object mask만 key 제외에 사용한다. Static A는 허용하고 novel query도 포함한다. Clean/transient query 그룹은 기존 helper와 같은 가중치로 계산한다. Flow group2의 token 분리는 기존 object+shadow mask를 사용한다. Mask head 학습은 이번 9개 실험에 없다.

LR은 2e-4, 기존 실험처럼 warmup_from_zero=false로 첫 15 epochs 동안 base LR을 유지하고 이후 linear decay, 최저 LR은 0.1배다. 모든 새 실험은 동일 schedule을 사용한다.

## 내일 로컬 검증

데이터 검증 후 GPU 0·1에서 전체 9개 조합의 실제 forward/backward와 작은 eval을 실행한다. Smoke에서는 global batch를 GPU 수로 줄이고, 각 family 안에서 모델을 재사용한다. 성능 비교 실험이 아닌 실행 가능성 검사다.

```bash
CUDA_VISIBLE_DEVICES=0,1 DATA_ROOT=/data/gard \
  bash run_scripts/JIHYE/handoff/20261013_smoke.bash
```

단일 조합을 두 optimizer updates까지 검사하려면 다음을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0,1 GLOBAL_BATCH_SIZE=2 DATA_ROOT=/data/gard \
  bash run_scripts/JIHYE/handoff/20261013_train.bash \
  --experiment nvs_legacy_target --max-steps 2 --workers 0
```

재개는 원래 run의 config와 latest.pt를 사용한다. 정확한 RNG 재개는 동일 GPU 수와 batch 설정을 요구한다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash run_scripts/JIHYE/handoff/20261013_train.bash \
  --resume /path/to/run/latest.pt
```

## 로컬 로그와 평가 산출물

W&B나 온라인 모니터링 없이 다음 파일로 학습을 확인한다.

- `console.log`: 모든 rank의 stdout, warning, 오류, eval 출력
- `train.log`: rank0의 timestamp, loss, source clean/transient loss, novel loss, alignment loss, LR, grad norm, memory, checkpoint, eval 수치와 이미지 경로
- `metrics.jsonl`, `status.json`, `data_audit.json`, 동결된 `config.yaml`
- `latest.pt`: model/EMA/optimizer/RNG와 resume 위치. Optimizer는 이 파일에만 저장한다.
- `epoch_001.pt`, `003`, `010`, `020`, `030`: EMA weights와 config만 저장
- `eval_visualization/epoch_NNN/`: per-view PSNR/SSIM/LPIPS CSV, pose, depth, P20/40/50 cloud 수치, decoded RGB/depth/camera/cloud 이미지, PLY, raw prediction arrays, `index.html`

기본 eval 시점은 1/3/10/20/30 epoch다. `--eval-epochs 1,3`처럼 지정할 수 있고, smoke나 빠른 검증에서는 `--eval-limit 1`을 사용한다. `--eval-now`는 학습 전에 현재 EMA를 평가한다.

현재 전달용 portable evaluator는 paired synthetic 입력과 camera_records schema를 지원한다. Depth/cloud는 joint-clean DA3 pseudo-GT로 표기하고, pose는 renderer camera GT, RGB는 실제 clean render를 사용한다. 그룹 실패는 `error.json`과 `EVAL FAILED` 로그로 기록하고 나머지 그룹을 평가한다. 평가 `status.json`은 실패가 있으면 `partial_failure`다.

원래 Hypersim의 metric geometry 파일은 이 release에 포함되지 않으므로 실제 geometry GT로 표시하지 않는다. Cloud 거리 단위는 clean DA3 gauge이며 meters가 아니다. Source first3의 RGB/pose/cloud는 별도로 저장해 NVS novel 출력의 효과와 구분한다.

## Git 전달 파일

```bash
python scripts/train/handoff/inventory.py > handoff_source_inventory.json
```

이 명령은 새 entrypoint와 필수 local dependency의 경로, hash, Git 상태를 출력한다. Git add/commit/push는 수행하지 않는다. 저장소 기존 tracked 파일도 필요하다. HF 데이터나 checkpoint, 서버 credential을 Git에 추가하지 않는다.
