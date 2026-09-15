# GARD-DFrecon 실험 안내

이제 학습·추론·평가 코드는 **GARD-DFrecon**에서 실행합니다. 기존 GARD는 원본 보관용입니다.

## 공통 환경

저장소 루트에서 실행합니다. 현재 서버에서는 기존 환경을 복사하고 경로를 수정한 `.venv`를 사용합니다. 새 서버에서는 README의 conda 설치 절차를 사용할 수 있습니다.

```bash
source .venv/bin/activate
export GARD_PYTHON="$PWD/.venv/bin/python"
export HYPERSIM_PAIRS_ROOT="$PWD/datasets/hypersim_pairs"
export CUDA_VISIBLE_DEVICES=0,1
python scripts/check_experiment_setup.py
```

현재 서버의 기존 Hypersim pairs는 `datasets/hypersim_pairs`로 연결되어 있습니다. 새 데이터로 학습하려면 `HYPERSIM_PAIRS_ROOT`를 변경하세요. 기존 실험 재현·재개는 원래 manifest와 동일한 데이터를 사용해야 합니다.

## 학습

| 실험 | 실행 명령 |
|---|---|
| DA3 Hypersim pairs | `bash run_scripts/train/train_GARD_hypersim_pairs.sh` |
| DA3 기존 10 epochs → 총 20 epochs | `bash run_scripts/train/resume_train_GARD_hypersim_pairs_ext20ep.sh` |
| Omega Hypersim pairs | `bash run_scripts/train/train_GARD_omega_hypersim_pairs.bash` |
| Omega 통계 재계산 | `bash run_scripts/train/compute_omega_latent_stats.bash` |
| 기존 일반 GARD 학습 | `bash run_scripts/train/train_GARD.sh` |

`GARD_CONFIG`로 설정 파일, `GARD_PYTHON`으로 Python, `CUDA_VISIBLE_DEVICES`로 GPU를 지정합니다. DA3와 Omega pairs 학습의 기본값은 GPU 2개·10 epochs입니다. DA3의 기존 파일명은 현재 DA3 trainer를 호출하는 호환 진입점으로 유지했습니다. `--background`도 유지합니다.

W&B를 사용하려면 먼저 `wandb login` 또는 환경변수 `WANDB_API_KEY`로 인증하고 `--wandb`를 추가하세요. 기존 스크립트의 인라인 키는 제거했습니다. 그 키는 노출된 것으로 간주하고 W&B에서 폐기·재발급해야 합니다.

## 추론 및 비교

```bash
# Omega: CKPT를 학습 체크포인트 경로로 지정
CKPT=/path/to/latest.pt CUDA_VISIBLE_DEVICES=0 \
  bash run_scripts/val/infer_GARD_omega.bash --eval-groups 0 26 53

# 기존 3개 그룹의 Omega/DA3 비교 (기존 DA3 결과 파일 사용)
CUDA_VISIBLE_DEVICES=0 bash run_scripts/val/compare_omega_da3.sh

# GARD, DA3, VGGT, VGGT-Omega, VGTW 비교
CUDA_VISIBLE_DEVICES=0 bash run_scripts/val/compare_five_methods.sh
```

비교 모델의 환경이 다르면 `VGGT_PYTHON`, `OMEGA_PYTHON`, `VGTW_PYTHON`으로 지정합니다. 원래 `run_gpu5.sh` / `run_gpu6.sh` 이름의 호환 스크립트도 `scripts/eval`에 있지만 GPU 번호를 강제하지 않습니다.

## 개별 평가·분석

코드는 `scripts/eval/`, 출력은 같은 상대 구조의 `result_eval/`에 있습니다. 공통 실행 형식:

```bash
bash run_scripts/val/run_experiment.sh <아래 경로> [추가 인자]
```

| 실험 | `scripts/eval` 기준 코드 경로 |
|---|---|
| Hypersim DA3 평가 | `hypersim_10ep_ai001/evaluate.py` |
| 5개 방법 추론·보고서 | `hypersim_10ep_ai001/five_methods/` |
| Omega/DA3 보고서 | `omega_vs_da3_hypersim_10ep/build_report.py` |
| Clean DA3 비교 | `omega_vs_da3_hypersim_10ep/infer_da3_clean.py` |
| 카메라 pose 비교 | `omega_vs_da3_hypersim_10ep/plot_camera_poses.py` |
| 비교 이미지 재배치 | `omega_vs_da3_hypersim_10ep/reorder_grid.py` |
| DA3 layer cosine | `omega_vs_da3_hypersim_10ep/da3_layer_cosine/analyze.py`, `plot.py` |
| Cupcake 기본 | `cupcake_hypersim_10ep/inference.py` |
| Cupcake 4 views | `cupcake_hypersim_10ep_4views/inference.py`, `compare.py` |
| Cupcake 378×504 | `cupcake_hypersim_10ep_4views_378x504/inference.py`, `compare.py` |
| Cupcake DA3 기본 해상도 | `cupcake_hypersim_10ep_4views_da3default/inference.py` |
| Attention 분석 | `cupcake_hypersim_10ep_4views_da3default/attention/analyze.py`, `plot.py` |
| 모든 layer self attention 제거 | `cupcake_hypersim_10ep_4views_da3default/all_layers_no_self/inference.py`, `compare.py` |
| Mask ablation | `cupcake_hypersim_10ep_4views_da3default/mask_ablation/inference.py --mode self_view` 또는 `--mode self_token`, 이후 `compare.py` |

기존 결과를 함께 복사했으므로 보고서 생성만 다시 실행할 수 있습니다. 추론을 다시 실행하면 GARD-DFrecon의 해당 결과 파일을 갱신합니다. 기존 GARD의 결과 파일은 갱신하지 않습니다.

## 외부 의존성

GARD 모델을 로드하는 평가 adapter 3개는 `mvr/eval_models`로 옮겨 기존 GARD 코드를 import하지 않도록 했습니다. 다음 외부 프로젝트·데이터는 원래 서버 경로를 계속 사용하며 환경변수로 변경할 수 있습니다.

| 환경변수 | 용도 |
|---|---|
| `VGGT_OMEGA_REPO`, `VGGT_OMEGA_CKPT` | Omega 소스·가중치 |
| `VGGT_REPO` | VGGT baseline 소스 |
| `VGTW_REPO`, `VGTW_CKPT` | VGTW baseline 소스·가중치 |
| `FEATURE_COMPLETION_ROOT` | Cupcake 데이터 정보 및 일부 데이터 준비 모듈 |
| `SYNTHETIC_DISTRACTOR_ROOT` | Hypersim GT 평가 모듈·데이터 |

Cupcake의 데이터 준비 모듈은 기존 `pilot_multiview_feature_diff`에도 의존합니다. 이 외부 데이터 파이프라인은 이번 GARD 이전 대상에 포함하지 않았습니다.

## 이전한 실험 상태와 보관 정책

- `result_eval`: 기존 결과 약 931MB를 복사했습니다. 실행 코드는 Git에서 추적할 수 있도록 `scripts/eval`로 분리했습니다.
- `result_train`: DA3/Omega의 설정·로그와 `latest.pt`를 복사했습니다.
- `ckpts`: `gard_denoiser.pt`, `mae_adapter_giant.pt`를 복사했습니다.
- `data`: 원래 manifest·Omega 정규화 통계를 복사하고 대용량 원본 데이터 경로를 연결했습니다.
- 복사한 JSON/YAML의 GARD 절대 경로는 GARD-DFrecon으로 수정했습니다. 변경 전 원본은 기존 GARD에 남아 있습니다.
- 대용량 가중치 4개(총 약 48.87GiB)는 공간 확보 후 **독립된 실제 파일로 복사**했습니다. 파일 크기와 SHA-256을 검증한 뒤 기존 심볼릭 링크를 교체했으며, 가중치 로딩은 기존 GARD 폴더에 의존하지 않습니다.
- 기존 GARD 원본은 변경하지 않았습니다. 데이터 및 외부 프로젝트 의존성은 위 안내대로 유지됩니다.
- 결과·가중치·데이터는 Git에서 제외됩니다. 코드·설정·실험 안내는 Git에 포함합니다.

이번 검증은 CPU 테스트, import/CLI 확인, 기존 결과로 보고서 재생성입니다. 모든 GPU 실험을 재학습하거나 재평가한 것은 아닙니다.

## Python 환경 이전

기존 `GARD/.venv`를 GARD-DFrecon의 `.venv`로 복사하고 activation, CLI shebang, editable 설치 경로를 수정했습니다. PyTorch는 `2.4.1+cu121`이며 실제 이미지 검사를 포함해 29개 테스트가 통과했습니다. 기존 GARD 환경은 더 이상 실행에 필요하지 않습니다. 일반적인 venv와 같이 기반 Python(`/home/cvlab20/anaconda3/envs/videophy`)에는 의존하므로 해당 conda 환경은 유지해야 합니다.
