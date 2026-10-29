#!/usr/bin/env bash
# V=4 (N=2~3, M=4-N): group2 MSE + attention alignment; mask-aware OFF.
# CUDA_VISIBLE_DEVICES=0,1,2,3 GLOBAL_BATCH_SIZE=16 PER_GPU_BATCH_SIZE=1 bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash
# Options: NVS_BRANCH=true|false, FLOWMAP_FROM=noise|noise_lq,
#          DISTRACTOR_AWARE_ATTN_ALIGN=true|false, ATTENTION_ALIGNMENT=true|false
#          DISTRACTOR_LOSS=none|push_only|infoNCE,
#          GEOMETRY_CONTRASTIVE_USE=true|false
# NVS_CONFIG를 지정하면 prepare_handoff_config.py로 생성한 config를 기반으로
# ablation/batch 옵션을 적용합니다.
# CPU 설정 준비만: 위 명령 끝에 --prepare-only 추가
# 재시작: 위 명령 끝에 --resume /absolute/run/latest.pt 추가
set -euo pipefail
set +x

# ==================== 여기서 실험 옵션 수정 ====================
BASE_EXP_NAME="${EXP_NAME:-nvs_v4_n2-3}"
ATTENTION_ALIGNMENT="${ATTENTION_ALIGNMENT:-true}"
MASK_AWARE_ATTENTION="${MASK_AWARE_ATTENTION:-false}"
DISTRACTOR_AWARE_ATTN_ALIGN="${DISTRACTOR_AWARE_ATTN_ALIGN:-false}"
DISTRACTOR_LOSS="${DISTRACTOR_LOSS:-none}"
DISTRACTOR_LOSS_LAMBDA="${DISTRACTOR_LOSS_LAMBDA:-0.01}"
DISTRACTOR_LOSS_MARGIN="${DISTRACTOR_LOSS_MARGIN:-0.1}"
DISTRACTOR_LOSS_TEMPERATURE="${DISTRACTOR_LOSS_TEMPERATURE:-0.07}"
FLOWMAP_FROM="${FLOWMAP_FROM:-noise_lq}"
NVS_BRANCH="${NVS_BRANCH:-true}"
PRECISION="${PRECISION:-fp32}"
# 비워두면 NVS_CONFIG의 negative manifest를 그대로 사용합니다.
NEGATIVE_MANIFEST="${NEGATIVE_MANIFEST:-}"
NVS_CONFIG="${NVS_CONFIG:-run_configs/train/train_GARD_da3_nvs_v4_n2_3.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpts/gard_denoiser.pt}"
EPOCHS="${EPOCHS:-30}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-1}"
ORIGINAL_GROUPS_PER_EPOCH="${ORIGINAL_GROUPS_PER_EPOCH:-10000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
NOVEL_WEIGHT="${NOVEL_WEIGHT:-1.0}"         # M의 전체 1+P token MSE 가중치
GEOMETRY_CONTRASTIVE_USE="${GEOMETRY_CONTRASTIVE_USE:-}"
GEOMETRY_CONTRASTIVE_LAMBDA="${GEOMETRY_CONTRASTIVE_LAMBDA:-}"
GEOMETRY_CONTRASTIVE_TEMPERATURE="${GEOMETRY_CONTRASTIVE_TEMPERATURE:-}"
GEOMETRY_CONTRASTIVE_MAX_NEGATIVE_TOKENS="${GEOMETRY_CONTRASTIVE_MAX_NEGATIVE_TOKENS:-}"
GEOMETRY_CONTRASTIVE_TIME_WEIGHT="${GEOMETRY_CONTRASTIVE_TIME_WEIGHT:-}"
EXP_NAME="${BASE_EXP_NAME}_nvs${NVS_BRANCH}_flow${FLOWMAP_FROM}_dattn${DISTRACTOR_AWARE_ATTN_ALIGN}_align${ATTENTION_ALIGNMENT}"
if [[ -n "$GEOMETRY_CONTRASTIVE_USE" ]]; then
  EXP_NAME="${EXP_NAME}_supcon${GEOMETRY_CONTRASTIVE_USE}"
fi
LOG_EVERY_OPTIMIZER_STEPS="${LOG_EVERY_OPTIMIZER_STEPS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
SAVE_EPOCHS="${SAVE_EPOCHS:-1,3,10,20,30}"         # EMA; latest.pt에는 optimizer/RNG 포함
EVAL_EPOCHS="${EVAL_EPOCHS:-1,3,10,20,30}" # registry V4+M1 RGB/depth/camera/cloud
EVAL_PLAN="${EVAL_PLAN:-result_eval/20261008_nvs_epoch003_v4_m1/eval_plan.json}"
WANDB_PROJECT="${WANDB_PROJECT:-GARD-DFrecon}"
WANDB_ENTITY="${WANDB_ENTITY:-yejichoi_kaist-kaist}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/home/cvlab20/.config/wandb/gard_yejichoi.env}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
PREPARE_ONLY=false
RESUME_CHECKPOINT=""
TRAIN_ARGS=()
# ==============================================================

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-only) PREPARE_ONLY=true; shift ;;
    --novel-weight|--resume|--out)
      if [[ $# -lt 2 ]]; then echo "$1 requires a value." >&2; exit 1; fi
      case "$1" in
        --novel-weight) NOVEL_WEIGHT="$2" ;;
        --resume) RESUME_CHECKPOINT="$2" ;;
        --out) NVS_OUT_DIR="$2" ;;
      esac
      shift 2 ;;
    *) TRAIN_ARGS+=("$1"); shift ;;
  esac
done

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_DEVICES[@]}
if (( NUM_GPUS < 1 || PER_GPU_BATCH_SIZE < 1 || GLOBAL_BATCH_SIZE < NUM_GPUS * PER_GPU_BATCH_SIZE || GLOBAL_BATCH_SIZE % (NUM_GPUS * PER_GPU_BATCH_SIZE) != 0 )); then
  echo 'GLOBAL_BATCH_SIZE must be a positive multiple of NUM_GPUS * PER_GPU_BATCH_SIZE.' >&2; exit 1
fi
export GARD_GRAD_ACCUM_STEPS="${GARD_GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (NUM_GPUS * PER_GPU_BATCH_SIZE)))}"
if (( GARD_GRAD_ACCUM_STEPS * NUM_GPUS * PER_GPU_BATCH_SIZE != GLOBAL_BATCH_SIZE )); then
  echo 'GARD_GRAD_ACCUM_STEPS * NUM_GPUS * PER_GPU_BATCH_SIZE must equal GLOBAL_BATCH_SIZE.' >&2; exit 1
fi
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export GARD_EVAL_ROOT="${GARD_EVAL_ROOT:-$ROOT/datasets/eval}"
export HYPERSIM_CAMERA_ROOT="${HYPERSIM_CAMERA_ROOT:-$GARD_EVAL_ROOT/hypersim}"
export HYPERSIM_CAMERA_PARAMETERS="${HYPERSIM_CAMERA_PARAMETERS:-$GARD_EVAL_ROOT/metadata_camera_parameters.csv}"
export HYPERSIM_GEOMETRY_ROOT="${HYPERSIM_GEOMETRY_ROOT:-$HYPERSIM_CAMERA_ROOT}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/RAE/src:$ROOT/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
export DA3_RUN_ID="${DA3_RUN_ID:-$(date +%Y%m%d_%H%M%S_%N)}"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  RESUME_CHECKPOINT="$(realpath -e "$RESUME_CHECKPOINT")"
  RESUME_PARENT="$(dirname -- "$RESUME_CHECKPOINT")"
  if [[ "$(basename -- "$RESUME_PARENT")" == checkpoints ]]; then
    EXPERIMENT_DIR="$(dirname -- "$RESUME_PARENT")"
    RUN_DIR="${NVS_OUT_DIR:-$(realpath "$EXPERIMENT_DIR/../../..")}"
  else
    EXPERIMENT_DIR="$RESUME_PARENT"
    RUN_DIR="${NVS_OUT_DIR:-$RESUME_PARENT}"
  fi
else
  RUN_DIR="${NVS_OUT_DIR:-$ROOT/result_train/${DA3_RUN_ID%%_*}_${EXP_NAME}}"
fi
mkdir -p "$RUN_DIR"
RUN_DIR="$(realpath "$RUN_DIR")"
EXPERIMENT_NAME="$("$ROOT/.venv/bin/python" - "$GLOBAL_BATCH_SIZE" "$PER_GPU_BATCH_SIZE" "$GARD_GRAD_ACCUM_STEPS" "$LEARNING_RATE" "$PRECISION" "$ATTENTION_ALIGNMENT" <<'PY'
import sys
batch, per_gpu, accumulation, lr = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
print(f'TRAIN_{sys.argv[5]}_hypersim_pairs_da3-GIANT-L17_bs{batch}-pgbs{per_gpu}-maxview4-accum{accumulation}_lr-{lr:.0e}_attnalign-{"on" if sys.argv[6]=="true" else "off"}_maskaware-off')
PY
)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$RUN_DIR/training/hypersim_pairs/$EXPERIMENT_NAME}"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
CONFIG="$RUN_DIR/launch_config.yaml"

export NVS_LAUNCH_CONFIG="$CONFIG" NVS_INIT_CHECKPOINT="$INIT_CHECKPOINT"
NVS_CONFIG="$(realpath -e "$NVS_CONFIG")"
export NVS_DATA_TEMPLATE="$NVS_CONFIG"
export PRECISION
export EPOCHS ORIGINAL_GROUPS_PER_EPOCH NUM_WORKERS LEARNING_RATE NOVEL_WEIGHT ATTENTION_ALIGNMENT MASK_AWARE_ATTENTION PER_GPU_BATCH_SIZE
export LOG_EVERY_OPTIMIZER_STEPS CHECKPOINT_INTERVAL SAVE_EPOCHS EVAL_EPOCHS NEGATIVE_MANIFEST WANDB_PROJECT WANDB_ENTITY WANDB_ENABLED
export DISTRACTOR_AWARE_ATTN_ALIGN DISTRACTOR_LOSS DISTRACTOR_LOSS_LAMBDA DISTRACTOR_LOSS_MARGIN DISTRACTOR_LOSS_TEMPERATURE FLOWMAP_FROM NVS_BRANCH
export GEOMETRY_CONTRASTIVE_USE GEOMETRY_CONTRASTIVE_LAMBDA GEOMETRY_CONTRASTIVE_TEMPERATURE GEOMETRY_CONTRASTIVE_MAX_NEGATIVE_TOKENS GEOMETRY_CONTRASTIVE_TIME_WEIGHT
"$ROOT/.venv/bin/python" - <<'PY'
import math, os
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.load(os.environ['NVS_DATA_TEMPLATE'])
cfg.precision = os.environ['PRECISION']
for name, value in [('NVS_BRANCH', os.environ['NVS_BRANCH']), ('DISTRACTOR_AWARE_ATTN_ALIGN', os.environ['DISTRACTOR_AWARE_ATTN_ALIGN'])]:
    if value.lower() not in ('true', 'false'): raise ValueError(f'{name} must be true/false')
cfg.nvs_branch = os.environ['NVS_BRANCH'].lower() == 'true'
cfg.flowmap_from = os.environ['FLOWMAP_FROM'].lower()
if cfg.flowmap_from not in ('noise', 'noise_lq'): raise ValueError('FLOWMAP_FROM must be noise or noise_lq')
for key, env in [('attention_alignment.use','ATTENTION_ALIGNMENT'), ('mask_aware_attention','MASK_AWARE_ATTENTION')]:
    value = os.environ[env].lower()
    if value not in ('true','false'): raise ValueError(f'{env} must be true/false')
    OmegaConf.update(cfg,key,value == 'true')
if cfg.mask_aware_attention: raise ValueError('This NVS experiment requires mask-aware attention OFF')
cfg.attention_alignment.distractor_aware.use = os.environ['DISTRACTOR_AWARE_ATTN_ALIGN'].lower() == 'true'
loss_mode = os.environ['DISTRACTOR_LOSS']
if loss_mode.lower() == 'infonce': loss_mode = 'infoNCE'
if loss_mode not in ('none', 'push_only', 'infoNCE'): raise ValueError('DISTRACTOR_LOSS must be none, push_only, or infoNCE')
cfg.distractor_loss = dict(mode=loss_mode,
                           lambda_coeff=float(os.environ['DISTRACTOR_LOSS_LAMBDA']),
                           margin=float(os.environ['DISTRACTOR_LOSS_MARGIN']),
                           temperature=float(os.environ['DISTRACTOR_LOSS_TEMPERATURE']))
if cfg.distractor_loss.lambda_coeff < 0 or not math.isfinite(cfg.distractor_loss.lambda_coeff): raise ValueError('Invalid DISTRACTOR_LOSS_LAMBDA')
if not 0 <= cfg.distractor_loss.margin < 1 or not math.isfinite(cfg.distractor_loss.margin): raise ValueError('DISTRACTOR_LOSS_MARGIN must be in [0,1)')
if cfg.distractor_loss.temperature <= 0 or not math.isfinite(cfg.distractor_loss.temperature): raise ValueError('DISTRACTOR_LOSS_TEMPERATURE must be positive')
geometry_env = os.environ['GEOMETRY_CONTRASTIVE_USE']
if geometry_env:
    geometry_use = geometry_env.lower()
    if geometry_use not in ('true', 'false'): raise ValueError('GEOMETRY_CONTRASTIVE_USE must be true/false')
    if 'geometry_contrastive_loss' not in cfg or cfg.geometry_contrastive_loss is None:
        cfg.geometry_contrastive_loss = {}
    OmegaConf.update(cfg, 'geometry_contrastive_loss.use', geometry_use == 'true')
    for key, env in [('lambda_coeff','GEOMETRY_CONTRASTIVE_LAMBDA'), ('temperature','GEOMETRY_CONTRASTIVE_TEMPERATURE')]:
        if not os.environ[env]: raise ValueError(f'{env} must be provided with GEOMETRY_CONTRASTIVE_USE')
        OmegaConf.update(cfg, f'geometry_contrastive_loss.{key}', float(os.environ[env]))
    max_neg = os.environ['GEOMETRY_CONTRASTIVE_MAX_NEGATIVE_TOKENS']
    if max_neg.lower() == 'none': max_neg = None
    else:
        max_neg = int(max_neg)
        if max_neg < 1: raise ValueError('GEOMETRY_CONTRASTIVE_MAX_NEGATIVE_TOKENS must be positive or none')
    OmegaConf.update(cfg, 'geometry_contrastive_loss.max_negative_tokens', max_neg)
    time_weight = os.environ['GEOMETRY_CONTRASTIVE_TIME_WEIGHT']
    if time_weight not in ('one_minus_t', 'none'): raise ValueError('GEOMETRY_CONTRASTIVE_TIME_WEIGHT must be one_minus_t or none')
    OmegaConf.update(cfg, 'geometry_contrastive_loss.time_weight', time_weight)
if cfg.precision not in ('fp32', 'bf16'): raise ValueError('PRECISION must be fp32 or bf16')
for key, env in [('epochs','EPOCHS'), ('workers','NUM_WORKERS'), ('accumulation','GARD_GRAD_ACCUM_STEPS'), ('per_gpu_batch_size','PER_GPU_BATCH_SIZE'), ('log_interval','LOG_EVERY_OPTIMIZER_STEPS'), ('checkpoint_interval','CHECKPOINT_INTERVAL')]:
    cfg[key] = int(os.environ[env])
    if cfg[key] < 1: raise ValueError(f'{key} must be positive')
for key, env in [('lr','LEARNING_RATE'), ('novel_weight','NOVEL_WEIGHT')]:
    cfg[key] = float(os.environ[env])
    if not math.isfinite(cfg[key]) or cfg[key] < 0: raise ValueError(f'Invalid {key}')
cfg.data.original_samples = int(os.environ['ORIGINAL_GROUPS_PER_EPOCH'])
cfg.save_ema_epochs = [int(x) for x in os.environ['SAVE_EPOCHS'].split(',') if x]
cfg.eval_epochs = [int(x) for x in os.environ['EVAL_EPOCHS'].split(',') if x]
if any(x < 1 or x > cfg.epochs for x in cfg.eval_epochs + list(cfg.save_ema_epochs)):
    raise ValueError('Save/eval epochs must be within 1..EPOCHS')
if os.environ['NEGATIVE_MANIFEST']:
    cfg.data.negative_manifest = str(Path(os.environ['NEGATIVE_MANIFEST']).resolve())
if not Path(cfg.data.negative_manifest).is_file(): raise FileNotFoundError(cfg.data.negative_manifest)
cfg.checkpoint = str(Path(os.environ['NVS_INIT_CHECKPOINT']).resolve())
if not Path(cfg.checkpoint).is_file(): raise FileNotFoundError(cfg.checkpoint)
cfg.wandb.entity = os.environ['WANDB_ENTITY']
cfg.wandb.project = os.environ['WANDB_PROJECT']
if os.environ['WANDB_ENABLED'].lower() not in ('true', 'false'):
    raise ValueError('WANDB_ENABLED must be true or false')
cfg.wandb.enabled = os.environ['WANDB_ENABLED'].lower() == 'true'
cfg.wandb.name = Path(os.environ['NVS_LAUNCH_CONFIG']).parent.name
OmegaConf.save(cfg, os.environ['NVS_LAUNCH_CONFIG'])
print(f'Prepared config: {os.environ["NVS_LAUNCH_CONFIG"]}')
print(f'Data config: {os.environ["NVS_DATA_TEMPLATE"]}')
geometry_cfg = cfg.get('geometry_contrastive_loss', {})
print(f'V=4, N=2~3, M=4-N: nvs_branch={cfg.nvs_branch}, flowmap_from={cfg.flowmap_from}, per_gpu_batch_size={cfg.per_gpu_batch_size}, accumulation={cfg.accumulation}, distractor_aware_attn_align={cfg.attention_alignment.distractor_aware.use}, distractor_loss={cfg.distractor_loss.mode}, geometry_supcon={geometry_cfg.get("use", False)}; N_group2 + {cfg.novel_weight}*M_all_tokens + {cfg.attention_alignment.lambda_coeff}*attention_alignment + {geometry_cfg.get("lambda_coeff", 0.)}*geometry_supcon; camera tokens are clean')
PY
if [[ "$PREPARE_ONLY" == true ]]; then exit 0; fi
if [[ "$WANDB_ENABLED" == true && -f "$WANDB_ENV_FILE" ]]; then
  set -a
  source "$WANDB_ENV_FILE"
  set +a
fi
if [[ "$WANDB_ENABLED" == true && -z "${WANDB_API_KEY:-}" ]]; then echo 'W&B API key missing. Set WANDB_ENABLED=false to disable W&B.' >&2; exit 1; fi
cp mvr/nvs_training.py "$RUN_DIR/nvs_training_snapshot.py"
cp scripts/train/nvs/train.py "$RUN_DIR/train_snapshot.py"
printf 'Run directory: %s\nGPUs: %s\n' "$RUN_DIR" "$CUDA_VISIBLE_DEVICES"
# A bounded smoke test finishes after optimizer updates without epoch evaluation.
for ARG in "${TRAIN_ARGS[@]}"; do
  if [[ "$ARG" == --max-steps || "$ARG" == --max-steps=* ]]; then
    RESUME_ARGS=()
    if [[ -n "$RESUME_CHECKPOINT" ]]; then RESUME_ARGS=(--resume "$RESUME_CHECKPOINT"); fi
    "$ROOT/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" --max_restarts=0 \
      scripts/train/nvs/train.py --config "$CONFIG" --out "$EXPERIMENT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" \
      "${RESUME_ARGS[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
    exit 0
  fi
done
# Release training GPU memory at eval epochs, evaluate EMA, then resume latest.pt.
IFS=',' read -r -a EVAL_SCHEDULE <<< "$EVAL_EPOCHS"
if [[ ${#EVAL_SCHEDULE[@]} -gt 0 ]]; then
  mapfile -t EVAL_SCHEDULE < <(printf '%s\n' "${EVAL_SCHEDULE[@]}" | sort -nu)
fi
for BOUNDARY in "${EVAL_SCHEDULE[@]}" "$EPOCHS"; do
  [[ -n "$BOUNDARY" ]] || continue
  EPOCH_FILE="$CHECKPOINT_DIR/epoch_$(printf '%03d' "$BOUNDARY").pt"
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    COMPLETED_EPOCHS="$("$ROOT/.venv/bin/python" - "$RESUME_CHECKPOINT" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)
print(int(checkpoint['epoch']))
PY
)"
  else
    COMPLETED_EPOCHS=0
  fi
  if (( COMPLETED_EPOCHS < BOUNDARY )); then
    RESUME_ARGS=()
    if [[ -n "$RESUME_CHECKPOINT" ]]; then RESUME_ARGS=(--resume "$RESUME_CHECKPOINT"); fi
    "$ROOT/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" --max_restarts=0 \
      scripts/train/nvs/train.py --config "$CONFIG" --out "$EXPERIMENT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --until-epoch "$BOUNDARY" \
      "${RESUME_ARGS[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
    RESUME_CHECKPOINT="$CHECKPOINT_DIR/latest.pt"
  fi
  if [[ ",$EVAL_EPOCHS," == *",$BOUNDARY,"* ]]; then
    EVAL_DIR="$EXPERIMENT_DIR/fixed_eval/epoch_$(printf '%03d' "$BOUNDARY")"
    if [[ ! -f "$EVAL_DIR/status.json" ]] || ! "$ROOT/.venv/bin/python" - "$EVAL_DIR/status.json" <<'PY'
import json, sys
status = json.load(open(sys.argv[1]))
raise SystemExit(0 if status.get('status') == 'complete' else 1)
PY
    then
      mkdir -p "$EVAL_DIR"
      if [[ -f "$EVAL_PLAN" && ! -f "$EVAL_DIR/eval_plan.json" ]]; then cp "$EVAL_PLAN" "$EVAL_DIR/eval_plan.json"; fi
      CUDA_VISIBLE_DEVICES="${GPU_DEVICES[0]}" "$ROOT/.venv/bin/python" scripts/eval/nvs/run.py \
        --checkpoint "$EPOCH_FILE" --output "$EVAL_DIR" 2>&1 | tee -a "$EVAL_DIR/eval.log"
    fi
  fi
  if (( BOUNDARY == EPOCHS )); then break; fi
done
"$ROOT/.venv/bin/python" - "$EXPERIMENT_DIR/status.json" "$EPOCHS" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
status = json.loads(path.read_text())
status.update(status='complete', completed_epochs=int(sys.argv[2]))
status.pop('reason', None)
path.write_text(json.dumps(status, indent=2) + '\n')
PY
