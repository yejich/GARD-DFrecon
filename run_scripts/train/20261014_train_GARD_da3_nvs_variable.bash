#!/usr/bin/env bash
# V=4 (N=2~3, M=4-N): group2 MSE + attention alignment; mask-aware OFF.
# CUDA_VISIBLE_DEVICES=2,3 bash run_scripts/train/20261014_train_GARD_da3_nvs_variable.bash
# CPU 설정 준비만: 위 명령 끝에 --prepare-only 추가
# 재시작: 위 명령 끝에 --resume /absolute/run/latest.pt 추가
set -euo pipefail
set +x

# ==================== 여기서 실험 옵션 수정 ====================
EXP_NAME="${EXP_NAME:-nvs_v4_n2-3_attn_align_on_mask_aware_off}"
ATTENTION_ALIGNMENT="${ATTENTION_ALIGNMENT:-true}"
MASK_AWARE_ATTENTION="${MASK_AWARE_ATTENTION:-false}"
PRECISION="${PRECISION:-fp32}"
NEGATIVE_MANIFEST="${NEGATIVE_MANIFEST:-manifests/hypersim_negative_mixed_snapshot.json}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpts/gard_denoiser.pt}"
EPOCHS="${EPOCHS:-30}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"  # GPU당 1그룹, accumulation 자동 계산
ORIGINAL_GROUPS_PER_EPOCH="${ORIGINAL_GROUPS_PER_EPOCH:-10000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
NOVEL_WEIGHT="${NOVEL_WEIGHT:-1.0}"         # M의 전체 1+P token MSE 가중치
LOG_EVERY_OPTIMIZER_STEPS="${LOG_EVERY_OPTIMIZER_STEPS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
SAVE_EPOCHS="${SAVE_EPOCHS:-1,3,10,20,30}"         # EMA; latest.pt에는 optimizer/RNG 포함
EVAL_EPOCHS="${EVAL_EPOCHS:-1,3,10,20,30}" # registry V4+M1 RGB/depth/camera/cloud
EVAL_PLAN="${EVAL_PLAN:-result_eval/20261008_nvs_epoch003_v4_m1/eval_plan.json}"
WANDB_PROJECT="${WANDB_PROJECT:-GARD-DFrecon}"
WANDB_ENTITY="${WANDB_ENTITY:-yejichoi_kaist-kaist}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/home/cvlab20/.config/wandb/gard_yejichoi.env}"
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=',' read -r -a GPU_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_DEVICES[@]}
if (( NUM_GPUS < 1 || GLOBAL_BATCH_SIZE < NUM_GPUS || GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo 'GLOBAL_BATCH_SIZE must be a positive multiple of NUM_GPUS.' >&2; exit 1
fi
export GARD_GRAD_ACCUM_STEPS="${GARD_GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / NUM_GPUS))}"
if (( GARD_GRAD_ACCUM_STEPS * NUM_GPUS != GLOBAL_BATCH_SIZE )); then
  echo 'GARD_GRAD_ACCUM_STEPS * NUM_GPUS must equal GLOBAL_BATCH_SIZE.' >&2; exit 1
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/nvme1n1/hyuna/hf_cache/hub}"
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
  RUN_DIR="${NVS_OUT_DIR:-$ROOT/result_train/${DA3_RUN_ID%%_*}_${EXP_NAME}_${DA3_RUN_ID#*_}}"
fi
mkdir -p "$RUN_DIR"
RUN_DIR="$(realpath "$RUN_DIR")"
EXPERIMENT_NAME="$("$ROOT/.venv/bin/python" - "$GLOBAL_BATCH_SIZE" "$GARD_GRAD_ACCUM_STEPS" "$LEARNING_RATE" "$PRECISION" "$ATTENTION_ALIGNMENT" <<'PY'
import sys
batch, accumulation, lr = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
print(f'TRAIN_{sys.argv[4]}_hypersim_pairs_da3-GIANT-L17_bs{batch}-maxview4-accum{accumulation}_lr-{lr:.0e}_attnalign-{"on" if sys.argv[5]=="true" else "off"}_maskaware-off')
PY
)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$RUN_DIR/training/hypersim_pairs/$EXPERIMENT_NAME}"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
CONFIG="$RUN_DIR/launch_config.yaml"

export NVS_LAUNCH_CONFIG="$CONFIG" NVS_INIT_CHECKPOINT="$INIT_CHECKPOINT"
export PRECISION
export EPOCHS ORIGINAL_GROUPS_PER_EPOCH NUM_WORKERS LEARNING_RATE NOVEL_WEIGHT ATTENTION_ALIGNMENT MASK_AWARE_ATTENTION
export LOG_EVERY_OPTIMIZER_STEPS CHECKPOINT_INTERVAL SAVE_EPOCHS EVAL_EPOCHS NEGATIVE_MANIFEST WANDB_PROJECT WANDB_ENTITY
"$ROOT/.venv/bin/python" - <<'PY'
import math, os
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.load('run_configs/train/train_GARD_da3_nvs_v4_n2_3.yaml')
cfg.precision = os.environ['PRECISION']
for key, env in [('attention_alignment.use','ATTENTION_ALIGNMENT'), ('mask_aware_attention','MASK_AWARE_ATTENTION')]:
    value = os.environ[env].lower()
    if value not in ('true','false'): raise ValueError(f'{env} must be true/false')
    OmegaConf.update(cfg,key,value == 'true')
if cfg.mask_aware_attention: raise ValueError('This NVS experiment requires mask-aware attention OFF')
cfg.attention_alignment.distractor_aware.use = False
if cfg.precision not in ('fp32', 'bf16'): raise ValueError('PRECISION must be fp32 or bf16')
for key, env in [('epochs','EPOCHS'), ('workers','NUM_WORKERS'), ('accumulation','GARD_GRAD_ACCUM_STEPS'), ('log_interval','LOG_EVERY_OPTIMIZER_STEPS'), ('checkpoint_interval','CHECKPOINT_INTERVAL')]:
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
cfg.data.negative_manifest = str(Path(os.environ['NEGATIVE_MANIFEST']).resolve())
if not Path(cfg.data.negative_manifest).is_file(): raise FileNotFoundError(cfg.data.negative_manifest)
cfg.checkpoint = str(Path(os.environ['NVS_INIT_CHECKPOINT']).resolve())
if not Path(cfg.checkpoint).is_file(): raise FileNotFoundError(cfg.checkpoint)
cfg.wandb.entity = os.environ['WANDB_ENTITY']
cfg.wandb.project = os.environ['WANDB_PROJECT']
cfg.wandb.name = Path(os.environ['NVS_LAUNCH_CONFIG']).parent.name
OmegaConf.save(cfg, os.environ['NVS_LAUNCH_CONFIG'])
print(f'Prepared config: {os.environ["NVS_LAUNCH_CONFIG"]}')
print(f'V=4, N=2~3, M=4-N: N_group2 + {cfg.novel_weight}*M_all_tokens + {cfg.attention_alignment.lambda_coeff}*attention_alignment; camera tokens are clean')
PY
if [[ "$PREPARE_ONLY" == true ]]; then exit 0; fi
if [[ -f "$WANDB_ENV_FILE" ]]; then
  set -a
  source "$WANDB_ENV_FILE"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then echo 'W&B API key missing.' >&2; exit 1; fi
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
