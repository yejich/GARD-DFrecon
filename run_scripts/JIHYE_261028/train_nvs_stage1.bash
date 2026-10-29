#!/usr/bin/env bash
# JIHYE 2026-10-28: clean NVS-only Phase-1 pretraining.
# Datasets: Hypersim + TartanAir + DL3DV.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export ROOT
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPUS[@]}

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
# PhaseCNVS has variable N=1/2/3 and currently requires microbatch 1.
PER_GPU_BATCH_SIZE=1
ACCUMULATION=$((GLOBAL_BATCH_SIZE / NUM_GPUS))
if (( NUM_GPUS < 1 || GLOBAL_BATCH_SIZE < NUM_GPUS || GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo 'GLOBAL_BATCH_SIZE must be divisible by the number of visible GPUs.' >&2
  exit 1
fi

MANIFEST="${PHASE1_MANIFEST:-$ROOT/data/phase1_nvs_manifest.json}"
CHECKPOINT="${INIT_CHECKPOINT:-$ROOT/ckpts/gard_denoiser.pt}"
EPOCHS="${EPOCHS:-30}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-2500}"
WORKERS="${NUM_WORKERS:-2}"
LR="${LEARNING_RATE:-0.0002}"
OUT="${OUT:-$ROOT/result_train/$(date +%Y%m%d)_nvs_stage1_phasec}"
CONFIG="$OUT/launch_config.yaml"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  echo "Run prepare_nvs_stage1_data.bash after downloading all three datasets." >&2
  exit 1
fi
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$OUT"

export PHASE1_MANIFEST="$MANIFEST" INIT_CHECKPOINT="$CHECKPOINT"
export EPOCHS STEPS_PER_EPOCH WORKERS LR ACCUMULATION PER_GPU_BATCH_SIZE
export CONFIG OUT
"$PYTHON" - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
root=Path(os.environ['ROOT']) if 'ROOT' in os.environ else Path.cwd()
cfg=OmegaConf.load(root/'run_configs/train/train_GARD_da3_phase_c_nvs.yaml')
cfg.checkpoint=str(Path(os.environ['INIT_CHECKPOINT']).resolve())
cfg.data.manifest=str(Path(os.environ['PHASE1_MANIFEST']).resolve())
cfg.data.name='phase_c'
cfg.per_gpu_batch_size=int(os.environ['PER_GPU_BATCH_SIZE'])
cfg.accumulation=int(os.environ['ACCUMULATION'])
cfg.steps_per_epoch=int(os.environ['STEPS_PER_EPOCH'])
cfg.epochs=int(os.environ['EPOCHS'])
cfg.workers=int(os.environ['WORKERS'])
cfg.lr=float(os.environ['LR'])
cfg.nvs_branch=True
cfg.flowmap_from=os.environ.get('FLOWMAP_FROM','noise_lq')
cfg.eval_epochs=[]
cfg.save_ema_epochs=[1,3,10,20,30]
cfg.attention_alignment.use=False
cfg.attention_alignment.distractor_aware.use=False
cfg.distractor_loss.mode='none'
cfg.full_view_loss=True
cfg.wandb.enabled=os.environ.get('WANDB_ENABLED','false').lower() == 'true'
cfg.wandb.name=Path(os.environ['OUT']).name
Path(os.environ['CONFIG']).parent.mkdir(parents=True,exist_ok=True)
OmegaConf.save(cfg, os.environ['CONFIG'])
print(f'Prepared Stage-1 config: {os.environ["CONFIG"]}')
print(f'global_batch={int(os.environ["PER_GPU_BATCH_SIZE"])*int(os.environ["ACCUMULATION"])*len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))}, per_gpu=1, accumulation={os.environ["ACCUMULATION"]}, NVS-only=true, eval=off')
PY

CHECKPOINT_DIR="$OUT/checkpoints"
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then RESUME_ARGS=(--resume "$(realpath -e "$RESUME")"); fi
if [[ "${PREPARE_ONLY:-false}" == true ]]; then exit 0; fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" --max_restarts=0 \
  scripts/train/nvs/train.py --config "$CONFIG" --out "$OUT" --checkpoint-dir "$CHECKPOINT_DIR" \
  "${RESUME_ARGS[@]}" "$@"
