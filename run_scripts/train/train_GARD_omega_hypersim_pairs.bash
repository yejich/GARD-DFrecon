#!/usr/bin/env bash
# GARD on VGGT-Omega (restore inter_frame_blocks[3] tokens), Hypersim clean/distractor pairs.
#   CUDA=0,1 bash run_scripts/train/train_GARD_omega_hypersim_pairs.bash                    # full run
#   CUDA=0,1 bash run_scripts/train/train_GARD_omega_hypersim_pairs.bash --max-steps 1      # smoke test
#   extra args are forwarded (--resume, --fixed-views, --wandb --wandb-run-name ...)
# global_batch_size (16) must be divisible by NUM_GPUS * grad_accum_steps (8).
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA:-0,1}}"
NUM_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export PYTHONPATH="$GARD_ROOT:$GARD_ROOT/Depth-Anything-3/src:$GARD_ROOT/RAE/src:$GARD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/gard_omega_hypersim_pairs_mpl
PYTHON_BIN="${GARD_PYTHON:-python}"
CONFIG="${GARD_CONFIG:-${CONFIG:-$GARD_ROOT/run_configs/train/train_GARD_omega_hypersim_pairs.yaml}}"

# Default to the same fixed HF release as DA3.
export OMEGA_PAIRS_MANIFEST="${OMEGA_PAIRS_MANIFEST:-$GARD_ROOT/manifests/hypersim_hf_a935b4e251fa.json}"


STATS=$("$PYTHON_BIN" - "$CONFIG" "$@" <<'PY'
import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
from mvr.omega_training import checkpoint_normalizer_state

cfg = OmegaConf.load(sys.argv[1])
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--resume')
args, _ = parser.parse_known_args(sys.argv[2:])
checkpoint = args.resume or cfg.stage_2.get('ckpt')
embedded = checkpoint_normalizer_state(checkpoint)
norm = cfg.mvrm.get('latent_norm', {})
stats_path = norm.get('stats_path') if norm.get('use', False) else None
if checkpoint and embedded is None and stats_path and not Path(stats_path).is_file():
    raise SystemExit('Legacy checkpoint requires its original latent stats; refusing to recompute them.')
print(stats_path if stats_path and embedded is None else '')
PY
)
if [[ -n "$STATS" && ! -f "$STATS" ]]; then
  echo "[omega] latent stats missing -> computing $STATS"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" CONFIG="$CONFIG" bash "$GARD_ROOT/run_scripts/train/compute_omega_latent_stats.bash"
fi

exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  RAE/src/train_omega_hypersim_pairs.py --config "$CONFIG" "$@"
