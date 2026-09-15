#!/usr/bin/env bash
# Per-channel latent stats of VGGT-Omega inter_frame_blocks[3] tokens (needed once before training
# when mvrm.latent_norm.use=true). Writes mvrm.latent_norm.stats_path from the config.
#   bash run_scripts/train/compute_omega_latent_stats.bash [--num-groups 300]
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GARD_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA:-0}}"
export PYTHONPATH="$GARD_ROOT:$GARD_ROOT/Depth-Anything-3/src:$GARD_ROOT/RAE/src:$GARD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${GARD_PYTHON:-python}"
CONFIG="${GARD_CONFIG:-${CONFIG:-$GARD_ROOT/run_configs/train/train_GARD_omega_hypersim_pairs.yaml}}"

# Default to the same fixed HF release as DA3.
export OMEGA_PAIRS_MANIFEST="${OMEGA_PAIRS_MANIFEST:-$GARD_ROOT/manifests/hypersim_hf_a935b4e251fa.json}"


"$PYTHON_BIN" RAE/src/compute_omega_latent_stats.py --config "$CONFIG" "$@"
