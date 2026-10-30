#!/usr/bin/env bash
# JIHYE 2026-10-30: mixed Hypersim + negative NVS with optional relative pose conditioning.
# The shared implementation lives in the repository's common NVS launcher.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Preserve the 261030 experiment family while exposing the new camera-pose
# convention explicitly. All other ablation options are handled by the
# shared 261028 launcher.
exec bash "$ROOT/run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash" \
  --camera-pose-mode "${CAMERA_POSE_MODE:-first_relative}" "$@"
