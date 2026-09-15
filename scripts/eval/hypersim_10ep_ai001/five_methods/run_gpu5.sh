#!/usr/bin/env bash
# Compatibility name; GPU is selected through CUDA_VISIBLE_DEVICES.
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec bash "$GARD_ROOT/run_scripts/val/compare_five_methods.sh" "$@"
