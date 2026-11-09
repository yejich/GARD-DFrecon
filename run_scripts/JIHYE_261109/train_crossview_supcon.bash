#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-}"
case "$MODE" in
  clean_gt|observed) shift ;;
  *) echo "Usage: bash $0 clean_gt|observed [shared training options]"; exit 2 ;;
esac
export NVS_CONFIG="$ROOT/run_configs/train/crossview_supcon_20261109/$MODE.yaml"
export EXP_NAME="crossview_supcon_allanchors_selfgt_cross_${MODE}"
exec bash "$ROOT/run_scripts/JIHYE_261028/train_nvs_supcon_softself.bash" "$@"
