#!/usr/bin/env bash
# Compatibility entrypoint. Maintain the canonical script in run_scripts/JIHYE/.
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$GARD_ROOT/run_scripts/JIHYE/train_GARD_da3_hypersim_20k.sh" "$@"
