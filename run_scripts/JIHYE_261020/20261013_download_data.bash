#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/run_scripts/JIHYE/handoff/20261013_download_data.bash" "$@"
