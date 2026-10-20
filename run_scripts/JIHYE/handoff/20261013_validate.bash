#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/RAE/src:$ROOT/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-$ROOT/.venv/bin/python}" scripts/train/handoff/validate.py --data "${DATA_ROOT:-$ROOT/data/handoff}" "$@"
