#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"${PYTHON:-$ROOT/.venv/bin/python}" scripts/train/handoff/cache_models.py --out "${CACHE_REPORT:-$ROOT/handoff_model_cache.json}"
