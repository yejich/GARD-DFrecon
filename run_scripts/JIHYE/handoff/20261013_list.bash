#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
"${PYTHON:-$ROOT/.venv/bin/python}" - "$ROOT/run_configs/JIHYE/handoff/20261013_experiments.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
c=OmegaConf.load(sys.argv[1])
for name,spec in c.experiments.items():print(f'{name:36s} dataset={spec.dataset:9s} NVS={str(spec.nvs):5s} flow={spec.flow:10s} camera={spec.camera:13s} distractor={spec.distractor_aware}')
PY
