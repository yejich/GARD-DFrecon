#!/usr/bin/env bash
# Official GARD repository: https://github.com/cvlab-kaist/GARD
# Pretrained weights are hosted at https://huggingface.co/jinlovespho/GARD
# Download the official checkpoints into this repository's ckpts/ directory.
set -euo pipefail

GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
hf download jinlovespho/GARD --local-dir "$GARD_ROOT/ckpts"
