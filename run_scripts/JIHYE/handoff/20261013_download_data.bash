#!/usr/bin/env bash
# Run only after the other publishing task marks the HF release complete.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export HANDOFF_DATA_ROOT="${DATA_ROOT:-$ROOT/data/handoff}"
export HANDOFF_HF_REPO="${HF_DATA_REPO:-cyjcyj91/Hypersim-Distractor}"
export HANDOFF_HF_RELEASE="${HF_DATA_RELEASE:-releases/20261013_full_synthetic_v4_n3_m1}"
export HANDOFF_HF_REVISION="${HF_DATA_REVISION:-main}"
"${PYTHON:-$ROOT/.venv/bin/python}" - <<'PY'
import json,os,subprocess,sys
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
repo=os.environ['HANDOFF_HF_REPO'];release=os.environ['HANDOFF_HF_RELEASE'];revision=HfApi().repo_info(repo,repo_type='dataset',revision=os.environ['HANDOFF_HF_REVISION']).sha
summary=hf_hub_download(repo,release+'/summary.json',repo_type='dataset',revision=revision)
if json.loads(Path(summary).read_text()).get('status')!='complete':raise RuntimeError('HF release is still being published; retry when complete')
script=hf_hub_download(repo,release+'/download_and_extract.py',repo_type='dataset',revision=revision)
subprocess.run([sys.executable,script,'--destination',os.environ['HANDOFF_DATA_ROOT'],'--revision',revision],check=True)
p=Path(os.environ['HANDOFF_DATA_ROOT'])/'hf_revision.json';p.write_text(json.dumps(dict(repo=repo,release=release,revision=revision),indent=2)+'\n')
PY
