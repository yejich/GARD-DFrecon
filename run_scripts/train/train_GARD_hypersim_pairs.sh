#!/usr/bin/env bash
# Compatibility entry point for the DA3 Hypersim-pair experiment.
set -euo pipefail
GARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export GARD_CONFIG="${GARD_CONFIG:-$GARD_ROOT/run_configs/train/train_GARD_hypersim_pairs.yaml}"
cd "$GARD_ROOT"
PYTHON_BIN="${GARD_PYTHON:-python}"
if [[ "${1:-}" == "--background" ]]; then
  shift
  "$PYTHON_BIN" - "$GARD_ROOT" "$@" <<'LAUNCHPY'
import json, os, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
folder = root / 'result_train/hypersim_pairs_group2_launch'
folder.mkdir(parents=True, exist_ok=True)
marker = folder / 'process.json'
if marker.exists():
    pid = json.loads(marker.read_text())['pid']
    proc = Path(f'/proc/{pid}/cmdline')
    if proc.exists() and b'JIHYE_train_GARD_da3_hypersim_20k.py' in proc.read_bytes():
        raise SystemExit(f'Training is already running: PID {pid}')
command = ['bash', str(root / 'run_scripts/train/train_GARD_hypersim_pairs.sh'), *sys.argv[2:]]
with (folder / 'train.log').open('a') as log:
    process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(pid=process.pid, command=command, gpus=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1'),
              started=time.strftime('%Y-%m-%d %H:%M:%S'), log=str(folder/'train.log'))
marker.write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
LAUNCHPY
  exit
fi
ARGS=()
if [[ -z "${HYPERSIM_PAIRS_ROOT:-}" ]]; then
  ARGS=(--manifest "$GARD_ROOT/data/hypersim_pairs/groups.json")
fi
exec bash "$GARD_ROOT/run_scripts/JIHYE/train_GARD_da3_hypersim_20k.sh" "${ARGS[@]}" "$@"
