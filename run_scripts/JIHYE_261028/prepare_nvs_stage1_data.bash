#!/usr/bin/env bash
# Prepare clean Phase-1 NVS data from Hypersim, TartanAir and DL3DV.
# Dataset downloads are intentionally explicit because all three datasets
# have separate licenses/access requirements.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
HYPERSIM_ROOT="${HYPERSIM_ROOT:-$ROOT/data/train/hypersim}"
TARTANAIR_ROOT="${TARTANAIR_ROOT:-$ROOT/data/train/tartanair}"
DL3DV_ROOT="${DL3DV_ROOT:-$ROOT/data/train/dl3dv}"
HYPERSIM_CAMERA_CSV="${HYPERSIM_CAMERA_CSV:-$ROOT/download_scripts/hypersim/metadata_camera_parameters.csv}"
MANIFEST="${PHASE1_MANIFEST:-$ROOT/data/phase1_nvs_manifest.json}"
MAX_SCENES="${MAX_SCENES:-0}"

usage() {
  cat <<'EOF'
Usage: bash run_scripts/JIHYE_261028/prepare_nvs_stage1_data.bash

Required dataset roots (override with environment variables):
  HYPERSIM_ROOT       raw Hypersim root containing data/ai_*/
  HYPERSIM_CAMERA_CSV Hypersim metadata_camera_parameters.csv
  TARTANAIR_ROOT      TartanAir root containing data/*/Easy/*/
  DL3DV_ROOT          DL3DV root containing scene */*/transforms.json
  PHASE1_MANIFEST     output manifest path

This script validates existing downloads and rebuilds the portable manifest.
Use the official dataset download tools separately; do not download large
datasets implicitly from a training command.
EOF
}

[[ "${1:-}" != --help ]] || { usage; exit 0; }
for path in "$HYPERSIM_ROOT" "$TARTANAIR_ROOT" "$DL3DV_ROOT" "$HYPERSIM_CAMERA_CSV"; do
  [[ -e "$path" ]] || { echo "Missing dataset asset: $path" >&2; exit 1; }
done

mkdir -p "$(dirname "$MANIFEST")"
"$PYTHON" "$ROOT/scripts/train/nvs/build_phase_c_manifest.py" \
  --hypersim-root "$HYPERSIM_ROOT" \
  --hypersim-camera-csv "$HYPERSIM_CAMERA_CSV" \
  --tartanair-root "$TARTANAIR_ROOT" \
  --dl3dv-root "$DL3DV_ROOT" \
  --output "$MANIFEST" \
  --max-scenes "$MAX_SCENES"

"$PYTHON" - "$MANIFEST" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
groups=d.get('groups', [])
counts=Counter(g.get('dataset') for g in groups)
if set(counts) != {'hypersim','tartanair','dl3dv'}:
    raise SystemExit(f'Expected all three datasets, got {dict(counts)}')
if any(len(g.get('views', [])) < 4 for g in groups):
    raise SystemExit('Every Phase-1 group must contain at least four views')
missing=[v['path'] for g in groups for v in g['views'] if not Path(v['path']).is_file()]
if missing: raise SystemExit(f'Missing {len(missing)} manifest image files; first={missing[0]}')
print(f'Validated {len(groups)} groups: {dict(counts)}')
PY
