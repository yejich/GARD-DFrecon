"""Explicit source handoff inventory; never stages, commits or pushes files."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
paths=[]
for pattern in ['mvr/handoff/*.py','scripts/train/handoff/*.py','run_scripts/JIHYE/handoff/*','run_configs/JIHYE/handoff/*','tests/test_handoff_experiments.py']:
 paths.extend(ROOT.glob(pattern))
for name in ['src/gard/GARD.py','src/gard/condition_encoder.py','RAE/src/stage2/models/model_utils.py',
 'mvr/nvs_camera.py','mvr/nvs_attention.py','mvr/grouped_mse.py','mvr/distractor_attention.py',
 'mvr/dataset/hypersim_pairs.py','mvr/dataset/hypersim_mixed.py',
 'mvr/eval_models/rgb_decode.py','scripts/eval/quant_eval/metrics.py',
 'Depth-Anything-3/src/depth_anything_3/model/dinov2/vision_transformer.py',
 'Depth-Anything-3/src/depth_anything_3/model/dinov2/vision_transformer_eccv.py']:
 paths.append(ROOT/name)
records=[]
for p in sorted(set(paths)):
 if not p.is_file():raise FileNotFoundError(p)
 rel=str(p.relative_to(ROOT))
 records.append(dict(path=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),git_status=subprocess.check_output(['git','status','--short','--',rel],cwd=ROOT,text=True).strip() or 'tracked unchanged'))
print(json.dumps(dict(note='Required handoff entrypoints and direct local dependencies. Existing tracked repo files also required; this is not a standalone archive. No weights/data/secrets included.',files=records),indent=2))
