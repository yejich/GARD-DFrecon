# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import json
from PIL import Image,ImageDraw
P=output_dir(__file__);H,W=336,504
for frames,name in [([1,2],'comparison.jpg'),([5,6],'clean_comparison_baseline.jpg')]:
 canvas=Image.new('RGB',(4*W,len(frames)*(H+30)),'white');d=ImageDraw.Draw(canvas)
 for r,f in enumerate(frames):
  for c,(label,path) in enumerate([('Input',P/f'input_{f:04d}.png'),('Original GARD',P.parent/f'gard_{f:04d}.png'),('All GARD layers: no own token',P/f'gard_{f:04d}.png'),('Clean reference',P/f'clean_{f:04d}.png')]):
   canvas.paste(Image.open(path),(c*W,r*(H+30)+30));d.text((c*W+6,r*(H+30)+8),f'{label} | {f:04d}',fill='black')
 canvas.save(P/name)
a=json.loads((P.parent/'results.json').read_text());b=json.loads((P/'results.json').read_text());rows=[]
for f in [1,2,5,6]:
 x=next(x for x in a['metrics'] if x['frame']==f);y=next(x for x in b['metrics'] if x['frame']==f)
 rows.append({'frame':f,'baseline_psnr':x['gard_psnr'],'masked_psnr':y['gard_psnr']})
(P/'comparison_metrics.json').write_text(json.dumps(rows,indent=2));print(rows)
audit=json.loads((P/'mask_audit.json').read_text());assert len(audit['layers'])==14 and all(x['calls']==49 and x['unmasked_forward_max_error']==0 for x in audit['layers'].values());print('Verified: all 14 layers, 49 calls each, exact unmasked-forward parity.')
