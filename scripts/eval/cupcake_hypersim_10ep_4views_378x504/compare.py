# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw
import cv2
p=output_dir(__file__);prev=p.parent/'cupcake_hypersim_10ep_4views';H,W=378,504
out=Image.new('RGB',(W*4,(H+30)*2),'white');draw=ImageDraw.Draw(out);stats=[]
for row,f in enumerate([1,2]):
 low=np.asarray(Image.open(p/f'gard_{f:04d}.png')).astype(float)/255
 high=cv2.resize(np.asarray(Image.open(prev/f'gard_{f:04d}.png')).astype(float)/255,(W,H),interpolation=cv2.INTER_AREA)
 gt=np.asarray(Image.open(p/f'clean_{f:04d}.png')).astype(float)/255
 for col,(title,a) in enumerate([('Input 378x504',Image.open(p/f'input_{f:04d}.png')),('GARD 588x896 (downsampled)',Image.fromarray(np.uint8(np.clip(high,0,1)*255))),('GARD 378x504',Image.open(p/f'gard_{f:04d}.png')),('Clean reference',Image.open(p/f'clean_{f:04d}.png'))]):
  out.paste(a,(col*W,row*(H+30)+30));draw.text((col*W+6,row*(H+30)+8),f'{title} | {f:04d}',fill='black')
 stats.append({'frame':f,'high_downsampled_psnr':float(-10*np.log10(((high-gt)**2).mean())),'low_psnr':float(-10*np.log10(((low-gt)**2).mean()))})
out.save(p/'resolution_comparison.jpg');(p/'resolution_comparison_metrics.json').write_text(json.dumps(stats,indent=2));print(stats)
