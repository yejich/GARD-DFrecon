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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=output_dir(__file__);H,W=336,504
methods=[('Original GARD',P.parent),('Block own token',P/'self_token'),('Block own view',P/'self_view')]
canvas=Image.new('RGB',(5*W,2*(H+30)),'white');d=ImageDraw.Draw(canvas);stats=[]
for row,(f,xy) in enumerate([(1,(175,259)),(2,(273,105))]):
 paths=[('Input',P.parent/f'input_{f:04d}.png')]+[(n,p/f'gard_{f:04d}.png') for n,p in methods]+[('Clean',P.parent/f'clean_{f:04d}.png')]
 for c,(n,p) in enumerate(paths):
  im=Image.open(p).convert('RGB');canvas.paste(im,(c*W,row*(H+30)+30));d.text((c*W+7,row*(H+30)+8),f'{n} | frame {f:04d}',fill='black')
  if c==0:
   x,y=xy;d.rectangle((x-7,row*(H+30)+30+y-7,x+7,row*(H+30)+30+y+7),outline='cyan',width=2)
 base=np.asarray(Image.open(paths[1][1])).astype(float)/255.;gt=np.asarray(Image.open(paths[-1][1])).astype(float)/255.;x,y=xy
 for n,p in methods:
  a=np.asarray(Image.open(p/f'gard_{f:04d}.png')).astype(float)/255.;err=(a-gt)**2
  stats.append(dict(frame=f,mode=n,full_psnr=float(-10*np.log10(err.mean())),query_patch_psnr=float(-10*np.log10(err[y-7:y+7,x-7:x+7].mean())),mean_abs_change_from_baseline=float(np.abs(a-base).mean())))
canvas.save(P/'comparison.jpg')
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained');diffs=[]
for f in [1,2]:
 b=np.asarray(Image.open(P.parent/f'gard_{f:04d}.png')).astype(float)/255.
 diffs.append([np.abs(np.asarray(Image.open(P/m/f'gard_{f:04d}.png')).astype(float)/255.-b).mean(-1) for m in ['self_token','self_view']])
limit=max(v.max() for r in diffs for v in r)
for r in range(2):
 for c in range(2):
  im=axes[r,c].imshow(diffs[r][c],vmin=0,vmax=limit,cmap='inferno');axes[r,c].set_title(f'Frame {r+1}: '+['own token blocked','own view blocked'][c]);axes[r,c].axis('off')
fig.colorbar(im,ax=axes,label='Mean absolute RGB change from original GARD (0–1)');fig.savefig(P/'change_maps.png',dpi=150)
(P/'comparison_metrics.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
