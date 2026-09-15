# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from PIL import Image
P=output_dir(__file__);m=json.loads((P/'metadata.json').read_text());a=np.load(P/'attention.npz')['attention'];ph,pw=m['patch_hw'];T=ph*pw+1;order=m['order']
imgs=[np.asarray(Image.open(P.parent/f'input_{f:04d}.png')) for f in order]
means=a.mean(1).reshape(3,2,4,T);stats=[]
# Fixed absolute log probability color scale across queries, times and destination views.
vmax=float(means[...,1:].max());norm=LogNorm(vmin=max(vmax*1e-4,1e-9),vmax=vmax)
for qi,q in enumerate(m['queries']):
 fig,axs=plt.subplots(3,5,figsize=(18,8.8),layout='constrained')
 for s in range(3):
  ax=axs[s,0];ax.imshow(imgs[q['view']]);ax.add_patch(Rectangle((q['col']*14,q['row']*14),14,14,edgecolor='cyan',facecolor='none',lw=2));ax.plot(q['x'],q['y'],'c+',ms=12,mew=2);ax.set_title(f"Query frame {q['frame']:04d} | ({q['x']},{q['y']})\nODE call {m['steps'][s]+1}/49, t={m['times'][s]:.4f}")
  for j,f in enumerate(order):
   w=means[s,qi,j,1:].reshape(ph,pw);mass=float(w.sum());cls=float(means[s,qi,j,0]);ax=axs[s,j+1];ax.imshow(imgs[j]);im=ax.imshow(w,extent=(-.5,503.5,335.5,-.5),cmap='magma',norm=norm,alpha=.68,interpolation='nearest');idx=np.unravel_index(w.argmax(),w.shape);ax.plot(idx[1]*14+7,idx[0]*14+7,'x',color='lime',ms=8)
   ax.set_title(f"Frame {f:04d} ({'clean' if f in [5,6] else 'distractor'})\nspatial {mass*100:.1f}% | CLS {cls*100:.2f}%")
   stats.append(dict(query_frame=q['frame'],call=m['steps'][s]+1,t=m['times'][s],destination_frame=f,spatial_mass=mass,cls_mass=cls,peak_xy=[int(idx[1]*14+7),int(idx[0]*14+7)]))
  for ax in axs[s]:ax.axis('off')
 fig.suptitle(f"Cupcake | GARD layer 9 | query on frame {q['frame']:04d} distractor | mean of 16 heads\nCyan: query patch. Green: highest-attention spatial key per view. Shared absolute log color scale.",fontsize=13)
 fig.colorbar(im,ax=axs[:,1:],shrink=.75,label='Attention probability per key token (log scale)')
 fig.savefig(P/f'query_frame{q["frame"]:04d}.png',dpi=140);plt.close(fig)
 # Final call per-head view mass: includes CLS and spatial tokens.
 masses=a[-1,:,qi,:].reshape(16,4,T).sum(-1)
 fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(masses,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(4),[f'{f:04d}' for f in order]);ax.set_yticks(range(16),range(16));ax.set(xlabel='Destination frame',ylabel='Head',title=f'Query frame {q["frame"]:04d}: final-call attention mass (incl. CLS)')
 for h in range(16):
  for j in range(4):ax.text(j,h,f'{masses[h,j]*100:.0f}%',ha='center',va='center',fontsize=8,color='white' if masses[h,j]<.5 else 'black')
 fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(P/f'head_mass_frame{q["frame"]:04d}.png',dpi=150);plt.close(fig)
(P/'attention_summary.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
