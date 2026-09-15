# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import json
import cv2,numpy as np
from PIL import Image,ImageDraw
OUT=output_dir(__file__);OLD=OUT.parent/'hypersim_10ep_ai001/five_methods';grids=[];scales={}
labels=['Clean image','Distractor image','DA3 clean depth','GARD-DA3 depth','VGGT-Omega clean depth','GARD-Omega depth']
def color(a,lo,hi):return cv2.cvtColor(cv2.applyColorMap(np.uint8(255*(1-np.clip((np.nan_to_num(a,nan=hi)-lo)/max(hi-lo,1e-8),0,1))),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
for g in [0,26,53]:
 folder=OUT/f'group_{g:02d}';d=np.load(folder/'comparison.npz');ref=d['gt_depth'];valid=d['valid'];bg=valid&~d['distractor_mask'];pred=np.load(folder/'da3_clean.npz')['depth'];sel=bg&np.isfinite(pred)&(pred>0);scale=float(np.median(ref[sel]/pred[sel]));pred=pred*scale;scales[str(g)]=scale
 omega_clean=np.load(OUT/f'omega/group_{g:02d}/pred.npz')['hq_depth']
 omega_clean=np.stack([cv2.resize(x,(ref.shape[2],ref.shape[1]),interpolation=cv2.INTER_NEAREST) for x in omega_clean])
 omega_sel=bg&np.isfinite(omega_clean)&(omega_clean>0)
 omega_scale=float(np.median(ref[omega_sel]/omega_clean[omega_sel]));omega_clean=omega_clean*omega_scale
 scales[str(g)]={'da3_clean':scale,'omega_clean':omega_scale}
 lo,hi=np.percentile(ref[valid],[2,98]);cols=[np.stack([np.asarray(Image.open(OLD/f'group_{g:02d}/{kind}_{v}.png')) for v in range(4)]) for kind in ['clean','input']]
 cols += [np.stack([color(x,lo,hi) for x in a]) for a in [pred,d['gard_da3'],omega_clean,d['gard_omega']]]
 frames=[r['frame'] for r in json.loads((folder/'inputs.json').read_text())['records']]
 w,h,pad=384,288,25;grid=Image.new('RGB',(6*w,4*(h+pad)),'white');draw=ImageDraw.Draw(grid)
 for v in range(4):
  for j,(label,imgs) in enumerate(zip(labels,cols)):
   grid.paste(Image.fromarray(imgs[v]).resize((w,h)),(j*w,v*(h+pad)+pad));draw.text((j*w+4,v*(h+pad)+5),f'{label} | frame {frames[v]:04d}',fill='black')
 grid.save(folder/'depth_comparison_reordered.jpg',quality=95);grids.append(grid)
canvas=Image.new('RGB',(grids[0].width,sum(x.height for x in grids)),'white');y=0
for grid in grids:canvas.paste(grid,(0,y));y+=grid.height
canvas.save(OUT/'overview_reordered.jpg',quality=95)
(OUT/'reordered_grid.json').write_text(json.dumps({'columns':labels,'groups':[0,26,53],'clean_depth_scales':scales,'note':'DA3 clean is inferred from clean RGB, not GT depth. Common GT-derived color range and clean-background scale alignment retained. Column 5 is the clean-input Omega prediction, not GT depth; column 6 is restored Omega depth. Input switches remain the fixed evaluation group switches.'},indent=2))
(OUT/'index_reordered.html').write_text('<!doctype html><meta charset="utf-8"><title>Reordered comparison</title><style>img{width:100%}body{font:16px sans-serif}</style><p>'+ ' | '.join(labels)+'</p>'+''.join(f'<h2>Group {g:02d}</h2><img src="group_{g:02d}/depth_comparison_reordered.jpg">' for g in [0,26,53]))
print('Saved overview_reordered.jpg and three group grids')
