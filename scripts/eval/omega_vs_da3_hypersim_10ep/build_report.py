# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import sys,json,csv
import numpy as np
import cv2
from PIL import Image,ImageDraw
ROOT=REPO_ROOT;OUT=output_dir(__file__)
sys.path.insert(0,str(ROOT))
from mvr.dataset.hypersim_pairs import HypersimPairs
OLD=ROOT/'result_eval/hypersim_10ep_ai001/five_methods'
ds=HypersimPairs(ROOT/'data/hypersim_pairs/groups.json',split='eval')
METHODS=[('da3','DA3 input'),('gard_da3','GARD-DA3 10ep'),('omega','Omega input'),('gard_omega','GARD-Omega 10ep')]
metrics=[];provenance=[];grids=[]
def color(a,lo,hi):
 a=np.nan_to_num(a,nan=hi,posinf=hi,neginf=lo)
 return cv2.cvtColor(cv2.applyColorMap(np.uint8(255*(1-np.clip((a-lo)/max(hi-lo,1e-8),0,1))),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
def draw_grid(columns,path,frames):
 w,h=320,240;pad=25
 canvas=Image.new('RGB',(w*len(columns),4*(h+pad)),'white');d=ImageDraw.Draw(canvas)
 for view in range(4):
  for col,(label,imgs) in enumerate(columns):
   a=Image.fromarray(np.asarray(imgs[view],dtype=np.uint8)).resize((w,h))
   canvas.paste(a,(col*w,view*(h+pad)+pad));d.text((col*w+5,view*(h+pad)+5),f'{label} | frame {frames[view]:04d}',fill='black')
 canvas.save(path,quality=93);return canvas
for group in [0,26,53]:
 folder=OUT/f'group_{group:02d}';folder.mkdir(exist_ok=True)
 old=OLD/f'group_{group:02d}';info=json.loads((old/'inputs.json').read_text());item=ds[group];records=[ds.records[i] for i in ds.manifest['eval_groups'][group]['indices']]
 assert [(r['scene'],r['camera'],r['frame']) for r in records]==[(r['scene'],r['camera'],r['frame']) for r in info['records']]
 assert info['distractor']==item['distractor_views']
 inputs=np.stack([np.asarray(Image.open(old/f'input_{i}.png')) for i in range(4)]);clean=np.stack([np.asarray(Image.open(old/f'clean_{i}.png')) for i in range(4)])
 np.testing.assert_array_equal(inputs,np.stack(item['lq_views']));np.testing.assert_array_equal(clean,np.stack(item['hq_views']))
 gt=np.load(old/'gt.npz');ref=gt['depth'];mask=np.asarray(item['pixel_masks'])>0;H,W=ref.shape[1:];valid=gt['valid'].astype(bool)&np.isfinite(ref)&(ref>0)
 omega=np.load(OUT/f'omega/group_{group:02d}/pred.npz')
 native={'da3':np.load(old/'da3.npz')['depth'],'gard_da3':np.load(old/'ours.npz')['depth'],'omega':omega['lq_depth'],'gard_omega':omega['restored_depth']}
 preds={}
 for name,a in native.items():
  if a.ndim==4:a=a[...,0]
  preds[name]=np.stack([cv2.resize(x,(W,H),interpolation=cv2.INTER_NEAREST) for x in a]);valid &= np.isfinite(preds[name])&(preds[name]>0)
 bg=valid&~mask
 if not bg.any():raise ValueError('No clean background for shared-scale evaluation')
 aligned={};errors={}
 for name,label in METHODS:
  pred=preds[name];scale=float(np.median(ref[bg]/pred[bg]));aligned[name]=pred*scale;errors[name]=np.abs(aligned[name]-ref)/np.maximum(ref,1e-8)
  row={'group':group,'method':name,'scale':scale,'native_shape':list(native[name].shape)}
  for region,sel in [('all',valid),('distractor',valid&mask),('background',bg)]:
   row['pixels_'+region]=int(sel.sum())
   row['absrel_'+region]=float(errors[name][sel].mean()) if sel.any() else None
   row['rmse_'+region]=float(np.sqrt(((aligned[name][sel]-ref[sel])**2).mean())) if sel.any() else None
   ratio=np.maximum(aligned[name][sel]/ref[sel],ref[sel]/aligned[name][sel]);row['delta1_'+region]=float((ratio<1.25).mean()) if sel.any() else None
  metrics.append(row)
 lo,hi=np.percentile(ref[valid],[2,98]);cols=[('Clean RGB',clean),('Input RGB',inputs),('Clean GT depth',np.stack([color(x,lo,hi) for x in ref]))]
 cols += [(label,np.stack([color(x,lo,hi) for x in aligned[name]])) for name,label in METHODS]
 frames=[r['frame'] for r in records];grids.append(draw_grid(cols,folder/'depth_comparison.jpg',frames))
 errcols=[('Input RGB',inputs)]+[(label+' AbsRel',np.stack([color(x,0,.5) for x in errors[name]])) for name,label in METHODS]
 draw_grid(errcols,folder/'error_comparison.jpg',frames)
 np.savez_compressed(folder/'comparison.npz',gt_depth=ref,valid=valid,distractor_mask=mask,**aligned)
 record={'group':group,'records':records,'distractor':item['distractor_views'],'cached_da3':str(old),'omega':str(OUT/f'omega/group_{group:02d}'),'input_identity_verified':True};provenance.append(record);(folder/'inputs.json').write_text(json.dumps(record,indent=2))
summary={name:{k:float(np.mean([r[k] for r in metrics if r['method']==name and r[k] is not None])) for k in ['absrel_all','absrel_distractor','absrel_background','rmse_all','delta1_all']} for name,_ in METHODS}
(OUT/'metrics.json').write_text(json.dumps({'summary_group_mean':summary,'per_group':metrics,'protocol':'One median GT/pred scale per method/group, fitted only to valid clean-background pixels; shared valid pixels across methods; nearest resize to 378x504; no confidence filtering.'},indent=2))
with (OUT/'metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
overview=Image.new('RGB',(grids[0].width,sum(im.height for im in grids)),'white');y=0
for im in grids:overview.paste(im,(0,y));y+=im.height
overview.save(OUT/'overview.jpg',quality=93)
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>GARD Omega vs DA3</title><style>body{font:16px sans-serif;margin:20px}img{width:100%}</style><h1>GARD: Omega vs DA3 — 10 epochs</h1><p>Same fixed groups 00,26,53. Depths scaled using clean-background GT; native model resolutions differ. RGB shown is input/GT, not an Omega RGB reconstruction.</p>'+''.join(f'<h2>Group {g:02d}</h2><img src="group_{g:02d}/depth_comparison.jpg"><img src="group_{g:02d}/error_comparison.jpg">' for g in [0,26,53]))
print(json.dumps(summary,indent=2))
