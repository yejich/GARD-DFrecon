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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
sys.path.insert(0,str(SYNTHETIC_DISTRACTOR_ROOT))
from hypersim_pipeline.hypersim_io import load_scene_camera,load_frames
OUT=output_dir(__file__)
METHODS=[('ours','Ours (GARD 10ep)'),('vggt','VGGT'),('vggt_omega','VGGT-Omega'),('da3','DA3'),('vgtw','VGTW')]

def resize(a,h,w):return cv2.resize(a,(w,h),interpolation=cv2.INTER_NEAREST)

def fit_similarity(x,y):
 valid=np.isfinite(x).all(-1)&np.isfinite(y).all(-1);x=x[valid];y=y[valid]
 rng=np.random.default_rng(42)
 if len(x)>8000:
  ix=rng.choice(len(x),8000,replace=False);x=x[ix];y=y[ix]
 keep=np.ones(len(x),bool)
 for _ in range(4):
  a=x[keep];b=y[keep];ac=a.mean(0);bc=b.mean(0);a=a-ac;b=b-bc
  u,d,vt=np.linalg.svd(a.T@b/len(a));flip=np.ones(3);flip[-1]=np.sign(np.linalg.det(u@vt))
  R=u@np.diag(flip)@vt;scale=(d*flip).sum()/np.square(a).sum(axis=1).mean();translation=bc-scale*ac@R
  error=np.linalg.norm(scale*x@R+translation-y,axis=1);keep=error<=np.quantile(error,.8)
 return scale,R,translation

def ply(path,pts,colors):
 colors=np.uint8(np.clip(colors,0,1)*255)
 a=np.empty(len(pts),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
 for i,k in enumerate(['x','y','z']):a[k]=pts[:,i]
 for i,k in enumerate(['red','green','blue']):a[k]=colors[:,i]
 with path.open('wb') as f:
  f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode());a.tofile(f)

def cloud(points,colors,valid):
 # Same deterministic spatial stride for equivalent source pixel density.
 v,h,w,_=points.shape;stride=max(1,int(np.sqrt(h*w/16000)))
 p=points[:,::stride,::stride].reshape(-1,3);c=colors[:,::stride,::stride].reshape(-1,3);m=valid[:,::stride,::stride].reshape(-1)&np.isfinite(p).all(1)
 return p[m],c[m]

metrics=[];summary=[]
for group in [0,26,53]:
 folder=OUT/f'group_{group:02d}';info=json.loads((folder/'inputs.json').read_text());clean=np.stack([np.array(Image.open(folder/f'clean_{i}.png'))/255 for i in range(4)]);inp=np.stack([np.array(Image.open(folder/f'input_{i}.png'))/255 for i in range(4)]);masks=np.load(folder/'masks.npy')>0
 h,w=clean.shape[1:3];gtp=[];gtd=[];gtv=[]
 for r in info['records']:
  cam=load_scene_camera(r['scene']);fr=load_frames(r['scene'],r['camera'],[r['frame']],cam)[0]
  depth=-(fr.position@fr.R_cam_from_world.T+fr.t_cam_from_world)[...,2]
  gtp.append(resize(fr.position,h,w));gtd.append(resize(depth,h,w));gtv.append(resize(fr.valid.astype(np.uint8),h,w)>0)
 gtp=np.array(gtp);gtd=np.array(gtd);gtv=np.array(gtv)&np.isfinite(gtd)&(gtd>0)
 np.savez_compressed(folder/'gt.npz',world_points=gtp,depth=gtd,valid=gtv,images=clean)
 pgt,cgt=cloud(gtp,clean,gtv);ply(folder/'gt.ply',pgt,cgt)
 lo,hi=np.quantile(pgt,[.01,.99],axis=0);center=(lo+hi)/2;radius=max(hi-lo)*.55
 lo=center-radius;hi=center+radius
 data=[];alignments={};depths=[]
 for name,label in METHODS:
  z=np.load(folder/f'{name}.npz');d=z['depth'];d=d[...,0] if d.ndim==4 else d
  pred=np.stack([resize(x,h,w) for x in d]);valid=gtv&np.isfinite(pred)&(pred>0);bg=valid&~masks
  scale_depth=float(np.median(gtd[bg]/pred[bg]));pred=pred*scale_depth;depths.append(pred)
  row=dict(group=group,method=name,depth_scale=scale_depth)
  for region,mask in [('all',valid),('distractor',valid&masks),('background',bg)]:
   row[f'absrel_{region}']=float(np.mean(np.abs(pred[mask]-gtd[mask])/gtd[mask]));row[f'rmse_{region}']=float(np.sqrt(np.mean((pred[mask]-gtd[mask])**2)))
  metrics.append(row)
  points=z['world_points'];colors=z['images'];H,W=points.shape[1:3]
  target=np.stack([resize(x,H,W) for x in gtp]);bg_native=np.stack([resize(x.astype(np.uint8),H,W)>0 for x in bg])
  sc,R,t=fit_similarity(points[bg_native],target[bg_native]);aligned=sc*points@R+t
  alignments[name]=dict(scale=float(sc),rotation=R.tolist(),translation=t.tolist(),fit='one robust similarity per group, background GT correspondences only')
  pv=np.isfinite(d)&(d>0);p,c=cloud(points,colors,pv);ply(folder/f'{name}_raw.ply',p,c)
  p,c=cloud(aligned,colors,pv);ply(folder/f'{name}_aligned.ply',p,c);data.append((name,label,p,c))
  if 'decoded_rgb' in z:
   pp,cc=cloud(aligned,z['decoded_rgb'],pv);ply(folder/f'{name}_decoded_aligned.ply',pp,cc)
   if name=='ours':ours_colored=(pp,cc)
  if 'distractor_valid_mask' in z:
   native=z['distractor_valid_mask'];native=native[...,0] if native.ndim==4 else native
   pp,cc=cloud(aligned,colors,pv&(native>.5));ply(folder/f'{name}_filtered_aligned.ply',pp,cc)
   data_extra=(name+'_filtered','VGTW (native mask)',pp,cc)
 (folder/'alignment.json').write_text(json.dumps(alignments,indent=2))
 # Shared depth scale after one multiplicative GT alignment per method/group.
 low,high=np.quantile(gtd[gtv],[.02,.98]);tilew,tileh=336,252
 canvas=Image.new('RGB',(tilew*7,(tileh+22)*4));draw=ImageDraw.Draw(canvas)
 for v in range(4):
  tiles=[('Input',inp[v])]+[(label,plt.get_cmap('turbo')(np.clip((dep[v]-low)/(high-low),0,1))[...,:3]) for (_,label),dep in zip(METHODS,depths)]+[('GT depth',plt.get_cmap('turbo')(np.clip((gtd[v]-low)/(high-low),0,1))[...,:3])]
  for col,(label,a) in enumerate(tiles):
   im=Image.fromarray(np.uint8(np.clip(np.nan_to_num(a),0,1)*255)).resize((tilew,tileh))
   canvas.paste(im,(col*tilew,v*(tileh+22)+22));draw.text((col*tilew+3,v*(tileh+22)+3),f'{label} | frame {info["records"][v]["frame"]}',fill='white')
 canvas.save(folder/'depth_comparison.jpg',quality=95)
 # Common axes and viewpoints for all five aligned reconstructions.
 fig=plt.figure(figsize=(20,8))
 for col,(name,label,p,c) in enumerate(data):
  take=np.linspace(0,len(p)-1,min(22000,len(p))).astype(int)
  for row,(elev,azim) in enumerate([(20,-65),(65,-65)]):
   ax=fig.add_subplot(2,5,row*5+col+1,projection='3d');ax.scatter(*p[take].T,c=c[take],s=.3,depthshade=False,rasterized=True)
   ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim);ax.set_axis_off();ax.set_title(label)
 fig.suptitle(f'Group {group:02d} | 4 input views | GT-aligned coordinates | input RGB colors | no confidence filtering')
 fig.subplots_adjust(left=0,right=1,bottom=0,top=.91,wspace=0,hspace=0);fig.savefig(folder/'pointcloud_comparison.jpg',dpi=150);plt.close(fig)
 traces=[];choices=data+[('ours_decoded','Ours (restored RGB)',*ours_colored),('gt','Clean GT',pgt,cgt)]
 if 'data_extra' in locals():choices.append(data_extra);del data_extra
 for name,label,p,c in choices:
  ix=np.linspace(0,len(p)-1,min(32000,len(p))).astype(int);rgb=np.uint8(c[ix]*255)
  traces.append(go.Scatter3d(x=p[ix,0],y=p[ix,1],z=p[ix,2],mode='markers',marker=dict(size=1.5,color=[f'rgb({r},{g},{b})' for r,g,b in rgb]),name=label,visible=len(traces)==0,hoverinfo='skip'))
 fig=go.Figure(traces);fig.update_layout(title=f'Group {group:02d}: choose method, drag to rotate',height=850,margin=dict(l=0,r=0,b=0,t=75),scene=dict(aspectmode='cube',xaxis=dict(range=[lo[0],hi[0]]),yaxis=dict(range=[lo[1],hi[1]]),zaxis=dict(range=[lo[2],hi[2]])),updatemenus=[dict(buttons=[dict(label=label,method='update',args=[{'visible':[j==i for j in range(len(traces))]}]) for i,(_,label,_,_) in enumerate(choices)])])
 fig.write_html(folder/'pointcloud_viewer.html',include_plotlyjs=True)
 summary.append(dict(group=group,views=[r['frame'] for r in info['records']],distractor=info['distractor']))
 print('report complete',group,flush=True)
(OUT/'depth_metrics.json').write_text(json.dumps(metrics,indent=2))
with (OUT/'depth_metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
(OUT/'groups_summary.json').write_text(json.dumps(summary,indent=2))
links=''.join(f'<h2>Group {g:02d}</h2><p><a href="group_{g:02d}/pointcloud_viewer.html">Interactive point cloud</a> | <a href="group_{g:02d}/depth_comparison.jpg">Depth grid</a></p><img src="group_{g:02d}/depth_comparison.jpg"><img src="group_{g:02d}/pointcloud_comparison.jpg">' for g in [0,26,53])
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Five-method comparison</title><style>body{background:#111;color:#eee;font:16px sans-serif}a{color:#8cf}img{width:100%;display:block}</style><h1>Hypersim: Ours / VGGT / VGGT-Omega / DA3 / VGTW</h1><p>Same four selected views per group. Depth uses one GT-fitted scale per method/group. Clouds use one robust GT-background similarity alignment for display. Main clouds have input RGB colors and no confidence/distractor filtering; restored-color and native VGTW-mask variants are available in the viewer. GT is used only for post-hoc evaluation/alignment.</p>'+links)
