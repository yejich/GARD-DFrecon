# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import sys,json,itertools
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
ROOT=REPO_ROOT;OUT=output_dir(__file__)/'camera_poses';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(SYNTHETIC_DISTRACTOR_ROOT))
from hypersim_pipeline.hypersim_io import load_scene_camera,DATA_ROOT
BASE=OUT.parent;OLD=ROOT/'result_eval/hypersim_10ep_ai001/five_methods'
METHODS=[('da3_clean','DA3 clean'),('da3_input','DA3 distractor input'),('gard_da3','GARD-DA3'),('omega_clean','VGGT-Omega clean'),('omega_input','VGGT-Omega distractor input'),('gard_omega','GARD-Omega')]
COLORS=['#2563eb','#f97316','#059669','#7c3aed','#dc2626','#0891b2'];allrows=[];images=[]
def h5(p):
 with h5py.File(p) as f:return f['dataset'][:]
def camera(ext):
 r=ext[:,:3,:3].astype(float);t=ext[:,:3,3].astype(float);wc=r.transpose(0,2,1);centers=-np.einsum('vij,vj->vi',wc,t)
 assert np.allclose(wc@wc.transpose(0,2,1),np.eye(3),atol=1e-3)
 return centers,wc
def angle(r):return float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))
for group in [0,26,53]:
 records=json.loads((BASE/f'group_{group:02d}/inputs.json').read_text())['records'];frames=[r['frame'] for r in records];centers=[];rots=[]
 for record in records:
  cam=load_scene_camera(record['scene']);d=DATA_ROOT/record['scene']/'_detail'/record['camera'];idx=list(h5(d/'camera_keyframe_frame_indices.hdf5').astype(int)).index(record['frame'])
  centers.append(h5(d/'camera_keyframe_positions.hdf5')[idx]*cam.meters_per_asset_unit)
  # Hypersim cameras use OpenGL axes; prediction cameras use OpenCV (+z forward, +y down).
  rots.append(h5(d/'camera_keyframe_orientations.hdf5')[idx]@np.diag([1.,-1.,-1.]))
 gt=np.array(centers);gr=np.array(rots);omega=np.load(BASE/f'omega/group_{group:02d}/pred.npz');old=OLD/f'group_{group:02d}'
 exts={'da3_clean':np.load(BASE/f'group_{group:02d}/da3_clean.npz')['extrinsics'],'da3_input':np.load(old/'da3.npz')['extrinsics'],'gard_da3':np.load(old/'ours.npz')['extrinsics'],'omega_clean':omega['hq_extrinsics'],'omega_input':omega['lq_extrinsics'],'gard_omega':omega['restored_extrinsics']}
 aligned={};data={'frames':frames,'gt_centers_m':gt.tolist(),'gt_camera_to_world_opencv':gr.tolist(),'methods':{}}
 for name,label in METHODS:
  c,r=camera(exts[name]);rotation=gr[0]@r[0].T;x=(c-c[0])@rotation.T;y=gt-gt[0];scale=float(np.sum(x*y)/np.sum(x*x))
  if scale<=0:raise ValueError('Nonpositive scale alignment')
  a=gt[0]+scale*x;ar=rotation[None]@r;aligned[name]=(a,ar)
  pairs=[]
  for i,j in itertools.combinations(range(4),2):
   relp=r[i].T@r[j];relg=gr[i].T@gr[j];pairs.append(angle(relp@relg.T))
  ce=np.linalg.norm(a-gt,axis=1);oe=np.array([angle(ar[i]@gr[i].T) for i in range(4)])
  row={'group':group,'method':name,'center_rmse_m':float(np.sqrt(np.mean(ce**2))),'center_rmse_nonanchor_m':float(np.sqrt(np.mean(ce[1:]**2))),'relative_rotation_mean_deg':float(np.mean(pairs)),'relative_rotation_max_deg':float(np.max(pairs)),'aligned_orientation_nonanchor_mean_deg':float(np.mean(oe[1:])),'scale':scale};allrows.append(row)
  data['methods'][name]={'aligned_centers_m':a.tolist(),'aligned_camera_to_world':ar.tolist(),'per_view_center_error_m':ce.tolist(),'per_view_orientation_deg':oe.tolist(),**row}
 (OUT/f'group_{group:02d}_poses.json').write_text(json.dumps(data,indent=2))
 allpts=np.concatenate([gt]+[x[0] for x in aligned.values()]);mid=(allpts.min(0)+allpts.max(0))/2;span=max(np.ptp(allpts,axis=0).max(),.5);arrow=span*.14;radius=span*.65
 fig=plt.figure(figsize=(17,10))
 for k,(name,label) in enumerate(METHODS):
  ax=fig.add_subplot(2,3,k+1,projection='3d');a,r=aligned[name]
  for pts,rr,col,marker in [(gt,gr,'#777777','x'),(a,r,COLORS[k],'o')]:
   ax.scatter(*pts.T,color=col,marker=marker,s=35)
   for i in range(4):
    forward=rr[i,:,2];ax.quiver(*pts[i],*forward,length=arrow,color=col,arrow_length_ratio=.25)
    ax.text(*(pts[i]+[0,0,arrow*.12]),str(frames[i]),fontsize=8,color=col)
  for p,q in zip(gt,a):ax.plot(*np.stack([p,q]).T,color='gray',alpha=.4,linestyle=':')
  row=next(x for x in allrows if x['group']==group and x['method']==name)
  ax.set_title(f"{label}\ncenter RMSE {row['center_rmse_m']:.3f} m | rel. rotation {row['relative_rotation_mean_deg']:.2f} deg",fontsize=10)
  ax.set(xlim=(mid[0]-radius,mid[0]+radius),ylim=(mid[1]-radius,mid[1]+radius),zlim=(mid[2]-radius,mid[2]+radius),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)');ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-60)
 fig.suptitle(f'Group {group:02d}: camera centers and viewing directions | gray x = Hypersim GT\nFirst camera pose aligned; one positive scale fitted per method/group. Labels = frame IDs (not temporal trajectory).',fontsize=13)
 fig.tight_layout(rect=(0,0,1,.93));path=OUT/f'group_{group:02d}.png';fig.savefig(path,dpi=140);plt.close(fig);images.append(Image.open(path).convert('RGB'))
 figure=go.Figure()
 for label,pts,rr,col in [('GT',gt,gr,'gray')]+[(label,*aligned[name],COLORS[k]) for k,(name,label) in enumerate(METHODS)]:
  figure.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers+text',text=[str(f) for f in frames],name=label,legendgroup=label,marker=dict(color=col,size=5)))
  for axis in range(3):
   xyz=[]
   for i in range(4):xyz.extend([pts[i],pts[i]+rr[i,:,axis]*arrow*(1 if axis==2 else .4),[None,None,None]])
   figure.add_trace(go.Scatter3d(x=[p[0] for p in xyz],y=[p[1] for p in xyz],z=[p[2] for p in xyz],mode='lines',name=label,legendgroup=label,showlegend=False,line=dict(color=col,width=4 if axis==2 else 2)))
 figure.update_layout(title=f'Group {group:02d}: aligned camera frames (long axis = viewing direction)',scene=dict(aspectmode='cube',xaxis=dict(range=[mid[0]-radius,mid[0]+radius]),yaxis=dict(range=[mid[1]-radius,mid[1]+radius]),zaxis=dict(range=[mid[2]-radius,mid[2]+radius])),legend=dict(groupclick='togglegroup'))
 figure.write_html(OUT/f'group_{group:02d}.html',include_plotlyjs=True)
canvas=Image.new('RGB',(images[0].width,sum(x.height for x in images)),'white');y=0
for im in images:canvas.paste(im,(0,y));y+=im.height
canvas.save(OUT/'overview.jpg',quality=93)
summary={name:{key:float(np.mean([r[key] for r in allrows if r['method']==name])) for key in ['center_rmse_m','center_rmse_nonanchor_m','relative_rotation_mean_deg']} for name,_ in METHODS}
(OUT/'metrics.json').write_text(json.dumps({'summary_group_mean':summary,'per_group':allrows},indent=2));print(json.dumps(summary,indent=2))
(OUT/'README.md').write_text('''# Camera pose comparison

Three fixed four-view groups, not a dense temporal trajectory. Figures show camera centers and optical-axis directions; labels are frame IDs. Interactive HTML additionally shows short camera x/y axes.

Hypersim GT camera positions are converted from asset units to meters. GT camera-to-world OpenGL axes are converted to OpenCV with diag(1,-1,-1). Prediction extrinsics are world-to-camera, so centers are -R^T t.

Each method is aligned to GT using the first camera orientation and center, then one positive scalar is fitted by least squares to the four center displacements. No separate view alignment and no rotation fit to camera centers. Anchor errors are zero by construction; both all-view and nonanchor position errors are reported. This comparison removes global pose/scale ambiguity, not absolute metric-pose accuracy. Pairwise relative rotation errors use all six view pairs and do not depend on alignment. Four views from one scene are insufficient for broad conclusions.

All predictions are reused from the depth comparison; no new GPU inference was required. Clean baselines use clean RGB. Input baselines and GARD restored methods use the same fixed clean/distractor switches. Read metrics.json and group_XX_poses.json for numeric values and exact coordinates.
''')
