"""Fixed-support scoring for restoration V and novel M; no geometry GT invention."""
import json,csv,itertools,os
from pathlib import Path
import numpy as np,cv2,torch,h5py
try:
 from skimage.metrics import structural_similarity
except ModuleNotFoundError:
 def structural_similarity(a,b,data_range=1.,channel_axis=-1,full=False):
  """Small dependency-free SSIM fallback for the fixed RGB evaluator."""
  a=np.asarray(a,dtype=np.float32);b=np.asarray(b,dtype=np.float32)
  if channel_axis is not None:
   a=np.moveaxis(a,channel_axis,-1);b=np.moveaxis(b,channel_axis,-1)
  k=(11,11);sigma=1.5
  mu_a=cv2.GaussianBlur(a,k,sigma);mu_b=cv2.GaussianBlur(b,k,sigma)
  var_a=cv2.GaussianBlur(a*a,k,sigma)-mu_a*mu_a
  var_b=cv2.GaussianBlur(b*b,k,sigma)-mu_b*mu_b
  cov=cv2.GaussianBlur(a*b,k,sigma)-mu_a*mu_b
  c1=(.01*data_range)**2;c2=(.03*data_range)**2
  score=((2*mu_a*mu_b+c1)*(2*cov+c2))/((mu_a*mu_a+mu_b*mu_b+c1)*(var_a+var_b+c2))
  score=np.clip(score,0,1)
  return (float(score.mean()),score) if full else float(score.mean())
from scripts.eval.quant_eval.metrics import centers,fit_sim,transform,pose_metrics,voxel,cloud_metrics,write_ply
from scripts.eval.registry_qual.run import grid,color_depth
import matplotlib.pyplot as plt

def safe(x):
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,np.ndarray):return safe(x.tolist())
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 return x

def pose_similarity(pred,ref):
 p,q=centers(pred),centers(ref)
 if np.linalg.svd(q-q.mean(0),compute_uv=False)[1]>1e-5:
  try:
   return fit_sim(p,q)
  except ValueError as exc:
   fallback_reason=f'degenerate prediction alignment: {exc}'
 else:
  fallback_reason='collinear reference camera centers'
 # Some failed/ablated checkpoints can collapse predicted camera centers.
 # Preserve all RGB/depth/cloud outputs and make the geometry limitation
 # explicit instead of aborting the whole fixed evaluation group.
 ratio=[]
 for i,j in itertools.combinations(range(len(p)),2):
  pd=np.linalg.norm(p[i]-p[j]);qd=np.linalg.norm(q[i]-q[j])
  if pd>1e-9 and np.isfinite(pd) and np.isfinite(qd):ratio.append(qd/pd)
 scale=float(np.median(ratio)) if ratio else 1.0
 if not np.isfinite(scale) or scale<=0:scale=1.0
 r=ref[0,:,:3].T@pred[0,:,:3];t=q[0]-scale*r@p[0]
 return scale,r,t,dict(fallback=fallback_reason,valid_baseline_ratios=len(ratio))

def rgb_scores(pred,target,valid,lp):
 rows=[]
 for i,(p,q,s) in enumerate(zip(pred,target,valid)):
  if not s.any():rows.append(dict(view=i,status='no_valid_reference'));continue
  mse=float(np.mean((p[s]-q[s])**2));_,ss=structural_similarity(p,q,data_range=1.,channel_axis=-1,full=True)
  interior=cv2.erode(s.astype(np.uint8),np.ones((7,7),np.uint8))>0
  a=torch.tensor(p).permute(2,0,1)[None].cuda()*2-1;b=torch.tensor(q).permute(2,0,1)[None].cuda()*2-1
  with torch.no_grad():lmap=lp(a,b).float()[0,0];mask=torch.tensor(s,dtype=torch.float32,device='cuda')[None,None];weight=torch.nn.functional.interpolate(mask,size=lmap.shape,mode='area')[0,0];lval=(lmap*weight).sum()/weight.sum().clamp_min(1e-8)
  rows.append(dict(view=i,psnr=-10*np.log10(max(mse,1e-12)),ssim=float(ss.mean(-1)[interior].mean()) if interior.any() else None,lpips_alex=float(lval),coverage=float(s.mean())))
 return dict(per_view=rows,**{k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None for k in ['psnr','ssim','lpips_alex']})

def reproject_reference(raw,item):
 """Register clean DA3 reference to supplied cameras then z-buffer into V+M.
 Reference pixels/points only; holes remain invalid, never inpainted.
 """
 gt_c2w=np.array(item['reference_c2w']);ge=np.linalg.inv(gt_c2w)[:,:3];sim=pose_similarity(raw['extrinsics'],ge);rgb=raw['input_rgb'];nv=len(item['source_target_c2w']);h,w=raw['depth'].shape[1:]
 yy,xx=np.mgrid[:h,:w];pixels=np.stack([xx,yy,np.ones_like(xx)],-1);p=[]
 for i,(c2w,k) in enumerate(zip(gt_c2w,item['reference_intrinsics'])):
  ref_dist=item.get('reference_distortion') or [[0.]*5]*len(gt_c2w);dist=np.array(ref_dist[i]);uv=cv2.undistortPoints(pixels[...,:2].astype(np.float64).reshape(-1,1,2),np.array(k),dist).reshape(h,w,2);rays=np.concatenate([uv,np.ones((h,w,1))],-1);camera=rays*(raw['depth'][i]*sim[0])[...,None];p.append(camera@c2w[:3,:3].T+c2w[:3,3])
 p=np.stack(p)
 points=p.reshape(-1,3);colors=rgb.reshape(-1,3);good=np.isfinite(points).all(-1)&np.isfinite(raw['depth'].reshape(-1))&(raw['depth'].reshape(-1)>0);points=points[good];colors=colors[good]
 # Paired views score only corresponding reference image; OnTheGo/absent pairs use clean extra cloud.
 ds=[];ps=[];rgbs=[];vs=[]
 paired=item['clean_source_paths'] is not None and len(item['reference_paths'])==nv
 for i,(pose,k) in enumerate(zip(item['source_target_c2w'],item['source_target_intrinsics'])):
  if paired:
   pp=p[i].reshape(-1,3);cc=rgb[i].reshape(-1,3);ok=np.isfinite(pp).all(-1)&(raw['depth'][i].reshape(-1)>0)
   vp=(item.get('clean_source_valid_paths') or [None]*nv)[i] if i<len(item.get('clean_source_paths') or []) else item.get('target_valid_path')
   if vp:ok&=np.load(vp).reshape(-1)
   pp=pp[ok];cc=cc[ok]
  else:pp=points;cc=colors
  e=np.linalg.inv(np.array(pose));cam=pp@e[:3,:3].T+e[:3,3];z=cam[:,2];src_dist=item.get('source_target_distortion') or [[0.]*5]*nv;uv=cv2.projectPoints(cam,np.zeros(3),np.zeros(3),np.array(k),np.array(src_dist[i]))[0].reshape(-1,2);ix=np.rint(np.clip(uv,-1e8,1e8)).astype(np.int64)
  valid=np.isfinite(z)&(z>0)&(ix[:,0]>=0)&(ix[:,0]<w)&(ix[:,1]>=0)&(ix[:,1]<h);idx=np.where(valid)[0];idx=idx[np.argsort(z[idx],kind='stable')];linear=ix[idx,1]*w+ix[idx,0];_,first=np.unique(linear,return_index=True);idx=idx[first];linear=linear[first]
  d=np.full(h*w,np.nan,np.float32);pr=np.full((h*w,3),np.nan,np.float32);c=np.zeros((h*w,3),np.float32);v=np.zeros(h*w,bool);d[linear]=z[idx];pr[linear]=pp[idx];c[linear]=cc[idx];v[linear]=True
  ds.append(d.reshape(h,w));ps.append(pr.reshape(h,w,3));rgbs.append(c.reshape(h,w,3));vs.append(v.reshape(h,w))
 return dict(depth=np.stack(ds),world_points=np.stack(ps),rgb=np.stack(rgbs),valid=np.stack(vs),extrinsics=np.linalg.inv(np.array(item['source_target_c2w']))[:,:3]),dict(kind='joint_clean_DA3_pseudo_GT_registered_to_dataset_cameras',sim3=safe(sim),reference_paths=item['reference_paths'])

def measured_hypersim(item,ref):
 g=item['group'];h,w=ref['depth'].shape[1:];scene=Path(g['input_paths'][0]).parents[2].name;camera=Path(g['input_paths'][0]).parents[1].name
 # Scene/camera from explicit frozen camera path, robust to dataset subdirectories.
 import re
 scene=re.search(r'ai_\d{3}_\d{3}',g['input_paths'][0]).group();camera=re.search(r'cam_\d{2}',g['input_paths'][0]).group()
 csvpath=os.environ.get('HYPERSIM_CAMERA_PARAMETERS',str(Path(__file__).resolve().parents[3]/'datasets/eval/metadata_camera_parameters.csv'))
 geometry_root=Path(os.environ.get('HYPERSIM_GEOMETRY_ROOT',str(Path(__file__).resolve().parents[3]/'datasets/eval/hypersim')))
 with open(csvpath) as f:scale=float(next(r for r in csv.DictReader(f) if r['scene_name']==scene)['settings_units_info_meters_scale'])
 for i,fid in enumerate(g['frame_order']):
  path=geometry_root/scene/'images'/f'scene_{camera}_geometry_hdf5'/f'frame.{fid:04d}.position.hdf5'
  with h5py.File(path) as f:p=f['dataset'][:].astype(np.float32)*scale
  p=cv2.resize(p,(w,h),interpolation=cv2.INTER_NEAREST_EXACT);e=ref['extrinsics'][i];depth=(p@e[:,:3].T+e[:,3])[...,2];v=np.isfinite(p).all(-1)&(depth>0)
  sm=Path(g['input_paths'][i]).with_name('static_mask.png')
  if sm.exists():v&=cv2.resize(np.array(__import__('PIL.Image',fromlist=['Image']).open(sm).convert('L')),(w,h),interpolation=cv2.INTER_NEAREST_EXACT)==0
  ref['depth'][i]=depth;ref['world_points'][i]=p;ref['valid'][i]=v
 return ref

def depth_metrics(pred,gt,support):
 n=int(support.sum())
 if not n:return dict(status='no_reference',coverage=0.)
 valid=np.isfinite(pred)&(pred>0);s=support&valid;nvalid=int(s.sum());ratio=np.maximum(pred[s]/gt[s],gt[s]/pred[s])
 return dict(pixels=n,invalid_prediction_pixels=n-nvalid,absrel=float(np.mean(abs(pred[s]-gt[s])/gt[s])) if nvalid==n else None,rmse=float(np.sqrt(np.mean((pred[s]-gt[s])**2))) if nvalid==n else None,delta1=float((ratio<1.25).sum()/n),coverage=float(support.mean()))

def plot_cloud(path,p,c,q):
 fig=plt.figure(figsize=(9,5));
 for j,(title,pts,col) in enumerate([('Prediction',p,c),('Reference',q,np.tile([.2,.65,.3],(len(q),1)))]):
  ax=fig.add_subplot(1,2,j+1,projection='3d');ids=np.linspace(0,len(pts)-1,min(len(pts),12000),dtype=int);ax.scatter(*pts[ids].T,c=col[ids],s=.4) if len(ids) else None;ax.set_title(title)
 fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)

def score(folder,item,base,gard,raw,lp):
 ref,info=reproject_reference(raw,item);ref['rgb_valid']=ref['valid'].copy()
 for i,p in enumerate(item.get('source_valid_paths') or []):
  mask=np.load(p);ref['valid'][i]&=mask;ref['rgb_valid'][i]&=mask
 if item['group']['dataset']=='Hypersim':ref=measured_hypersim(item,ref);info['V_geometry_kind']='measured_Hypersim_GT_unchanged_surfaces';info['M_geometry_kind']='joint_clean_DA3_pseudo_GT'
 else:info['V_geometry_kind']=info['M_geometry_kind']='joint_clean_DA3_pseudo_GT'
 g=item['group'];V=len(g['input_paths']);prediction_views=len(gard['depth']);M=max(prediction_views-V,0);metric=dict(V=V,M=M,methods={},reference=info,source_rgb_reference_kind=item['source_rgb_reference_kind'],target_rgb_reference_kind=item['target_rgb_reference_kind'])
 # Normalize pseudo geometry units to median V reference-camera baseline.
 unit='meter' if g['dataset'].startswith('Hypersim') else 'median_dataset_reference_camera_baseline'
 e=ref['extrinsics'];camera_centers=centers(e[:V]);baseline=float(np.median([np.linalg.norm(camera_centers[i]-camera_centers[j]) for i,j in itertools.combinations(range(V),2)]));factor=1. if unit=='meter' else baseline
 assert factor>1e-9
 ref['depth']/=factor;ref['world_points']/=factor;ref['extrinsics'][:,:,3]/=factor
 metric['geometry_registration']='Clean DA3 depth scaled by clean-camera baselines, unprojected/projected through dataset cameras including stored distortion; one V-pose Sim3 per prediction reused at all cutoffs';metric['M_condition_camera_model']='pinhole Plucker rays, matching training; reference scoring accounts for distortion';metric['distance_units']=unit;metric['reference_baseline_original_units']=baseline
 np.savez_compressed(folder/'reference.npz',**ref);(folder/'reference_provenance.json').write_text(json.dumps(safe(metric['reference']),indent=2))
 finite=ref['depth'][ref['valid']];lo,hi=np.percentile(finite,[2,98]);depthcols=[('Input',gard['input_rgb'][:V]),('Reference',color_depth(ref['depth'][:V],lo,hi))]
 poseplot=plt.figure(figsize=(7,6));ax=poseplot.add_subplot(111,projection='3d')
 qcent=centers(ref['extrinsics']);ax.plot(*qcent.T,'o-',label='Reference')
 for method,data in [('da3',base),('gard_nvs',gard)]:
  nv=len(data['depth']);sim=pose_similarity(data['extrinsics'][:V],ref['extrinsics'][:V]);scale,r,t=sim[:3];p=transform(data['world_points'],sim);d=data['depth']*scale
  mm=dict(pose_V=pose_metrics(data['extrinsics'][:V],ref['extrinsics'][:V]),alignment=safe(sim),depth_V=depth_metrics(d[:V],ref['depth'][:V],ref['valid'][:V]))
  if nv>V:mm['pose_allM']=pose_metrics(data['extrinsics'],ref['extrinsics']);mm['depth_M']=depth_metrics(d[V:],ref['depth'][V:],ref['valid'][V:])
  if item['clean_source_paths'] is None:mm['rgb_V']=dict(status='N/A',reason='No paired clean RGB for frozen V inputs')
  else:mm['rgb_V']=rgb_scores(data['rgb'][:V],ref['rgb'][:V],ref['rgb_valid'][:V],lp)
  if nv>V:
   # Actual M photograph, no geometry registration or confidence filtering.
   mm['rgb_M']=rgb_scores(data['rgb'][V:],gard['target_rgb'][None],gard['target_valid'][None],lp)
  conf=data['depth_conf'];conf=conf[...,0] if conf.ndim==4 else conf;pop=conf[:V][np.isfinite(conf[:V])];good=np.isfinite(data['depth'])&(data['depth']>0)&np.isfinite(p).all(-1)&np.isfinite(conf)
  support=ref['valid'][:V];mm['confidence_cutoffs']={}
  for level in [20,40,50]:
   threshold=float(np.percentile(pop,level));keep=good[:V]&(conf[:V]>=threshold);sel=keep&support
   pp,cc=voxel(p[:V][sel],data['rgb'][:V][sel],.005);qq,_=voxel(ref['world_points'][:V][support],None,.005)
   ratio=np.maximum(d[:V]/ref['depth'][:V],ref['depth'][:V]/np.maximum(d[:V],1e-12))
   fixed_delta=float(((ratio<1.25)&sel).sum()/max(support.sum(),1))
   cut=dict(threshold=threshold,retained_fraction=float(keep.mean()),reference_coverage=float(sel.sum()/max(support.sum(),1)),depth=depth_metrics(d[:V],ref['depth'][:V],sel),delta1_fixed_support=fixed_delta,cloud=cloud_metrics(pp,qq,[.01,.02,.05]))
   mm['confidence_cutoffs'][str(level)]=cut
   write_ply(folder/f'{method}_decoded_rgb_colored_p{level}.ply',pp,cc);plot_cloud(folder/f'{method}_point_cloud_comparison_p{level}.png',pp,cc,qq)
  if nv>V:
   sel=good[V]&ref['valid'][V];pp,cc=voxel(p[V][sel],data['rgb'][V][sel],.005);qq,_=voxel(ref['world_points'][V][ref['valid'][V]],None,.005);mm['cloud_M_full_valid_reference_support']=cloud_metrics(pp,qq,[.01,.02,.05]);write_ply(folder/'M_decoded_rgb_colored.ply',pp,cc)
  pc=transform(centers(data['extrinsics']),sim);ax.plot(*pc.T,'o-',label=method);depthcols.append((method,color_depth(d[:V],lo,hi)));metric['methods'][method]=mm
 ax.legend();ax.set_title('Cameras: one V-pose Sim3 per method');poseplot.savefig(folder/'camera_comparison.png',dpi=120);plt.close(poseplot)
 grid(folder/'depth_comparison.jpg',depthcols)
 grid(folder/'rgb_comparison.jpg',[('Input',gard['input_rgb'][:V]),('DA3 decoded',base['rgb']),('GARD NVS decoded',gard['rgb'][:V])]+([('Registered clean',ref['rgb'][:V])] if item['clean_source_paths'] is not None else []))
 if len(gard['depth'])>V:
  grid(folder/'M_rgb_comparison.jpg',[('M target',gard['target_rgb'][None]),('M generated',gard['rgb'][V:])])
  gm=metric['methods']['gard_nvs'];scale=gm['alignment'][0];grid(folder/'M_depth_comparison.jpg',[('M target',gard['target_rgb'][None]),('Clean DA3 pseudo depth',color_depth(ref['depth'][V:],lo,hi)),('NVS depth',color_depth(gard['depth'][V:]*scale,lo,hi))])
 np.savez_compressed(folder/'prediction_arrays.npz',**gard);np.save(folder/'depth.npy',gard['depth']);np.save(folder/'depth_conf.npy',gard['depth_conf']);np.savez_compressed(folder/'poses.npz',da3=base['extrinsics'],gard=gard['extrinsics'],reference=ref['extrinsics'])
 for i,im in enumerate(gard['rgb']):__import__('PIL.Image',fromlist=['Image']).fromarray(np.uint8(np.clip(im,0,1)*255)).save(folder/f'decoded_view{i}.png')
 (folder/'metrics.json').write_text(json.dumps(safe(metric),indent=2,allow_nan=False));return metric
