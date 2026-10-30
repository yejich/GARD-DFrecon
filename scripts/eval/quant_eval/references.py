import sys,json,csv
from pathlib import Path
import numpy as np
import cv2,h5py
from PIL import Image
from metrics import centers,fit_sim,transform,confidence_keep
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'result_eval/20260923_quant_eval'
sys.path.insert(0,str(ROOT/'Depth-Anything-3/src'))
from depth_anything_3.utils.read_write_model import read_images_binary,read_cameras_binary

def resize(a,hw,nearest=False):return cv2.resize(a,(hw[1],hw[0]),interpolation=cv2.INTER_NEAREST_EXACT if nearest else cv2.INTER_LINEAR)
def stackresize(a,hw,nearest=False):return np.stack([resize(v,hw,nearest) for v in a])
def h5(p):
 if not Path(p).exists() and str(p).startswith('/mnt/dataset1/'):
  q=Path('/mnt/nvme1n1')/Path(p).relative_to('/mnt/dataset1')
  if q.exists(): p=q
 with h5py.File(p) as f:return f['dataset'][:]
def masks(g,hw):
 obj=np.zeros((4,*hw),bool);union=obj.copy()
 for vi,paths in g.get('mask_paths',{}).items():
  for p in paths:
   a=resize(np.asarray(Image.open(p).convert('L')),hw,True)>0;union[int(vi)]|=a
   if 'shadow' not in Path(p).name:obj[int(vi)]|=a
 return obj,union,bool(g.get('mask_paths'))
def hypersim(g,hw):
 suite=json.loads((ROOT/'run_configs/eval/d2_fixed_4view_2clutter_rgb.json').read_text());cfg=suite['correspondence'];scene=g['id'][:10];cam='cam_00'
 with open(cfg['hypersim_camera_parameters']) as f:row=next(x for x in csv.DictReader(f) if x['scene_name']==scene)
 scale=float(row['settings_units_info_meters_scale']);proj=np.array([[float(row[f'M_proj_{i}{j}']) for j in range(4)] for i in range(4)])
 root=Path(cfg['hypersim_data_root'])/scene
 if not root.exists() and str(root).startswith('/mnt/dataset1/'):
  root=Path('/mnt/nvme1n1')/root.relative_to('/mnt/dataset1')
 detail=root/'_detail'/cam;ids=h5(detail/'camera_keyframe_frame_indices.hdf5').reshape(-1).tolist();rs=h5(detail/'camera_keyframe_orientations.hdf5');cs=h5(detail/'camera_keyframe_positions.hdf5')*scale
 ps=[];es=[];ds=[];rgb=[];fix=np.diag([1,-1,-1])
 for frame,path in zip(g['frame_order'],g['input_paths']):
  i=ids.index(frame);r=fix@rs[i].T;t=-r@cs[i];e=np.c_[r,t];p=h5(root/'images'/f'scene_{cam}_geometry_hdf5'/f'frame.{frame:04d}.position.hdf5').astype(np.float32)*scale
  # Half-pixel nearest mapping from processed pixel centers to native GT.
  nh,nw=p.shape[:2];ys=np.clip(np.rint((np.arange(hw[0])+.5)*nh/hw[0]-.5).astype(int),0,nh-1);xs=np.clip(np.rint((np.arange(hw[1])+.5)*nw/hw[1]-.5).astype(int),0,nw-1);p=p[ys[:,None],xs[None,:]]
  ps.append(p);es.append(e);ds.append((p@r.T+t)[...,2]);rgb.append(resize(np.asarray(Image.open(Path(path).with_name('clean.png')).convert('RGB')).astype(np.float32)/255,hw))
 points=np.stack(ps);depth=np.stack(ds);valid=np.isfinite(points).all(-1)&(depth>0);obj,union,known=masks(g,hw)
 return dict(points=points,depth=depth,valid=valid,extrinsics=np.stack(es),rgb=np.stack(rgb),object_mask=obj,union_mask=union,mask_known=np.array(known)),dict(reference='Hypersim clean GT',unit='meter',meters_per_asset_unit=scale,sheared_projection=bool(np.any(abs(proj[3,:2])>1e-9)))
def colmap(g,paths,hw,return_distortion=False):
 root=Path(g['input_paths'][0]).parent.parent
 ims={x.name:x for x in read_images_binary(root/'sparse/0/images.bin').values()};cams=read_cameras_binary(root/'sparse/0/cameras.bin');es=[];ks=[];distortions=[]
 for p in paths:
  im=ims[Path(p).name];cam=cams[im.camera_id];pars=cam.params
  if cam.model in ['SIMPLE_PINHOLE','SIMPLE_RADIAL','RADIAL']:fx=fy=pars[0];cx,cy=pars[1:3]
  elif cam.model in ['PINHOLE','OPENCV','FULL_OPENCV']:fx,fy,cx,cy=pars[:4]
  else:raise ValueError(f'Unsupported camera {cam.model}')
  # These references require undistorted pixels; nonzero distortion must be handled explicitly.
  distortion=pars[3:] if cam.model in ['SIMPLE_RADIAL','RADIAL'] else (pars[4:] if cam.model in ['OPENCV','FULL_OPENCV'] else [])
  dist=np.zeros(8 if cam.model=='FULL_OPENCV' else 5);dist[:len(distortion)]=distortion;distortions.append(dist)
  sx=hw[1]/cam.width;sy=hw[0]/cam.height;k=np.array([[fx*sx,0,(cx+.5)*sx-.5],[0,fy*sy,(cy+.5)*sy-.5],[0,0,1.]])
  es.append(np.c_[im.qvec2rotmat(),im.tvec]);ks.append(k)
 return (np.stack(es),np.stack(ks),distortions) if return_distortion else (np.stack(es),np.stack(ks))
def real(g,hw):
 folder=OUT/g['id'];raw=np.load(folder/'clean_reference.npz');meta=json.loads((folder/'clean_reference.json').read_text());paths=meta['paths'];refext,refk=colmap(g,paths,raw['depth'].shape[1:]);ext,k,distortions=colmap(g,g['input_paths'],hw,return_distortion=True)
 sim=fit_sim(centers(raw['extrinsics']),centers(refext));base=np.median([np.linalg.norm(centers(ext)[i]-centers(ext)[j]) for i in range(4) for j in range(i)])
 if base<=1e-10:raise ValueError('Degenerate reference baseline')
 # Geometry reference stays in the clean DA3 reconstruction's own coordinate frame.
 # Select corresponding evaluation clean views from the jointly inferred clean set.
 names=[Path(x).name for x in paths];selected=[names.index(Path(x).name.replace('2clutter','0clean')) for x in g['input_paths']]
 native_centers=centers(raw['extrinsics'])[selected]
 geometry_base=float(np.median([np.linalg.norm(native_centers[i]-native_centers[j]) for i in range(4) for j in range(i)]))
 if geometry_base<=1e-10:raise ValueError('Degenerate clean DA3 evaluation baseline')
 ext[:,:3,3]/=base
 geometry_ext=raw['extrinsics'][selected].copy();geometry_ext[:,:3,3]/=geometry_base
 assert tuple(raw['depth'].shape[1:])==tuple(hw), 'Reference intrinsics resize must be explicit'
 keep,thr=confidence_keep(raw)
 ds=[resize(raw['depth'][i],hw,True)/geometry_base for i in selected]
 ps=[resize(raw['world_points'][i],hw,True)/geometry_base for i in selected]
 vs=[resize(keep[i].astype(np.uint8),hw,True)>0 for i in selected]
 rgbs=[];decs=[]
 # RGB reference: use corresponding clean view, not a mixture of confidence-filtered cloud colors.
 # Warp via COLMAP cameras + scale-aligned clean Z depth; same camera gives identity.
 rgb_valid=[];rgb_registration=[];direct_rgbs=[];direct_decs=[]
 refext_rgb,refk_rgb,refdist=colmap(g,paths,hw,return_distortion=True)
 names=[Path(x).name for x in paths]
 for vi,path in enumerate(g['input_paths']):
  target_name=Path(path).name.replace('2clutter','0clean')
  ci=names.index(target_name);source_rgb=resize(raw['input_rgb'][ci],hw);source_dec=resize(raw['rgb'][ci],hw);source_z=resize(raw['depth'][ci],hw)*sim[0]/base
  yy,xx=np.mgrid[:hw[0],:hw[1]];pixels=np.stack([xx,yy],-1).astype(np.float64).reshape(-1,1,2)
  normalized=cv2.undistortPoints(pixels,refk_rgb[ci],refdist[ci]).reshape(-1,2)
  cam=np.c_[normalized,np.ones(len(normalized))]*source_z.reshape(-1,1)
  se=refext_rgb[ci].copy();se[:,3]/=base;world=(cam-se[:,3])@se[:,:3]
  target=world@ext[vi,:,:3].T+ext[vi,:,3];tz=target[:,2]
  uv=cv2.projectPoints(target,np.zeros(3),np.zeros(3),k[vi],distortions[vi])[0].reshape(-1,2)
  displacement=np.linalg.norm(uv-pixels.reshape(-1,2),axis=-1)
  pix=np.rint(np.clip(uv,-1e9,1e9)).astype(np.int64);valid=np.isfinite(tz)&(tz>0)&(pix[:,0]>=0)&(pix[:,0]<hw[1])&(pix[:,1]>=0)&(pix[:,1]<hw[0])
  ids=np.where(valid)[0];ids=ids[np.argsort(tz[ids],kind='stable')];linear=pix[ids,1]*hw[1]+pix[ids,0];_,first=np.unique(linear,return_index=True);ids=ids[first];linear=linear[first]
  a=np.zeros((hw[0]*hw[1],3),np.float32);b=a.copy();v=np.zeros(hw[0]*hw[1],bool);a[linear]=source_rgb.reshape(-1,3)[ids];b[linear]=source_dec.reshape(-1,3)[ids];v[linear]=True
  direct_rgbs.append(a.reshape(*hw,3));direct_decs.append(b.reshape(*hw,3));rgb_valid.append(v.reshape(hw));rgb_registration.append(dict(source=paths[ci],coverage=float(v.mean()),median_displacement_px=float(np.median(displacement)),p99_displacement_px=float(np.percentile(displacement,99))))
 rgbs=direct_rgbs;decs=direct_decs
 obj,union,known=masks(g,hw)
 return dict(depth=np.stack(ds),points=np.stack(ps),valid=np.stack(vs),extrinsics=ext,intrinsics=k,geometry_extrinsics=geometry_ext,geometry_intrinsics=raw['intrinsics'][selected],rgb=np.stack(rgbs),decoded_rgb=np.stack(decs),rgb_valid=np.stack(rgb_valid),object_mask=obj,union_mask=union,mask_known=np.array(known)),dict(reference='Joint clean DA3 native geometry: corresponding four clean-view outputs define visible reference support; RGB uses corresponding clean view with COLMAP camera warp',unit='median_clean_DA3_eval_camera_baseline',geometry_baseline=geometry_base,pose_unit='median_COLMAP_eval_camera_baseline',colmap_baseline=float(base),registration_diagnostic_not_used_for_geometry=sim[3],reference_confidence_threshold=thr,clean_views=len(paths),target_camera_distortion=[d.tolist() for d in distortions],rgb_registration=rgb_registration,reference_coverage=[float(x.mean()) for x in vs])
