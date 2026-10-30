"""Freeze one novel camera per frozen registry group without changing V."""
import sys,json,re,hashlib,itertools,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src')]
import numpy as np
from mvr.fixed_suite_training_eval import load_groups
from mvr.fixed_pose_eval import resolve_reference_path
from mvr.nvs_camera import HypersimCameraReferences
from scripts.eval.quant_eval.metrics import centers
from depth_anything_3.utils.read_write_model import read_images_binary,read_cameras_binary

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ext4(e):
 e=np.asarray(e);return np.concatenate([e,np.broadcast_to([0.,0.,0.,1.],(*e.shape[:-2],1,4))],-2)
def freeze(out):
 camera=HypersimCameraReferences(os.environ.get('HYPERSIM_CAMERA_ROOT',str(ROOT/'datasets/eval/hypersim')),os.environ.get('HYPERSIM_CAMERA_PARAMETERS',str(ROOT/'datasets/eval/metadata_camera_parameters.csv')))
 eval_root=Path(os.environ.get('GARD_EVAL_ROOT',str(ROOT/'datasets/eval')))
 negative={g['id']:g for g in json.loads((eval_root/'HypersimNegative/nvs/manifest.json').read_text())['groups']};rows=[]
 for name,g,proto in load_groups([eval_root/'registry.json']):
  hw=g['image_hw'];dataset=g['dataset'];clean=None;valid=None
  if dataset.startswith('Hypersim'):
   scene=re.search(r'ai_\d{3}_\d{3}',g['input_paths'][0]).group();cam=re.search(r'cam_\d{2}',g['input_paths'][0]).group()
   records=[dict(scene=scene,camera=cam,frame=f) for f in g['frame_order']];poses,k=camera.cameras(records,hw)
   if dataset=='HypersimNegative':
    nr=negative[g['id']];target=nr['target_path'];tid=nr['target_frame']
   else:
    candidates=[]
    for p in Path(g['input_paths'][0]).parent.parent.glob('frame.*/clean.png'):
     f=int(p.parent.name.split('.')[1])
     if f in g['frame_order']:continue
     pose,_=camera.cameras([dict(scene=scene,camera=cam,frame=f)],hw);dist=np.linalg.norm(poses[:,:3,3]-pose[0,:3,3],axis=-1)
     if dist.min()>.03:candidates.append((float(dist.mean()),f,str(p)))
    _,tid,target=min(candidates)
   poses,k=camera.cameras(records+[dict(scene=scene,camera=cam,frame=tid)],hw)
   clean=g.get('clean_paths',[str(Path(p).with_name('clean.png')) for p in g['input_paths']]);refpaths=clean+[target];refposes=poses;refk=k;selection='Existing generated static target' if dataset=='HypersimNegative' else 'Distinct original frame, >=3cm baseline, smallest mean V-center distance'
  elif dataset in ['RealX3D','OnTheGo']:
   sm=json.loads(Path(g['source_manifest']).read_text());sg=next(x for x in sm['groups'] if x['id']==g['id']);tid=sg['target_index'];tr=sg['frames'][str(tid)];target=str(resolve_reference_path(tr['path']));valid=str(resolve_reference_path(tr['valid_path']));e=np.array(g['pose_reference_extrinsics']);poses=np.linalg.inv(ext4(np.concatenate([e,np.array(tr['extrinsics'])[None]])))
   k=np.array([sg['frames'][str(i)]['intrinsics'] for i in sg['input_indices']]+[tr['intrinsics']]);selection='Existing held-out target_index from source manifest'
   if dataset=='RealX3D':
    cr=sg['paired_clean_frames'];clean=[str(resolve_reference_path(d['path'])) for d in cr];refpaths=clean+[target];refposes=np.linalg.inv(ext4(np.array([d['extrinsics'] for d in cr]+[tr['extrinsics']])));refk=np.array([d['intrinsics'] for d in cr]+[tr['intrinsics']]);cleanvalid=[str(resolve_reference_path(d['valid_path'])) for d in cr]
   else:
    cr=[sg['frames'][str(i)] for i in sg['clean_reference_indices']];refpaths=[str(resolve_reference_path(d['path'])) for d in cr];refposes=np.linalg.inv(ext4(np.array([d['extrinsics'] for d in cr])));refk=np.array([d['intrinsics'] for d in cr]);cleanvalid=None
  else:
   base=Path(g['input_paths'][0]).parent.parent;ims={im.name:im for im in read_images_binary(base/'sparse/0/images.bin').values()};cams=read_cameras_binary(base/'sparse/0/cameras.bin')
   def camera_for(path):
    im=ims[Path(path).name];cam=cams[im.camera_id];p=cam.params
    if cam.model in ['SIMPLE_PINHOLE','SIMPLE_RADIAL','RADIAL']:fx=fy=p[0];cx,cy=p[1:3]
    else:fx,fy,cx,cy=p[:4]
    sx=hw[1]/cam.width;sy=hw[0]/cam.height;k=np.array([[fx*sx,0,(cx+.5)*sx-.5],[0,fy*sy,(cy+.5)*sy-.5],[0,0,1.]])
    return np.linalg.inv(ext4(np.c_[im.qvec2rotmat(),im.tvec])),k
   vcam=[camera_for(p) for p in g['input_paths']];source_centers=np.array([x[0][:3,3] for x in vcam]);cand=[]
   for p in sorted((base/'images').glob('0clean*')):
    if p.name not in ims:continue
    pose,kk=camera_for(p);distance=np.linalg.norm(source_centers-pose[:3,3],axis=-1)
    if distance.min()>.03:cand.append((float(distance.mean()),str(p),pose,kk))
   _,target,tp,tk=min(cand,key=lambda x:(x[0],x[1]));tid=Path(target).name;poses=np.stack([x[0] for x in vcam]+[tp]);k=np.stack([x[1] for x in vcam]+[tk]);selection='COLMAP distinct clean camera >=0.03 SfM units from V; smallest mean V-center distance'
   clean=[str(Path(p).with_name(Path(p).name.replace('2clutter','0clean'))) for p in g['input_paths']]
   if not all(Path(p).exists() and Path(p).name in ims for p in clean):
    clean=None;refpaths=[str(p) for p in (base/'images').glob('0clean*') if p.name in ims];refpaths=sorted(refpaths,key=lambda p:np.linalg.norm(camera_for(p)[0][:3,3]-source_centers.mean(0)))[:12]
    if target not in refpaths:refpaths=[target]+refpaths[:11]
   else:refpaths=clean+[target]
   rc=[camera_for(p) for p in refpaths];refposes=np.stack([x[0] for x in rc]);refk=np.stack([x[1] for x in rc])
  cleanvalid=cleanvalid if dataset=='RealX3D' else None
  assert target not in g['input_paths'];assert Path(target).exists()
  item=dict(name=name,group=g,protocol=proto,target_path=target,target_id=tid,target_sha256=sha(target),target_valid_path=valid,source_target_c2w=poses.tolist(),source_target_intrinsics=k.tolist(),clean_source_paths=clean,clean_source_valid_paths=cleanvalid,reference_paths=refpaths,reference_c2w=refposes.tolist(),reference_intrinsics=refk.tolist(),reference_sha256=[sha(p) for p in refpaths],selection=selection,target_rgb_reference_kind='heldout_capture_clean_not_guaranteed' if dataset=='OnTheGo' else 'clean_RGB',source_rgb_reference_kind='no_paired_clean' if clean is None else 'paired_clean_registration_required' if dataset in ['RobustNeRF','RealX3D'] else 'synthetic_clean_GT')
  rows.append(item)
 data=dict(V=4,M=1,N=5,groups=rows,registry_sha256=sha(eval_root/'registry.json'));Path(out).write_text(json.dumps(data,indent=2)+'\n');print('Frozen',len(rows),'NVS groups',flush=True)
if __name__=='__main__':freeze(sys.argv[1])
