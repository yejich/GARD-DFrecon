"""Renderer target cameras placed into source-only predicted DA3 coordinates."""
import csv
from pathlib import Path
import h5py
import numpy as np
import torch

class HypersimCameraReferences:
 def __init__(self,root,camera_parameters):
  self.root=Path(root);self.rows={r['scene_name']:r for r in csv.DictReader(open(camera_parameters))};self.cache={}
 def cameras(self,records,image_size):
  h,w=image_size;poses=[];ks=[]
  for record in records:
   scene,camera,frame=record['scene'],record['camera'],record['frame'];key=(scene,camera)
   if key not in self.cache:
    arrays=[]
    for name in ['frame_indices','orientations','positions']:
     with h5py.File(self.root/scene/'_detail'/camera/f'camera_keyframe_{name}.hdf5') as f:arrays.append(f['dataset'][:])
    self.cache[key]=(arrays[0].reshape(-1).tolist(),arrays[1],arrays[2])
   ids,rotations,centers=self.cache[key];idx=ids.index(frame);row=self.rows[scene]
   pose=np.eye(4);pose[:3,:3]=rotations[idx]@np.diag([1.,-1.,-1.]);pose[:3,3]=centers[idx]*float(row['settings_units_info_meters_scale']);poses.append(pose)
   # NDC projection converted to processed pixel coordinates, pixel centers.
   k=np.eye(3);k[0,0]=float(row['M_proj_00'])*w/2;k[1,1]=float(row['M_proj_11'])*h/2
   k[0,2]=(1-float(row['M_proj_02']))*w/2-.5;k[1,2]=(1+float(row['M_proj_12']))*h/2-.5;ks.append(k)
  return np.stack(poses),np.stack(ks)

def align_target_camera(gt_c2w,predicted_source_w2c):
 """Anchor rotation/translation to first V; scale from median V baselines.

 No target RGB or teacher prediction is used. Source poses are calibration
 references for M only, never source camera-conditioning tokens.
 """
 pred=np.asarray(predicted_source_w2c,dtype=np.float64)
 if pred.shape[-2:]==(3,4):
  bottom=np.zeros((len(pred),1,4));bottom[:,:,3]=1;pred=np.concatenate([pred,bottom],1)
 pc=np.linalg.inv(pred);v=len(pc);gt=gt_c2w[:v]
 ratios=[np.linalg.norm(pc[i,:3,3]-pc[j,:3,3])/np.linalg.norm(gt[i,:3,3]-gt[j,:3,3]) for i in range(v) for j in range(i) if np.linalg.norm(gt[i,:3,3]-gt[j,:3,3])>1e-8]
 if not ratios:raise ValueError('Degenerate source baseline')
 scale=float(np.median(ratios))
 if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid source scale')
 rotation=pc[0,:3,:3]@gt[0,:3,:3].T;translation=pc[0,:3,3]-scale*rotation@gt[0,:3,3]
 result=gt_c2w[v:].copy();result[:,:3,:3]=rotation@result[:,:3,:3];result[:,:3,3]=scale*(result[:,:3,3]@rotation.T)+translation
 residual=np.linalg.norm(scale*(gt[:,:3,3]@rotation.T)+translation-pc[:,:3,3],axis=-1).mean()
 return result,scale,float(residual)
