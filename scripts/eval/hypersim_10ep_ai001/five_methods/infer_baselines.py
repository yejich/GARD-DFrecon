# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import argparse,sys,json
import numpy as np
import torch
OUT=output_dir(__file__)
parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['vggt','vggt_omega','vgtw'],required=True);args=parser.parse_args()
if args.method=='vggt':
 sys.path.insert(0,str(VGGT_REPO))
 from vggt.models.vggt import VGGT
 from vggt.utils.load_fn import load_and_preprocess_images as loader
 from vggt.utils.pose_enc import pose_encoding_to_extri_intri as cameras
 model=VGGT.from_pretrained('facebook/VGGT-1B').cuda().eval()
elif args.method=='vggt_omega':
 sys.path.insert(0,str(VGGT_OMEGA_REPO))
 from vggt_omega.models import VGGTOmega
 from vggt_omega.utils.load_fn import load_and_preprocess_images as loader
 from vggt_omega.utils.pose_enc import encoding_to_camera as cameras
 model=VGGTOmega().eval();model.load_state_dict(torch.load(str(VGGT_OMEGA_CKPT),map_location='cpu',weights_only=True));model.cuda()
else:
 sys.path[:0]=[str(VGTW_REPO),str(VGTW_REPO / "scripts")]
 from vgtw_common import load_model,run_model
 from vgtw.utils.load_fn import load_and_preprocess_images as loader
 model=load_model(str(VGTW_CKPT),'cuda')

def unproject(d,e,k):
 v,h,w=d.shape;y,x=np.mgrid[:h,:w]
 cam=np.stack([(x[None]-k[:,0,2,None,None])/k[:,0,0,None,None]*d,(y[None]-k[:,1,2,None,None])/k[:,1,1,None,None]*d,d],-1)
 return np.einsum('vji,vhwj->vhwi',e[:,:3,:3],cam-e[:,:3,3,None,None].transpose(0,2,3,1))

for group in [0,26,53]:
 folder=OUT/f'group_{group:02d}';paths=[str(folder/f'input_{i}.png') for i in range(4)]
 if args.method=='vgtw':
  images=loader(paths);pred=run_model(paths,model,'cuda');e=pred['extrinsic'];k=pred['intrinsic']
 else:
  images=loader(paths);x=images.cuda()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x if args.method=='vggt_omega' else x[None])
  e,k=cameras(pred['pose_enc'],images.shape[-2:]);e=e[0].float().cpu().numpy();k=k[0].float().cpu().numpy()
  pred={key:val[0].float().cpu().numpy() for key,val in pred.items() if torch.is_tensor(val)}
 d=pred['depth'];d=d[...,0] if d.ndim==4 else d
 conf=pred.get('depth_conf',np.ones_like(d));conf=conf[...,0] if conf.ndim==4 else conf
 output=dict(depth=d,depth_conf=conf,extrinsics=e,intrinsics=k,world_points=unproject(d,e,k),images=images.permute(0,2,3,1).numpy())
 if 'distractor_valid_mask' in pred:output['distractor_valid_mask']=pred['distractor_valid_mask']
 np.savez_compressed(folder/f'{args.method}.npz',**output)
 print(args.method,group,d.shape,'finite',np.isfinite(d).mean(),flush=True)
