"""N+M alignment with transient-only key exclusion; legacy helper unchanged."""
import torch
from depth_anything_3.utils.geometry import unproject_depth,affine_inverse,as_homogeneous
from mvr.distractor_attention import distractor_aware_target,distractor_aware_attn_loss
from mvr.nvs_attention import attention_loss,geometry_target

@torch.no_grad()
def target(depth,extrinsics,intrinsics,cfg,transient_mask):
    ordinary=geometry_target(depth,extrinsics,intrinsics,cfg)
    if not cfg.distractor_aware.use:return ordinary,None
    points=unproject_depth(depth[...,None].float(),intrinsics.float(),affine_inverse(as_homogeneous(extrinsics.float())))
    b,v,h,w,_=points.shape
    points=points.reshape(b,v,h//14,14,w//14,14,3).mean((3,5)).reshape(b,-1,3)
    distance=-torch.cdist(points,points)
    # Match the legacy cycle-consistent visibility rule, not softmax nonzero support.
    n=(h//14)*(w//14);pw=w//14
    visibility=torch.ones_like(ordinary,dtype=torch.bool)
    if cfg.da3_point_cloud.visibility_mask=='cycle_consistency':
        reference=torch.arange(n,device=depth.device)[None].expand(b,-1)
        for a in range(v):
            for c in range(v):
                if a==c:continue
                sa=slice(a*n,(a+1)*n);sc=slice(c*n,(c+1)*n)
                forward=distance[:,sa,sc].argmax(-1);back=distance[:,sc,sa].argmax(-1)
                rt=torch.gather(back,1,forward)
                d=torch.maximum((rt//pw-reference//pw).abs(),(rt%pw-reference%pw).abs())
                visibility[:,sa,sc]=(d<=cfg.da3_point_cloud.vis_pc_cycle_threshold)[...,None]
    return distractor_aware_target(distance,ordinary,visibility,transient_mask[:,:,1:],
                                  cfg.da3_point_cloud.vis_pc_temperature,cfg.distractor_aware)

def loss(model,targets,valid,cfg,mask):
    if not cfg.distractor_aware.use:
        return attention_loss(model,targets,cfg,mask.shape[1],mask.shape[2])
    layer=int(cfg.mvrm_layer_idx);module=model.module if hasattr(model,'module') else model
    idx,kind,attention=module.blocks[layer].attn.attn_map
    if idx!=layer or kind!='global':raise ValueError('Expected D2 global map')
    prediction=attention.mean(1);del module.blocks[layer].attn.attn_map
    spatial=torch.arange(mask.shape[1]*mask.shape[2],device=mask.device)%mask.shape[2]!=0
    prediction=prediction[:,spatial][:,:,spatial]
    return distractor_aware_attn_loss(prediction,targets,valid,mask[:,:,1:],cfg.distractor_aware)
