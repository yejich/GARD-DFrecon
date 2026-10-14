"""Legacy DA3 point-cloud attention alignment over all N+M views."""
import torch
from depth_anything_3.utils.geometry import unproject_depth, affine_inverse, as_homogeneous

@torch.no_grad()
def geometry_target(depth, extrinsics, intrinsics, cfg, patch_size=14):
    if not cfg.get('da3_point_cloud', {}).get('use', False):
        raise ValueError('NVS alignment requires clean DA3 point-cloud targets')
    pc=cfg['da3_point_cloud']
    points=unproject_depth(depth.unsqueeze(-1).float(), intrinsics.float(),
                           affine_inverse(as_homogeneous(extrinsics.float())))
    b,v,h,w,_=points.shape
    ph,pw=h//patch_size,w//patch_size;n=ph*pw
    points=points.reshape(b,v,ph,patch_size,pw,patch_size,3).mean((3,5)).reshape(b,v*n,3)
    distances=-torch.linalg.vector_norm(points.unsqueeze(1)-points.unsqueeze(2),dim=-1)
    temperature=float(pc.get('vis_pc_temperature',1.))
    if temperature==-1:
        blocked=distances.reshape(b,v*n,v,n)
        target=torch.zeros_like(blocked).scatter_(-1,blocked.argmax(-1,keepdim=True),1.).reshape(b,v*n,v*n)
    elif temperature>0:target=(distances/temperature).softmax(-1)
    else:raise ValueError('Invalid geometry temperature')
    visibility=pc.get('visibility_mask','none')
    if visibility=='cycle_consistency':
        valid=torch.zeros_like(target,dtype=torch.bool)
        reference=torch.arange(n,device=depth.device)[None].expand(b,-1)
        threshold=pc.get('vis_pc_cycle_threshold',0)
        for a in range(v):
            for c in range(v):
                sa=slice(a*n,(a+1)*n);sc=slice(c*n,(c+1)*n)
                if a==c:valid[:,sa,sc]=True;continue
                forward=target[:,sa,sc].argmax(-1);backward=target[:,sc,sa].argmax(-1)
                roundtrip=torch.gather(backward,1,forward)
                distance=torch.maximum((roundtrip//pw-reference//pw).abs(),(roundtrip%pw-reference%pw).abs())
                valid[:,sa,sc]=(distance<=threshold)[...,None]
        target=target*valid
        target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
    elif visibility!='none':raise ValueError('Unsupported NVS visibility mode')
    return target.detach()

def attention_loss(model, target, cfg, views, tokens):
    layer=int(cfg['mvrm_layer_idx']);module=model.module if hasattr(model,'module') else model
    index,kind,attention=module.blocks[layer].attn.attn_map
    if index!=layer or kind!='global':raise ValueError('Expected configured global attention layer')
    prediction=attention.mean(1)
    # Do not retain a previous step's attention graph on the model.
    del module.blocks[layer].attn.attn_map
    spatial=torch.arange(views*tokens,device=prediction.device)%tokens!=0
    prediction=prediction[:,spatial][:,:,spatial].float()
    if prediction.shape!=target.shape:raise ValueError('Attention/geometry target shapes differ')
    per_row=-(target*(prediction+1e-8).log()).sum(-1)
    valid=target.sum(-1)>0
    return per_row[valid].mean()

def clear_attention(model, cfg):
    """Checkpoint recomputation may recreate the map during backward."""
    if not cfg.get('use',False):return
    module=model.module if hasattr(model,'module') else model
    attention=module.blocks[int(cfg['mvrm_layer_idx'])].attn
    if hasattr(attention,'attn_map'):del attention.attn_map
