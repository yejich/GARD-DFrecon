"""Pixel camera conditioning and explicit, matched train/inference flow paths."""
import torch
from torch import nn
import torch.nn.functional as F
from gard.GARD import GARD

class PixelRayGARD(GARD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera_patch = nn.Conv2d(7, self.encoder_hidden_size, 14, 14)
        nn.init.zeros_(self.camera_patch.weight)
        nn.init.zeros_(self.camera_patch.bias)

    def forward(self, x, t, model_img_size, analysis=None, camera_map=None, camera_views=None, **kwargs):
        condition = None
        if camera_map is not None:
            b,v,c,h,w = camera_map.shape
            if (h,w) != tuple(model_img_size) or c != 7 or camera_views.shape != (b,v):
                raise ValueError('Camera map shape must be B,V,7,H,W')
            patches = self.camera_patch(camera_map.reshape(b*v,c,h,w)).flatten(2).transpose(1,2)
            patches = patches.reshape(b,v,-1,self.encoder_hidden_size)
            patches = patches * camera_views[:,:,None,None]
            # GLD-style: camera condition is added to spatial patches only.
            condition = torch.cat([torch.zeros_like(patches[:,:,:1]), patches],2)
        return super().forward(x,t,model_img_size,analysis=analysis,
                               encoder_token_condition=condition,**kwargs)


def pixel_rays(c2w, intrinsics, image_size, novel_mask):
    """OpenCV rays, direction + origin cross direction + novel-role channel."""
    h,w=image_size
    yy,xx=torch.meshgrid(torch.arange(h,device=c2w.device),torch.arange(w,device=c2w.device),indexing='ij')
    pix=torch.stack([xx,yy,torch.ones_like(xx)],-1).float()
    camera=torch.einsum('hwj,bvij->bvhwi',pix,torch.linalg.inv(intrinsics))
    direction=F.normalize(torch.einsum('bvhwj,bvij->bvhwi',camera,c2w[:,:,:3,:3]),dim=-1)
    origin=c2w[:,:,:3,3][:,:,None,None,:].expand_as(direction)
    role=novel_mask[:,:,None,None,None].expand(*direction.shape[:-1],1).float()
    return torch.cat([direction,torch.cross(origin,direction,dim=-1),role],-1).permute(0,1,4,2,3).contiguous()


def flow_endpoints(lq, clean, flow, source_views, pseudo=None, noise=None, noise_lvl=.3):
    """Returns start state and additive condition; same function used in sampling."""
    if noise is None:noise=torch.randn_like(clean)
    condition=torch.zeros_like(clean)
    condition[:,:source_views]=lq.detach()
    if flow=='legacy':
        start=noise.clone()
        start[:,:source_views]=lq+noise_lvl*noise[:,:source_views]
    elif flow=='pure_noise':
        # Unit Gaussian everywhere; source LQ is still an additive condition.
        start=noise
    elif flow=='pseudo_lq':
        if pseudo is None:raise ValueError('pseudo_lq flow needs target pseudo-LQ features')
        condition[:,source_views:]=pseudo.detach()
        start=condition+noise_lvl*noise
    else:raise ValueError(flow)
    return start,condition


def flow_batch(lq,clean,t,flow,source_views,pseudo=None,noise=None):
    start,condition=flow_endpoints(lq,clean,flow,source_views,pseudo,noise)
    t=t[:,None,None,None]
    return (1-t)*clean+t*start+condition,start-clean


@torch.no_grad()
def warp_rgb(rgb,depth,source_c2w,source_k,target_c2w,target_k):
    """Source-only z-buffer splat; holes filled with ImageNet mean, no target RGB."""
    b,v,_,h,w=rgb.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=rgb.device),torch.arange(w,device=rgb.device),indexing='ij')
    pixels=torch.stack([xx,yy,torch.ones_like(xx)],-1).float().reshape(-1,3)
    results=[];coverage=[]
    for bi in range(b):
        world=[];colors=[];source_valid=[]
        for vi in range(v):
            xyz=(pixels@torch.linalg.inv(source_k[bi,vi]).T)*depth[bi,vi].reshape(-1,1)
            xyz=xyz@source_c2w[bi,vi,:3,:3].T+source_c2w[bi,vi,:3,3]
            source_valid.append(torch.isfinite(depth[bi,vi].reshape(-1)) & (depth[bi,vi].reshape(-1)>0))
            world.append(xyz);colors.append(rgb[bi,vi].permute(1,2,0).reshape(-1,3))
        points=torch.cat(world);colors=torch.cat(colors)
        inv=torch.linalg.inv(target_c2w[bi,0]);cam=points@inv[:3,:3].T+inv[:3,3]
        uv=cam@target_k[bi,0].T;uv=uv[:,:2]/uv[:,2:];xy=uv.round().long()
        valid=torch.cat(source_valid)&torch.isfinite(cam).all(-1)&torch.isfinite(uv).all(-1)&(cam[:,2]>1e-6)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        ids=(xy[valid,1]*w+xy[valid,0]);z=cam[valid,2];col=colors[valid]
        nearest=torch.full((h*w,),torch.inf,device=rgb.device);nearest.scatter_reduce_(0,ids,z,reduce='amin',include_self=True)
        # Resolve exact depth ties deterministically by first source pixel.
        eligible=z==nearest[ids];pick=torch.full((h*w,),len(ids),device=rgb.device,dtype=torch.long)
        order=torch.arange(len(ids),device=rgb.device);pick.scatter_reduce_(0,ids[eligible],order[eligible],reduce='amin',include_self=True)
        seen=pick<len(ids);result=rgb.new_tensor([.485,.456,.406]).expand(h*w,3).clone();result[seen]=col[pick[seen]]
        results.append(result.reshape(h,w,3).permute(2,0,1));coverage.append(seen.float().mean())
    return torch.stack(results)[:,None],torch.stack(coverage)
