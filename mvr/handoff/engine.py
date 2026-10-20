"""Shared extraction and sampling semantics for training and visualization."""
from types import SimpleNamespace
import numpy as np
import torch
from mvr.nvs_camera import HypersimCameraReferences,align_target_camera
from mvr.grouped_mse import token_groups
from mvr.handoff.model import pixel_rays,warp_rgb,flow_endpoints
from mvr.handoff.alignment import target

class Engine:
    def __init__(self,encoder,cfg,device):
        self.encoder,self.cfg,self.device=encoder,cfg,device
        self.cameras=HypersimCameraReferences(cfg.camera.root,cfg.camera.parameters)
        self.mean=torch.tensor([.485,.456,.406],device=device)[None,None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=device)[None,None,:,None,None]
    def normalize(self,x):return (x-self.mean)/self.std
    def encode(self,images):
        output,_=self.encoder.forward(self.normalize(images),export_feat_layers=[17],ref_view_strategy='first')
        if output.ref_b_idx is not None and not torch.all(output.ref_b_idx==0):raise ValueError('Reference changed')
        return output
    @torch.no_grad()
    def extract(self,batch,supervise=True):
        cfg=self.cfg;src=3
        images=batch['lq_views'].to(self.device)[:,:src];h,w=images.shape[-2:]
        source=self.encode(images);lq=source.aux['feat_layer_17'].float().clone()
        mask,flags=token_groups(batch['pixel_masks'].to(self.device),batch['distractor_views'].to(self.device),None)
        transient,_=token_groups(batch['object_pixel_masks'].to(self.device),batch['distractor_views'].to(self.device),None)
        if cfg.nvs:mask[:,3:]=False;transient[:,3:]=False;flags[:,3:]=False
        else:mask=mask[:,:3];transient=transient[:,:3];flags=flags[:,:3]
        camera_map,camera_views,pseudo=None,None,None
        if cfg.nvs:
            src_ext=source.extrinsics.float();bottom=torch.zeros(*src_ext.shape[:-2],1,4,device=self.device);bottom[...,0,3]=1
            src_c2w=torch.linalg.inv(torch.cat([src_ext,bottom],-2));ks=source.intrinsics.float();targets=[];targetks=[]
            for i,records in enumerate(batch['camera_records']):
                gt,k=self.cameras.cameras(records,(h,w));c2w,_,_=align_target_camera(gt,src_ext[i].cpu().numpy())
                targets.append(c2w);targetks.append(k[3:])
            target_c2w=torch.as_tensor(np.stack(targets),device=self.device,dtype=torch.float32)
            target_k=torch.as_tensor(np.stack(targetks),device=self.device,dtype=torch.float32)
            cams=torch.cat([src_c2w,target_c2w],1);allk=torch.cat([ks,target_k],1)
            role=torch.zeros(cams.shape[:2],device=self.device,dtype=torch.bool);role[:,3:]=True
            camera_map=pixel_rays(cams,allk,(h,w),role)
            camera_views=torch.ones_like(role) if cfg.camera_mode=='source_target' else role
            if cfg.flow=='pseudo_lq':
                warped,coverage=warp_rgb(images,source.depth.float(),src_c2w,ks,target_c2w,target_k)
                pseudoout=self.encode(torch.cat([images,warped],1));pseudo=pseudoout.aux['feat_layer_17'][:,3:].float().clone();del pseudoout
        teacher,targets,valid=None,None,None
        if supervise:
            hq=self.encode(batch['hq_views'].to(self.device)[:,:4 if cfg.nvs else 3]);teacher=hq.aux['feat_layer_17'].float().clone()
            if cfg.attention_alignment.use:targets,valid=target(hq.depth,hq.extrinsics,hq.intrinsics,cfg.attention_alignment,transient)
            del hq
        geometry={'depth':source.depth.float().cpu().numpy(),'extrinsics':source.extrinsics.float().cpu().numpy(),'intrinsics':source.intrinsics.float().cpu().numpy()}
        del source
        return dict(lq=lq,teacher=teacher,mask=mask,flags=flags,transient=transient,targets=targets,valid=valid,
                    camera_map=camera_map,camera_views=camera_views,pseudo=pseudo,image_size=(h,w),source_views=src,source_geometry=geometry)

    @torch.no_grad()
    def sample(self,model,ex,steps=50,seed=42):
        cfg=self.cfg;src=ex['source_views'];shape=(ex['lq'].shape[0],4 if cfg.nvs else 3,*ex['lq'].shape[2:])
        gen=torch.Generator(device=self.device).manual_seed(seed);noise=torch.randn(shape,device=self.device,generator=gen)
        state,condition=flow_endpoints(ex['lq'],torch.empty_like(noise),cfg.flow,src,ex['pseudo'],noise,cfg.noise_lvl)
        shift=np.sqrt(1492992/4096);times=1-torch.linspace(0,1,steps,device=self.device);times=shift*times/(1+(shift-1)*times)
        for i in range(steps-1):
            kwargs={} if not cfg.nvs else dict(camera_map=ex['camera_map'],camera_views=ex['camera_views'])
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg.precision=='bf16'):
                velocity=model(state+condition,times[i:i+1].expand(shape[0]),ex['image_size'],**kwargs)
            state=state+(times[i+1]-times[i])*velocity.float()
        return state

    @torch.no_grad()
    def decode_geometry(self,batch,latent):
        src=3;image=batch['lq_views'].to(self.device)[:,:4 if self.cfg.nvs else 3].clone()
        if self.cfg.nvs:image[:,src:]=self.mean
        output,_=self.encoder.forward(self.normalize(image),export_feat_layers=[],ref_view_strategy='first',mode='val',
            mvrm_cfg=SimpleNamespace(restore_feat_layers=[17],concat_feat=False,skip_and_input_feat=False),
            mvrm_result={('restored_latent',17):latent},export_rgb_feat_layers=True)
        return output
