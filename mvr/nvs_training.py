"""Opt-in restoration+NVS primitives with target-only camera conditioning.

GLD reference: https://github.com/cvlab-kaist/GLD
Source L17 MUST come from a source-only DA3 forward. Teacher L17 comes from
all-clean source+novel views, in the same reference/order. Never encode novel
RGB into the source condition. Cameras are OpenCV and already aligned to the
source-only DA3 coordinate system; this module does not infer camera scale.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from gard.GARD import GARD


class NVSGARD(GARD):
    """Shared GARD attention over N input views + M novel-view grids."""
    def __init__(self, **kwargs):
        if kwargs.get('mask_head_enabled', False) or kwargs.get('c_feat_enabled', False):
            raise ValueError('This experiment disables mask head and C_feat')
        kwargs['mask_head_enabled'] = False
        super().__init__(**kwargs)
        # Zero projection preserves pretrained GARD behavior on introduction.
        # The extra role channel is ONLY nonzero on novel grids, never on N.
        self.nvs_camera_embedder = nn.Linear(7, self.encoder_hidden_size)
        nn.init.zeros_(self.nvs_camera_embedder.weight)
        nn.init.zeros_(self.nvs_camera_embedder.bias)

    def forward(self, x, t, model_img_size, analysis=None, return_aux=False,
                lq_latent=None, c_feat=None, condition_mask=None,
                nvs_rays=None, nvs_view_mask=None):
        condition = None
        if nvs_rays is not None:
            b,v,n,_=x.shape
            if nvs_rays.shape != (b,v,n-1,6) or nvs_view_mask is None or nvs_view_mask.shape != (b,v):
                raise ValueError('Expected Plucker rays [B,V,patches,6] and novel mask [B,V]')
            mask=nvs_view_mask.to(device=x.device,dtype=torch.bool)
            # Clear N rays before projection, and clear all projected N tokens
            # after projection as well: learned bias cannot introduce V pose.
            rays=torch.where(mask[:,:,None,None],nvs_rays.to(x.device),0.)
            role=mask[:,:,None,None].expand(b,v,n-1,1).to(rays.dtype)
            patches=self.nvs_camera_embedder(torch.cat([rays,role],-1))
            cls=self.nvs_camera_embedder(torch.cat([rays.mean(2,keepdim=True),role[:,:,:1]],-1))
            condition=torch.cat([cls,patches],2)*mask[:,:,None,None].to(patches.dtype)
        elif nvs_view_mask is not None:
            raise ValueError('Novel view mask requires target rays')
        return super().forward(x,t,model_img_size,analysis,return_aux,
                               lq_latent,c_feat,condition_mask,encoder_token_condition=condition)


@dataclass
class NVSFlowBatch:
    model_input: torch.Tensor
    target_velocity: torch.Tensor
    source_condition: torch.Tensor
    novel_view_mask: torch.Tensor
    time: torch.Tensor


def prepare_nvs_flow(source_lq, clean_teacher, time, *, noise=None, noise_lvl=.3, double_condition=True):
    """t=0 clean teacher; t=1 restoration start / pure-noise novel grids.

    Shapes: source_lq [B,V,T,C], clean_teacher [B,V+M,T,C]. CLS included.
    clean_teacher/source_lq are frozen DA3 outputs in the SAME feature units.
    """
    if source_lq.ndim!=4 or clean_teacher.ndim!=4 or source_lq.shape[0]!=clean_teacher.shape[0] or source_lq.shape[2:]!=clean_teacher.shape[2:]:
        raise ValueError('Source/teacher feature dimensions must match')
    b,v=source_lq.shape[:2];total=clean_teacher.shape[1]
    if total<=v or time.shape!=(b,) or not torch.isfinite(time).all() or not ((time>=0)&(time<=1)).all():
        raise ValueError('Need at least one novel grid and t in [0,1] per sample')
    if noise_lvl<0:raise ValueError('noise_lvl must be nonnegative')
    if noise is None:noise=torch.randn_like(clean_teacher)
    if noise.shape!=clean_teacher.shape:raise ValueError('Noise must match teacher')
    condition=torch.zeros_like(clean_teacher);condition[:,:v]=source_lq.detach()
    start=noise.clone();start[:,:v]=noise_lvl*noise[:,:v]+condition[:,:v]
    teacher=clean_teacher.detach();t=time[:,None,None,None]
    state=(1-t)*teacher+t*start
    if double_condition:state=state+condition
    novel=torch.zeros((b,total),device=teacher.device,dtype=torch.bool);novel[:,v:]=True
    return NVSFlowBatch(state,start-teacher,condition,novel,time)


def nvs_flow_losses(prediction,batch,*,novel_weight=1.,source_distractor_mask=None,group_mse_cfg=None):
    """N: legacy clean/distractor group2 MSE; M: all-clean 1+P MSE.

    Camera tokens always belong to clean, with no independent camera term.
    """
    from mvr.grouped_mse import grouped_mse
    if not math.isfinite(float(novel_weight)) or novel_weight < 0:
        raise ValueError('novel_weight must be finite and nonnegative')
    if prediction.shape!=batch.target_velocity.shape:raise ValueError('Prediction/velocity shapes must match')
    if prediction.shape[2]<2:raise ValueError('Need camera token and at least one patch')
    b,total,tokens,_=prediction.shape
    n=batch.source_condition.shape[1]-int(batch.novel_view_mask[0].sum())
    if not torch.equal(batch.novel_view_mask, batch.novel_view_mask[:1].expand_as(batch.novel_view_mask)):
        raise ValueError('All samples must use the same N/M layout')
    if source_distractor_mask is None:
        source_distractor_mask=torch.zeros(b,n,tokens,device=prediction.device,dtype=torch.bool)
    if source_distractor_mask.shape!=(b,n,tokens) or source_distractor_mask.dtype!=torch.bool:
        raise ValueError('Expected boolean distractor mask [B,N,1+P]')
    if source_distractor_mask[:,:,0].any():raise ValueError('Camera tokens must be clean')
    cfg=group_mse_cfg or dict(mode='group2',lambda_clean=1.,lambda_distractor=1.)
    if cfg.get('mode','group2')!='group2':raise ValueError('NVS restoration requires group2')
    restoration,parts=grouped_mse(prediction[:,:n],batch.target_velocity[:,:n],source_distractor_mask,
                                source_distractor_mask.any(-1),cfg)
    novel=(prediction[:,n:].float()-batch.target_velocity[:,n:].float()).square().mean()
    weighted_novel=novel_weight*novel
    return dict(loss=restoration+weighted_novel,restoration_loss=restoration,
                nvs_loss=novel,weighted_nvs_loss=weighted_novel,
                restoration_clean_loss=parts['clean'],restoration_distractor_loss=parts['distractor'],
                attention_alignment_loss=novel.new_zeros(()),weighted_attention_alignment_loss=novel.new_zeros(()))


class NVSLossTracker:
    """Detached, sample-weighted logging across microbatches and DDP ranks.

    update() never modifies tensors used for backward. write() must be called
    by ALL ranks at the same logging interval if distributed=True; only rank0
    writes sinks. The tracker can also be used for validation via prefix='eval'.
    """
    keys = ('loss', 'restoration_loss', 'nvs_loss', 'weighted_nvs_loss',
            'restoration_clean_loss', 'restoration_distractor_loss',
            'attention_alignment_loss', 'weighted_attention_alignment_loss')

    def __init__(self):
        self.totals = None
        from mvr.common_loss_metrics import CommonLossMetrics
        self.common_metrics = CommonLossMetrics()

    def update(self, losses, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        values = torch.stack([losses[key].detach().to(torch.float64)
                              for key in self.keys])
        if values.shape != (len(self.keys),):
            raise ValueError('Expected scalar mean losses')
        counts = values.new_tensor([batch_size])
        update = torch.cat([values * batch_size, counts])
        if self.totals is None:
            self.totals = torch.zeros_like(update)
        self.totals += update

    def compute(self, *, distributed=True, prefix='train'):
        if self.totals is None:
            raise ValueError('No NVS losses have been recorded')
        totals = self.totals.clone()
        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        means = (totals[:-1] / totals[-1]).cpu().tolist()
        result = {f'{prefix}/{key}': value for key, value in zip(self.keys, means)}
        result[f'{prefix}/samples'] = int(totals[-1].item())
        return result

    def write(self, step, *, logger=None, tracker=None, jsonl_path=None,
              distributed=True, prefix='train', reset=True):
        """logger=Python logger; tracker=W&B-like .log(payload, step=step)."""
        payload = self.compute(distributed=distributed, prefix=prefix)
        payload.update(self.common_metrics.compute(distributed=distributed, prefix=prefix, reset=reset))
        rank = (torch.distributed.get_rank() if distributed and
                torch.distributed.is_initialized() else 0)
        if rank == 0:
            if logger is not None:
                common = ' '.join(f"{name}_loss={payload[f'{prefix}/{name}_loss']:.4f}" if f'{prefix}/{name}_loss' in payload else f'{name}_loss=NA' for name in self.common_metrics.names)
                logger.info('step=%s %s %s', step, common, json.dumps(payload))
            if tracker is not None:
                tracker.log(payload, step=step)
            if jsonl_path is not None:
                path = Path(jsonl_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as stream:
                    stream.write(json.dumps(dict(step=int(step), **payload)) + '\n')
        if reset:
            self.totals = None
        return payload


def target_plucker_rays(target_c2w,intrinsics,image_size,patch=14):
    """Patch-center [direction, origin cross direction] in shared DA3 frame.

    target_c2w [B,M,4,4], intrinsics [B,M,3,3] for processed image pixels.
    V camera matrices are NOT accepted or embedded here.
    """
    if target_c2w.ndim!=4 or target_c2w.shape[-2:]!=(4,4) or intrinsics.shape!=(*target_c2w.shape[:2],3,3):
        raise ValueError('Expected target c2w [B,M,4,4], K [B,M,3,3]')
    h,w=image_size
    if h%patch or w%patch:raise ValueError('Image must fit patch grid')
    y,x=torch.meshgrid(torch.arange(h//patch,device=intrinsics.device,dtype=intrinsics.dtype),torch.arange(w//patch,device=intrinsics.device,dtype=intrinsics.dtype),indexing='ij')
    pixels=torch.stack([x*patch+(patch-1)/2,y*patch+(patch-1)/2,torch.ones_like(x)],-1).reshape(-1,3)
    cam=pixels@torch.linalg.inv(intrinsics).transpose(-1,-2)
    direction=F.normalize(cam@target_c2w[:,:,:3,:3].transpose(-1,-2),dim=-1)
    origin=target_c2w[:,:,:3,3][:,:,None,:].expand_as(direction)
    return torch.cat([direction,torch.cross(origin,direction,dim=-1)],-1)

@torch.no_grad()
def encode_nvs_training_features(encoder, source_images, clean_source_images, novel_clean_images, train_cfg):
    """Frozen teacher preparation through existing DA3 early-return API.

    Images are already ImageNet normalized. Encoder returns (out, feats, ref).
    M supervision is present only in the teacher forward, not in source pass.
    The source-selected reference is forced on the teacher forward.
    """
    from omegaconf import OmegaConf
    cfg=OmegaConf.create(OmegaConf.to_container(train_cfg,resolve=True))
    if len(cfg.extract_feat_layers)!=1:raise ValueError('NVS requires one raw feature layer')
    if source_images.shape!=clean_source_images.shape:raise ValueError('Paired V images must match')
    if source_images.shape[0]!=novel_clean_images.shape[0] or source_images.shape[2:]!=novel_clean_images.shape[2:]:raise ValueError('Novel images must use the same batch and pixel grid')
    cfg.break_and_return_feat=True
    out,source,ref=encoder(image=source_images,export_feat_layers=[],mvrm_cfg=cfg,mode='train')
    if out is not None:raise RuntimeError('Expected feature-only encoder')
    teacher_images=torch.cat([clean_source_images,novel_clean_images],dim=1)
    out,teacher,_=encoder(image=teacher_images,export_feat_layers=[],mvrm_cfg=cfg,mode='train',ref_b_idx=ref)
    if out is not None:raise RuntimeError('Expected feature-only teacher')
    key=('extract_feat',cfg.extract_feat_layers[0])
    return source[key].detach().float(),teacher[key].detach().float(),ref


def nvs_training_step(model, source_lq, clean_teacher, target_rays, time, image_size,
                      *, noise=None,noise_lvl=.3,novel_weight=1.,source_distractor_mask=None,group_mse_cfg=None,alignment_target=None,alignment_cfg=None,loss_tracker=None):
    """Single shared-model forward: N restoration and M NVS velocity losses."""
    batch=prepare_nvs_flow(source_lq,clean_teacher,time,noise=noise,noise_lvl=noise_lvl)
    b,v=source_lq.shape[:2];m=clean_teacher.shape[1]-v
    if target_rays.shape!=(b,m,clean_teacher.shape[2]-1,6):raise ValueError('Target rays must match only M patch grids')
    rays=torch.cat([target_rays.new_zeros(b,v,target_rays.shape[2],6),target_rays],dim=1)
    analysis=None
    if alignment_cfg and alignment_cfg.get('use',False):
        from omegaconf import OmegaConf
        if alignment_cfg.get('distractor_aware',{}).get('use',False):raise ValueError('Mask-aware alignment must be OFF')
        if alignment_target is None:raise ValueError('Missing clean geometry attention target')
        analysis=OmegaConf.create(dict(vis_attn_map=True,mvrm_attn_map=dict(extract_idx=[alignment_cfg['mvrm_layer_idx']])))
    pred=model(batch.model_input,time,image_size,analysis=analysis,nvs_rays=rays,nvs_view_mask=batch.novel_view_mask)
    losses = nvs_flow_losses(pred,batch,novel_weight=novel_weight,source_distractor_mask=source_distractor_mask,group_mse_cfg=group_mse_cfg)
    alignment=losses['loss'].new_zeros(())
    if analysis is not None:
        from mvr.nvs_attention import attention_loss
        alignment=attention_loss(model,alignment_target,alignment_cfg,v+m,clean_teacher.shape[2])
    losses['attention_alignment_loss']=alignment
    losses['weighted_attention_alignment_loss']=float((alignment_cfg or {}).get('lambda_coeff',1.))*alignment
    losses['loss']=losses['loss']+losses['weighted_attention_alignment_loss']
    if loss_tracker is not None:
        common_mask = source_distractor_mask
        if common_mask is None:
            common_mask = torch.zeros_like(pred[:, :v, :, 0], dtype=torch.bool)
        loss_tracker.common_metrics.update(pred[:, :v], batch.target_velocity[:, :v], common_mask)
        loss_tracker.update(losses, source_lq.shape[0])
    return losses
