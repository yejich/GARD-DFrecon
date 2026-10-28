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
        # Maps GARD encoder features to the DA3 feature space used by the
        # clean destination tokens in the InfoNCE ablation.
        self.destination_projector = nn.Linear(self.encoder_hidden_size, self.in_channels, bias=False)
        if self.encoder_hidden_size == self.in_channels:
            nn.init.eye_(self.destination_projector.weight)

    def forward(self, x, t, model_img_size, analysis=None, return_aux=False,
                lq_latent=None, c_feat=None, condition_mask=None,
                nvs_rays=None, nvs_view_mask=None, return_features=False):
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
                               lq_latent,c_feat,condition_mask,
                               encoder_token_condition=condition,
                               return_features=return_features)


@dataclass
class NVSFlowBatch:
    model_input: torch.Tensor
    target_velocity: torch.Tensor
    source_condition: torch.Tensor
    novel_view_mask: torch.Tensor
    time: torch.Tensor


def prepare_nvs_flow(source_lq, clean_teacher, time, *, noise=None, noise_lvl=.3,
                     double_condition=True, flowmap_from='noise_lq'):
    """t=0 clean teacher; t=1 restoration start / pure-noise novel grids.

    Shapes: source_lq [B,V,T,C], clean_teacher [B,V+M,T,C]. CLS included.
    clean_teacher/source_lq are frozen DA3 outputs in the SAME feature units.
    """
    if source_lq.ndim!=4 or clean_teacher.ndim!=4 or source_lq.shape[0]!=clean_teacher.shape[0] or source_lq.shape[2:]!=clean_teacher.shape[2:]:
        raise ValueError('Source/teacher feature dimensions must match')
    b,v=source_lq.shape[:2];total=clean_teacher.shape[1]
    if total<v or time.shape!=(b,) or not torch.isfinite(time).all() or not ((time>=0)&(time<=1)).all():
        raise ValueError('Invalid source/teacher layout or t outside [0,1]')
    if noise_lvl<0:raise ValueError('noise_lvl must be nonnegative')
    if flowmap_from not in ('noise', 'noise_lq'):
        raise ValueError("flowmap_from must be 'noise' or 'noise_lq'")
    if noise is None:noise=torch.randn_like(clean_teacher)
    if noise.shape!=clean_teacher.shape:raise ValueError('Noise must match teacher')
    condition=torch.zeros_like(clean_teacher);condition[:,:v]=source_lq.detach()
    start=noise.clone()
    if flowmap_from == 'noise_lq':
        start[:,:v]=noise_lvl*noise[:,:v]+condition[:,:v]
    else:
        start[:,:v]=noise[:,:v]
    teacher=clean_teacher.detach();t=time[:,None,None,None]
    state=(1-t)*teacher+t*start
    if double_condition:state=state+condition
    novel=torch.zeros((b,total),device=teacher.device,dtype=torch.bool);novel[:,v:]=True
    return NVSFlowBatch(state,start-teacher,condition,novel,time)


def nvs_flow_losses(prediction,batch,*,novel_weight=1.,source_distractor_mask=None,group_mse_cfg=None,
                    full_view_loss=False):
    """Compute source restoration and novel-view flow losses.

    ``full_view_loss`` is the GLD-style Phase-C mode: source and novel views
    are concatenated and receive the same per-view/token MSE.  Camera tokens
    are included in the same average; there is no separate camera loss.
    The legacy branch is retained for the existing restoration+NVS runs.
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
    if full_view_loss:
        errors=(prediction.float()-batch.target_velocity.float()).square().mean(-1)
        all_view_loss=errors.mean()
        source_errors=errors[:,:n]
        novel_errors=errors[:,n:] if n < total else errors.new_zeros((errors.shape[0],0,errors.shape[2]))
        restoration=source_errors.mean()
        novel=novel_errors.mean() if novel_errors.numel() else errors.new_zeros(())
        zero=errors.new_zeros(())
        return dict(loss=all_view_loss,restoration_loss=restoration,
                    nvs_loss=novel,weighted_nvs_loss=novel,
                    restoration_clean_loss=restoration,restoration_distractor_loss=zero,
                    attention_alignment_loss=zero,weighted_attention_alignment_loss=zero,
                    full_view_flow_loss=all_view_loss)

    restoration,parts=grouped_mse(prediction[:,:n],batch.target_velocity[:,:n],source_distractor_mask,
                                source_distractor_mask.any(-1),cfg)
    novel=(prediction[:,n:].float()-batch.target_velocity[:,n:].float()).square().mean() if n < total else prediction.new_zeros(())
    weighted_novel=novel_weight*novel
    return dict(loss=restoration+weighted_novel,restoration_loss=restoration,
                nvs_loss=novel,weighted_nvs_loss=weighted_novel,
                restoration_clean_loss=parts['clean'],restoration_distractor_loss=parts['distractor'],
                attention_alignment_loss=novel.new_zeros(()),weighted_attention_alignment_loss=novel.new_zeros(()))


def distractor_suppression_loss(features, distractor_mask, margin=0.1):
    """Suppress cross-view similarity from GT distractor query tokens.

    ``features`` are encoder patch tokens [B,V,P,C]. A source distractor
    query is compared against every patch in every other view. Only the most
    similar cross-view feature is penalized when cosine similarity exceeds
    ``margin``. The mask selects queries, not candidate keys.
    """
    if features.ndim != 4 or distractor_mask.ndim != 3:
        raise ValueError('Expected features [B,V,P,C] and mask [B,N,1+P]')
    b, v, p, _ = features.shape
    n = distractor_mask.shape[1]
    if distractor_mask.shape[0] != b or n > v or distractor_mask.shape[2] != p + 1:
        raise ValueError('Feature/mask shapes are incompatible')
    if not 0 <= float(margin) < 1:
        raise ValueError('Feature suppression margin must be in [0,1)')
    queries = distractor_mask[:, :, 1:].bool()
    if not queries.any():
        return features.new_zeros(())
    normalized = F.normalize(features.float(), dim=-1)
    losses=[]
    for view in range(n):
        other_views = [i for i in range(v) if i != view]
        if not other_views:
            continue
        candidates = torch.cat([normalized[:, i] for i in other_views], dim=1)
        similarity = torch.bmm(normalized[:, view], candidates.transpose(1, 2))
        best = similarity.amax(-1)
        selected = queries[:, view]
        losses.append(F.relu(best - float(margin))[selected])
    values=torch.cat([x for x in losses if x.numel()])
    return values.mean() if values.numel() else features.new_zeros(())


def distractor_destination_infonce_loss(features, clean_teacher, distractor_mask, temperature=0.07):
    """Route distractor queries to their same-view, same-patch clean target.

    The positive is the clean HQ token at the query's original location. The
    negatives are distractor tokens from other source views, which directly
    targets repeated semantic distractors without treating static clean scene
    correspondences as false negatives.
    """
    if features.ndim != 4 or clean_teacher.ndim != 4 or distractor_mask.ndim != 3:
        raise ValueError('Expected features [B,V,P,C], clean teacher [B,V+M,1+P,C], and mask [B,N,1+P]')
    b, v, p, c = features.shape
    n = distractor_mask.shape[1]
    if n > v or clean_teacher.shape[1] < n or clean_teacher.shape[2] != p + 1:
        raise ValueError('Feature, teacher, and mask shapes are incompatible')
    if clean_teacher.shape[-1] != c:
        raise ValueError('Projected query and clean teacher channels must match')
    temperature = float(temperature)
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError('InfoNCE temperature must be positive')
    query_mask = distractor_mask[:, :, 1:].bool()
    if not query_mask.any():
        return features.new_zeros(())
    q = F.normalize(features.float(), dim=-1)
    keys = F.normalize(clean_teacher[:, :n, 1:].detach().float(), dim=-1)
    values = []
    for view in range(n):
        selected = query_mask[:, view]
        if not selected.any():
            continue
        negative_views = [i for i in range(n) if i != view]
        if not negative_views:
            continue
        negative_mask = torch.cat([query_mask[:, i] for i in negative_views], dim=1)
        negative_keys = torch.cat([keys[:, i] for i in negative_views], dim=1)
        positive = keys[:, view]
        negative_logits = torch.einsum('bpc,bkc->bpk', q[:, view], negative_keys)
        negative_logits = negative_logits.masked_fill(~negative_mask[:, None, :], float('-inf'))
        logits = torch.cat([
            (q[:, view] * positive).sum(-1, keepdim=True),
            negative_logits,
        ], dim=-1) / temperature
        valid = selected & (negative_mask.any(-1)[:, None])
        if valid.any():
            values.append((-logits.log_softmax(-1)[..., 0])[valid])
    values = [x for x in values if x.numel()]
    return torch.cat(values).mean() if values else features.new_zeros(())


def geometry_correspondence_supcon_loss(prediction, batch, source_lq,
                                        clean_teacher, distractor_mask,
                                        geometry_aux, *, temperature=0.07,
                                        max_negative_tokens=None,
                                        time_weight='one_minus_t'):
    """Geometry-positive / mask-defined-global-negative SupCon loss.

    The anchor is the predicted clean latent x0-hat of a source distractor
    token.  Its positives are the same-location clean teacher token and all
    valid geometry-corresponding clean teacher tokens in other source views.
    Its negatives are distractor input tokens from all source views, capped
    per anchor.  Geometry visibility only gates cross-view positives; an
    invalid reprojection is never treated as a negative.

    ``geometry_aux`` is the ``(target, distances, visibility)`` tuple returned
    by ``geometry_target``.  All teacher/input keys are detached.  Positive
    terms are averaged per positive, so anchors with different numbers of
    visible views have the same loss scale.
    """
    if prediction.ndim != 4 or batch.model_input.ndim != 4:
        raise ValueError('Expected prediction and model_input [B,V,T,C]')
    if source_lq.ndim != 4 or clean_teacher.ndim != 4 or distractor_mask.ndim != 3:
        raise ValueError('Expected source_lq/teacher [B,V,T,C] and mask [B,N,T]')
    if geometry_aux is None or len(geometry_aux) != 3:
        raise ValueError('Geometry correspondence SupCon requires geometry_aux')
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('SupCon temperature must be positive')
    if max_negative_tokens is not None:
        max_negative_tokens = int(max_negative_tokens)
    if max_negative_tokens is not None and max_negative_tokens < 1:
        raise ValueError('max_negative_tokens must be positive')
    b, total, tokens, channels = prediction.shape
    n = source_lq.shape[1]
    if total < n or clean_teacher.shape[1] < n:
        raise ValueError('Prediction/teacher view count is smaller than source view count')
    if source_lq.shape[2:] != (tokens, channels) or clean_teacher.shape[2:] != (tokens, channels):
        raise ValueError('SupCon feature shapes must match prediction [B,V,T,C]')
    if distractor_mask.shape != (b, n, tokens) or distractor_mask[:, :, 0].any():
        raise ValueError('Expected distractor mask [B,N,T] with clean CLS tokens')

    # x_t contains the additive source condition in the current flow recipe;
    # subtract it before recovering the clean endpoint x0 from velocity.
    time = batch.time.to(device=prediction.device, dtype=prediction.dtype)
    x0_hat = batch.model_input - batch.source_condition - time[:, None, None, None] * prediction
    anchor = F.normalize(x0_hat[:, :n, 1:].float(), dim=-1)
    clean_keys = F.normalize(clean_teacher[:, :n, 1:].detach().float(), dim=-1)
    distractor_keys = F.normalize(source_lq[:, :n, 1:].detach().float(), dim=-1)
    query_mask = distractor_mask[:, :, 1:].bool()
    if not query_mask.any():
        return prediction.new_zeros(())

    geo_target, _, visibility = geometry_aux
    patch_count = tokens - 1
    if geo_target.shape[1] < n * patch_count or geo_target.shape[2] < n * patch_count:
        raise ValueError('Geometry target is smaller than source patch-token layout')
    geo_target = geo_target[:, :n * patch_count, :n * patch_count]
    visibility = visibility[:, :n * patch_count, :n * patch_count].bool()

    losses=[]
    weights=[]
    for bi in range(b):
        negative_positions = query_mask[bi].nonzero(as_tuple=False)
        if negative_positions.numel() == 0:
            continue
        # One global negative pool per sample: all distractor input tokens in
        # all source views. Keep the anchor's own input token whenever capped.
        flat_negative = (negative_positions[:, 0] * patch_count + negative_positions[:, 1])
        for view, patch in negative_positions.tolist():
            row = view * patch_count + patch
            positive_keys=[clean_keys[bi, view, patch]]  # self-positive

            # Geometry correspondence is selected independently per target
            # view. Visibility gates the candidate; mask=0 makes it positive.
            for target_view in range(n):
                if target_view == view:
                    continue
                cols = slice(target_view * patch_count, (target_view + 1) * patch_count)
                row_weights = geo_target[bi, row, cols]
                row_valid = visibility[bi, row, cols]
                if not row_valid.any():
                    continue
                masked_weights = row_weights.masked_fill(~row_valid, float('-inf'))
                target_patch = int(masked_weights.argmax().item())
                if not bool(distractor_mask[bi, target_view, target_patch + 1]):
                    positive_keys.append(clean_keys[bi, target_view, target_patch])

            # Cap negatives without instance matching. Keep self-negative and
            # sample the rest uniformly; within the sampled set, InfoNCE's
            # exponential weighting automatically emphasizes hard negatives.
            keep = flat_negative == row
            remaining = (~keep).nonzero(as_tuple=False).flatten()
            slots = None if max_negative_tokens is None else max_negative_tokens - 1
            if slots is not None and slots > 0 and remaining.numel() > slots:
                remaining = remaining[torch.randperm(remaining.numel(), device=remaining.device)[:slots]]
            selected_indices = torch.cat([keep.nonzero(as_tuple=False).flatten(), remaining])
            if max_negative_tokens is not None:
                selected_indices = selected_indices[:max_negative_tokens]
            negative_keys = distractor_keys[bi].reshape(-1, channels)[flat_negative[selected_indices]]

            pos = torch.stack(positive_keys, dim=0)
            logits = torch.cat([
                anchor[bi, view, patch][None] @ pos.transpose(0, 1),
                anchor[bi, view, patch][None] @ negative_keys.transpose(0, 1),
            ], dim=1).squeeze(0) / temperature
            log_prob = logits - torch.logsumexp(logits, dim=0)
            # Average over positives, not sum them, so visibility does not
            # change the scale of an anchor's loss.
            losses.append(-log_prob[:pos.shape[0]].mean())
            if time_weight == 'one_minus_t':
                weights.append((1. - time[bi]).float().clamp_min(0.))
            elif time_weight in ('none', None):
                weights.append(losses[-1].new_ones(()))
            else:
                raise ValueError(f'Unsupported SupCon time_weight={time_weight!r}')
    if not losses:
        return prediction.new_zeros(())
    values = torch.stack(losses)
    weights = torch.stack(weights).to(values.device, values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


class NVSLossTracker:
    """Detached, sample-weighted logging across microbatches and DDP ranks.

    update() never modifies tensors used for backward. write() must be called
    by ALL ranks at the same logging interval if distributed=True; only rank0
    writes sinks. The tracker can also be used for validation via prefix='eval'.
    """
    keys = ('loss', 'restoration_loss', 'nvs_loss', 'weighted_nvs_loss',
            'restoration_clean_loss', 'restoration_distractor_loss',
            'attention_alignment_loss', 'weighted_attention_alignment_loss',
            'distractor_suppression_loss', 'weighted_distractor_suppression_loss',
            'distractor_infonce_loss', 'weighted_distractor_infonce_loss',
            'geometry_contrastive_loss', 'weighted_geometry_contrastive_loss')

    def __init__(self):
        self.totals = None
        from mvr.common_loss_metrics import CommonLossMetrics
        self.common_metrics = CommonLossMetrics()

    def update(self, losses, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        reference = losses['loss']
        values = torch.stack([(losses[key] if key in losses else reference.new_zeros(())).detach().to(torch.float64)
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
                      *, noise=None, noise_lvl=.3, flowmap_from='noise_lq',
                      nvs_branch=True, novel_weight=1., source_distractor_mask=None,
                      group_mse_cfg=None, alignment_target=None, alignment_aux=None, alignment_cfg=None,
                      distractor_suppression_cfg=None,
                      geometry_contrastive_cfg=None,
                      full_view_loss=False,
                      loss_tracker=None):
    """Single shared-model forward: N restoration and M NVS velocity losses."""
    b,v=source_lq.shape[:2]
    if not nvs_branch:
        clean_teacher=clean_teacher[:,:v]
        if alignment_target is not None:
            # Geometry targets contain patch tokens only, whereas DA3 feature
            # tensors include one CLS token per view.
            patch_tokens=clean_teacher.shape[2]-1
            alignment_target=alignment_target[:,:v*patch_tokens,:v*patch_tokens]
        if alignment_aux is not None:
            patch_tokens=clean_teacher.shape[2]-1
            alignment_aux=(alignment_aux[0][:,:v*patch_tokens,:v*patch_tokens],
                           alignment_aux[1][:,:v*patch_tokens,:v*patch_tokens],
                           alignment_aux[2][:,:v*patch_tokens,:v*patch_tokens])
    batch=prepare_nvs_flow(source_lq,clean_teacher,time,noise=noise,noise_lvl=noise_lvl,
                           # flowmap_from selects the interpolation start;
                           # LQ remains an external model condition in both modes.
                           double_condition=True,
                           flowmap_from=flowmap_from)
    m=clean_teacher.shape[1]-v
    if nvs_branch and target_rays.shape!=(b,m,clean_teacher.shape[2]-1,6):raise ValueError('Target rays must match only M patch grids')
    rays=torch.cat([target_rays.new_zeros(b,v,target_rays.shape[2],6),target_rays],dim=1) if nvs_branch else None
    analysis=None
    hard_self_view = bool((alignment_cfg or {}).get('hard_self_view_for_distractor', False))
    if alignment_cfg and alignment_cfg.get('use',False):
        from omegaconf import OmegaConf
        if alignment_target is None:raise ValueError('Missing clean geometry attention target')
        analysis=OmegaConf.create(dict(vis_attn_map=True,mvrm_attn_map=dict(extract_idx=[alignment_cfg['mvrm_layer_idx']])), flags={'allow_objects': True})
        if hard_self_view:
            from mvr.distractor_attention import build_hard_self_view_mask
            full_distractor_mask = torch.cat([
                source_distractor_mask,
                torch.zeros((b, m, source_distractor_mask.shape[2]), device=source_distractor_mask.device, dtype=torch.bool)
            ], dim=1) if m else source_distractor_mask
            analysis.mvrm_hard_attn_layer = int(alignment_cfg['mvrm_layer_idx'])
            analysis.mvrm_hard_attn_mask = build_hard_self_view_mask(full_distractor_mask)
    model_kwargs = dict(analysis=analysis)
    if nvs_branch:
        model_kwargs.update(nvs_rays=rays, nvs_view_mask=batch.novel_view_mask)
    distractor_cfg=distractor_suppression_cfg or {}
    geometry_cfg=geometry_contrastive_cfg or {}
    loss_mode=str(distractor_cfg.get('mode','none'))
    if loss_mode.lower() == 'infonce': loss_mode='infoNCE'
    if loss_mode not in ('none','push_only','infoNCE'):
        raise ValueError('distractor loss mode must be none, push_only, or infoNCE')
    use_geometry_contrastive=bool(geometry_cfg.get('use',False))
    use_distractor_loss=loss_mode != 'none'
    pred_output=model(batch.model_input,time,image_size,
                      return_features=use_distractor_loss, **model_kwargs)
    if use_distractor_loss:
        if not isinstance(pred_output, dict) or 'features' not in pred_output:
            raise RuntimeError('Distractor feature loss requires encoder features from GARD')
        features=pred_output['features']
        pred=pred_output['pred']
    else:
        features=None
        pred=pred_output
    losses = nvs_flow_losses(pred,batch,novel_weight=novel_weight,source_distractor_mask=source_distractor_mask,group_mse_cfg=group_mse_cfg,full_view_loss=full_view_loss)
    suppression=losses['loss'].new_zeros(())
    infonce=losses['loss'].new_zeros(())
    geometry_contrastive=losses['loss'].new_zeros(())
    if loss_mode == 'push_only':
        suppression=distractor_suppression_loss(
            features, source_distractor_mask,
            margin=float(distractor_cfg.get('margin',0.1)))
    elif loss_mode == 'infoNCE':
        projected = model.destination_projector(features) if hasattr(model, 'destination_projector') else features
        infonce=distractor_destination_infonce_loss(
            projected, clean_teacher, source_distractor_mask,
            temperature=float(distractor_cfg.get('temperature',0.07)))
    losses['distractor_suppression_loss']=suppression
    losses['weighted_distractor_suppression_loss']=float(distractor_cfg.get('lambda_coeff',0.01))*suppression
    losses['distractor_infonce_loss']=infonce
    losses['weighted_distractor_infonce_loss']=float(distractor_cfg.get('lambda_coeff',0.01))*infonce
    losses['loss']=losses['loss']+losses['weighted_distractor_suppression_loss']+losses['weighted_distractor_infonce_loss']
    if use_geometry_contrastive:
        geometry_contrastive=geometry_correspondence_supcon_loss(
            pred, batch, source_lq, clean_teacher, source_distractor_mask,
            alignment_aux,
            temperature=float(geometry_cfg.get('temperature',0.07)),
            max_negative_tokens=geometry_cfg.get('max_negative_tokens',64),
            time_weight=geometry_cfg.get('time_weight','one_minus_t'))
    losses['geometry_contrastive_loss']=geometry_contrastive
    losses['weighted_geometry_contrastive_loss']=float(geometry_cfg.get('lambda_coeff',0.05))*geometry_contrastive
    losses['loss']=losses['loss']+losses['weighted_geometry_contrastive_loss']
    alignment=losses['loss'].new_zeros(())
    if analysis is not None:
        from mvr.nvs_attention import attention_loss
        alignment=attention_loss(model,alignment_target,alignment_cfg,v+m,clean_teacher.shape[2],
                                 distractor_mask=torch.cat([source_distractor_mask,
                                                             torch.zeros(source_distractor_mask.shape[0], m, source_distractor_mask.shape[2], device=source_distractor_mask.device, dtype=torch.bool)], dim=1)
                                 if (alignment_cfg.get('distractor_aware',{}).get('use',False) or hard_self_view) else None,
                                 alignment_aux=alignment_aux)
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
