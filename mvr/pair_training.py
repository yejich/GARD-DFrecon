"""Fixed held-out flow-loss evaluation and resumable epoch checkpoints."""
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from mvr.dataset.hypersim_pairs import HypersimPairs, pair_collate_fn
from mvr.grouped_mse import token_groups, grouped_mse
from mvr.mask_supervision import mask_targets, mask_bce
from mvr.dataset.pho_concat_ds import multiview_collate_fn


def load_weights(path, models):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    weights = state.get('ema', state.get('model', state))
    model = models['denoiser']
    expected = set(model.state_dict())
    missing = expected - set(weights)
    extra = set(weights) - expected
    optional_groups = [{key for key in expected if key.startswith(prefix)}
                       for prefix in ('mask_head.', 'condition_encoder.', 'condition_projector.')]
    allowed_missing = set().union(*(group for group in optional_groups if group <= missing))
    # Only entire optional modules may be absent; corrupt/partial checkpoints fail.
    if extra or missing != allowed_missing:
        raise ValueError(f'Checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}')
    model.load_state_dict(weights, strict=not bool(missing))
    # Include the exact same newly initialized head in EMA.
    models['ema_denoiser'].load_state_dict(model.state_dict(), strict=True)


@torch.no_grad()
def evaluate_pairs(cfg, models, transport, device, rank, world_size, max_groups=None, return_metrics=False, mask_output_dir=None):
    object_only = cfg.mvrm.loss.get('mask', {}).get('target', 'object_shadow') == 'object'
    dataset = HypersimPairs(cfg.data.train.pairs.manifest, split='eval', include_object_masks=object_only)
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.training.guidance.lq_drop = 0.
    eval_cfg.mvrm.analysis.vis_attn_map = False
    norm_mean = torch.tensor([.485,.456,.406], device=device)[None,None,:,None,None]
    norm_std = torch.tensor([.229,.224,.225], device=device)[None,None,:,None,None]
    stats = torch.zeros(2, device=device, dtype=torch.float64)
    mask_stats = torch.zeros(8, device=device, dtype=torch.float64)
    with torch.random.fork_rng(devices=[device.index]):
        count = len(dataset.manifest['eval_groups'])
        if max_groups is not None:
            count = min(count, max_groups)
        for i in range(rank, count, world_size):
            torch.manual_seed(dataset.manifest['seed'] + i)
            batch = pair_collate_fn([dataset[i]])
            lq = (batch['lq_views'].to(device)-norm_mean)/norm_std
            hq = (batch['hq_views'].to(device)-norm_mean)/norm_std
            lq_out, lf = models['encoder'](image=lq, export_feat_layers=[], mvrm_cfg=cfg.mvrm.train, mode='train')
            _, hf = models['encoder'](image=hq, export_feat_layers=[], mvrm_cfg=cfg.mvrm.train,
                                     mode='train', ref_b_idx=lq_out.ref_b_idx)
            key = ('extract_feat', cfg.mvrm.train.extract_feat_layers[0])
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.training.precision == 'bf16'):
                out = transport.training_losses_mvrm(models['ema_denoiser'], hf[key].clone().float(),
                                                     lf[key].clone().float(), hq.shape[-2:], eval_cfg)
            mask, flags = token_groups(batch['pixel_masks'].to(device),
                                       batch['distractor_views'].to(device), lq_out.ref_b_idx)
            loss, _ = grouped_mse(out['pred'], out['target_velocity'], mask, flags, cfg.mvrm.loss.group_mse)
            stats[0] += loss.double()
            stats[1] += 1
            if 'mask_logits' in out:
                target = mask_targets(batch['object_pixel_masks' if object_only else 'pixel_masks'].to(device),
                                      batch['distractor_views'].to(device), lq_out.ref_b_idx,
                                      resolution=cfg.stage_2.params.get('mask_head_resolution', 'patch'))
                mask_stats[0] += mask_bce(out['mask_logits'], target).double()
                # Use the real inference endpoint (t=1, LQ+noise, optional LQ addition).
                # Preserve RNG so auxiliary evaluation cannot alter the fixed flow metric.
                with torch.random.fork_rng(devices=[device.index]):
                    lq_latent = lf[key].clone().float()
                    start = lq_latent + float(cfg.mvrm.noise_lvl) * torch.randn_like(lq_latent)
                    if cfg.mvrm.get('training_lq_cond', False):
                        start = start + lq_latent
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.training.precision == 'bf16'):
                        condition_kwargs = ({'lq_latent': lq_latent}
                            if cfg.stage_2.params.get('c_feat_enabled', False) else {})
                        initial = models['ema_denoiser'](start, torch.ones(lq.shape[0], device=device),
                                                        hq.shape[-2:], return_aux=True, **condition_kwargs)
                logits = initial['mask_logits']
                if mask_output_dir is not None:
                    from mvr.mask_prediction_outputs import save_mask_grid
                    save_mask_grid(Path(mask_output_dir)/f'group_{i:03d}', batch['lq_views'],
                        batch['object_pixel_masks' if object_only else 'pixel_masks'],
                        batch['distractor_views'], lq_out.ref_b_idx, logits, dict(group_index=i, source='synthetic',weights='EMA'))
                mask_stats[1] += mask_bce(logits, target).double()
                pred, truth = logits >= 0, target >= .5
                mask_stats[2] += (pred & truth).sum()
                mask_stats[3] += (pred & ~truth).sum()
                mask_stats[4] += (~pred & truth).sum()
                clean = (~flags).reshape(*flags.shape, *([1] * (pred.ndim - 2))).expand_as(pred)
                mask_stats[5] += (pred & clean).sum()
                mask_stats[6] += clean.sum()
                mask_stats[7] += 1
    dist.all_reduce(stats)
    flow = (stats[0]/stats[1].clamp_min(1)).item()
    if not return_metrics:
        return flow
    dist.all_reduce(mask_stats)
    metrics = {'fixed_flow_loss': flow}
    if mask_stats[7] > 0:
        sampled_bce, initial_bce, tp, fp, fn, clean_fp, clean_count, count = mask_stats.tolist()
        metrics.update(mask_bce=sampled_bce/count, mask_start_bce=initial_bce/count,
                       mask_start_iou=tp/max(tp+fp+fn, 1),
                       mask_start_precision=tp/max(tp+fp, 1),
                       mask_start_recall=tp/max(tp+fn, 1),
                       mask_start_clean_fpr=clean_fp/max(clean_count, 1))
    return metrics


def save_training_checkpoint(path, next_epoch, step, optimizer_step, models, optimizer, scheduler, rank, manifest_fingerprint=None, extra_state=None):
    rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
               python=random.getstate(), numpy=np.random.get_state())
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, rng)
    if rank == 0:
        data = dict(model=models['denoiser'].state_dict(), ema=models['ema_denoiser'].state_dict(),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
                    next_epoch=next_epoch, step=step, optimizer_step=optimizer_step, rng=states,
                    manifest_fingerprint=manifest_fingerprint)
        if getattr(models['denoiser'], 'mask_head', None) is not None:
            data['mask_head_layer'] = models['denoiser'].mask_head_layer
            data['mask_head_source'] = getattr(models['denoiser'], 'mask_head_source', ('c_feat' if getattr(models['denoiser'], 'c_feat_enabled', False) else 'encoder'))
        if extra_state:
            if data.keys() & extra_state.keys():
                raise ValueError("Extra checkpoint state overwrites training state")
            data.update(extra_state)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(data, temporary)
        os.replace(temporary, path)
    dist.barrier()


def save_epoch_weights(checkpoint_dir, completed_epoch, models, rank, save_epochs, extra_state=None):
    """Export inference EMA weights at selected completed epochs, alongside latest.pt."""
    if completed_epoch not in save_epochs:
        return
    if rank == 0:
        data = {'ema': models['ema_denoiser'].state_dict()}
        if getattr(models['ema_denoiser'], 'mask_head', None) is not None:
            data['mask_head_layer'] = models['ema_denoiser'].mask_head_layer
            data['mask_head_source'] = getattr(models['ema_denoiser'], 'mask_head_source', ('c_feat' if getattr(models['ema_denoiser'], 'c_feat_enabled', False) else 'encoder'))
        if extra_state:
            if data.keys() & extra_state.keys():
                raise ValueError('Extra checkpoint state overwrites EMA weights')
            data.update(extra_state)
        path = Path(checkpoint_dir) / f'epoch_{completed_epoch:03d}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(data, temporary)
        os.replace(temporary, path)
    dist.barrier()


def resume_training(path, models, optimizer, scheduler, rank, manifest_fingerprint=None):
    data = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    saved_fingerprint = data.get('manifest_fingerprint')
    if saved_fingerprint and manifest_fingerprint and saved_fingerprint != manifest_fingerprint:
        raise ValueError('Resume dataset differs from the checkpoint. Resume with the original manifest/data, or start a new fine-tuning run.')
    if len(data['rng']) != dist.get_world_size():
        raise ValueError('Resume requires the same world size')
    model = models['denoiser']
    if getattr(model, 'mask_head', None) is not None:
        # Older mask checkpoints always used the final encoder block.
        source = getattr(model, 'mask_head_source', 'c_feat' if getattr(model, 'c_feat_enabled', False) else 'encoder')
        if data.get('mask_head_source', 'encoder') != source:
            raise ValueError('Resume mask head source differs from checkpoint')
        saved_layer = data.get('mask_head_layer', model.num_encoder_blocks)
        if model.mask_head_layer != saved_layer:
            raise ValueError(f'Resume mask head layer differs: checkpoint={saved_layer}, model={model.mask_head_layer}')
    models['denoiser'].load_state_dict(data['model'], strict=True)
    models['ema_denoiser'].load_state_dict(data['ema'], strict=True)
    optimizer.load_state_dict(data['optimizer'])
    if scheduler and data['scheduler']:
        scheduler.load_state_dict(data['scheduler'])
    state = data['rng'][rank]
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    return data['next_epoch'], data['step'], data['optimizer_step']
