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
from mvr.dataset.pho_concat_ds import multiview_collate_fn


def load_weights(path, models):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    weights = state.get('ema', state.get('model', state))
    models['denoiser'].load_state_dict(weights, strict=True)
    models['ema_denoiser'].load_state_dict(weights, strict=True)


@torch.no_grad()
def evaluate_pairs(cfg, models, transport, device, rank, world_size, max_groups=None):
    dataset = HypersimPairs(cfg.data.train.pairs.manifest, split='eval')
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.training.guidance.lq_drop = 0.
    eval_cfg.mvrm.analysis.vis_attn_map = False
    norm_mean = torch.tensor([.485,.456,.406], device=device)[None,None,:,None,None]
    norm_std = torch.tensor([.229,.224,.225], device=device)[None,None,:,None,None]
    stats = torch.zeros(2, device=device, dtype=torch.float64)
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
    dist.all_reduce(stats)
    return (stats[0]/stats[1].clamp_min(1)).item()


def save_training_checkpoint(path, next_epoch, step, optimizer_step, models, optimizer, scheduler, rank):
    rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
               python=random.getstate(), numpy=np.random.get_state())
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, rng)
    if rank == 0:
        data = dict(model=models['denoiser'].state_dict(), ema=models['ema_denoiser'].state_dict(),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
                    next_epoch=next_epoch, step=step, optimizer_step=optimizer_step, rng=states)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(data, temporary)
        os.replace(temporary, path)
    dist.barrier()


def resume_training(path, models, optimizer, scheduler, rank):
    data = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if len(data['rng']) != dist.get_world_size():
        raise ValueError('Resume requires the same world size')
    models['denoiser'].load_state_dict(data['model'], strict=True)
    models['ema_denoiser'].load_state_dict(data['ema'], strict=True)
    optimizer.load_state_dict(data['optimizer'])
    if scheduler and data['scheduler']:
        scheduler.load_state_dict(data['scheduler'])
    state = data['rng'][rank]
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    return data['next_epoch'], data['step'], data['optimizer_step']
