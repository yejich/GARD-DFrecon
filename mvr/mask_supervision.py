"""Patch occupancy supervision for the optional GARD encoder mask branch."""
import torch
import torch.nn.functional as F


def mask_targets(pixel_masks, distractor_views, reference_indices, patch_size=14, resolution='patch'):
    """Soft supplied-mask occupancy in DA3 reference-first order; no CLS target."""
    b, v, h, w = pixel_masks.shape
    if h % patch_size or w % patch_size:
        raise ValueError('Mask dimensions must be divisible by backbone patch size')
    if distractor_views.shape != (b, v):
        raise ValueError('Distractor flags must match mask batch and views')
    if resolution == 'pixel':
        target = pixel_masks.float() * distractor_views[..., None, None].float()
    elif resolution == 'patch':
        target = F.avg_pool2d(pixel_masks.float().reshape(b*v, 1, h, w), patch_size, patch_size)
        target = target.reshape(b, v, -1) * distractor_views[..., None].float()
    else:
        raise ValueError('Unknown mask resolution')
    if reference_indices is None:
        reference_indices = torch.zeros(b, dtype=torch.long, device=target.device)
    refs = torch.as_tensor(reference_indices, device=target.device).reshape(-1)
    if refs.numel() != b or ((refs < 0) | (refs >= v)).any():
        raise ValueError('One valid reference index is required per sample')
    order = torch.tensor([[r] + [i for i in range(v) if i != r] for r in refs.tolist()],
                         device=target.device)
    indices = order.reshape(b, v, *([1] * (target.ndim - 2))).expand_as(target)
    return target.gather(1, indices)


def mask_bce(logits, targets, conditioned=None):
    """Mean soft-target BCE per group; CFG-dropped groups contribute zero.

    Divide by the full batch size so DDP/accumulation has the same weighting
    regardless of where dropped groups land. Keep the all-dropped graph live.
    """
    if logits.shape != targets.shape:
        raise ValueError(f'Mask shape mismatch: {logits.shape} vs {targets.shape}')
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none')
    per_group = loss.flatten(1).mean(1)
    if conditioned is not None:
        if conditioned.shape != per_group.shape:
            raise ValueError('Conditioning flags must have one entry per sample')
        per_group = per_group * conditioned.to(per_group.dtype)
    return per_group.mean()


def validate_mask_config(cfg):
    enabled = bool(cfg.stage_2.params.get('mask_head_enabled', False))
    weight = float(cfg.mvrm.loss.get('mask', {}).get('lambda_coeff', 0.))
    if weight < 0 or enabled != (weight > 0):
        raise ValueError('Enable mask_head_enabled with a positive mask.lambda_coeff; disable both for baseline')
    if cfg.mvrm.loss.get('mask', {}).get('target', 'object_shadow') not in ('object','object_shadow'):
        raise ValueError('Synthetic mask target must be object or object_shadow')
    real = cfg.mvrm.loss.get('mask', {}).get('real', {})
    if real.get('use', False) and (not enabled or not float(real.get('lambda_coeff',0)) > 0):
        raise ValueError('Real mask supervision requires an enabled head and positive weight')
    return enabled, weight
