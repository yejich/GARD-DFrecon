"""Shared distractor-aware attention targets and loss for DA3 and Omega."""
import torch


def distractor_aware_target(neg_l2, geo_target, visibility, distractor_mask, temperature, cfg):
    """Filter only the HQ target. Distances are in the HQ backbone's 3D units."""
    b, v, n = distractor_mask.shape
    distractor = distractor_mask.reshape(b, v * n).bool()
    max_distance = float(cfg.get("max_correspondence_distance", 0.03))
    if not 0 < max_distance < float("inf"):
        raise ValueError("max_correspondence_distance must be finite and positive")
    allowed = torch.isfinite(neg_l2) & (neg_l2 >= -max_distance)
    allowed &= ~distractor[:, None, :]  # no distractor keys, for any query
    if visibility is not None:
        allowed &= visibility
    if cfg.get("cross_view_only_for_distractor", True):
        view_ids = torch.arange(v, device=neg_l2.device).repeat_interleave(n)
        same_view = view_ids[:, None] == view_ids[None, :]
        allowed &= ~(distractor[:, :, None] & same_view[None])
    if temperature == -1:
        target = geo_target * allowed
        mass = target.sum(-1, keepdim=True)
        valid = mass.squeeze(-1) > 0
        target = target / mass.clamp_min(1e-8)
    else:
        if temperature <= 0:
            raise ValueError("Attention temperature must be positive or -1")
        valid = allowed.any(-1)
        # Equivalent to masking/renormalizing the original softmax, but avoids
        # underflow from self-dominated rows. Empty rows stay zero, never NaN.
        logits = (neg_l2 / temperature).masked_fill(~allowed, -torch.inf)
        logits = torch.where(valid[..., None], logits, torch.zeros_like(logits))
        target = logits.softmax(-1) * allowed
    return target, valid


def distractor_aware_attn_loss(pred, target, valid, distractor_mask, cfg):
    # Do not mask or renormalize predictions: attention spent on excluded keys
    # must still reduce the probability available to target correspondences.
    per_row = -(target * (pred.float() + 1e-8).log()).sum(-1)
    if not cfg.get("balance_query_groups", True):
        return (per_row * valid).sum() / valid.sum().clamp_min(1)
    distractor = distractor_mask.flatten(1).bool()
    loss = per_row.sum() * 0
    for name, selected in (("distractor", distractor), ("clean", ~distractor)):
        weight = float(cfg.get(f"lambda_{name}", 1.0))
        if not 0 <= weight < float("inf"):
            raise ValueError(f"lambda_{name} must be finite and nonnegative")
        selected = selected & valid
        # Match group2 MSE: equal group weights per sample, empty groups = zero.
        group_mean = (per_row * selected).sum(-1) / selected.sum(-1).clamp_min(1)
        loss = loss + weight * group_mean.mean()
    return loss


