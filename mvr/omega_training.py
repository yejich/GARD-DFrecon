"""VGGT-Omega counterparts of the DA3-specific pieces used by GARD Hypersim-pair training.

- OmegaEncoder            : frozen VGGT-Omega with GARD hooks (extract / inject after inter block `layer`)
- LatentNormalizer        : per-channel stats (special tokens per position, patches shared)
- token_groups_omega      : distractor token mask for (S special + N patch) layout, no view reordering
- attn_align_loss_omega   : point-cloud attention alignment with S special tokens per view
- load_train_data_omega   : HypersimPairs resized to a patch-16 grid
- evaluate_pairs_omega    : fixed held-out flow loss (same protocol as mvr.pair_training.evaluate_pairs)
"""
import sys
import hashlib
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, unproject_depth
from mvr.dataset.hypersim_pairs import HypersimPairs, PairBatchSampler, pair_collate_fn
from mvr.grouped_mse import grouped_mse


# ----------------------------------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------------------------------
class OmegaEncoder(nn.Module):
    """Images are expected in [0, 1] (VGGT-Omega normalises internally; do NOT apply ImageNet norm)."""

    def __init__(self, repo: str, ckpt: str, layer: int = 3):
        super().__init__()
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from mvr.omega_backbone.vggt_omega_gard import VGGTOmegaGARD

        self.model = VGGTOmegaGARD()
        state = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(state, strict=True)
        self.layer = int(layer)
        self.patch_size = self.model.aggregator.patch_size
        self.num_special = self.model.patch_token_start
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.extract(images, layer=self.layer)  # (B, V, S+N, D)

    @torch.no_grad()
    def forward_full(self, images: torch.Tensor, restored: torch.Tensor | None = None, extract: bool = False):
        """Full forward; optionally inject `restored` (B, V, S+N, D) after block `layer`."""
        return self.model(
            images,
            extract_layers=(self.layer,) if extract else (),
            restored_latents={self.layer: restored} if restored is not None else None,
        )


def load_omega_encoder(cfg, device) -> OmegaEncoder:
    oc = cfg.stage_1.vggt_omega
    return OmegaEncoder(oc.repo, oc.ckpt, layer=cfg.mvrm.train.extract_feat_layers[0]).to(device)


# ----------------------------------------------------------------------------------------------------
# Latent normalisation
# ----------------------------------------------------------------------------------------------------
class LatentNormalizer(nn.Module):
    """z = (x - mean) / std. mean/std: (S+1, D) -> rows 0..S-1 special tokens, row S shared by patches."""

    def __init__(self, stats_path: str | None, num_special: int, dim: int, eps: float = 1e-6,
                 *, stats=None, metadata=None):
        super().__init__()
        self.num_special = num_special
        mean = torch.zeros(num_special + 1, dim)  # (S+1, D)
        std = torch.ones(num_special + 1, dim)  # (S+1, D)
        self.metadata = metadata or {}
        self.enabled = stats.get("enabled", True) if stats is not None else stats_path is not None
        if self.enabled:
            if stats is None:
                stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            for key in ("mean", "std"):
                if stats[key].shape != mean.shape or not torch.isfinite(stats[key]).all():
                    raise ValueError(f"Invalid latent {key}: expected finite tensor {tuple(mean.shape)}")
            if (stats["std"] < 0).any():
                raise ValueError("Latent std must be nonnegative")
            mean, std = stats["mean"].float(), stats["std"].float().clamp_min(eps)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def checkpoint_state(self):
        return dict(enabled=self.enabled, mean=self.mean.detach().cpu().clone(),
                    std=self.std.detach().cpu().clone(), **self.metadata)

    def _expand(self, t: torch.Tensor, num_tokens: int) -> torch.Tensor:
        S = self.num_special
        return torch.cat([t[:S], t[S:].expand(num_tokens - S, -1)], dim=0)  # (T, D)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        T = x.shape[-2]
        return (x - self._expand(self.mean, T)) / self._expand(self.std, T)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return z
        T = z.shape[-2]
        return z * self._expand(self.std, T) + self._expand(self.mean, T)


def omega_latent_metadata(cfg):
    """Content identity remains valid if the backbone checkpoint is relocated."""
    digest = hashlib.sha256()
    with open(cfg.stage_1.vggt_omega.ckpt, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(layer=int(cfg.mvrm.train.extract_feat_layers[0]),
                backbone_sha256=digest.hexdigest())


def checkpoint_normalizer_state(checkpoint_path):
    if not checkpoint_path:
        return None
    # Same trust contract as resume_training: user-supplied local training checkpoints.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    return state.get("omega_latent_norm")


def build_normalizer(cfg, encoder: OmegaEncoder, device, checkpoint_path=None) -> LatentNormalizer:
    norm_cfg = cfg.mvrm.get("latent_norm", None)
    enabled = bool(norm_cfg is not None and norm_cfg.get("use", False))
    stats = checkpoint_normalizer_state(checkpoint_path)
    metadata = omega_latent_metadata(cfg)
    metadata.update(num_special=encoder.num_special, patch_size=encoder.patch_size)
    if stats is not None:
        if stats["enabled"] != enabled:
            raise ValueError("latent_norm.use differs from checkpoint; use the original training setting")
        for key, expected in metadata.items():
            if stats.get(key) != expected:
                raise ValueError(f"Checkpoint latent normalization {key} does not match this backbone/layout")
    elif checkpoint_path:
        warnings.warn("Legacy checkpoint has no latent statistics; supply the original training stats file.")
    stats_path = None
    if enabled and stats is None:
        stats_path = norm_cfg.stats_path
        if not Path(stats_path).is_file():
            raise FileNotFoundError(
                f"Latent stats {stats_path} missing; provide original stats for existing checkpoints, "
                "or compute stats before a new training run")
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        for key, expected in metadata.items():
            if key in stats and stats[key] != expected:
                raise ValueError(f"Latent stats {key} does not match this backbone/layout")
        if "backbone_sha256" not in stats:
            warnings.warn("Legacy stats have no backbone fingerprint; verify their origin before training.")
    dim = encoder.model.aggregator.camera_token.shape[-1]
    return LatentNormalizer(stats_path, encoder.num_special, dim, stats=stats, metadata=metadata).to(device)


# ----------------------------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------------------------
def token_groups_omega(pixel_masks, distractor_views, patch_size=16, num_special=17):
    """pixel_masks (b, v, H, W), distractor_views (b, v) -> token mask (b, v, S+N); special tokens = clean."""
    b, v, h, w = pixel_masks.shape
    mask = F.avg_pool2d(pixel_masks.float().reshape(b * v, 1, h, w), patch_size, patch_size)  # (b*v, 1, h/p, w/p)
    mask = (mask.flatten(1) >= 0.5).reshape(b, v, -1)  # (b, v, N)
    mask &= distractor_views[..., None]
    special = torch.zeros(b, v, num_special, dtype=torch.bool, device=mask.device)  # (b, v, S)
    return torch.cat([special, mask], dim=-1), distractor_views  # (b, v, S+N), (b, v)


def cross_entropy_attn(pred, target, row_mask=None, eps=1e-8):
    per_row = -(target * (pred + eps).log()).sum(dim=-1)  # (b, v*n)
    if row_mask is not None:
        return per_row[row_mask].mean()
    return per_row.mean()


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


def attn_align_loss_omega(pred_map, depth, extrinsics, intrinsics, pc_cfg, patch_size=16, num_special=17,
                          *, distractor_cfg=None, token_mask=None):
    """
    pred_map  : (b, v*(S+n), v*(S+n)) head-averaged denoiser attention
    depth     : (b, v, H, W, 1) HQ depth ; extrinsics (b, v, 3, 4) w2c ; intrinsics (b, v, 3, 3)
    Same target construction as the DA3 recipe (train_hypersim_pairs.py), without reference reordering.
    """
    device = depth.device
    aware = distractor_cfg is not None and distractor_cfg.get("use", False)
    c2w = affine_inverse(as_homogeneous(extrinsics.float()))  # (b, v, 4, 4)
    pts = unproject_depth(depth.float(), intrinsics.float(), c2w)  # (b, v, H, W, 3)
    b, v, H, W, _ = pts.shape
    Ph, Pw = H // patch_size, W // patch_size
    n = Ph * Pw
    pts_patch = pts.reshape(b, v, Ph, patch_size, Pw, patch_size, 3).mean(dim=(3, 5))  # (b, v, Ph, Pw, 3)
    pts_flat = pts_patch.reshape(b, v * n, 3)  # (b, v*n, 3)

    neg_l2 = -torch.cdist(pts_flat, pts_flat)  # (b, v*n, v*n)
    T = pc_cfg.get("vis_pc_temperature", 1.0)
    if T == -1:
        blocked = neg_l2.reshape(b, v * n, v, n)  # (b, v*n, v, n)
        geo_target = torch.zeros_like(blocked).scatter_(-1, blocked.argmax(dim=-1, keepdim=True), 1.0)
        geo_target = geo_target.reshape(b, v * n, v * n)  # (b, v*n, v*n)
    else:
        geo_target = (neg_l2 / T).softmax(dim=-1)  # (b, v*n, v*n)

    vis_mask_type = pc_cfg.get("visibility_mask", "none")
    valid_rows = None
    vis_mask = None
    if vis_mask_type != "none":
        if vis_mask_type != "cycle_consistency":
            raise NotImplementedError(f"visibility_mask={vis_mask_type} not ported for VGGT-Omega")
        vis_mask = torch.zeros(b, v * n, v * n, dtype=torch.bool, device=device)  # (b, v*n, v*n)
        ref_idx = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)  # (b, n)
        cycle_thresh = pc_cfg.get("vis_pc_cycle_threshold", 0)
        for va in range(v):
            for vb in range(v):
                sa, sb = slice(va * n, (va + 1) * n), slice(vb * n, (vb + 1) * n)
                if va == vb:
                    vis_mask[:, sa, sb] = True
                    continue
                # Raw distances avoid underflow when a cross-view softmax block is all zero.
                correspondence = neg_l2 if aware else geo_target
                fwd = correspondence[:, sa, sb].argmax(dim=-1)  # (b, n)
                bwd = correspondence[:, sb, sa].argmax(dim=-1)  # (b, n)
                roundtrip = torch.gather(bwd, 1, fwd)  # (b, n)
                if cycle_thresh == 0:
                    visible = roundtrip == ref_idx
                else:
                    dist_ = torch.max((roundtrip // Pw - ref_idx // Pw).abs(), (roundtrip % Pw - ref_idx % Pw).abs())
                    visible = dist_ <= cycle_thresh
                vis_mask[:, sa, sb] = visible.unsqueeze(-1).expand(-1, -1, n)
        geo_target = geo_target * vis_mask.float()
        row_sum = geo_target.sum(dim=-1, keepdim=True)  # (b, v*n, 1)
        valid_rows = row_sum.squeeze(-1) > 0  # (b, v*n)
        geo_target = geo_target / row_sum.clamp(min=1e-8)

    per_view = num_special + n
    spatial = torch.ones(pred_map.shape[-1], dtype=torch.bool, device=device)  # (v*(S+n),)
    for vi in range(v):
        spatial[vi * per_view: vi * per_view + num_special] = False
    pred_spatial = pred_map[:, spatial][:, :, spatial]  # (b, v*n, v*n)
    if aware:
        if token_mask is None or token_mask.shape != (b, v, per_view):
            raise ValueError("Distractor-aware attention requires a (B,V,special+patches) token_mask")
        distractor_mask = token_mask[:, :, num_special:]
        target, valid = distractor_aware_target(
            neg_l2, geo_target, vis_mask, distractor_mask, T, distractor_cfg)
        return distractor_aware_attn_loss(pred_spatial, target, valid, distractor_mask, distractor_cfg)
    return cross_entropy_attn(pred_spatial.float(), geo_target, row_mask=valid_rows)


# ----------------------------------------------------------------------------------------------------
# Data / eval
# ----------------------------------------------------------------------------------------------------
def pair_dataset(cfg, split, patch_size):
    return HypersimPairs(cfg.data.train.pairs.manifest, split=split,
                         process_res=cfg.data.train.pairs.get("process_res", 512), patch_size=patch_size)


def load_train_data_omega(cfg, batch_size, rank, world_size, patch_size=16):
    ds = pair_dataset(cfg, "train", patch_size)
    sampler = PairBatchSampler(ds, batch_size, rank, world_size,
                               accumulation=cfg.training.grad_accum_steps,
                               seed=cfg.training.global_seed,
                               min_views=cfg.data.train.get("min_num_input_view", 1),
                               max_views=cfg.data.train.max_num_input_view)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.training.num_workers,
                        pin_memory=True, collate_fn=pair_collate_fn)
    return loader, sampler


@torch.no_grad()
def evaluate_pairs_omega(cfg, models, normalizer, transport, device, rank, world_size, max_groups=None):
    encoder = models["encoder"]
    dataset = pair_dataset(cfg, "eval", encoder.patch_size)
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.training.guidance.lq_drop = 0.0
    eval_cfg.mvrm.analysis.vis_attn_map = False
    stats = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.random.fork_rng(devices=[device.index]):
        count = len(dataset.manifest["eval_groups"])
        if max_groups is not None:
            count = min(count, max_groups)
        for i in range(rank, count, world_size):
            torch.manual_seed(dataset.manifest["seed"] + i)
            batch = pair_collate_fn([dataset[i]])
            lq = batch["lq_views"].to(device)  # (1, v, 3, H, W) in [0, 1]
            hq = batch["hq_views"].to(device)  # (1, v, 3, H, W) in [0, 1]
            lq_latent = normalizer.normalize(encoder.extract(lq))  # (1, v, S+N, D)
            hq_latent = normalizer.normalize(encoder.extract(hq))  # (1, v, S+N, D)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.training.precision == "bf16"):
                out = transport.training_losses_mvrm(models["ema_denoiser"], hq_latent, lq_latent,
                                                     hq.shape[-2:], eval_cfg)
            mask, flags = token_groups_omega(batch["pixel_masks"].to(device), batch["distractor_views"].to(device),
                                             encoder.patch_size, encoder.num_special)
            loss, _ = grouped_mse(out["pred"], out["target_velocity"], mask, flags, cfg.mvrm.loss.group_mse)
            stats[0] += loss.double()
            stats[1] += 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats)
    return (stats[0] / stats[1].clamp_min(1)).item()
