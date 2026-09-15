"""Multi-view frozen-DA3 wrapper that can INJECT a completed feature back at
layer 17 (mode='val', mvrm_cfg.restore_feat_layers=[17]) and let the frozen
backbone finish through the remaining layers (18-39) to produce real
geometry output (depth/pose/point cloud) -- this is GARD's own restore
mechanism (see api.py's `_run_model_forward` orchestration and
vision_transformer_eccv.py's per-block `kwargs['mode']=='val'` branch,
2026-09-02 research), reused here so we can qualitatively check what the
completion module's output actually does to DA3's depth prediction, not
just the abstract feature-space loss.

Same model instance also does plain (mode=None) forward passes for
baseline/reference comparisons -- avoids loading DA3-GIANT twice.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class MultiViewDA3RestoreModel:
    def __init__(self, pretrained: str = "depth-anything/DA3-GIANT-1.1", device: str = "cuda"):
        from depth_anything_3.api import DepthAnything3

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained(pretrained)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.model.device = self.device

    def _to_tensor(self, image01: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(image01).permute(2, 0, 1).float()
        return (t - IMAGENET_MEAN) / IMAGENET_STD

    def _img_tensor(self, views01: List[np.ndarray]) -> torch.Tensor:
        imgs = torch.stack([self._to_tensor(v) for v in views01], dim=0).to(self.device)  # (V,3,H,W)
        return imgs.unsqueeze(0)  # (1,V,3,H,W)

    def _check_ref(self, output):
        ref_b_idx = getattr(output, "ref_b_idx", None)
        if ref_b_idx is not None and not torch.all(ref_b_idx == 0):
            raise RuntimeError(f"ref_b_idx={ref_b_idx.tolist()} != 0 -- view-axis reorder safety violated.")

    def _geometry_from_output(self, output) -> Dict[str, np.ndarray]:
        from depth_anything_3.utils.geometry import affine_inverse, unproject_depth

        depth = output.depth
        if depth.dim() == 5:
            depth = depth[..., 0]
        depth_conf = output.depth_conf
        extrinsics = output.extrinsics
        intrinsics = output.intrinsics
        bottom = torch.zeros(*extrinsics.shape[:-2], 1, 4, device=extrinsics.device, dtype=extrinsics.dtype)
        bottom[..., 0, 3] = 1.0
        ext4x4 = torch.cat([extrinsics, bottom], dim=-2)
        c2w = affine_inverse(ext4x4)
        world_points = unproject_depth(depth[..., None], intrinsics, c2w=c2w)
        return {
            "depth": depth[0].float().cpu().numpy(),
            "depth_conf": depth_conf[0].float().cpu().numpy(),
            "extrinsics": extrinsics[0].float().cpu().numpy(),
            "intrinsics": intrinsics[0].float().cpu().numpy(),
            "world_points": world_points[0].float().cpu().numpy(),
        }

    @torch.no_grad()
    def geometry_plain(self, views01: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Normal forward pass, no MVRM involvement -- baseline/reference."""
        img_t = self._img_tensor(views01)
        output, _ = self.model.forward(img_t, extrinsics=None, intrinsics=None, export_feat_layers=[], ref_view_strategy="first")
        self._check_ref(output)
        return self._geometry_from_output(output)

    @torch.no_grad()
    def geometry_with_restored_feature(self, views01: List[np.ndarray], layer: int, restored_feat: torch.Tensor) -> Dict[str, np.ndarray]:
        """views01: the DISTRACTED view stack (same images used to compute
        x0 for this layer). restored_feat: (V, N+1, C) -- e.g. the
        completion module's output (fcmodels.gard_completion.ode_complete),
        no batch dim. Injects it at `layer` (mode='val', restore_feat_layers
        =[layer]) and lets the frozen backbone finish -> real depth output.
        """
        img_t = self._img_tensor(views01)
        mvrm_val_cfg = SimpleNamespace(restore_feat_layers=[layer], concat_feat=False)
        mvrm_result = {("restored_latent", layer): restored_feat.unsqueeze(0).to(self.device)}  # (1,V,N+1,C)
        output, _ = self.model.forward(
            img_t, extrinsics=None, intrinsics=None, export_feat_layers=[],
            ref_view_strategy="first", mvrm_cfg=mvrm_val_cfg, mvrm_result=mvrm_result, mode="val",
        )
        self._check_ref(output)
        return self._geometry_from_output(output)
