"""Wraps GARD's own pretrained RGB decoder (RAE_DA3.mae_decoder, a frozen
ViT-XL MAE-style decoder trained to reconstruct pixels from DA3-Giant's own
intermediate features at layers [19, 27, 33, 39]) so our completion module's
output -- injected at layer 17, then let flow through the rest of the frozen
backbone -- can be turned into an actual viewable RGB image and compared
directly against the real clean photo, not just depth.

2026-09-04 investigation of Depth-Anything-3/src/depth_anything_3/api.py's
`inference()` (lines ~380-403) and GLD/src/stage1/rae_da3.py found:
- `RAE_DA3.decode()` (rae_da3.py:476-572) looks like the intended one-call
  entry point, but it's missing a required step for the giant backbone: it
  never applies `self.adapter` (the 12288->6144 `FeatureProjectionAdapter`)
  before calling `self.mae_decoder`, so calling it as-is on da3-giant feats
  raises a shape mismatch in `mae_decoder`'s `decoder_embed` Linear (which
  expects hidden_size=6144, i.e. 1536*4, not the raw 3072*4=12288 concat).
  api.py's own inline copy of this same logic (its three `if rgb_decoder is
  not None:` blocks) DOES apply the adapter first -- that's the version this
  module replicates (`decode_rgb_from_feats` below), not `RAE_DA3.decode()`.
- `normalization_stat_path` in the repo's own `run_configs/val/
  val_GARD_da3_bench.yaml` (GLD/model_stats/da3/normalization_stats_level0.pt)
  doesn't exist anywhere in this GARD checkout -- passing it would crash
  `RAE_DA3._init_normalization`'s `torch.load`. Not needed for RGB decode
  (only referenced by an unrelated latent-normalization code path), so
  `build_rae_da3` below passes `normalization_stat_path=None`.
- Only `rae.mae_decoder` and `rae.adapter` need `.to(device)` -- `RAE_DA3`
  itself is not moved wholesale (mirrors evaluator.py:288-291). This
  conveniently keeps the incidental extra DA3-Base encoder that `RAE_DA3.
  __init__` loads (needed by its constructor, unused by us) on CPU, saving
  GPU memory.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch

GARD_ROOT = Path(__file__).resolve().parents[2]
if str(GARD_ROOT) not in sys.path:
    sys.path.insert(0, str(GARD_ROOT))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def build_rae_da3(device: str = "cuda"):
    """Builds RAE_DA3 exactly per run_configs/val/val_GARD_da3_bench.yaml's
    `MVRM_EVAL.rgb_decoder.stage_1.params` (absolute paths substituted for
    the yaml's GARD_ROOT-relative ones, normalization_stat_path dropped --
    see module docstring), then moves only mae_decoder+adapter to `device`.
    """
    from GLD.src.stage1.rae_da3 import RAE_DA3

    rae = RAE_DA3(
        encoder_pretrained_path="depth-anything/DA3-Base",
        encoder_input_size=252,
        encoder_type="DA3EncoderDirect",
        decoder_config_path=str(GARD_ROOT / "GLD" / "configs" / "decoder" / "ViTXL"),
        dpt_decoder_path=None,
        da3_weights_path=None,
        dpt_model_type="da3-base",
        mae_weight=str(GARD_ROOT / "ckpts" / "mae_adapter_giant.pt"),
        noise_tau=0.0,
        reshape_to_2d=True,
        normalization_stat_path=None,
    )
    rae.mae_decoder = rae.mae_decoder.to(device)
    rae.mae_decoder.eval()
    if getattr(rae, "adapter", None) is not None:
        rae.adapter = rae.adapter.to(device)
        rae.adapter.eval()
    return rae


@torch.no_grad()
def decode_rgb_from_feats(rae, feats: List[Tuple[torch.Tensor, torch.Tensor]], H: int, W: int) -> torch.Tensor:
    """feats: output.feat from DepthAnything3Net.forward(..., export_rgb_feat_layers=True)
    -- a list of (patches, cls_token) tuples, one per out_layers entry
    (4 for da3-giant: [19,27,33,39]), patches shaped (B,V,N,C).

    Replicates api.py's inline rgb_decoder call (NOT RAE_DA3.decode(), see
    module docstring) -- concat all levels' patches -> project 12288->6144
    via rae.adapter -> mae_decoder -> unpatchify -> ImageNet denorm -> clamp
    to [0,1]. Returns (B,V,3,H,W) float32 on CPU.
    """
    mae_feats = [patches for (patches, _cls) in feats]
    z_cat = torch.cat(mae_feats, dim=-1)  # (B,V,N,C*4)
    b, v, n, c_tot = z_cat.shape
    z_cat = z_cat.reshape(b * v, n, c_tot)

    with torch.autocast(device_type=z_cat.device.type, enabled=True, dtype=torch.bfloat16):
        if getattr(rae, "adapter", None) is not None:
            z_cat = rae.adapter(z_cat)
        logits = rae.mae_decoder(z_cat, input_size=(H, W), drop_cls_token=False).logits
        x_rec = rae.mae_decoder.unpatchify(logits, (H, W))  # (B*V, 3, H, W)
        x_rec = x_rec.reshape(b, v, 3, H, W)

    x_rec = x_rec.float().cpu()
    rgb = (x_rec * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)  # (B,V,3,H,W)
    return rgb
