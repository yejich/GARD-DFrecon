"""GARD denoiser for VGGT-Omega aggregator tokens.

`GARD.forward` hard-codes DA3 assumptions (ViT patch 14, one cls/cam token). This subclass keeps
every parameter name of `GARD` and only generalises the token layout:

    x: (b, v, S + N, in_channels), S = num_special_tokens (VGGT-Omega: 1 camera + 16 registers = 17)
       N = (H // vit_patch_size) * (W // vit_patch_size)             (VGGT-Omega: patch 16)

Because only the in/out projections depend on `in_channels`, weights of a DA3 GARD checkpoint with
the same hidden sizes can be partially transferred with `load_shape_matched_state_dict`.
"""

import torch
import torch.nn as nn
from einops import rearrange

from gard.GARD import GARD


class GARDOmega(GARD):
    def __init__(self, *args, vit_patch_size: int = 16, num_special_tokens: int = 17, **kwargs):
        super().__init__(*args, **kwargs)
        self.vit_patch_size = vit_patch_size
        self.num_cls_tkn = num_special_tokens

    def _prepare_rope(self, B, S, num_pH, num_pW, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(B * S, num_pH, num_pW, device=device)  # (b*v, N, 2)
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)  # (b, v, N, 2)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)  # (b, v, N, 2)
            pos = pos + 1  # special tokens keep position 0
            pos_special = torch.zeros(B, S, self.num_cls_tkn, 2, device=device, dtype=pos.dtype)  # (b, v, S, 2)
            pos = torch.cat([pos_special, pos], dim=2)  # (b, v, S+N, 2)
            pos_nodiff = torch.cat([pos_special, pos_nodiff + 1], dim=2)  # (b, v, S+N, 2)
        return pos, pos_nodiff

    def forward(self, x, t, model_img_size, analysis=None):
        if self.use_global_residual:
            global_residual = x  # (b, v, S+N, d)

        model_H, model_W = model_img_size
        num_pH = model_H // self.vit_patch_size
        num_pW = model_W // self.vit_patch_size
        S = self.num_cls_tkn

        special_tkns = x[:, :, :S]  # (b, v, S, d)
        patch_tkns = x[:, :, S:]  # (b, v, N, d)
        b, v, n, d = patch_tkns.shape
        if n != num_pH * num_pW:
            raise ValueError(f"{n} patch tokens do not match image {model_img_size} / patch {self.vit_patch_size}")
        x = rearrange(patch_tkns, "b v (h w) d -> (b v) d h w", h=num_pH, w=num_pW)  # (b*v, d, h, w)

        # time condition
        t = self.t_embedder(t)  # (b, enc_d)
        c = nn.functional.silu(t)  # (b, enc_d)

        # encoder input: special + patch embeddings
        s_special = self.s_cls_embedder(special_tkns.reshape(b * v, S, d))  # (b*v, S, enc_d)
        s = self.s_embedder(x)  # (b*v, N, enc_d)
        s = torch.cat([s_special, s], dim=1)  # (b*v, S+N, enc_d)
        s = rearrange(s, "(b v) n d -> b v n d", b=b, v=v)  # (b, v, S+N, enc_d)

        pos, pos_nodiff = self._prepare_rope(b, v, num_pH, num_pW, s.device)  # (b, v, S+N, 2)
        g_pos = pos_nodiff
        l_pos = pos

        for i in range(self.num_encoder_blocks):
            attn_type = "local" if i % 2 == 0 else "global"
            s = self.process_attention_encoder(
                s, c, self.blocks[i], attn_type, pos=l_pos if attn_type == "local" else g_pos,
                layer_idx=i, analysis=analysis,
            )  # (b, v, S+N, enc_d)

        s = rearrange(s, "b v n d -> (b v) n d")  # (b*v, S+N, enc_d)
        t = t.repeat_interleave(s.shape[0] // t.shape[0], dim=0)  # (b*v, enc_d)
        t = t.unsqueeze(1).expand(-1, s.shape[1], -1)  # (b*v, S+N, enc_d)
        s = nn.functional.silu(t + s)  # (b*v, S+N, enc_d)
        s = self.s_projector(s)  # (b*v, S+N, dec_d)

        # decoder input: special + patch embeddings
        x_special = self.x_cls_embedder(special_tkns.reshape(b * v, S, d))  # (b*v, S, dec_d)
        x = self.x_embedder(x)  # (b*v, N, dec_d)
        x = torch.cat([x_special, x], dim=1)  # (b*v, S+N, dec_d)

        x = rearrange(x, "(b v) n d -> b v n d", b=b, v=v)  # (b, v, S+N, dec_d)
        s = rearrange(s, "(b v) n d -> b v n d", b=b, v=v)  # (b, v, S+N, dec_d)

        for i in range(self.num_encoder_blocks, self.num_blocks):
            attn_type = "local" if i % 2 == 0 else "global"
            x = self.process_attention_decoder(
                x, s, self.blocks[i], attn_type, pos=l_pos if attn_type == "local" else g_pos,
                layer_idx=i, analysis=analysis,
            )  # (b, v, S+N, dec_d)

        x = rearrange(x, "b v n d -> (b v) n d")  # (b*v, S+N, dec_d)
        s = rearrange(s, "b v n d -> (b v) n d")  # (b*v, S+N, dec_d)
        x = self.final_layer(x, s)  # (b*v, S+N, in_channels)
        x = rearrange(x, "(b v) n d -> b v n d", b=b, v=v)  # (b, v, S+N, in_channels)

        if self.use_global_residual:
            x = x + global_residual
        return x


def load_shape_matched_state_dict(model: nn.Module, state_dict: dict) -> tuple[list[str], list[str]]:
    """Load every tensor whose name and shape match; returns (loaded, skipped) key lists."""
    own = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(k for k in own if k not in matched)
    model.load_state_dict(matched, strict=False)
    return sorted(matched), skipped
