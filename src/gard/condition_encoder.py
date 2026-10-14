"""Timestep-independent, variable-view LQ condition encoder."""
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from RAE.src.stage2.models.model_utils import NormAttentionMVRM


class ConditionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, rope, qk_norm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NormAttentionMVRM(dim, heads, qkv_bias=True, qk_norm=qk_norm, rope=rope)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x, pos):
        x = x + self.attn(self.norm1(x), pos=pos)
        return x + self.mlp(self.norm2(x))


class LQConditionEncoder(nn.Module):
    """One local and one global block; no timestep or noisy-state input."""
    def __init__(self, in_dim, dim, heads, mlp_ratio, rope, qk_norm):
        super().__init__()
        self.input_proj = nn.Identity() if in_dim == dim else nn.Linear(in_dim, dim)
        self.local = ConditionBlock(dim, heads, mlp_ratio, rope, qk_norm)
        self.global_block = ConditionBlock(dim, heads, mlp_ratio, rope, qk_norm)

    def forward(self, lq, local_pos, global_pos):
        b, v, n, _ = lq.shape
        x = self.input_proj(lq)
        for block, shape, pos in ((self.local, (b*v, n, -1), local_pos),
                                  (self.global_block, (b, v*n, -1), global_pos)):
            x = x.reshape(*shape)
            pos = pos.reshape(*shape[:2], 2) if pos is not None else None
            if self.training:
                x = checkpoint(block, x, pos, use_reentrant=False)
            else:
                x = block(x, pos)
            x = x.reshape(b, v, n, -1)
        return x
