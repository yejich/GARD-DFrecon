"""Reuse GARD's exact denoiser architecture + pretrained (EMA) weights and
flow-matching machinery, repurposed for cross-view distractor completion
instead of motion-blur denoising (2026-09-02, per handoff.md section 3.3 and
user decision: reuse GARD's architecture as-is + GARD's pretrained weights
as init, only swap what x0/x1 mean).

Original GARD training: x0 = noise, x1 = HQ (sharp) feature, model learns to
map noise -> HQ conditioned on the LQ (blurry) feature via `lq_latent_cond`.
Here: x0 = z_distracted (real distractor feature, deterministic -- NOT
noise), x1 = z_clean (same-pose clean-view feature). This is a standard
paired/deterministic-coupling rectified-flow reformulation (see CondOT /
paired rectified flow): xt = (1-t)x0 + t*x1, target velocity ut = x1 - x0,
loss = MSE(model(xt, t) - ut). At t->1 the ODE solution is exactly x1, so a
well-trained model directly maps a distracted feature to its clean
counterpart -- this is genuinely deterministic regression, dressed in
GARD's existing forward()/checkpoint interface so we can reuse both without
modification.

Import note: GARD's own package (`gard.GARD`) does `from RAE.src.stage2....`
which needs the GARD repo ROOT on sys.path (for the top-level `RAE` package)
in addition to `GARD/src` (for the `gard` package) -- see models/GARD.py's
own import statement, verified by hand 2026-09-02.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

GARD_ROOT = Path(__file__).resolve().parents[2]


def _ensure_gard_on_path():
    for p in (str(GARD_ROOT / "src"), str(GARD_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_gard_denoiser(
    pretrained_ckpt: Optional[Path] = GARD_ROOT / "ckpts" / "gard_denoiser.pt",
    in_channels: int = 1536,
    device: str = "cuda",
) -> nn.Module:
    """Constructs GARD with the EXACT config used for the shipped checkpoint
    (encoder depth 8 @ hidden 1536, decoder depth 6 @ hidden 3072, 16 heads,
    RoPE, SwiGLU, RMSNorm -- see run_configs/train/train_GARD.yaml, verified
    against the checkpoint's own tensor shapes). Loads the EMA weights if
    pretrained_ckpt is given (strict=True -- fail loudly on any mismatch,
    since silent partial-load would corrupt the whole point of reusing
    GARD's pretrained init).
    """
    _ensure_gard_on_path()
    from gard.GARD import GARD

    model = GARD(
        input_size=None,
        patch_size=1,
        in_channels=in_channels,
        hidden_size=[1536, 3072],
        depth=[8, 6],
        num_heads=[16, 16],
        mlp_ratio=4.0,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed=False,
        use_global_residual=False,
    )
    if pretrained_ckpt is not None:
        state = torch.load(str(pretrained_ckpt), map_location="cpu", weights_only=False)
        sd = state["ema"] if "ema" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, f"unexpected checkpoint mismatch: missing={missing} unexpected={unexpected}"
    return model.to(device)


def masked_paired_flow_targets(x0: torch.Tensor, x1: torch.Tensor, distractor_mask: torch.Tensor) -> torch.Tensor:
    """x0: (V, N+1, C) distracted feature (CLS included). x1: (V, N+1, C)
    clean feature, same shape. distractor_mask: (V, N+1) bool, True at
    distractor TOKEN positions (CLS and non-distractor tokens False).

    Returns x1_effective: (V, N+1, C) -- equals x1 at distractor tokens, and
    equals x0 (identity target) everywhere else. This bakes BOTH terms of
    handoff.md section 7's loss into one masked-target regression:
      - at distractor tokens: target=x1 (real completion supervision)
      - at clean/CLS tokens: target=x0 (identity-preservation term, exactly
        zero velocity once learned -- avoids needing a separate lambda-
        weighted term or two loss computations).
    """
    mask = distractor_mask.unsqueeze(-1)  # (V, N+1, 1)
    return torch.where(mask, x1, x0)


@torch.no_grad()
def ode_complete(denoiser: nn.Module, x0: torch.Tensor, model_img_size: tuple, num_steps: int = 1, device: str = "cuda") -> torch.Tensor:
    """Euler-integrate the trained denoiser's velocity field from t=0 (x0 =
    real distracted feature) to t=1.

    num_steps=1 (default) is the CORRECT choice here, not a shortcut: the
    paired/deterministic path used during training (see
    sample_paired_flow_batch) is a straight line, xt=(1-t)x0+t*x1, whose
    TRUE target velocity ut=x1-x0 is constant along the whole line -- a
    perfectly-trained model needs exactly one Euler step from (x0, t=0).
    Multi-step integration was tried and found to DIVERGE (verified
    2026-09-02 on the step250 checkpoint: v magnitude grew ~10x per step,
    overflowed to NaN by step 6/10) -- an undertrained model's velocity
    prediction is only reliable near the real training distribution (real
    x0 at t=0); each extra Euler step pushes x further off that
    distribution into where the model was never trained and extrapolates
    unpredictably, and the error compounds. Only raise num_steps if you have
    a specific reason to believe the model needs it (e.g. after much more
    training) -- verify it doesn't diverge before trusting the output.
    x0: (V, N+1, C) (no batch dim). Returns (V, N+1, C).
    """
    was_training = denoiser.training
    denoiser.eval()
    x = x0.unsqueeze(0).clone()  # (1, V, N+1, C)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((1,), step * dt, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v = denoiser(x, t, model_img_size)
        x = x + dt * v.float()
    if was_training:
        denoiser.train()
    return x[0]


def sample_paired_flow_batch(x0: torch.Tensor, x1_effective: torch.Tensor, t: torch.Tensor) -> tuple:
    """x0, x1_effective: (V, N+1, C). t: (1,) or scalar in [0,1] (one timestep
    shared across the whole multi-view sample, matching GARD's own
    `t_embedder` call convention of one t per training example).
    Returns (xt, ut) both (V, N+1, C): xt = (1-t)x0 + t*x1_effective (linear/
    CondOT path, matching GARD's own `path_type: Linear` config), ut =
    x1_effective - x0 (target velocity).
    """
    t_ = t.view(-1, *([1] * (x0.dim() - 1)))
    xt = (1 - t_) * x0 + t_ * x1_effective
    ut = x1_effective - x0
    return xt, ut
