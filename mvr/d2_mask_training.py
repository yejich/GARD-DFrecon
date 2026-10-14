"""Opt-in fixed-t=1 D2 mask training without changing the original GARD flow.

Both branches execute inside ONE DDP forward. The sampled-t branch produces
the original velocity/attention; the independent mask branch is always LQ
conditioned and ends after decoder block 2. State-dict names remain compatible
with GARD, including its optional mask_head.* parameters.
"""
import math

import torch
import torch.nn.functional as F

from gard.GARD import GARD


class _D2Finished(Exception):
    def __init__(self, features):
        self.features = features


class FixedT1D2GARD(GARD):
    def __init__(self, mask_head_resolution='patch', mask_head_pixel_patch_size=14, attention_head_chunk_size=16, attention_inner_checkpoint=False, **kwargs):
        kwargs.setdefault('mask_head_enabled', True)
        kwargs.setdefault('mask_head_source', 'decoder')
        kwargs.setdefault('mask_head_layer', 2)
        if (not kwargs['mask_head_enabled'] or kwargs['mask_head_source'] != 'decoder'
                or kwargs['mask_head_layer'] != 2 or kwargs.get('c_feat_enabled', False)):
            raise ValueError('Fixed-t1 training requires a post-D2 head without C_feat')
        super().__init__(**kwargs)
        if mask_head_resolution not in ('patch', 'pixel'):
            raise ValueError('mask_head_resolution must be patch or pixel')
        self.mask_head_resolution = mask_head_resolution
        self.mask_head_pixel_patch_size = int(mask_head_pixel_patch_size)
        if self.mask_head_pixel_patch_size < 1:
            raise ValueError('Pixel patch size must be positive')
        if mask_head_resolution == 'pixel':
            # One distinct logit per subpixel, not interpolation of one patch logit.
            with torch.random.fork_rng(devices=[]):
                self.mask_head[-1] = torch.nn.Linear(
                    self.mask_head[-1].in_features, self.mask_head_pixel_patch_size ** 2)
        # Preserve the alignment head mean while avoiding full-head attention
        # allocations in the opt-in fixed-t1 experiment only.
        if not isinstance(attention_head_chunk_size, int) or attention_head_chunk_size < 1:
            raise ValueError("Attention head chunk size must be a positive integer")
        for block in self.blocks:
            block.attn.checkpoint_attention_heads = bool(attention_inner_checkpoint)
            block.attn.retain_attention_head_mean = True
            block.attn.attention_head_chunk_size = attention_head_chunk_size
        self._mask_only_active = False

    def _spatial_mask_logits(self, logits, image_size):
        if self.mask_head_resolution == 'patch':
            return logits
        p = self.mask_head_pixel_patch_size
        h, w = map(int, image_size)
        if h % p or w % p or logits.shape[-2:] != (h // p * (w // p), p * p):
            raise ValueError('Pixel mask tokens must match image size and patch grid')
        b, v = logits.shape[:2]
        return logits.reshape(b, v, h // p, w // p, p, p).permute(
            0, 1, 2, 4, 3, 5).reshape(b, v, h, w)

    def process_attention_decoder(self, *args, **kwargs):
        # Stop OUTSIDE the block/checkpoint call. Raising from a block hook
        # would interrupt checkpoint internals and break training gradients.
        output = super().process_attention_decoder(*args, **kwargs)
        if self._mask_only_active and kwargs['layer_idx'] == self.num_encoder_blocks + 1:
            raise _D2Finished(output)
        return output

    def mask_logits_at_t1(self, model_input, model_img_size):
        """model_input is already 2*LQ + noise_lvl*epsilon, including CLS."""
        if self._mask_only_active:
            raise RuntimeError('Nested/concurrent mask forwards are unsupported')
        self._mask_only_active = True
        try:
            super().forward(model_input, model_input.new_ones(model_input.shape[0]),
                            model_img_size, analysis=None, return_aux=False)
        except _D2Finished as captured:
            # process_attention_decoder has restored (B,V,N+CLS,C) order.
            features = captured.features[:, :, self.num_cls_tkn:]
        else:
            raise RuntimeError('D2 was not reached')
        finally:
            self._mask_only_active = False
        return self._spatial_mask_logits(self.mask_head(features).squeeze(-1), model_img_size)

    def forward(self, x, t, model_img_size, analysis=None, return_aux=False,
                lq_latent=None, c_feat=None, condition_mask=None,
                mask_lq_latent=None, mask_noise_lvl=0.3, mask_only=False,
                real_mask_lq_latent=None, real_mask_img_size=None):
        if real_mask_lq_latent is not None and mask_lq_latent is None:
            raise ValueError("Real supervision requires the joint flow/t1-mask forward")
        if mask_only:
            if mask_lq_latent is not None or not torch.all(t == 1):
                raise ValueError('mask_only expects an already-conditioned input at t=1')
            return {'mask_logits': self.mask_logits_at_t1(x, model_img_size)}
        if mask_lq_latent is None:
            # Original inference/sampling and auxiliary-evaluation API.
            output = super().forward(x, t, model_img_size, analysis, return_aux,
                                     lq_latent, c_feat, condition_mask)
            if return_aux:
                output['mask_logits'] = self._spatial_mask_logits(output['mask_logits'], model_img_size)
            return output
        if not return_aux:
            raise ValueError('Joint flow/t1-mask forward requires return_aux=True')
        if mask_lq_latent.shape != x.shape:
            raise ValueError('LQ and flow latent shapes must match')
        # Head is never evaluated on sampled-t features in this branch.
        prediction = super().forward(x, t, model_img_size, analysis=analysis,
                                     return_aux=False, lq_latent=lq_latent,
                                     c_feat=c_feat, condition_mask=condition_mask)
        mask_input = fixed_t1_input(mask_lq_latent, mask_noise_lvl)
        logits = self.mask_logits_at_t1(mask_input, model_img_size)
        output = {'pred': prediction, 'mask_logits': logits}
        if real_mask_lq_latent is not None:
            if real_mask_img_size is None:
                raise ValueError('Real mask branch needs its own image size')
            output['real_mask_logits'] = self.mask_logits_at_t1(
                fixed_t1_input(real_mask_lq_latent, mask_noise_lvl), real_mask_img_size)
        return output


def fixed_t1_input(lq_latent, noise_lvl, noise=None):
    """Inference endpoint for the existing additive/double-LQ recipe."""
    if not math.isfinite(float(noise_lvl)) or noise_lvl < 0:
        raise ValueError('noise_lvl must be finite and nonnegative')
    if noise is None:
        noise = torch.randn_like(lq_latent)
    if noise.shape != lq_latent.shape:
        raise ValueError('Noise shape must match LQ')
    return 2 * lq_latent + float(noise_lvl) * noise


class _JointForward:
    """Preserve the original transport's sampled-t/CFG/attention logic."""
    def __init__(self, model, lq_latent, noise_lvl, real_mask_lq_latent=None, real_mask_img_size=None):
        self.model, self.lq_latent, self.noise_lvl = model, lq_latent, noise_lvl
        self.real_mask_lq_latent, self.real_mask_img_size = real_mask_lq_latent, real_mask_img_size

    @property
    def module(self):
        return getattr(self.model, 'module', self.model)

    def __call__(self, *args, **kwargs):
        if self.real_mask_lq_latent is not None:
            kwargs.update(real_mask_lq_latent=self.real_mask_lq_latent, real_mask_img_size=self.real_mask_img_size)
        output = self.model(*args, mask_lq_latent=self.lq_latent,
                            mask_noise_lvl=self.noise_lvl, **kwargs)
        self.real_mask_logits = output.get('real_mask_logits')
        return output


class FixedT1MaskTransport:
    def __init__(self, transport):
        self.transport = transport

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def training_losses_mvrm(self, model, x1, xcond, model_img_size, cfg,
                            real_mask_lq_latent=None, real_mask_img_size=None, mask_update=True):
        if not mask_update:
            from omegaconf import OmegaConf
            flow_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            flow_cfg.stage_2.params.mask_head_enabled = False
            return self.transport.training_losses_mvrm(model, x1, xcond, model_img_size, flow_cfg)
        original_lq = xcond  # Transport clones before CFG dropping.
        proxy = _JointForward(model, original_lq, float(cfg.mvrm.noise_lvl), real_mask_lq_latent, real_mask_img_size)
        output = self.transport.training_losses_mvrm(
            proxy,
            x1, xcond, model_img_size, cfg)
        if real_mask_lq_latent is not None:
            output['real_mask_logits'] = proxy.real_mask_logits
        # Independent mask branch always uses original LQ, including samples
        # whose FLOW conditioning was dropped. No dropout-based mask skipping.
        output['mask_conditioned'] = torch.ones(x1.shape[0], dtype=torch.bool, device=x1.device)
        return output


def configure_mask_transport(transport, cfg):
    """Identity for every existing config; opt-in only for the new model."""
    option = cfg.mvrm.loss.get('mask', {}).get('fixed_t1', {})
    if not option.get('use', False):
        if cfg.stage_2.target == 'mvr.d2_mask_training.FixedT1D2GARD':
            raise ValueError('FixedT1D2GARD must enable mask.fixed_t1.use')
        return transport
    every = cfg.mvrm.loss.mask.get("update_every", 1)
    if not isinstance(every, int) or every < 1:
        raise ValueError("mask.update_every must be a positive integer")
    params = cfg.stage_2.params
    if (cfg.stage_2.target != 'mvr.d2_mask_training.FixedT1D2GARD'
            or not params.get('mask_head_enabled', False)
            or params.get('mask_head_source') != 'decoder'
            or params.get('mask_head_layer') != 2
            or params.get('c_feat_enabled', False)
            or option.get('timestep') != 1
            or option.get('conditioning') != 'always_lq'
            or cfg.mvrm.lq_latent_cond != 'addition'
            or not cfg.mvrm.get('training_lq_cond', False)):
        raise ValueError('Invalid fixed-t1 D2 configuration/conditioning recipe')
    if float(cfg.mvrm.loss.mask.lambda_coeff) <= 0:
        raise ValueError('Fixed-t1 mask weight must be positive')
    if not math.isfinite(float(cfg.mvrm.noise_lvl)) or float(cfg.mvrm.noise_lvl) < 0:
        raise ValueError('Invalid mask noise level')
    return FixedT1MaskTransport(transport)


def mask_checkpoint_metadata(cfg):
    option = cfg.mvrm.loss.get('mask', {}).get('fixed_t1', {})
    if not option.get('use', False):
        return None
    protocol = dict(mode='separate_fixed_t1', timestep=1,
        position='post_d2', conditioning='always_lq', noise_lvl=float(cfg.mvrm.noise_lvl),
        weight=float(cfg.mvrm.loss.mask.lambda_coeff))
    every = int(cfg.mvrm.loss.mask.get('update_every', 1))
    if every != 1: protocol['update_every_optimizer_steps'] = every
    target = cfg.mvrm.loss.mask.get('target', 'object_shadow')
    if target != 'object_shadow':
        protocol['synthetic_target'] = target
    real = cfg.mvrm.loss.mask.get('real', {})
    if real.get('use', False):
        import hashlib
        from pathlib import Path
        protocol['real_mask'] = dict(weight=float(real.lambda_coeff),
            manifest_sha256=hashlib.sha256(Path(real.manifest).read_bytes()).hexdigest(),
            sampling=('one_real_group_per_synthetic_group_per_microbatch' if every==1 else 'one_real_group_per_synthetic_group_on_mask_update_steps'))
    if cfg.stage_2.params.get('mask_head_resolution', 'patch') == 'pixel':
        protocol.update(resolution='pixel', pixel_patch_size=int(
            cfg.stage_2.params.get('mask_head_pixel_patch_size', 14)))
    return {'mask_training_protocol': protocol}


def validate_mask_resume(path, cfg):
    """Resume is stricter than initializing a new run from baseline EMA."""
    expected = mask_checkpoint_metadata(cfg)
    if expected is None:
        return
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if saved.get('mask_training_protocol') != expected['mask_training_protocol']:
        raise ValueError('Resume mask protocol differs; use --init-checkpoint for a new run')


def annotated_mask_bce(logits, targets, known):
    """Mask-only real-data loss: ignore unlabeled views/pixels, no clean RGB.

    Inputs are spatial patch targets in reference-first order (no CLS).
    This helper does not add a real loader to the synthetic trainer. When
    integrating DDP real batches, include them inside a joint forward rather
    than issuing a separate DDP forward before the same backward.
    """
    if logits.shape != targets.shape or known.shape != targets.shape:
        raise ValueError('Logits, targets and known-label masks must match')
    known = known.bool()
    # Missing real labels may be NaN; they must not poison the loss/gradient.
    safe_targets = torch.where(known, targets, torch.zeros_like(targets))
    loss = F.binary_cross_entropy_with_logits(logits.float(), safe_targets.float(), reduction='none')
    known = known.to(loss.dtype)
    counts = known.flatten(1).sum(1)
    per_group = (loss * known).flatten(1).sum(1) / counts.clamp_min(1)
    return per_group.sum() / (counts > 0).sum().clamp_min(1)


def mask_update_due(optimizer_step, every):
    """Completed optimizer steps are zero-based; updates 4,8,... for every=4."""
    if int(every) != every or every < 1:
        raise ValueError('mask update interval must be a positive integer')
    return (optimizer_step + 1) % every == 0
