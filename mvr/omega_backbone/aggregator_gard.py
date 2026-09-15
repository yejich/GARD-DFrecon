"""GARD feature extraction / injection hooks for the VGGT-Omega aggregator.

The original `Aggregator` is left untouched; this subclass only re-implements `forward`
with two extra hooks placed right after `inter_frame_blocks[i]`:

    extract_layers    : inter-frame block indices whose output tokens are returned
    restored_latents  : {block_idx: tokens} that overwrite the output of that block

Token layout at every hook: (B, V, 1 + num_register + N, D) = camera, registers, patches.
Injecting after block 3 (default for GARD) sits right before the first DPT-cached layer 4,
so all four DenseHead inputs and the CameraHead see the restored tokens.
"""

import torch

from vggt_omega.models.aggregator import Aggregator, slice_expand_and_flatten


class AggregatorGARD(Aggregator):
    def forward(
        self,
        images: torch.Tensor,
        extract_layers: tuple[int, ...] = (),
        restored_latents: dict[int, torch.Tensor] | None = None,
        stop_after_extract: bool = False,
    ) -> tuple[list[torch.Tensor | None] | None, int, dict[int, torch.Tensor]]:
        restored_latents = restored_latents or {}
        for idx in list(extract_layers) + list(restored_latents):
            if idx in self.cached_layer_indices:
                # frame_tokens of a cached layer are concatenated before the hook would run,
                # so the DPT input of that layer would still carry the un-restored features.
                raise ValueError(f"GARD hook at cached layer {idx} leaks degraded frame tokens; use {idx - 1}")

        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)  # (B*V, 3, H, W)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)  # (B*V, 1, D)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)  # (B*V, R, D)

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]  # (B*V, N, D)

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)  # (B*V, T, D), T = 1 + R + N
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        last_needed = max(extract_layers) if (stop_after_extract and extract_layers) else self.depth - 1
        extracted = {}
        outputs = []
        for block_idx in range(last_needed + 1):
            tokens, frame_tokens = self._run_frame_block(
                tokens, batch_size, num_frames, num_tokens, embed_dim, block_idx, frame_rope
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
            )  # (B, V, T, D)

            if block_idx in extract_layers:
                extracted[block_idx] = tokens.float()  # (B, V, T, D)
            if block_idx in restored_latents:
                restored = restored_latents[block_idx]  # (B, V, T, D)
                if restored.shape != tokens.shape:
                    raise ValueError(f"Restored latent {tuple(restored.shape)} != tokens {tuple(tokens.shape)}")
                tokens = restored.to(dtype=tokens.dtype)

            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))  # (B, V, T, 2D)
            else:
                outputs.append(None)

        if last_needed < self.depth - 1:
            return None, self.patch_token_start, extracted
        return outputs, self.patch_token_start, extracted
