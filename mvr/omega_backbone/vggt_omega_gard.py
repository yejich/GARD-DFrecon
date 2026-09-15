"""VGGT-Omega with GARD hooks (same state_dict as `VGGTOmega`).

Usage:
    model = VGGTOmegaGARD()
    model.load_state_dict(torch.load("vggt_omega_1b_512.pt"))

    # 1) extract only (runs DINOv3 + aggregator blocks 0..3, no heads)
    feats = model.extract(images, layer=3)                              # (B, V, T, D)
    # 2) full forward with the restored tokens injected after block 3
    preds = model(images, restored_latents={3: restored})
"""

import torch

from .aggregator_gard import AggregatorGARD
from vggt_omega.models.vggt_omega import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera


class VGGTOmegaGARD(VGGTOmega):
    def __init__(self, patch_size: int = 16, embed_dim: int = 1024, **kwargs) -> None:
        super().__init__(patch_size=patch_size, embed_dim=embed_dim, **kwargs)
        # same parameter names as Aggregator -> checkpoint loads unchanged
        self.aggregator = AggregatorGARD(patch_size=patch_size, embed_dim=embed_dim)

    @property
    def patch_token_start(self) -> int:
        return self.aggregator.patch_token_start

    def _amp_dtype(self) -> torch.dtype:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def extract(self, images: torch.Tensor, layer: int = 3) -> torch.Tensor:
        """images in [0, 1], (B, V, 3, H, W) -> float32 tokens after inter_frame_blocks[layer]."""
        if images.ndim == 4:
            images = images.unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=self._amp_dtype()):
            _, _, extracted = self.aggregator(images, extract_layers=(layer,), stop_after_extract=True)
        return extracted[layer]  # (B, V, T, D)

    def forward(
        self,
        images: torch.Tensor,
        extract_layers: tuple[int, ...] = (),
        restored_latents: dict[int, torch.Tensor] | None = None,
        with_cameras: bool = True,
    ) -> dict[str, torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(0)

        with torch.autocast(device_type="cuda", dtype=self._amp_dtype()):
            aggregated_tokens_list, patch_token_start, extracted = self.aggregator(
                images, extract_layers=extract_layers, restored_latents=restored_latents
            )

        predictions = {
            "camera_and_register_tokens": aggregated_tokens_list[-1][:, :, :patch_token_start].contiguous(),
            "gard_feats": extracted,
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)
                if with_cameras:
                    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
                    predictions["extrinsics"] = extrinsics  # (B, V, 3, 4) camera-from-world, OpenCV
                    predictions["intrinsics"] = intrinsics  # (B, V, 3, 3)
            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list, images=images, patch_token_start=patch_token_start
                )
                predictions["depth"] = depth  # (B, V, H, W, 1)
                predictions["depth_conf"] = depth_conf  # (B, V, H, W)
            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(aggregated_tokens_list, patch_token_start=patch_token_start)
                )
        return predictions
