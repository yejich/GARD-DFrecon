"""Per-channel mean/std of VGGT-Omega GARD latents (tokens after inter_frame_blocks[layer]).

Output: {"mean": (S+1, D), "std": (S+1, D), "layer", "num_images"}; rows 0..S-1 = special tokens
(camera, registers) per position, row S = all patch tokens pooled. Uses clean and distractor images
from the Hypersim-pair train split.
"""
import argparse
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from mvr.dataset.hypersim_pairs import pair_collate_fn
from mvr.omega_training import load_omega_encoder, pair_dataset, omega_latent_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-groups", type=int, default=300, help="random train records (each gives clean + distractor)")
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda")
    encoder = load_omega_encoder(cfg, device)
    S = encoder.num_special
    ds = pair_dataset(cfg, "train", encoder.patch_size)
    rng = random.Random(args.seed)

    anchors = [i for i, r in enumerate(ds.records) if len(r["candidates"]) >= args.views - 1]
    D = encoder.model.aggregator.camera_token.shape[-1]
    sum1 = torch.zeros(S + 1, D, dtype=torch.float64, device=device)  # (S+1, D)
    sum2 = torch.zeros(S + 1, D, dtype=torch.float64, device=device)  # (S+1, D)
    count = torch.zeros(S + 1, 1, dtype=torch.float64, device=device)  # (S+1, 1)
    num_images = 0

    for _ in tqdm(range(args.num_groups)):
        item = (rng.choice(anchors), args.views, rng.getrandbits(63))
        batch = pair_collate_fn([ds[item]])
        for key in ("hq_views", "lq_views"):
            feats = encoder.extract(batch[key].to(device)).double()  # (1, v, S+N, D)
            feats = feats.flatten(0, 1)  # (v, S+N, D)
            special, patch = feats[:, :S], feats[:, S:].reshape(-1, D)  # (v, S, D), (v*N, D)
            sum1[:S] += special.sum(0)
            sum2[:S] += special.square().sum(0)
            count[:S] += special.shape[0]
            sum1[S] += patch.sum(0)
            sum2[S] += patch.square().sum(0)
            count[S] += patch.shape[0]
            num_images += feats.shape[0]

    mean = sum1 / count  # (S+1, D)
    std = (sum2 / count - mean.square()).clamp_min(0).sqrt()  # (S+1, D)
    out = Path(cfg.mvrm.latent_norm.stats_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = omega_latent_metadata(cfg)
    metadata.update(num_special=S, patch_size=encoder.patch_size)
    torch.save(dict(mean=mean.float().cpu(), std=std.float().cpu(), num_images=num_images, **metadata), out)
    print(f"saved {out} | images {num_images} | patch mean|.| {mean[S].abs().mean():.4f} std avg {std[S].mean():.4f} "
          f"min {std[S].min():.4f} max {std[S].max():.4f} | special std avg {std[:S].mean():.4f}")


if __name__ == "__main__":
    main()
