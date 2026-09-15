"""GARD restoration inference on VGGT-Omega (restore inter_frame_blocks[3] tokens, then finish the forward).

Modes
  --eval-groups 0 1 ...      Hypersim-pair held-out groups from the manifest (HQ available -> metrics)
  --lq-images a.png b.png    arbitrary images (no GT; saves predictions only)

Per group it saves   <out>/<name>/pred.npz   (depth/conf/extrinsics/intrinsics for lq | restored | hq)
                     <out>/<name>/grid.jpg   (rows: images, depth LQ, depth restored, depth HQ)
and <out>/metrics.json with depth AbsRel / delta1 of LQ and restored vs the HQ (clean-input) prediction,
split into all / distractor / clean pixels (median-scale aligned per group).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from stage2.transport import Sampler, create_transport
from utils.model_utils import instantiate_from_config

from mvr.dataset.hypersim_pairs import pair_collate_fn
from mvr.omega_training import build_normalizer, load_omega_encoder, pair_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="training checkpoint (latest.pt) or raw denoiser state_dict")
    p.add_argument("--out", required=True)
    p.add_argument("--eval-groups", type=int, nargs="*", default=None, help="manifest eval group ids; empty = all")
    p.add_argument("--lq-images", nargs="*", default=None)
    p.add_argument("--image-resolution", type=int, default=512, help="--lq-images mode (balanced resize)")
    p.add_argument("--num-steps", type=int, default=None, help="override sampler.params.num_steps")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def colorize(depth, lo, hi):
    d = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_TURBO)


def save_grid(path, images, depth_rows):
    """images (v, 3, H, W) in [0, 1]; depth_rows: list of (v, H, W) numpy."""
    valid = np.concatenate([r.reshape(-1) for r in depth_rows])
    lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    rows = [np.concatenate([cv2.cvtColor((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR) for img in images], axis=1)]
    rows += [np.concatenate([colorize(d, lo, hi) for d in r], axis=1) for r in depth_rows]
    cv2.imwrite(str(path), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 90])


def depth_metrics(pred, ref, mask):
    """pred/ref (v, H, W) ; mask (v, H, W) bool. Median scale aligned on all valid pixels."""
    valid = (ref > 0) & np.isfinite(pred)
    scale = np.median(ref[valid]) / max(np.median(pred[valid]), 1e-8)
    pred = pred * scale
    out = {}
    for name, sel in (("all", valid), ("distractor", valid & mask), ("clean", valid & ~mask)):
        if sel.sum() == 0:
            continue
        p, r = pred[sel], ref[sel]
        out[f"absrel_{name}"] = float(np.mean(np.abs(p - r) / r))
        out[f"delta1_{name}"] = float(np.mean(np.maximum(p / r, r / p) < 1.25))
    return out


def to_np(pred, prefix):
    return {f"{prefix}_depth": pred["depth"][0, ..., 0].float().cpu().numpy(),  # (v, H, W)
            f"{prefix}_conf": pred["depth_conf"][0].float().cpu().numpy(),  # (v, H, W)
            f"{prefix}_extrinsics": pred["extrinsics"][0].float().cpu().numpy(),  # (v, 3, 4)
            f"{prefix}_intrinsics": pred["intrinsics"][0].float().cpu().numpy()}  # (v, 3, 3)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.num_steps is not None:
        cfg.sampler.params.num_steps = args.num_steps
    cfg.mvrm.analysis.vis_attn_map = False
    device = torch.device("cuda")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    encoder = load_omega_encoder(cfg, device)
    normalizer = build_normalizer(cfg, encoder, device, checkpoint_path=args.ckpt)
    denoiser = instantiate_from_config(dict(target=cfg.stage_2.target, params=cfg.stage_2.params)).to(device).eval()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    denoiser.load_state_dict(state.get("ema", state.get("model", state)), strict=True)

    shift = math.sqrt(cfg.misc.time_dist_shift_dim / cfg.misc.time_dist_shift_base)
    transport = create_transport(**cfg.transport.params, time_dist_shift=shift)
    eval_sampler = Sampler(transport).sample_ode(**cfg.sampler.params)
    noise_lvl = cfg.mvrm.get("noise_lvl", None)
    generator = torch.Generator(device=device)

    # ------------------------------------------------------------------ inputs
    jobs = []
    if args.lq_images:
        sys.path.insert(0, cfg.stage_1.vggt_omega.repo)
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        imgs = load_and_preprocess_images(args.lq_images, image_resolution=args.image_resolution)  # (v, 3, H, W)
        jobs.append(("custom", imgs[None], None, None))
    else:
        ds = pair_dataset(cfg, "eval", encoder.patch_size)
        ids = args.eval_groups if args.eval_groups else range(len(ds.manifest["eval_groups"]))
        for i in ids:
            batch = pair_collate_fn([ds[i]])
            mask = (batch["pixel_masks"][0].numpy() > 0) & batch["distractor_views"][0].numpy()[:, None, None]
            jobs.append((f"group_{i:02d}", batch["lq_views"], batch["hq_views"], mask))

    all_metrics = {}
    for name, lq, hq, mask in jobs:
        lq = lq.to(device)  # (1, v, 3, H, W) in [0, 1]
        H, W = lq.shape[-2:]

        lq_pred = encoder.forward_full(lq, extract=True)
        lq_latent = normalizer.normalize(lq_pred["gard_feats"][encoder.layer])  # (1, v, S+N, D)

        generator.manual_seed(args.seed)
        noise = torch.randn(lq_latent.shape, generator=generator, device=device)  # (1, v, S+N, D)
        x0 = noise * noise_lvl + lq_latent if noise_lvl is not None else noise
        model_kwargs = dict(mvrm_cfg=cfg.mvrm, model_img_size=(H, W), lq_latent=lq_latent)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            restored = eval_sampler(x0, denoiser.forward, **model_kwargs)[-1]  # (1, v, S+N, D)
        restored = normalizer.denormalize(restored.float())  # (1, v, S+N, D)
        res_pred = encoder.forward_full(lq, restored=restored)

        save_dir = out_root / name
        save_dir.mkdir(parents=True, exist_ok=True)
        arrays = {**to_np(lq_pred, "lq"), **to_np(res_pred, "restored")}
        rows = [arrays["lq_depth"], arrays["restored_depth"]]
        if hq is not None:
            hq = hq.to(device)
            hq_pred = encoder.forward_full(hq, extract=True)
            arrays.update(to_np(hq_pred, "hq"))
            rows.append(arrays["hq_depth"])
            hq_latent = hq_pred["gard_feats"][encoder.layer]  # (1, v, S+N, D)
            lq_raw = lq_pred["gard_feats"][encoder.layer]
            feat_err = lambda a: ((a - hq_latent).norm() / hq_latent.norm()).item()  # noqa: E731
            m = {"lq": depth_metrics(arrays["lq_depth"], arrays["hq_depth"], mask),
                 "restored": depth_metrics(arrays["restored_depth"], arrays["hq_depth"], mask),
                 "feat_relL2_lq": feat_err(lq_raw), "feat_relL2_restored": feat_err(restored)}
            all_metrics[name] = m
            print(f"{name}: absrel lq {m['lq']['absrel_all']:.4f} -> restored {m['restored']['absrel_all']:.4f} | "
                  f"feat relL2 {m['feat_relL2_lq']:.3f} -> {m['feat_relL2_restored']:.3f}")
        np.savez_compressed(save_dir / "pred.npz", **arrays)
        save_grid(save_dir / "grid.jpg", lq[0], rows)

    if all_metrics:
        keys = sorted({k for m in all_metrics.values() for k in m["lq"]})
        summary = {src: {k: float(np.mean([m[src][k] for m in all_metrics.values() if k in m[src]])) for k in keys}
                   for src in ("lq", "restored")}
        for k in ("feat_relL2_lq", "feat_relL2_restored"):
            summary[k] = float(np.mean([m[k] for m in all_metrics.values()]))
        (out_root / "metrics.json").write_text(json.dumps(dict(summary=summary, per_group=all_metrics), indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
