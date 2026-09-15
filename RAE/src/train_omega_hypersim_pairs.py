# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Hypersim saved-pair GARD training on VGGT-Omega features (inter_frame_blocks[3] output).
VGGT-Omega counterpart of train_hypersim_pairs.py (DA3); the DA3 script is left unchanged.
"""
import argparse
import math
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from stage2.transport import create_transport
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import update_ema
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import configure_experiment_dirs, save_worktree
from utils.dist_utils import setup_distributed, cleanup_distributed

from gard.GARD_omega import load_shape_matched_state_dict
from mvr.grouped_mse import grouped_mse
from mvr.omega_training import (attn_align_loss_omega, build_normalizer, evaluate_pairs_omega,
                                load_omega_encoder, load_train_data_omega, token_groups_omega)
from mvr.pair_training import resume_training, save_training_checkpoint, save_epoch_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GARD on VGGT-Omega latents (Hypersim pairs).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many optimizer updates; 0 = full")
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="Save and stop at this completed epoch; keep the configured full LR schedule (0 = full)")
    parser.add_argument("--fixed-views", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--resume", type=str)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cross-view-feature-completion")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    if args.stop_after_epoch < 0:
        parser.error('--stop-after-epoch must be nonnegative')
    return args


def load_denoiser(cfg, device, logger):
    # target/params only: instantiate_from_config would otherwise non-strictly load stage_2.ckpt itself
    denoiser = instantiate_from_config(dict(target=cfg.stage_2.target, params=cfg.stage_2.params)).to(device)
    ckpt, init_ckpt = cfg.stage_2.get("ckpt", None), cfg.stage_2.get("init_from_da3_ckpt", None)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)  # trusted local training checkpoint (includes RNG)
        denoiser.load_state_dict(state.get("ema", state.get("model", state)), strict=True)
        logger.info(f"Denoiser loaded (strict) from {ckpt}")
    elif init_ckpt:
        state = torch.load(init_ckpt, map_location="cpu", weights_only=True, mmap=True)
        loaded, skipped = load_shape_matched_state_dict(denoiser, state.get("ema", state.get("model", state)))
        logger.info(f"Denoiser partially initialised from DA3 GARD {init_ckpt}: {len(loaded)} tensors loaded, "
                    f"{len(skipped)} kept at init: {skipped}")
    else:
        logger.info("Denoiser trained from scratch")
    denoiser.requires_grad_(True)
    ddp = DDP(denoiser, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False, find_unused_parameters=False)
    # DDP broadcasts rank-0 parameters; copy only after that synchronization.
    ema = deepcopy(ddp.module)
    ema.requires_grad_(False)
    ema.eval()
    ddp._set_static_graph()
    ddp.train()
    return dict(denoiser=ddp.module, ema_denoiser=ema, ddp_denoiser=ddp)


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    full_cfg = OmegaConf.load(args.config)
    if args.fixed_views:
        full_cfg.data.train.min_num_input_view = args.fixed_views
        full_cfg.data.train.max_num_input_view = args.fixed_views
    if args.wandb:
        full_cfg.log.tracker.name = "wandb"
        full_cfg.log.tracker.wandb.project = args.wandb_project
        full_cfg.log.tracker.wandb.run_name = args.wandb_run_name
        full_cfg.log.tracker.wandb.entity = args.wandb_entity
        full_cfg.log.tracker.wandb.unique_run = True
    training_cfg = full_cfg.training

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(full_cfg, rank)

    time_dist_shift = math.sqrt(full_cfg.misc.time_dist_shift_dim / full_cfg.misc.time_dist_shift_base)
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad) if clip_grad and clip_grad > 0 else None
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.epochs)
    stop_epoch = min(num_epochs, args.stop_after_epoch or num_epochs)
    global_batch_size = int(training_cfg.global_batch_size)
    assert global_batch_size % (world_size * grad_accum_steps) == 0, "Effective batch must be divisible"
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    log_interval = int(training_cfg.get("log_interval", 100))
    global_seed = int(training_cfg.get("global_seed", 0))
    seed = global_seed * world_size + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # models
    encoder = load_omega_encoder(full_cfg, device)
    normalizer = build_normalizer(full_cfg, encoder, device,
                                  checkpoint_path=args.resume or full_cfg.stage_2.get("ckpt"))
    models = dict(encoder=encoder, **load_denoiser(full_cfg, device, logger))
    patch_size, num_special = encoder.patch_size, encoder.num_special
    assert models["denoiser"].num_cls_tkn == num_special and models["denoiser"].vit_patch_size == patch_size
    assert models["denoiser"].in_channels == encoder.model.aggregator.camera_token.shape[-1]

    train_loader, train_sampler = load_train_data_omega(full_cfg, micro_batch_size, rank, world_size, patch_size)
    from mvr.dataset.hypersim_manifest import fingerprint
    manifest_fingerprint = fingerprint(train_loader.dataset.manifest)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)

    optimizer, optim_msg = build_optimizer([p for p in models["denoiser"].parameters() if p.requires_grad], training_cfg)
    for group in optimizer.param_groups:
        group["foreach"] = False
    scheduler, sched_msg = (build_scheduler(optimizer, steps_per_epoch, training_cfg)
                            if training_cfg.get("scheduler") else (None, None))

    transport = create_transport(**full_cfg.transport.params, time_dist_shift=time_dist_shift)

    start_epoch, global_train_step, optimizer_step = 0, 0, 0
    running_loss, running_loss_count = 0.0, 0
    if args.resume:
        start_epoch, global_train_step, optimizer_step = resume_training(args.resume, models, optimizer, scheduler, rank,
                                                                          manifest_fingerprint=manifest_fingerprint)
    if rank == 0:
        save_worktree(experiment_dir, full_cfg)
        import json
        (Path(experiment_dir) / "pairs_manifest.json").write_text(
            json.dumps(train_loader.dataset.manifest, indent=2))
        logger.info(f"Stage-1 VGGT-Omega parameters: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M "
                    f"(GARD hook after inter_frame_blocks[{encoder.layer}], S={num_special}, patch={patch_size})")
        logger.info(f"Stage-2 Denoiser parameters: {sum(p.numel() for p in models['denoiser'].parameters())/1e6:.2f}M")
        logger.info(f"Latent normalisation: {'on' if normalizer.enabled else 'off'}")
        logger.info(optim_msg)
        logger.info(sched_msg if sched_msg else "No LR scheduler.")
        logger.info(f"Training {num_epochs} epochs, micro batch {micro_batch_size}/GPU, accum {grad_accum_steps}, "
                    f"{len(train_loader.dataset)} samples, {steps_per_epoch} steps/epoch, world {world_size}")

    attn_cfg = full_cfg.mvrm.loss.attn_align
    use_attn_align = attn_cfg.use and full_cfg.mvrm.analysis.vis_attn_map

    dist.barrier()
    if rank == 0:
        logger.info(f"This invocation stops after epoch {stop_epoch}; configured LR schedule remains {num_epochs} epochs.")
    for epoch in range(start_epoch, stop_epoch):
        models["ddp_denoiser"].train()
        train_sampler.set_epoch(epoch)
        epoch_loss = torch.zeros(1, device=device)
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for train_step, batch in enumerate(train_loader):
            hq_views = batch["hq_views"].to(device)  # (b, v, 3, H, W) in [0, 1]
            lq_views = batch["lq_views"].to(device)  # (b, v, 3, H, W) in [0, 1]
            train_h, train_w = hq_views.shape[-2:]

            with torch.no_grad():
                lq_latent = normalizer.normalize(encoder.extract(lq_views))  # (b, v, S+N, D)
                if use_attn_align:
                    hq_out = encoder.forward_full(hq_views, extract=True)
                    hq_latent = hq_out["gard_feats"][encoder.layer]  # (b, v, S+N, D)
                else:
                    hq_out = None
                    hq_latent = encoder.extract(hq_views)  # (b, v, S+N, D)
                hq_latent = normalizer.normalize(hq_latent)
            assert lq_latent.shape == hq_latent.shape

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=training_cfg.precision == "bf16"):
                transport_output = transport.training_losses_mvrm(
                    model=models["ddp_denoiser"], x1=hq_latent, xcond=lq_latent,
                    model_img_size=(train_h, train_w), cfg=full_cfg)

            token_mask, view_flags = token_groups_omega(
                batch["pixel_masks"].to(device), batch["distractor_views"].to(device), patch_size, num_special)
            transport_loss, group_parts = grouped_mse(
                transport_output["pred"], transport_output["target_velocity"], token_mask, view_flags,
                full_cfg.mvrm.loss.group_mse)
            loss = transport_loss

            attn_loss = torch.zeros((), device=device)
            if use_attn_align and attn_cfg.da3_point_cloud.use:
                pred_map = transport_output["mvrm_maps"][("mvrm", attn_cfg.mvrm_layer_idx, "global")]
                attn_loss = attn_align_loss_omega(
                    pred_map, hq_out["depth"], hq_out["extrinsics"], hq_out["intrinsics"],
                    attn_cfg.da3_point_cloud, patch_size, num_special,
                    distractor_cfg=attn_cfg.get("distractor_aware"), token_mask=token_mask)
                loss = transport_loss + attn_cfg.lambda_coeff * attn_loss

            finite = torch.isfinite(loss.detach()).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Nonfinite training loss on at least one rank")
            (loss / grad_accum_steps).backward()
            if (train_step + 1) % grad_accum_steps == 0:
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(models["ddp_denoiser"].parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                update_ema(models["ema_denoiser"], models["denoiser"], decay=ema_decay)
                optimizer_step += 1

            running_loss += loss.item()
            running_loss_count += 1
            epoch_loss += loss.detach()
            if rank == 0 and global_train_step % log_interval == 0:
                avg_loss = running_loss / running_loss_count
                stats = {
                    "train/loss_interval_avg": avg_loss,
                    "train/loss_transport": transport_loss.item(),
                    "train_attn/loss_attn": attn_loss.item(),
                    "train_etc/lr": optimizer.param_groups[0]["lr"],
                    "train_etc/peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                }
                stats.update({f"train_group/{k}": v.item() for k, v in group_parts.items()})
                logger.info(f"[Epoch {epoch} | Step {global_train_step}] loss={avg_loss:.4f} "
                            f"mse={transport_loss.item():.4f} dist={group_parts.get('distractor', transport_loss).item():.4f} "
                            f"clean={group_parts.get('clean', transport_loss).item():.4f} attn={attn_loss.item():.4f} "
                            f"lr={optimizer.param_groups[0]['lr']:.2e} views={hq_views.shape[1]} "
                            f"mem={stats['train_etc/peak_mem_gb']:.1f}G")
                if full_cfg.log.tracker.name == "wandb":
                    wandb_utils.log(stats, step=global_train_step)
                running_loss, running_loss_count = 0.0, 0

            num_batches += 1
            global_train_step += 1
            if args.max_steps and optimizer_step >= args.max_steps:
                eval_loss = evaluate_pairs_omega(full_cfg, models, normalizer, transport, device, rank, world_size,
                                                 max_groups=2)
                logger.info(f"Smoke eval/fixed_flow_loss={eval_loss:.6f}; no checkpoint written.")
                dist.barrier()
                if rank == 0 and full_cfg.log.tracker.name == "wandb":
                    wandb_utils.log({"eval/smoke_group_mse": eval_loss}, step=global_train_step)
                    wandb_utils.wandb.finish()
                cleanup_distributed()
                return

        eval_loss = evaluate_pairs_omega(full_cfg, models, normalizer, transport, device, rank, world_size)
        if rank == 0:
            logger.info(f"[Epoch {epoch}] eval/fixed_flow_loss={eval_loss:.6f} "
                        f"epoch/loss={epoch_loss.item() / max(num_batches, 1):.4f}")
            if full_cfg.log.tracker.name == "wandb":
                wandb_utils.log({"eval/group_mse": eval_loss, "epoch": epoch,
                                 "epoch/loss": epoch_loss.item() / max(num_batches, 1)}, step=global_train_step)
        save_training_checkpoint(Path(checkpoint_dir) / "latest.pt", epoch + 1,
                                 global_train_step, optimizer_step, models, optimizer, scheduler, rank,
                                 manifest_fingerprint=manifest_fingerprint,
                                 extra_state={"omega_latent_norm": normalizer.checkpoint_state()})
        save_epoch_weights(checkpoint_dir, epoch + 1, models, rank,
                           training_cfg.get('save_weight_epochs', []),
                           extra_state={"omega_latent_norm": normalizer.checkpoint_state()})

    dist.barrier()
    logger.info("Done!")
    if rank == 0 and full_cfg.log.tracker.name == "wandb":
        wandb_utils.wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
