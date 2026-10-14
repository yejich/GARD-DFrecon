"""Evaluate the established fixed inputs using the live EMA at saved epochs.

Each DDP rank evaluates a disjoint set with no extra backbone/denoiser copy.
Sampling, reference order, queries and RGB decoder follow run_fixed_suite.py.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


def load_groups(suite_paths):
    groups = []
    seen = set()
    expanded=[]
    for item in suite_paths:
        path=Path(item)
        if path.is_dir():path=path/'registry.json'
        if path.name=='registry.json':
            registry=json.loads(path.read_text())
            expanded.extend(path.parent/relative for relative in registry['manifests'])
        else:expanded.append(path)
    for path in expanded:
        path = Path(path)
        suite = json.loads(path.read_text())
        for group in suite['groups']:
            identity = tuple(group['input_paths'])
            if identity in seen:
                continue
            seen.add(identity)
            for filename, expected in zip(group['input_paths'], group['input_sha256'], strict=True):
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Eval input changed: {filename}')
            group = dict(group)
            if 'correspondence' in suite:
                group['pose_correspondence'] = suite['correspondence']
            label=path.parent.name if path.name=='manifest.json' else path.stem
            groups.append((f'{label}/{group["id"]}', group, suite['protocol']))
    return groups


@torch.no_grad()
def evaluate_fixed_suites(cfg, models, device, rank, world_size, experiment_dir, epoch, step):
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from mvr.eval_models.rgb_decode import build_rae_da3, decode_rgb_from_feats
    from stage2.transport import create_transport, Sampler
    from scripts.eval.layer_query_attention.query_capture import QueryAttentionCapture
    from utils import wandb_utils
    from mvr.fixed_pose_eval import pose_reference, score_poses

    groups = load_groups(cfg.fixed_eval.suites)
    output = Path(experiment_dir)/'fixed_eval'/f'epoch_{epoch:03d}'
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.mvrm.analysis.vis_attn_map = False
    eval_cfg.mvrm.val.analysis = {'vis_attn_map': False}
    encoder, ema = models['encoder'], models['ema_denoiser']
    results = []
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    # Decoder construction and evaluation must not change subsequent training RNG.
    with torch.random.fork_rng(devices=[device.index]):
        decoder = build_rae_da3(device='cpu')
        for name, group, protocol in groups[rank::world_size]:
            folder = output/name
            folder.mkdir(parents=True, exist_ok=True)
            norm, _, _ = InputProcessor()(group['input_paths'], process_res=protocol['process_res'],
                process_res_method=protocol['process_res_method'], sequential=True)
            h, w = norm.shape[-2:]
            assert [h, w] == group['image_hw']
            image = norm[None].to(device)
            base, features = encoder(image=image, export_feat_layers=[], ref_view_strategy='first',
                                     mvrm_cfg=eval_cfg.mvrm.train, mode='train')
            assert torch.all(base.ref_b_idx == 0), 'Fixed eval requires the first reference view'
            cond = features[('extract_feat', 17)].clone().float()
            baseline_pose = base.extrinsics[0].float().cpu().numpy()
            del base, features
            generator = torch.Generator(device=device).manual_seed(protocol['seed'])
            initial = cond + protocol['noise_lvl'] * torch.randn(cond.shape, device=device, generator=generator)
            capture = QueryAttentionCapture(ema, (h, w), group['queries'], [0, 1, 2, 3], layers=[9])
            calls = [-1]
            def track(module, args):
                calls[0] += 1
                capture.enabled = calls[0] in protocol['capture_calls']
                capture.context = dict(call=calls[0], t=float(args[1][0]))
            hook = ema.register_forward_pre_hook(track)
            shift = (cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base)**.5
            sample = Sampler(create_transport(**cfg.transport.params, time_dist_shift=shift)).sample_ode(**cfg.sampler.params)
            try:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    final = sample(initial, ema, mvrm_cfg=eval_cfg.mvrm, model_img_size=(h, w), lq_latent=cond)[-1]
                assert calls[0] == 48 and len(capture.records) == 3 and torch.isfinite(final).all()
                np.savez_compressed(folder/'attention.npz', attention=np.stack([r['attention'] for r in capture.records]),
                                    calls=protocol['capture_calls'])
            finally:
                hook.remove()
                capture.close()
            restored, features = encoder(image=image, export_feat_layers=[], ref_view_strategy='first',
                mvrm_cfg=eval_cfg.mvrm.val, mode='val', export_rgb_feat_layers=True,
                mvrm_result={('restored_latent', 17): final})
            feats = [(a.cpu(), b.cpu()) for a, b in restored.feat]
            restored_pose = restored.extrinsics[0].float().cpu().numpy()
            reference_pose, reference_info = pose_reference(group)
            pose_scalars, pose_details = score_poses(baseline_pose, restored_pose, reference_pose, reference_info)
            np.savez_compressed(folder/'poses.npz', da3=baseline_pose, gard=restored_pose, reference=reference_pose)
            del restored, features, final, cond, initial, image, sample, capture
            torch.cuda.empty_cache()
            decoder.mae_decoder.to(device)
            decoder.adapter.to(device)
            rgb = []
            for view in range(4):
                part = [(a[:, view:view+1].to(device), b[:, view:view+1].to(device)) for a, b in feats]
                rgb.append(decode_rgb_from_feats(decoder, part, h, w)[0, 0])
                del part
            decoder.mae_decoder.cpu()
            decoder.adapter.cpu()
            del feats
            rgb = torch.stack(rgb)
            columns = [('Input', (norm*std+mean).clamp(0, 1)), ('GARD EMA', rgb)]
            metrics = dict(pose_scalars)
            if group.get('clean_paths'):
                clean, _, _ = InputProcessor()(group['clean_paths'], process_res=protocol['process_res'],
                    process_res_method=protocol['process_res_method'], sequential=True)
                clean = (clean*std+mean).clamp(0, 1)
                columns.append(('HQ', clean))
                mse = (rgb-clean).square().mean().item()
                metrics.update(rgb_mse=mse, rgb_psnr=float(-10*np.log10(max(mse, 1e-12))))
            panel = Image.new('RGB', (w*len(columns), (h+24)*4), 'white')
            draw = ImageDraw.Draw(panel)
            for col, (label, values) in enumerate(columns):
                for view in range(4):
                    pixels = values[view].permute(1, 2, 0).mul(255).byte().numpy()
                    panel.paste(Image.fromarray(pixels), (col*w, view*(h+24)+24))
                    draw.text((col*w+4, view*(h+24)+4), f'{label} | view {view}', fill='black')
            panel.save(folder/'rgb.jpg', quality=92)
            record = dict(name=name, epoch=epoch, metrics=metrics, image=str(folder/'rgb.jpg'),
                          group=group, protocol=protocol, weights='EMA',
                          pose_reference=reference_info, pose_details=pose_details)
            (folder/'result.json').write_text(json.dumps(record, indent=2)+'\n')
            results.append(record)
            print(f'[Fixed eval epoch {epoch}] {name}: {metrics}', flush=True)
        del decoder
    gathered = [None]*world_size
    dist.all_gather_object(gathered, results)
    if rank == 0:
        records = [r for partition in gathered for r in partition]
        (output/'results.json').write_text(json.dumps(records, indent=2)+'\n')
        if cfg.log.tracker.name == 'wandb':
            payload = {'fixed_eval/epoch': epoch}
            for record in records:
                key = 'fixed_eval/'+record['name']
                payload[key] = wandb_utils.wandb.Image(record['image'], caption=f'EMA epoch {epoch}')
                payload.update({key+'/'+k: v for k, v in record['metrics'].items()})
            wandb_utils.wandb.log(payload, step=step, commit=False)
    dist.barrier()
