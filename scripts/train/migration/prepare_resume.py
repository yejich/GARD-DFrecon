"""Relocate a full NVS training checkpoint without changing its experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf


def prepare(bundle, output, gpu_count):
    bundle = Path(bundle).resolve()
    checkpoint = bundle / 'checkpoints/resume.pt'
    meta = json.loads((bundle / 'migration.json').read_text())
    if checkpoint.stat().st_size != meta['checkpoint_bytes']:
        raise ValueError('Checkpoint size differs from migration manifest')
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    required = {'model', 'ema', 'optimizer', 'rng', 'step', 'epoch', 'next_batch', 'config'}
    if not required <= state.keys():
        raise ValueError(f'Not a full resume checkpoint: missing {required - state.keys()}')
    if len(state['rng']) != gpu_count:
        raise ValueError(f"Exact resume requires {len(state['rng'])} GPUs, got {gpu_count}")
    cfg = OmegaConf.create(state['config'])
    for field, filename in [('manifest', 'transient_only.json'),
                            ('negative_manifest', 'transient_static.json'),
                            ('scene_split', 'scene_split_policy.json')]:
        path = bundle / 'manifests' / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['manifest_sha256'][filename]:
            raise ValueError(f'Manifest changed: {filename}')
        cfg.data[field] = str(path)
    mapping = [('data.root', 'HYPERSIM_PAIRS_ROOT', True),
               ('data.negative_root', 'HYPERSIM_NEGATIVE_ROOT', True),
               ('camera.root', 'HYPERSIM_CAMERA_ROOT', True),
               ('camera.parameters', 'HYPERSIM_CAMERA_PARAMETERS', False)]
    for field, env, directory in mapping:
        value = os.environ.get(env) or OmegaConf.select(cfg, field)
        path = Path(value).expanduser().resolve()
        if not (path.is_dir() if directory else path.is_file()):
            raise FileNotFoundError(f'Set {env} to an existing path on this server: {path}')
        OmegaConf.update(cfg, field, str(path))
    cfg.checkpoint = str(checkpoint)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output)
    print(json.dumps(dict(config=str(output), step=state['step'], epoch=state['epoch'],
                         next_batch=state['next_batch'], world_size=gpu_count,
                         global_batch=gpu_count*cfg.per_gpu_batch_size*cfg.accumulation,
                         camera_loss=OmegaConf.to_container(cfg.group_mse),
                         geometry_supcon=OmegaConf.to_container(cfg.geometry_contrastive_loss)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('bundle', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-count', type=int, required=True)
    a = p.parse_args()
    prepare(a.bundle, a.output, a.gpu_count)
