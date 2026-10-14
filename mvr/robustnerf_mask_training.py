"""Deterministic Crab2/Yoda mask-only groups; never construct a clean flow target."""
import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mvr.mask_supervision import mask_targets
from mvr.d2_mask_training import fixed_t1_input, annotated_mask_bce

ROOT = Path(__file__).resolve().parents[1]


def protected_pair_ids(paths):
    result = {'crab2': set(), 'yoda': set()}
    def visit(value):
        if isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value, str):
            match = re.search(r'/(crab2|yoda)/(?:images(?:_[248])?)/(?:0clean|2clutter)(\d+)\.', value)
            if match: result[match[1]].add(int(match[2]))
    def visit_file(path):
        path=Path(path)
        data=json.loads(path.read_text())
        if data.get('schema_version')=='eval_registry_v1':
            for relative in data['manifests']:visit_file(path.parent/relative)
        else:visit(data)
    for path in paths:
        visit_file(path)
    return result


def build_manifest(root, output, protected_paths, seed=42, holdout_pairs=22):
    root = Path(root); protected = protected_pair_ids(protected_paths)
    spec = importlib.util.spec_from_file_location('robust_mask_colmap',
        ROOT/'Depth-Anything-3/src/depth_anything_3/utils/read_write_model.py')
    colmap = importlib.util.module_from_spec(spec); spec.loader.exec_module(colmap)
    scenes = {}
    for scene in ['crab2', 'yoda']:
        folder = root/scene
        poses = {im.name: im for im in colmap.read_images_binary(folder/'sparse/0/images.bin').values()}
        records = []
        for i in range(1,110):
            paths = dict(clean=str(folder/'images_4'/f'0clean{i}.jpg'),
                         clutter=str(folder/'images_4'/f'2clutter{i}.jpg'),
                         mask=str(folder/'masks'/f'2clutter{i}.png'))
            for path in paths.values():
                if not Path(path).is_file(): raise FileNotFoundError(path)
            pose = poses[f'2clutter{i}.jpg']
            records.append(dict(id=i, center=(-pose.qvec2rotmat().T@pose.tvec).tolist(), **paths,
                sha256={key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key,path in paths.items()}))
        held = set(protected[scene]);rng = random.Random(seed+len(scene))
        remaining = [i for i in range(1,110) if i not in held];rng.shuffle(remaining)
        held.update(remaining[:max(0,holdout_pairs-len(held))])
        train = [r['id'] for r in records if r['id'] not in held]
        if len(train)<4 or len(held)<4: raise ValueError('Both splits need at least four poses')
        scenes[scene] = dict(records=records,train_ids=train,val_ids=sorted(held),
                             protected_eval_ids=sorted(protected[scene]))
    manifest = dict(schema='robustnerf_mask_only_v1',seed=seed,scenes=scenes,
        views=4,process_res=504,mask_semantics='saved SAM2-assisted object masks; no shadow union',
        groups='four distinct poses from nearest16 within one split; clean/clutter/clutter/clean',
        flow_target=False,validation='same-scene held-out poses, not zero-shot')
    output = Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(manifest,indent=2,sort_keys=True))
    return manifest


class RealMaskSource:
    def __init__(self, manifest):
        self.manifest_path = Path(manifest)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.seed = self.manifest['seed']
        for scene in self.manifest['scenes'].values():
            if set(scene['train_ids']) & set(scene['val_ids']): raise ValueError('Real pose split overlap')
            if not set(scene['protected_eval_ids']) <= set(scene['val_ids']):
                raise ValueError('Existing eval poses must remain held out')

    def group(self, scene_name, split, anchor, seed):
        scene = self.manifest['scenes'][scene_name]
        rows = {r['id']:r for r in scene['records']}
        allowed = scene[f'{split}_ids']
        if anchor not in allowed: raise ValueError('Anchor outside split')
        center = np.array(rows[anchor]['center'])
        neighbors = sorted((i for i in allowed if i!=anchor),
                           key=lambda i:float(np.linalg.norm(np.array(rows[i]['center'])-center)))[:16]
        ids = [anchor]+random.Random(seed).sample(neighbors,3)
        images,masks,paths = [],[],[]
        for i,clutter in zip(ids,[False,True,True,False]):
            record = rows[i];path = record['clutter' if clutter else 'clean']
            with Image.open(path) as im:
                if im.width!=im.height: raise ValueError('Expected square RobustNeRF captures')
                rgb = np.asarray(im.convert('RGB').resize((504,504),Image.Resampling.BICUBIC)).copy()
            if clutter:
                with Image.open(record['mask']) as im:
                    mask = (np.asarray(im.convert('L').resize((504,504),Image.Resampling.NEAREST))>0).astype(np.float32)
            else: mask=np.zeros((504,504),dtype=np.float32)
            images.append(rgb);masks.append(mask);paths.append(path)
        # There is deliberately no HQ image/latent or flow target in this batch.
        return dict(lq_views=torch.from_numpy(np.stack(images)).permute(0,3,1,2).float()/255,
                    pixel_masks=torch.from_numpy(np.stack(masks)),
                    distractor_views=torch.tensor([False,True,True,False]),
                    scene=scene_name,pair_ids=ids,input_paths=paths)

    def batch(self, step, rank, batch_size):
        # Local RNG: deterministic resume and no changes to synthetic sampling RNG.
        rng = random.Random(self.seed+step*1000003+rank*7919)
        groups=[]
        for _ in range(batch_size):
            scene=rng.choice(sorted(self.manifest['scenes']))
            anchor=rng.choice(self.manifest['scenes'][scene]['train_ids'])
            groups.append(self.group(scene,'train',anchor,rng.getrandbits(63)))
        return {key:torch.stack([g[key] for g in groups]) for key in ['lq_views','pixel_masks','distractor_views']}

    def validation_groups(self):
        return [(scene,anchor,self.seed+index) for scene in sorted(self.manifest['scenes'])
                for index,anchor in enumerate(self.manifest['scenes'][scene]['val_ids'][:8])]


def encode_real_batch(batch, models, cfg, device, return_reference=False):
    images=batch['lq_views'].to(device)
    mean=images.new_tensor([.485,.456,.406])[None,None,:,None,None]
    std=images.new_tensor([.229,.224,.225])[None,None,:,None,None]
    from mvr.da3_feature_training import encode_training_features
    latent,reference_indices=encode_training_features(models['encoder'],(images-mean)/std,cfg.mvrm.train)
    targets=mask_targets(batch['pixel_masks'].to(device),batch['distractor_views'].to(device),
                         reference_indices,resolution=cfg.stage_2.params.get('mask_head_resolution','patch'))
    result=(latent,targets,tuple(images.shape[-2:]))
    return (*result,reference_indices) if return_reference else result


@torch.no_grad()
def evaluate_real_masks(source,models,cfg,device,rank,world_size,max_groups=None,mask_output_dir=None):
    groups=source.validation_groups()
    if max_groups is not None:groups=groups[:max_groups]
    stats=torch.zeros(5,device=device,dtype=torch.float64)
    with torch.random.fork_rng(devices=[device.index]):
        for i in range(rank,len(groups),world_size):
            scene,anchor,seed=groups[i];g=source.group(scene,'val',anchor,seed)
            batch={key:g[key][None] for key in ['lq_views','pixel_masks','distractor_views']}
            latent,target,hw,reference=encode_real_batch(batch,models,cfg,device,return_reference=True)
            torch.manual_seed(seed)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg.training.precision=='bf16'):
                logits=models['ema_denoiser'](fixed_t1_input(latent,cfg.mvrm.noise_lvl),
                    latent.new_ones(1),hw,mask_only=True)['mask_logits']
            if mask_output_dir is not None:
                from mvr.mask_prediction_outputs import save_mask_grid
                save_mask_grid(Path(mask_output_dir)/scene/f'anchor_{anchor:03d}',batch['lq_views'],
                    batch['pixel_masks'],batch['distractor_views'],reference,logits,
                    dict(source='real',scene=scene,pair_ids=g['pair_ids'],weights='EMA'))
            stats[0]+=annotated_mask_bce(logits,target,torch.ones_like(target,dtype=torch.bool)).double()
            stats[1]+=1;pred=logits>=0;truth=target>=.5
            stats[2]+=(pred&truth).sum();stats[3]+=(pred&~truth).sum();stats[4]+=(~pred&truth).sum()
    torch.distributed.all_reduce(stats)
    return dict(real_mask_bce=float(stats[0]/stats[1].clamp_min(1)),
                real_mask_iou=float(stats[2]/(stats[2]+stats[3]+stats[4]).clamp_min(1)),
                real_mask_validation_groups=int(stats[1]))
