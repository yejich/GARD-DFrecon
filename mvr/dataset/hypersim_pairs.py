"""Saved Hypersim pairs: clean covisibility sampling, independent LQ switches."""
import argparse
import json
import os
import random
from pathlib import Path
from collections import Counter

import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from .pho_concat_ds import multiview_collate_fn


class HypersimPairs(Dataset):
    ds_name = 'hypersim_pairs'

    def __init__(self, manifest, split='train', process_res=504, patch_size=14):
        self.manifest = json.loads(Path(manifest).read_text())
        self.root = Path(os.environ.get('HYPERSIM_PAIRS_ROOT', self.manifest['root'])).expanduser()
        self.records = self.manifest[split]
        self.split = split
        self.process_res, self.patch_size = process_res, patch_size
        self.datasets = [self]  # Training logger compatibility.

    def __len__(self):
        return len(self.records)

    def image(self, record, name):
        with Image.open(self.root / record['directory'] / name) as image:
            image = np.array(image.convert('RGB'))
        h, w = image.shape[:2]
        scale = self.process_res / max(h, w)
        nw, nh = round(w * scale), round(h * scale)
        p = self.patch_size
        nw, nh = max(p, ((nw + p//2)//p)*p), max(p, ((nh + p//2)//p)*p)
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    def __getitem__(self, item):
        if self.split == 'eval':
            group = self.manifest['eval_groups'][int(item)]
            indices, flags = group['indices'], group['distractor']
        else:
            anchor, n, seed = item
            rng = random.Random(seed)
            indices = [anchor] + rng.sample(self.records[anchor]['candidates'], n - 1)
            flags = [rng.random() < self.manifest['distractor_probability'] for _ in indices]
        records = [self.records[i] for i in indices]
        hq = [self.image(record, 'clean.png') for record in records]
        lq = [self.image(record, 'distractor.png') if flag else clean.copy()
              for record, flag, clean in zip(records, flags, hq)]
        masks = []
        for record, flag, clean in zip(records, flags, hq):
            mask = np.zeros(clean.shape[:2], dtype=np.float32)
            if flag:
                folder = self.root / record['directory']
                with Image.open(folder / 'mask_object.png') as im:
                    raw = np.array(im) > 0
                with Image.open(folder / 'mask_shadow.png') as im:
                    raw |= np.array(im) > 0
                mask = cv2.resize(raw.astype(np.uint8), (clean.shape[1], clean.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(np.float32)
            masks.append(mask)
        ids = [record['directory'].replace('/', '_') for record in records]
        return dict(frame_ids=indices, hq_ids=ids,
                    lq_ids=[f'{key}_{"distractor" if flag else "clean"}' for key, flag in zip(ids, flags)],
                    hq_views=hq, lq_views=lq, distractor_views=flags, pixel_masks=np.stack(masks))


class PairBatchSampler(Sampler):
    """Same N across ranks; independent random groups, equal full accumulation windows."""
    def __init__(self, dataset, batch_size, rank, world_size, accumulation=4, seed=42, min_views=1, max_views=4):
        self.dataset, self.batch_size, self.rank = dataset, batch_size, rank
        self.seed, self.min_views, self.max_views = seed, min_views, max_views
        self.epoch = 0
        self.batches = len(dataset) // (world_size * batch_size * accumulation) * accumulation
        if self.batches < 1:
            raise ValueError('Insufficient groups for one distributed accumulation window')
        self.pools = {}
        for n in range(min_views, max_views + 1):
            scenes = {}
            for i, record in enumerate(dataset.records):
                if len(record['candidates']) >= n - 1:
                    scenes.setdefault(record['scene'], []).append(i)
            if not scenes:
                raise ValueError(f'No anchor supports N={n}')
            self.pools[n] = scenes

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.batches

    def __iter__(self):
        views_rng = random.Random(self.seed + self.epoch * 1000003)
        rng = random.Random(self.seed + self.epoch * 1000003 + 7919 * (self.rank + 1))
        for _ in range(self.batches):
            n = views_rng.randint(self.min_views, self.max_views)
            scenes = self.pools[n]
            batch = []
            for _ in range(self.batch_size):
                scene = rng.choice(sorted(scenes))
                anchor = rng.choice(scenes[scene])
                batch.append((anchor, n, rng.getrandbits(63)))
            yield batch


def pair_collate_fn(batch):
    output = multiview_collate_fn(batch)
    output['distractor_views'] = torch.tensor([x['distractor_views'] for x in batch], dtype=torch.bool)
    output['pixel_masks'] = torch.from_numpy(np.stack([x['pixel_masks'] for x in batch]))
    return output


def load_train_data(cfg, batch_size, rank, world_size):
    ds = HypersimPairs(cfg.data.train.pairs.manifest)
    sampler = PairBatchSampler(ds, batch_size, rank, world_size,
                               accumulation=cfg.training.grad_accum_steps,
                               seed=cfg.training.global_seed,
                               min_views=cfg.data.train.get('min_num_input_view', 1),
                               max_views=cfg.data.train.max_num_input_view)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.training.num_workers,
                        pin_memory=True, collate_fn=pair_collate_fn)
    return loader, sampler


def build_manifest(root, output, threshold=.25, probability=.7, eval_scene='ai_001_001', seed=42):
    root = Path(root).resolve()
    splits = {'train': [], 'eval': []}
    for complete in sorted(root.glob('*/cam_*/frame.*/complete.json')):
        folder = complete.parent
        if not all((folder / name).is_file() for name in ('clean.png', 'distractor.png')):
            raise ValueError(f'Missing pair images: {folder}')
        relative = folder.relative_to(root)
        scene, camera, frame = relative.parts
        splits['eval' if scene == eval_scene else 'train'].append(
            dict(directory=str(relative), scene=scene, camera=camera, frame=int(frame.split('.')[1])))
    if not splits['train'] or not splits['eval']:
        raise ValueError('Both train and eval splits must be nonempty')
    cache = {}
    for records in splits.values():
        lookup = {(r['scene'], r['camera'], r['frame']): i for i, r in enumerate(records)}
        for record in records:
            key = (record['scene'], record['camera'])
            if key not in cache:
                cache[key] = json.loads((root / key[0] / key[1] / 'clean_multiview.json').read_text())
            candidates = cache[key][str(record['frame'])]['eligible_candidates']
            record['candidates'] = sorted({lookup[(*key, c['frame_id'])] for c in candidates
                if c['covisibility'] >= threshold and (*key, c['frame_id']) in lookup
                and c['frame_id'] != record['frame']})
    rng = random.Random(seed)
    groups = []
    for i, record in enumerate(splits['eval']):
        if len(record['candidates']) >= 3:
            indices = [i] + rng.sample(record['candidates'], 3)
            groups.append(dict(indices=indices, distractor=[rng.random() < probability for _ in indices]))
    data = dict(root=str(root), eval_scene=eval_scene, minimum_covisibility=threshold,
                overlap_definition='fraction of clean anchor GT points visible in candidate; same camera trajectory',
                distractor_probability=probability, seed=seed, eval_views=4, eval_groups=groups, **splits)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    summary = dict(train_pairs=len(splits['train']), eval_pairs=len(splits['eval']),
                   eval_groups=len(groups), split_scene_counts={k: len({r['scene'] for r in v}) for k,v in splits.items()},
                   eligible_anchors={k: {n: sum(len(r['candidates']) >= n-1 for r in v) for n in range(1,5)} for k,v in splits.items()})
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build_manifest(args.root, args.output)
