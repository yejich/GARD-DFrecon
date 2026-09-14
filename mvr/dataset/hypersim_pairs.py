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


# Compatibility import for existing manifest-building commands.
from .hypersim_manifest import build_manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build_manifest(args.root, args.output)
