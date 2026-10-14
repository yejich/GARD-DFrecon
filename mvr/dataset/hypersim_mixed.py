"""Original dynamic groups + exactly one visit to every negative variant per epoch."""
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, Sampler

from .hypersim_pairs import HypersimPairs, pair_collate_fn


def filter_scenes(records, held_out):
    mapping = {i: k for k, i in enumerate(i for i, r in enumerate(records)
                                        if r['scene'] not in held_out)}
    rows = []
    for i in mapping:
        row = dict(records[i])
        for c in row['candidates']:
            if not 0 <= c < len(records) or (records[c]['scene'], records[c]['camera']) != (row['scene'], row['camera']):
                raise ValueError('Candidates must reference valid records in the same scene/camera')
        row['candidates'] = [mapping[c] for c in row['candidates'] if c in mapping]
        rows.append(row)
    return rows, mapping


class MixedHypersimPairs(Dataset):
    ds_name = 'hypersim_original_and_negative'

    def __init__(self, original_manifest, negative_manifest, process_res=504, patch_size=14,
                 original_root=None, negative_root=None, scene_split_policy=None, include_object_masks=False):
        original = HypersimPairs(original_manifest, process_res=process_res,
                                 patch_size=patch_size, root=original_root, include_object_masks=include_object_masks)
        # Do not let HYPERSIM_PAIRS_ROOT redirect negative images to the original tree.
        negative = HypersimPairs(negative_manifest, process_res=process_res,
                                 patch_size=patch_size, include_object_masks=include_object_masks)
        negative.root = Path(negative_root or negative.manifest['root']).expanduser()
        held_out = {r['scene'] for r in original.manifest.get('eval', [])}
        held_out.update(negative.manifest.get('eval_scenes', []))
        held_out.update(r['scene'] for r in negative.manifest.get('eval', []))
        if original.manifest.get('eval_scene'):
            held_out.add(original.manifest['eval_scene'])
        if scene_split_policy:
            held_out.update(json.loads(Path(scene_split_policy).read_text())['eval_scenes'])
        elif not negative.manifest.get('eval_scenes'):
            raise ValueError('Use a scene-split negative snapshot or specify scene_split_policy')
        original.records, _ = filter_scenes(original.records, held_out)
        negative.records, index_map = filter_scenes(negative.records, held_out)
        self.negative_groups = []
        for group in negative.manifest['negative_fixed_groups']:
            indices = group['indices']
            if not any(i in index_map for i in indices):
                continue
            if not all(i in index_map for i in indices):
                raise ValueError('A negative group crosses the scene split')
            self.negative_groups.append(dict(group, indices=[index_map[i] for i in indices]))
        self.datasets = [original, negative]
        original.ds_name, negative.ds_name = 'hypersim_original', 'hypersim_negative'
        self._validate_groups()
        if not original.records or not self.negative_groups:
            raise ValueError('Both original records and negative train variants are required')
        # Audit snapshot understood by the trainer's fingerprint/logging path.
        self.manifest = copy.deepcopy(original.manifest)
        offset = len(original.records)
        combined = [dict(r, data_source='original') for r in original.records]
        combined += [dict(r, data_source='negative', candidates=[offset+c for c in r['candidates']])
                     for r in negative.records]
        self.manifest.update(schema_version='mixed_hypersim_v1', train=combined,
                             source_roots={'original': str(original.root), 'negative': str(negative.root)},
                             negative_fixed_groups=[dict(g, indices=[offset+i for i in g['indices']])
                                                    for g in self.negative_groups],
                             mixed_sampling={'held_out_scenes': sorted(held_out),
                                             'negative_probability': negative.manifest['distractor_probability'],
                                             'negative_policy': 'each_group_variant_once'},
                             note='Mixed-loader audit snapshot; source_roots/data_source select image roots.')
        self.epoch_samples = None

    def _validate_groups(self):
        rows = self.datasets[1].records
        seen = set()
        modes = defaultdict(list)
        backgrounds = {}
        for g in self.negative_groups:
            indices = g['indices']
            members = set(indices)
            if len(indices) != 4 or len(members) != 4 or seen & members:
                raise ValueError('Negative variants must contain four distinct, non-reused records')
            if any(i < 0 or i >= len(rows) for i in indices):
                raise ValueError('Invalid negative record index')
            records = [rows[i] for i in indices]
            if len({(r['scene'], r['camera']) for r in records}) != 1 or len({r['frame'] for r in records}) != 4:
                raise ValueError('Negative variant must contain four views from one scene/camera')
            signature = {(r['scene'], r['camera'], r['frame']) for r in records}
            if g['id'] in backgrounds and backgrounds[g['id']] != signature:
                raise ValueError('The two B variants must share the same four HQ views')
            backgrounds[g['id']] = signature
            for i, r in zip(indices, records):
                if r['background_variant_id'] != g['id'] or r['distractor_variant'] != g['mode']:
                    raise ValueError('Negative group/record identity mismatch')
                candidates = r['candidates']
                if len(set(candidates)) != len(candidates) or not set(candidates) <= members - {i}:
                    raise ValueError('Negative candidates must stay inside their static group and B variant')
            seen.update(members)
            modes[g['id']].append(g['mode'])
        if seen != set(range(len(rows))):
            raise ValueError('Negative records missing from fixed groups')
        if any(sorted(v) != ['independent_assets', 'shared_asset'] for v in modes.values()):
            raise ValueError('Each negative static group must have both B variants exactly once')

    def __len__(self):
        return self.epoch_samples if self.epoch_samples is not None else sum(map(len, self.datasets))

    def __getitem__(self, item):
        source, anchor, views, seed = item
        sample = self.datasets[source][(anchor, views, seed)]
        name = 'original' if source == 0 else 'negative'
        sample['data_source'] = name
        # Prevent a static-A target from sharing an ID with the original clean target.
        for key in ('hq_ids', 'lq_ids'):
            sample[key] = [f'{name}::{value}' for value in sample[key]]
        if source == 1:
            sample['frame_ids'] = [i + len(self.datasets[0]) for i in sample['frame_ids']]
        return sample


class MixedPairBatchSampler(Sampler):
    """Global shuffled epoch, sharded across ranks with common N per microbatch.

    Full accumulation windows are padded with ORIGINAL samples only. Negative
    variants are never dropped or repeated for DDP alignment.
    """
    def __init__(self, dataset, batch_size, rank, world_size, accumulation=4, seed=42,
                 min_views=1, max_views=4, original_samples=10000):
        if min(batch_size, world_size, accumulation, original_samples) < 1:
            raise ValueError('Batch/world/accumulation/original_samples must be positive')
        if not 0 <= rank < world_size or not 1 <= min_views <= max_views <= 4:
            raise ValueError('Invalid rank or view range')
        self.dataset, self.batch_size, self.rank, self.world_size = dataset, batch_size, rank, world_size
        self.accumulation, self.seed, self.epoch = accumulation, seed, 0
        self.min_views, self.max_views = min_views, max_views
        self.original_pools, self.negative_anchors = {}, {}
        for n in range(min_views, max_views+1):
            pool = defaultdict(list)
            for i, r in enumerate(dataset.datasets[0].records):
                if len(r['candidates']) >= n-1:
                    pool[r['scene']].append(i)
            if not pool:
                raise ValueError(f'Original data has no eligible {n}-view anchors')
            self.original_pools[n] = dict(sorted(pool.items()))
            eligible = [[i for i in g['indices'] if len(dataset.datasets[1].records[i]['candidates']) >= n-1]
                        for g in dataset.negative_groups]
            if any(not anchors for anchors in eligible):
                raise ValueError(f'A negative variant cannot support {n} views; lower max_num_input_view')
            self.negative_anchors[n] = eligible
        variants = len(dataset.negative_groups)
        window = world_size * batch_size * accumulation
        self.total_samples = math.ceil((original_samples+variants)/window)*window
        self.original_samples = self.total_samples-variants
        self.batches = self.total_samples // (world_size*batch_size)
        self.summary = dict(negative_static_groups=variants//2, negative_variants=variants,
                            original_requested=original_samples, original_samples=self.original_samples,
                            original_padding=self.original_samples-original_samples,
                            total_samples=self.total_samples, microbatches_per_rank=self.batches,
                            optimizer_steps=self.batches//accumulation)
        dataset.epoch_samples = self.total_samples
        dataset.manifest['mixed_sampling'].update(self.summary, seed=seed, min_views=min_views,
                                                  max_views=max_views, world_size=world_size,
                                                  batch_size=batch_size, accumulation=accumulation)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1000003)
        slots = list(range(len(self.dataset.negative_groups))) + [-1]*self.original_samples
        rng.shuffle(slots)
        width = self.world_size*self.batch_size
        for start in range(0, len(slots), width):
            n = rng.randint(self.min_views, self.max_views)
            tokens = []
            for group in slots[start:start+width]:
                if group == -1:
                    pool = self.original_pools[n]
                    scene = rng.choice(list(pool))
                    anchor = rng.choice(pool[scene])
                    source = 0
                else:
                    anchor = rng.choice(self.negative_anchors[n][group])
                    source = 1
                tokens.append((source, anchor, n, rng.getrandbits(63)))
            lo = self.rank*self.batch_size
            yield tokens[lo:lo+self.batch_size]


def load_mixed_train_data(cfg, batch_size, rank, world_size, process_res=504, patch_size=14):
    pairs = cfg.data.train.pairs
    ds = MixedHypersimPairs(pairs.manifest, pairs.negative_manifest,
                           process_res=pairs.get('process_res', process_res), patch_size=patch_size,
                           original_root=pairs.get('root'), negative_root=pairs.get('negative_root'),
                           scene_split_policy=pairs.get('scene_split_policy'),
                           include_object_masks=cfg.get('mvrm', {}).get('loss', {}).get('mask', {}).get('target', 'object_shadow') == 'object')
    sampler = MixedPairBatchSampler(ds, batch_size, rank, world_size,
                                   accumulation=int(cfg.training.grad_accum_steps),
                                   seed=cfg.training.global_seed,
                                   min_views=cfg.data.train.get('min_num_input_view', 1),
                                   max_views=cfg.data.train.max_num_input_view,
                                   original_samples=int(pairs.get('original_groups_per_epoch', 10000)))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg.training.num_workers,
                        pin_memory=True, collate_fn=pair_collate_fn)
    return loader, sampler
