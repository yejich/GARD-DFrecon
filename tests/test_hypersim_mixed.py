import json
import os
from pathlib import Path
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
from PIL import Image

from mvr.dataset.hypersim_mixed import MixedHypersimPairs, MixedPairBatchSampler
from mvr.dataset.hypersim_pairs import pair_collate_fn, load_train_data
from mvr.dataset.hypersim_manifest import fingerprint


class MixedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = []
        for source in ('original', 'negative'):
            root = self.root/source
            rows, groups = [], []
            for scene in ('train_a', 'train_b', 'held_out'):
                for mode in (('original',) if source == 'original' else ('shared_asset', 'independent_assets')):
                    indices = list(range(len(rows), len(rows)+4))
                    group = f'{scene}/static_00'
                    groups.append(dict(id=group, mode=mode, indices=indices))
                    for frame, i in enumerate(indices):
                        directory = f'{group}/{mode}/frame.{frame:04d}'
                        rows.append(dict(directory=directory, scene=scene, camera='cam_00', frame=frame,
                                         background_variant_id=group, distractor_variant=mode,
                                         candidates=[j for j in indices if i != j]))
                        folder = root/directory
                        folder.mkdir(parents=True)
                        for name, value in [('clean.png', 40 if source == 'original' else 180),
                                            ('distractor.png', 230)]:
                            Image.fromarray(np.full((28, 28, 3), value, np.uint8)).save(folder/name)
                        for name in ('mask_object.png', 'mask_shadow.png'):
                            Image.fromarray(np.full((28, 28), 255, np.uint8)).save(folder/name)
            manifest = dict(root=str(root), train=rows, eval=[], eval_groups=[],
                            eval_scene='held_out', eval_scenes=['held_out'], minimum_covisibility=.25,
                            distractor_probability=.7, seed=42, eval_views=4)
            if source == 'negative':
                manifest['negative_fixed_groups'] = groups
            path = self.root/f'{source}.json'
            path.write_text(json.dumps(manifest))
            self.paths.append(path)

    def dataset(self):
        return MixedHypersimPairs(*self.paths, process_res=28)

    def test_variable_nvs_fixed_teacher_and_variant_coverage(self):
        from mvr.dataset.hypersim_nvs import MixedHypersimNVS, mixed_nvs_sampler, nvs_collate_fn
        ds = self.dataset()
        wrapped = MixedHypersimNVS(ds)
        samplers = [mixed_nvs_sampler(ds, 1, rank, 2, accumulation=4, original_samples=100)
                    for rank in range(2)]
        batches = [list(s) for s in samplers]
        self.assertEqual({b[0][2] for b in batches[0]}, {2, 3})
        visits = Counter()
        lookup = {i: (g['id'], g['mode']) for g in ds.negative_groups for i in g['indices']}
        for aligned in zip(*batches):
            self.assertEqual(len({b[0][2] for b in aligned}), 1)
            for tokens in aligned:
                token = tokens[0]
                sample = wrapped[token]
                n = token[2]
                self.assertEqual(len(sample['hq_views']) + len(sample['nvs_hq_views']), 4)
                ids = sample['frame_ids'] + sample['nvs_frame_ids']
                self.assertEqual(len(set(ids)), 4)
                self.assertEqual(len(sample['camera_records']), 4)
                batch = nvs_collate_fn([sample])
                self.assertEqual(batch['lq_views'].shape[1], n)
                self.assertEqual(batch['nvs_hq_views'].shape[1], 4-n)
                if token[0] == 1: visits[lookup[token[1]]] += 1
        self.assertEqual(visits, Counter({(g['id'], g['mode']): 1 for g in ds.negative_groups}))
        self.assertEqual(batches[0], list(samplers[0]))

    def test_variable_nvs_rejects_original_anchors_without_four_views(self):
        from mvr.dataset.hypersim_nvs import mixed_nvs_sampler
        ds = self.dataset()
        for record in ds.datasets[0].records: record['candidates'] = record['candidates'][:2]
        with self.assertRaisesRegex(ValueError, 'No original anchors support V=4'):
            mixed_nvs_sampler(ds, 1, 0, 1, original_samples=4)

    def test_global_coverage_and_ddp_alignment(self):
        for world, batch, accumulation in [(1, 1, 4), (2, 2, 3), (4, 1, 4)]:
            ds = self.dataset()
            samplers = [MixedPairBatchSampler(ds, batch, rank, world, accumulation,
                                             original_samples=7) for rank in range(world)]
            all_batches = [list(s) for s in samplers]
            self.assertEqual({len(b) for b in all_batches}, {len(samplers[0])})
            self.assertEqual(len(samplers[0]) % accumulation, 0)
            for batches in zip(*all_batches):
                self.assertEqual(len({t[2] for b in batches for t in b}), 1)
            tokens = [t for batches in all_batches for b in batches for t in b]
            lookup = {i: (g['id'], g['mode']) for g in ds.negative_groups for i in g['indices']}
            visits = Counter(lookup[t[1]] for t in tokens if t[0] == 1)
            self.assertEqual(visits, Counter({(g['id'], g['mode']): 1 for g in ds.negative_groups}))
            self.assertEqual(sum(t[0] == 0 for t in tokens), samplers[0].original_samples)
            self.assertLess(samplers[0].summary['original_padding'], world*batch*accumulation)

    def test_epoch_reproducibility_and_view_variation(self):
        ds = self.dataset()
        s = MixedPairBatchSampler(ds, 1, 0, 1, original_samples=100)
        first = list(s)
        self.assertEqual(first, list(s))
        self.assertEqual({b[0][2] for b in first}, {1, 2, 3, 4})
        s.set_epoch(1)
        self.assertNotEqual(first, list(s))
        self.assertEqual(list(s), list(s))

    def test_held_out_and_source_roots_masks_ids(self):
        with patch.dict(os.environ, {'HYPERSIM_PAIRS_ROOT': str(self.root/'original')}):
            ds = self.dataset()
        self.assertEqual(ds.datasets[1].root, self.root/'negative')
        self.assertTrue(all(r['scene'] != 'held_out' for d in ds.datasets for r in d.records))
        for d in ds.datasets:
            for r in d.records:
                self.assertTrue(all(d.records[i]['scene'] == r['scene'] for i in r['candidates']))
        ds.datasets[1].manifest['distractor_probability'] = 0
        clean = ds[(1, 0, 4, 5)]
        self.assertTrue(all(np.all(img == 180) for img in clean['hq_views']))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(clean['hq_views'], clean['lq_views'])))
        self.assertFalse(clean['pixel_masks'].any())
        ds.datasets[1].manifest['distractor_probability'] = 1
        clutter = ds[(1, 0, 4, 5)]
        self.assertTrue(all(np.all(img == 230) for img in clutter['lq_views']))
        self.assertTrue(clutter['pixel_masks'].all())
        batch = pair_collate_fn([ds[(0, 0, 4, 5)], clutter])
        self.assertEqual(tuple(batch['hq_views'].shape), (2, 4, 3, 28, 28))
        self.assertEqual(batch['data_source'], ['original', 'negative'])
        self.assertTrue(clutter['hq_ids'][0].startswith('negative::'))

    def test_reject_cross_variant_candidates(self):
        m = json.loads(self.paths[1].read_text())
        m['train'][0]['candidates'] = [1, 2, 4]
        self.paths[1].write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'candidates must stay'):
            self.dataset()

    def test_reject_missing_variant(self):
        m = json.loads(self.paths[1].read_text())
        m['negative_fixed_groups'].pop(0)
        self.paths[1].write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            self.dataset()

    def test_reject_ineligible_variant(self):
        ds = self.dataset()
        for i in ds.negative_groups[0]['indices']:
            ds.datasets[1].records[i]['candidates'] = []
        with self.assertRaisesRegex(ValueError, 'cannot support'):
            MixedPairBatchSampler(ds, 1, 0, 1)

    def test_fingerprint_tracks_sampling_budget(self):
        ds = self.dataset()
        MixedPairBatchSampler(ds, 1, 0, 1, original_samples=7)
        before = fingerprint(ds.manifest)
        MixedPairBatchSampler(ds, 1, 0, 1, original_samples=17)
        self.assertNotEqual(before, fingerprint(ds.manifest))

    def test_train_loader_dispatch(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(dict(data=dict(train=dict(min_num_input_view=1, max_num_input_view=4,
            pairs=dict(manifest=str(self.paths[0]), negative_manifest=str(self.paths[1]),
                       original_groups_per_epoch=7, process_res=28))),
            training=dict(grad_accum_steps=2, global_seed=42, num_workers=0)))
        loader, sampler = load_train_data(cfg, 2, 0, 1)
        self.assertIsInstance(loader.dataset, MixedHypersimPairs)
        self.assertEqual(len(loader), len(sampler))
        self.assertEqual(next(iter(loader))['hq_views'].shape[0], 2)


if __name__ == '__main__':
    unittest.main()
