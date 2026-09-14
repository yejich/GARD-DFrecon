import json
import random
import unittest
from pathlib import Path
import numpy as np
from mvr.dataset.hypersim_pairs import HypersimPairs, PairBatchSampler
from mvr.dataset.pho_concat_ds import multiview_collate_fn

MANIFEST = Path(__file__).resolve().parents[1] / 'manifests/hypersim_001_groups.json'

class PairTests(unittest.TestCase):
    def setUp(self):
        self.ds = HypersimPairs(MANIFEST)

    def test_split_and_all_edges(self):
        m = self.ds.manifest
        self.assertEqual((len(m['train']),len(m['eval'])),(6743,80))
        for split, records in [('train',m['train']),('eval',m['eval'])]:
            for r in records:
                self.assertEqual(r['scene']=='ai_001_001', split=='eval')
                self.assertEqual(len(r['candidates']),len(set(r['candidates'])))
                for i in r['candidates']:
                    c=records[i]
                    self.assertEqual((r['scene'],r['camera']),(c['scene'],c['camera']))
                    self.assertNotEqual(r['frame'],c['frame'])

    def test_ddp_equal_steps_dynamic_views_repeated_anchors(self):
        ss=[PairBatchSampler(self.ds,1,r,4) for r in range(4)]
        batches=[list(s) for s in ss]
        self.assertEqual(len(batches[0])%4,0)
        self.assertEqual({len(x) for x in batches},{len(batches[0])})
        self.assertEqual([b[0][1] for b in batches[0]],[b[0][1] for b in batches[3]])
        self.assertNotEqual(batches[0],batches[1])
        ss[0].set_epoch(1)
        self.assertNotEqual(list(ss[0]),batches[0])
        self.assertEqual(len(list(ss[0])),len(ss[0]))

    def test_two_gpu_accumulation(self):
        sampler=PairBatchSampler(self.ds,1,0,2,accumulation=8)
        self.assertEqual(len(sampler)%8,0)
        self.assertEqual(2*1*8,16)

    def test_p07_and_distinct_random_views(self):
        sampler=PairBatchSampler(self.ds,1,0,4)
        flags=[]
        for batch in sampler:
            anchor,n,seed=batch[0];rng=random.Random(seed)
            indices=[anchor]+rng.sample(self.ds.records[anchor]['candidates'],n-1)
            self.assertEqual(len(set(indices)),n)
            flags.extend(rng.random()<.7 for _ in indices)
        self.assertLess(abs(np.mean(flags)-.7),.025)

    def test_actual_images_and_fixed_eval(self):
        if not self.ds.root.is_dir():
            self.skipTest("Set HYPERSIM_PAIRS_ROOT to enable real-image integration test")
        sampler=PairBatchSampler(self.ds,1,0,4,min_views=4)
        item=self.ds[next(iter(sampler))[0]]
        collated=multiview_collate_fn([item])
        self.assertEqual(tuple(collated['hq_views'].shape),(1,4,3,378,504))
        self.assertTrue(np.isfinite(collated['lq_views'].numpy()).all())
        ev=HypersimPairs(MANIFEST,split='eval')
        a,b=ev[0],ev[0]
        self.assertEqual(a['lq_ids'],b['lq_ids'])
        for x,y in zip(a['lq_views'],b['lq_views']):np.testing.assert_array_equal(x,y)
        self.assertEqual(len(ev.manifest['eval_groups']),80)

if __name__=='__main__':unittest.main()
