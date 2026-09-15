import json,tempfile,unittest
from pathlib import Path
from mvr.dataset.hypersim_manifest import collect_manifest, fingerprint, PAIR_FILES

class DownloadedManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.add('scenes_v2','ai_001_001')
        self.add('scenes_v2','ai_002_001')

    def add(self,collection,scene,frames=range(4)):
        camera=self.root/collection/scene/'cam_00';camera.mkdir(parents=True,exist_ok=True)
        cache={str(fid):{'eligible_candidates':[{'frame_id':other,'covisibility':.5} for other in range(6) if other!=fid]} for fid in range(6)}
        (camera/'clean_multiview.json').write_text(json.dumps(cache))
        for fid in frames:
            folder=camera/f'frame.{fid:04d}';folder.mkdir(exist_ok=True)
            for name in PAIR_FILES:(folder/name).touch()
        return camera

    def test_expansion_preserves_fixed_evaluation(self):
        before,_=collect_manifest(self.root)
        self.add('scenes_002','ai_001_002')
        after,summary=collect_manifest(self.root)
        self.assertEqual(len(after['train']),8)
        self.assertEqual(after['eval'],before['eval'])
        self.assertEqual(after['eval_groups'],before['eval_groups'])
        self.assertNotEqual(after['fingerprint'],before['fingerprint'])
        self.assertEqual(summary['split_scene_counts'],{'train':2,'eval':1})

    def test_incomplete_views_never_enter_candidates(self):
        camera=self.add('scenes_003','ai_004_003',range(5))
        (camera/'frame.0004/mask_shadow.png').unlink()
        data,summary=collect_manifest(self.root)
        self.assertEqual(summary['skipped_incomplete_pairs'],1)
        records=data['train']
        for record in records:
            for i in record['candidates']:
                candidate=records[i]
                self.assertEqual(candidate['scene'],record['scene'])
                self.assertNotEqual(candidate['frame'],4)

    def test_legacy_collection_is_not_counted_twice(self):
        self.add('scenes','ai_002_001')
        data,_=collect_manifest(self.root)
        self.assertEqual(len(data['train']),4)
        self.add('scenes_001','ai_002_001')
        with self.assertRaisesRegex(ValueError,'Duplicate source frame'):collect_manifest(self.root)

    def test_completion_marker_excludes_in_progress_pair(self):
        for folder in self.root.glob('*/ai_*/cam_*/frame.*'):
            (folder/'complete.json').write_text('{}')
        camera=self.add('scenes_002','ai_003_001',range(5))
        for i in range(4):
            (camera/f'frame.{i:04d}'/'complete.json').write_text('{}')
        # Frame 4 already has every PNG, but the producer has not finished it.
        data,summary=collect_manifest(self.root,require_complete=True)
        self.assertEqual(len(data['train']),8)
        self.assertEqual(summary['skipped_incomplete_pairs'],1)
        self.assertTrue(all(r['frame']!=4 for r in data['train']))
        for r in data['train']:
            self.assertTrue(all(data['train'][i]['frame']!=4 for i in r['candidates']))

    def test_covisibility_cache_required(self):
        (self.root/'scenes_v2/ai_002_001/cam_00/clean_multiview.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError,'clean_multiview.json'):collect_manifest(self.root)

    def test_single_collection_and_relocation(self):
        data,_=collect_manifest(self.root/'scenes_v2')
        self.assertTrue(all(r['directory'].startswith('ai_') for r in data['train']))
        original=fingerprint(data);data['root']='/some/other/server'
        self.assertEqual(fingerprint(data),original)

    def test_holdout_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            camera=Path(d)/'ai_001_002/cam_00/frame.0000';camera.mkdir(parents=True)
            for name in PAIR_FILES:(camera/name).touch()
            with self.assertRaisesRegex(ValueError,'held-out'):collect_manifest(d)

if __name__=='__main__':unittest.main()
