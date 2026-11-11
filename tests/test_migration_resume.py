import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from scripts.train.migration.prepare_resume import prepare


class MigrationResumeTests(unittest.TestCase):
    def test_preserves_experiment_and_requires_same_world_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = Path(tmp); (b/'checkpoints').mkdir(); (b/'manifests').mkdir()
            csv = b/'camera.csv'; csv.write_text('scene_name\n')
            cfg = dict(data=dict(root='/old/pairs', negative_root='/old/negative'),
                       camera=dict(root='/old/cameras', parameters='/old/camera.csv'),
                       per_gpu_batch_size=1, accumulation=8, epochs=30,
                       group_mse=dict(camera_mode='separate', lambda_camera=1.),
                       geometry_contrastive_loss=dict(use=True, lambda_coeff=.05))
            state = dict(model={}, ema={}, optimizer={'state': {}}, rng=[{}, {}],
                         step=15000, epoch=12, next_batch=6048, config=cfg)
            checkpoint = b/'checkpoints/resume.pt'; torch.save(state, checkpoint)
            hashes = {}
            for name in ('transient_only.json','transient_static.json','scene_split_policy.json'):
                p = b/'manifests'/name; p.write_text('{}')
                hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
            (b/'migration.json').write_text(json.dumps(dict(checkpoint_bytes=checkpoint.stat().st_size,
                                                           manifest_sha256=hashes)))
            env = dict(HYPERSIM_PAIRS_ROOT=str(b),HYPERSIM_NEGATIVE_ROOT=str(b),
                       HYPERSIM_CAMERA_ROOT=str(b),HYPERSIM_CAMERA_PARAMETERS=str(csv))
            output = b/'resumed/config.yaml'
            with patch.dict('os.environ',env):
                prepare(b,output,2)
                result = OmegaConf.load(output)
                for key in ('per_gpu_batch_size','accumulation','epochs','group_mse','geometry_contrastive_loss'):
                    self.assertEqual(OmegaConf.to_container(result)[key],cfg[key])
                self.assertEqual(result.checkpoint,str(checkpoint))
                self.assertEqual(result.data.root,str(b))
                with self.assertRaisesRegex(ValueError,'requires 2 GPUs'):prepare(b,output,4)
                (b/'manifests/transient_only.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'Manifest changed'):prepare(b,output,2)
