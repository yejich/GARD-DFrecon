import os
from pathlib import Path
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from mvr.dataset.hypersim_pairs import load_train_data
from mvr.omega_training import load_train_data_omega


class AccumulationConfigTests(unittest.TestCase):
    def test_real_loaders_with_environment_accumulation(self):
        root = Path(__file__).resolve().parents[1]
        for model, loader_fn in [('da3', load_train_data), ('omega', load_train_data_omega)]:
            for world_size in (2, 4, 8):
                accumulation = 8 // world_size
                with self.subTest(model=model, gpus=world_size), patch.dict(
                        os.environ, GARD_GRAD_ACCUM_STEPS=str(accumulation)):
                    cfg = OmegaConf.load(root / 'run_configs' / 'JIHYE' /
                                         f'train_GARD_{model}_hypersim_completed_260915.yaml')
                    cfg.data.train.pairs.manifest = str(root / 'manifests/hypersim_hf_a935b4e251fa.json')
                    self.assertIsInstance(cfg.training.grad_accum_steps, int)
                    # Include configs saved before the typed environment resolver fix.
                    for value in (accumulation, str(accumulation)):
                        cfg.training.grad_accum_steps = value
                        loader, sampler = loader_fn(cfg, 1, 0, world_size)
                        self.assertEqual(len(loader), 2610 * accumulation)
                        self.assertEqual(len(sampler) % accumulation, 0)
                        batch = next(iter(sampler))
                        self.assertEqual(len(batch), 1)
                        self.assertIn(batch[0][1], (1, 2, 3, 4))


if __name__ == '__main__':
    unittest.main()
