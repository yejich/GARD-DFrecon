"""Omega checkpoint compatibility and distributed EMA regression checks."""
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from mvr.omega_training import build_normalizer
from mvr.pair_training import save_training_checkpoint, resume_training
from train_omega_hypersim_pairs import load_denoiser
from utils.train_utils import update_ema


def tiny_cfg(root):
    return OmegaConf.create(dict(
        stage_1=dict(vggt_omega=dict(ckpt=str(root / 'backbone.pt'))),
        stage_2=dict(target='gard.GARD_omega.GARDOmega', ckpt=None, init_from_da3_ckpt=None,
                     params=dict(in_channels=16, hidden_size=[32, 64], depth=[2, 2],
                                 num_heads=[4, 4], use_pos_embed=False, use_rope=True)),
        mvrm=dict(train=dict(extract_feat_layers=[3]),
                  latent_norm=dict(use=True, stats_path=str(root / 'stats.pt')))))


def fake_encoder():
    return SimpleNamespace(num_special=17, patch_size=16,
                           model=SimpleNamespace(aggregator=SimpleNamespace(camera_token=torch.empty(1, 2, 1, 16))))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = tiny_cfg(self.root)
        (self.root / 'backbone.pt').write_bytes(b'backbone-v1')
        torch.save(dict(mean=torch.randn(18, 16), std=torch.rand(18, 16) + .1, layer=3), self.root / 'stats.pt')
        self.encoder = fake_encoder()

    def normalizer(self, checkpoint=None):
        return build_normalizer(self.cfg, self.encoder, torch.device('cpu'), checkpoint_path=checkpoint)

    def save_norm(self, norm):
        path = self.root / 'latest.pt'
        torch.save(dict(omega_latent_norm=norm.checkpoint_state(), rng=[dict(numpy=np.random.get_state())]), path)
        return path

    def test_embedded_stats_survive_deleted_file(self):
        norm = self.normalizer()
        path = self.save_norm(norm)
        (self.root / 'stats.pt').unlink()
        restored = self.normalizer(path)
        x = torch.randn(1, 4, 21, 16)
        torch.testing.assert_close(restored.normalize(x), norm.normalize(x), rtol=0, atol=0)
        torch.testing.assert_close(restored.denormalize(restored.normalize(x)), x)

    def test_reject_backbone_change(self):
        path = self.save_norm(self.normalizer())
        (self.root / 'backbone.pt').write_bytes(b'backbone-v2')
        with self.assertRaisesRegex(ValueError, 'backbone_sha256'):
            self.normalizer(path)

    def test_reject_layer_change(self):
        path = self.save_norm(self.normalizer())
        self.cfg.mvrm.train.extract_feat_layers = [5]
        with self.assertRaisesRegex(ValueError, 'layer'):
            self.normalizer(path)

    def test_disabled_and_mismatched_mode(self):
        self.cfg.mvrm.latent_norm.use = False
        path = self.save_norm(self.normalizer())
        self.assertFalse(self.normalizer(path).enabled)
        self.cfg.mvrm.latent_norm.use = True
        with self.assertRaisesRegex(ValueError, 'latent_norm.use'):
            self.normalizer(path)

    def test_legacy_needs_original_stats(self):
        path = self.root / 'legacy.pt'
        torch.save(dict(model={}, rng=[dict(numpy=np.random.get_state())]), path)
        with self.assertWarnsRegex(UserWarning, 'Legacy checkpoint'):
            self.normalizer(path)
        (self.root / 'stats.pt').unlink()
        with self.assertRaises(FileNotFoundError):
            self.normalizer(path)

    def test_real_save_and_resume(self):
        norm = self.normalizer()
        model = torch.nn.Linear(2, 2)
        ema = torch.nn.Linear(2, 2)
        models = dict(denoiser=model, ema_denoiser=ema)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        path = self.root / 'latest.pt'
        expected = model.weight.detach().clone()
        with patch('mvr.pair_training.dist.get_world_size', return_value=1), \
             patch('mvr.pair_training.dist.all_gather_object', side_effect=lambda dst, obj: dst.__setitem__(0, obj)), \
             patch('mvr.pair_training.dist.barrier'), \
             patch('mvr.pair_training.torch.cuda.get_rng_state', return_value=torch.get_rng_state()):
            save_training_checkpoint(path, 2, 16, 2, models, optimizer, None, 0,
                                     extra_state={'omega_latent_norm': norm.checkpoint_state()})
        model.weight.data.zero_()
        with patch('mvr.pair_training.dist.get_world_size', return_value=1), \
             patch('mvr.pair_training.torch.cuda.set_rng_state'):
            self.assertEqual(resume_training(path, models, optimizer, None, 0), (2, 16, 2))
        torch.testing.assert_close(model.weight, expected)
        torch.testing.assert_close(self.normalizer(path).mean, norm.mean)


def distributed_check():
    import os
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    # Different initializations reproduce the bug if EMA is copied before DDP.
    torch.manual_seed(123 + rank)
    cfg = tiny_cfg(Path('/tmp'))
    models = load_denoiser(cfg, device, logging.getLogger('test'))
    optimizer = torch.optim.AdamW(models['denoiser'].parameters())
    def assert_synced():
        for p, e in zip(models['denoiser'].parameters(), models['ema_denoiser'].parameters()):
            for value in (p, e):
                ref = value.detach().clone()
                dist.broadcast(ref, src=0)
                torch.testing.assert_close(value, ref, rtol=0, atol=0)
    assert_synced()
    for p, e in zip(models['denoiser'].parameters(), models['ema_denoiser'].parameters()):
        torch.testing.assert_close(p, e, rtol=0, atol=0)
    for _ in range(2):
        out = models['ddp_denoiser'](torch.randn(1, 4, 21, 16, device=device),
                                    torch.ones(1, device=device) * .5, (32, 32))
        (out - 1).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_ema(models['ema_denoiser'], models['denoiser'], .9995)
        assert_synced()
    # Exercise the stage_2.ckpt loader with the RNG-bearing training format.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'latest.pt'
        torch.save(dict(ema=models['ema_denoiser'].state_dict(),
                        rng=[dict(numpy=np.random.get_state())]), path)
        cfg.stage_2.ckpt = str(path)
        reloaded = load_denoiser(cfg, device, logging.getLogger('test'))
        for p, e in zip(reloaded['denoiser'].parameters(), models['ema_denoiser'].parameters()):
            torch.testing.assert_close(p, e, rtol=0, atol=0)
    if rank == 0:
        print('PASS: two-GPU EMA equality, two 4-view optimizer updates, RNG checkpoint weight loading')
    dist.destroy_process_group()


if __name__ == '__main__':
    import sys
    if '--distributed' in sys.argv:
        distributed_check()
    else:
        unittest.main()
