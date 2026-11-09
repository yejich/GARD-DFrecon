"""Numerical checks for the Jihye self-GT/cross-key ablation."""
from types import SimpleNamespace
import unittest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pathlib import Path
from mvr.nvs_training import geometry_correspondence_supcon_loss


class CrossViewSupConTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.source = torch.randn(1, 3, 3, 5, requires_grad=True)
        self.teacher = torch.randn(1, 4, 3, 5, requires_grad=True)
        self.pred = torch.randn(1, 4, 3, 5, requires_grad=True)
        self.batch = SimpleNamespace(model_input=torch.randn(1, 4, 3, 5),
                                     source_condition=torch.zeros(1, 4, 3, 5),
                                     time=torch.tensor([.5]))
        self.mask = torch.zeros(1, 3, 3, dtype=torch.bool)
        self.mask[0, 0, 1] = True
        self.mask[0, 1, 2] = True
        self.geo = torch.zeros(1, 6, 6)
        self.geo[:, :, ::2] = 1.  # First patch is correspondence in each view.
        self.visible = torch.ones_like(self.geo, dtype=torch.bool)

    def loss(self, mode='clean_gt', **kwargs):
        options = dict(anchor_scope='all', cross_positive_source=mode,
                       positive_reduction='self_plus_cross_mean',
                       max_negative_tokens=None, temperature=.7, time_weight='none')
        options.update(kwargs)
        return geometry_correspondence_supcon_loss(
            self.pred, self.batch, self.source, self.teacher, self.mask,
            (self.geo, None, self.visible), **options)

    def reference(self, mode, legacy=False):
        anchors = F.normalize((self.batch.model_input - .5*self.pred)[0, :3, 1:], dim=-1)
        teacher = F.normalize(self.teacher[0, :3, 1:].detach(), dim=-1)
        observed = F.normalize(self.source[0, :, 1:].detach(), dim=-1)
        cross = teacher if mode == 'clean_gt' else observed
        negatives = observed[self.mask[0, :, 1:]]
        values = []
        for view in range(3):
            for patch in range(2):
                if legacy and not self.mask[0, view, patch+1]:
                    continue
                positives = [teacher[view, patch]]
                for other in range(3):
                    if other != view and not self.mask[0, other, 1] and self.visible[0, view*2+patch, other*2]:
                        positives.append(cross[other, 0])
                keys = torch.cat([torch.stack(positives), negatives])
                nll = -(keys @ anchors[view, patch] / .7).log_softmax(0)[:len(positives)]
                values.append(nll.mean() if legacy else nll[0] + (nll[1:].mean() if len(positives)>1 else 0.))
        return torch.stack(values).mean()

    def test_exact_formula_both_sources_and_gradient_boundaries(self):
        for mode in ('clean_gt', 'observed'):
            with self.subTest(mode=mode):
                self.pred.grad = None
                loss = self.loss(mode)
                torch.testing.assert_close(loss, self.reference(mode))
                loss.backward()
                self.assertIsNone(self.source.grad)
                self.assertIsNone(self.teacher.grad)
                self.assertTrue(torch.isfinite(self.pred.grad).all())
                self.assertGreater(self.pred.grad[0, 2, 1:].abs().sum().item(), 0.)  # clean anchors
                self.assertEqual(self.pred.grad[0, 3].abs().sum().item(), 0.)  # novel
                self.assertEqual(self.pred.grad[:, :, 0].abs().sum().item(), 0.)  # CLS
        self.assertNotAlmostEqual(self.loss('clean_gt').item(), self.loss('observed').item())

    def test_no_cross_support_keeps_gt_self_in_both_modes(self):
        self.visible.zero_()
        torch.testing.assert_close(self.loss('clean_gt'), self.reference('clean_gt'))
        torch.testing.assert_close(self.loss('observed'), self.loss('clean_gt'))

    def test_no_negatives_is_differentiable_zero(self):
        self.mask.zero_()
        loss = self.loss()
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(self.pred.grad.abs().sum().item(), 0.)

    def test_legacy_defaults_match_original_mean(self):
        loss = geometry_correspondence_supcon_loss(
            self.pred, self.batch, self.source, self.teacher, self.mask,
            (self.geo, None, self.visible), temperature=.7, time_weight='none')
        torch.testing.assert_close(loss, self.reference('clean_gt', legacy=True))

    def test_negative_cap_one_and_invalid_options(self):
        self.assertTrue(torch.isfinite(self.loss(max_negative_tokens=1)))
        for kwargs in (dict(anchor_scope='bad'), dict(cross_positive_source='bad'),
                       dict(positive_reduction='bad'), dict(include_self_positive=False)):
            with self.assertRaises(ValueError):
                self.loss(**kwargs)

    def test_training_step_four_variants(self):
        from mvr.nvs_training import NVSGARD, nvs_training_step
        for mode in ('clean_gt', 'observed'):
            for nvs in (True, False):
                with self.subTest(mode=mode, nvs=nvs):
                    model = NVSGARD(in_channels=16, hidden_size=[32, 32], depth=[2, 2],
                                    num_heads=[2, 2], mlp_ratio=2., use_pos_embed=False)
                    model._nonreentrant_checkpoint = True
                    source = torch.randn(1, 3, 5, 16)
                    teacher = torch.randn(1, 4, 5, 16)
                    mask = torch.zeros(1, 3, 5, dtype=torch.bool)
                    mask[0, 0, 1] = True
                    cfg = dict(use=True, anchor_scope='all', cross_positive_source=mode,
                               include_self_positive=True, positive_reduction='self_plus_cross_mean')
                    geo = torch.eye(16)[None]
                    result = nvs_training_step(
                        model, source, teacher, torch.randn(1, 1, 4, 6),
                        torch.tensor([.5]), (28, 28), nvs_branch=nvs,
                        source_distractor_mask=mask,
                        alignment_aux=(geo, torch.zeros_like(geo), torch.ones_like(geo, dtype=torch.bool)),
                        geometry_contrastive_cfg=cfg)
                    self.assertTrue(torch.isfinite(result['loss']))
                    self.assertGreater(result['geometry_contrastive_loss'].item(), 0.)
                    result['loss'].backward()
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_configs_only_change_supcon_from_previous_handoff(self):
        root = Path(__file__).resolve().parents[1]
        base = OmegaConf.to_container(OmegaConf.load(root/'run_configs/train/train_GARD_da3_nvs_v4_n2_3_mixed_supcon_softself.yaml'))
        for mode in ('clean_gt', 'observed'):
            cfg = OmegaConf.to_container(OmegaConf.load(root/f'run_configs/train/crossview_supcon_20261109/{mode}.yaml'))
            gc = cfg['geometry_contrastive_loss']
            self.assertEqual(gc.pop('anchor_scope'), 'all')
            self.assertEqual(gc.pop('cross_positive_source'), mode)
            self.assertTrue(gc.pop('include_self_positive'))
            self.assertEqual(gc.pop('positive_reduction'), 'self_plus_cross_mean')
            self.assertEqual(cfg, base)
