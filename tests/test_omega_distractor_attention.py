import unittest
import torch
from mvr.omega_training import (
    attn_align_loss_omega, distractor_aware_target, distractor_aware_attn_loss,
)


class DistractorAttentionTests(unittest.TestCase):
    def test_target_masks_keys_and_same_view(self):
        mask = torch.tensor([[[True, False], [False, True]]])
        distances = -torch.tensor([[[0., .01, .02, .01], [.01, 0., .01, .01],
                                    [.02, .01, 0., .01], [.01, .01, .01, 0.]]])
        cfg = {'max_correspondence_distance': .03}
        target, valid = distractor_aware_target(distances, (distances / .01).softmax(-1), None, mask, .01, cfg)
        self.assertTrue(valid.all())
        self.assertTrue((target[..., [0, 3]] == 0).all())
        self.assertEqual(target[0, 0, 2].item(), 1.)
        self.assertEqual(target[0, 3, 1].item(), 1.)
        torch.testing.assert_close(target.sum(-1), torch.ones(1, 4))
        self.assertGreater(target[0, 1, 1].item(), 0.)  # clean query can attend itself

    def test_no_visible_or_near_correspondence(self):
        mask = torch.tensor([[[True], [False]]])
        distances = torch.tensor([[[0., -1.], [-1., 0.]]])
        target, valid = distractor_aware_target(distances, distances.softmax(-1), None, mask, .01, {})
        self.assertFalse(valid[0, 0])  # far key must not become a false correspondence
        self.assertEqual(target[0, 0].sum().item(), 0.)
        target, valid = distractor_aware_target(distances, distances.softmax(-1), torch.zeros_like(distances, dtype=torch.bool), mask, .01, {})
        self.assertFalse(valid.any())
        logits = torch.randn(1, 2, 2, requires_grad=True)
        loss = distractor_aware_attn_loss(logits.softmax(-1), target, valid, mask, {})
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad.abs().sum().item(), 0.)

    def test_all_distractor_and_single_view(self):
        for mask in (torch.ones(1, 2, 2, dtype=torch.bool), torch.ones(1, 1, 4, dtype=torch.bool)):
            d = torch.zeros(1, 4, 4)
            target, valid = distractor_aware_target(d, d.softmax(-1), None, mask, .01, {})
            self.assertFalse(valid.any())
            self.assertEqual(target.sum().item(), 0.)

    def test_prediction_still_penalizes_distractor_keys(self):
        # A clean target remains fully supervised despite the corrupt key having zero target mass.
        target = torch.tensor([[[1., 0.], [0., 0.]]])
        valid = torch.tensor([[True, False]])
        mask = torch.tensor([[[False, True]]])
        logits = torch.tensor([[[0., 2.], [0., 0.]]], requires_grad=True)
        loss = distractor_aware_attn_loss(logits.softmax(-1), target, valid, mask, {})
        loss.backward()
        self.assertGreater(logits.grad[0, 0, 1].item(), 0.)
        better = torch.tensor([[[2., 0.], [0., 0.]]]).softmax(-1)
        self.assertGreater(loss.item(), distractor_aware_attn_loss(better, target, valid, mask, {}).item())

    def test_query_weights(self):
        target = torch.eye(2)[None]
        pred = torch.tensor([[[.5, .5], [.25, .75]]])
        mask = torch.tensor([[[True, False]]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        loss = distractor_aware_attn_loss(pred, target, valid, mask, {'lambda_distractor': 2., 'lambda_clean': 3.})
        torch.testing.assert_close(loss, -2 * torch.log(torch.tensor(.5)) - 3 * torch.log(torch.tensor(.75)))

    def test_hard_targets_do_not_invent_new_matches(self):
        d = torch.zeros(1, 2, 2)
        hard = torch.tensor([[[1., 0.], [0., 1.]]])
        mask = torch.tensor([[[True], [False]]])
        target, valid = distractor_aware_target(d, hard, None, mask, -1, {})
        self.assertFalse(valid[0, 0])
        self.assertEqual(target[0, 1, 1].item(), 1.)

    def test_disabled_matches_legacy_and_integration_backward(self):
        depth = torch.ones(1, 2, 16, 16, 1)
        extrinsics = torch.eye(4)[:3].expand(1, 2, 3, 4)
        intrinsics = torch.eye(3).expand(1, 2, 3, 3)
        logits = torch.randn(1, 36, 36, requires_grad=True)
        pred = logits.softmax(-1)
        pc = {'vis_pc_temperature': .01, 'visibility_mask': 'cycle_consistency', 'vis_pc_cycle_threshold': 1.5}
        baseline = attn_align_loss_omega(pred, depth, extrinsics, intrinsics, pc)
        disabled = attn_align_loss_omega(pred, depth, extrinsics, intrinsics, pc,
                                        distractor_cfg={'use': False}, token_mask=torch.ones(1))
        torch.testing.assert_close(baseline, disabled, rtol=0, atol=0)
        # Identical cameras => two coincident patches; legacy target is [0.5, 0.5].
        expected = -(pred[:, [17, 35]][:, :, [17, 35]] + 1e-8).log().mean()
        torch.testing.assert_close(baseline, expected)
        mask = torch.zeros(1, 2, 18, dtype=torch.bool)
        mask[0, 0, 17] = True
        loss = attn_align_loss_omega(pred, depth, extrinsics, intrinsics, pc,
                                    distractor_cfg={'use': True}, token_mask=mask)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0, 17, 17].item(), 0.)
        with self.assertRaisesRegex(ValueError, 'token_mask'):
            attn_align_loss_omega(pred, depth, extrinsics, intrinsics, pc, distractor_cfg={'use': True})


if __name__ == '__main__':
    unittest.main()
