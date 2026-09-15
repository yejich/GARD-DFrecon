import unittest

import torch

from mvr.distractor_attention import distractor_aware_target, distractor_aware_attn_loss
from mvr.grouped_mse import token_groups


class DA3DistractorAttentionTests(unittest.TestCase):
    def test_reference_order_cls_exclusion_and_backward(self):
        # Original view 1 becomes the reference and contains a distractor patch.
        pixels = torch.zeros(1, 2, 14, 28)
        pixels[0, 1, :, :14] = 1
        masks, _ = token_groups(pixels, torch.tensor([[False, True]]), torch.tensor([1]))
        spatial = masks[:, :, 1:]
        self.assertEqual(spatial.tolist(), [[[True, False], [False, False]]])
        distances = torch.zeros(1, 4, 4)
        visibility = torch.ones_like(distances, dtype=torch.bool)
        target, valid = distractor_aware_target(
            distances, distances.softmax(-1), visibility, spatial, .01, {})
        # Distractor query can use only the other view's clean patches.
        torch.testing.assert_close(target[0, 0], torch.tensor([0., 0., .5, .5]))
        self.assertTrue((target[..., 0] == 0).all())
        logits = torch.zeros(1, 6, 6, requires_grad=True)  # CLS + 2 patches per view
        pred = logits.softmax(-1)[:, [1, 2, 4, 5]][:, :, [1, 2, 4, 5]]
        loss = distractor_aware_attn_loss(pred, target, valid, spatial, {})
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0, 1, 1].item(), 0.)
        self.assertLess(logits.grad[0, 1, 4].item(), 0.)
        # Predictions retain CLS mass, so attention spent on CLS is penalized too.
        self.assertGreater(logits.grad[0, 1, 0].item(), 0.)


if __name__ == '__main__':
    unittest.main()
