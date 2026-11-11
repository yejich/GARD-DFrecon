import unittest
import torch
from mvr.grouped_mse import grouped_mse, camera_loss_options
from mvr.nvs_training import prepare_nvs_flow, nvs_flow_losses, NVSLossTracker


class CameraFlowLossTests(unittest.TestCase):
    def example(self, novel=True):
        source = torch.zeros(1, 2, 3, 1)
        teacher = torch.zeros(1, 3 if novel else 2, 3, 1)
        batch = prepare_nvs_flow(source, teacher, torch.ones(1), noise=torch.zeros_like(teacher))
        pred = torch.tensor([[[[2.], [1.], [3.]], [[2.], [1.], [3.]], [[4.], [5.], [5.]]]])
        pred = pred[:, :teacher.shape[1]].clone().requires_grad_()
        mask = torch.tensor([[[False, False, True], [False, False, True]]])
        return batch, pred, mask

    def test_explicit_joint_matches_legacy_value_and_gradient(self):
        batch, pred, mask = self.example()
        kwargs = dict(source_distractor_mask=mask, novel_weight=.5)
        a = nvs_flow_losses(pred, batch, **kwargs)['loss']
        b = nvs_flow_losses(pred, batch, group_mse_cfg=dict(camera_mode='joint', novel_camera_mode='joint'), **kwargs)['loss']
        self.assertEqual(a.item(), 22.5)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(a, pred, retain_graph=True)[0], torch.autograd.grad(b, pred)[0], rtol=0, atol=0)

    def test_separate_values_and_no_double_count_camera_gradient(self):
        batch, pred, mask = self.example()
        cfg = dict(camera_mode='separate', lambda_camera=2., novel_camera_mode='separate', lambda_novel_camera=3.)
        losses = nvs_flow_losses(pred, batch, source_distractor_mask=mask, group_mse_cfg=cfg, novel_weight=.5)
        for key, expected in [('restoration_clean_loss', 1), ('restoration_distractor_loss', 9), ('restoration_camera_loss', 4), ('restoration_loss', 18), ('nvs_camera_loss', 16), ('nvs_patch_loss', 25), ('nvs_loss', 73), ('weighted_nvs_camera_loss', 24), ('loss', 54.5)]:
            self.assertEqual(losses[key].item(), expected)
        losses['loss'].backward()
        torch.testing.assert_close(pred.grad[0, :, :, 0], torch.tensor([[4., 1., 3.], [4., 1., 3.], [12., 2.5, 2.5]]))
        tracker = NVSLossTracker(); tracker.update(losses, 1)
        self.assertEqual(tracker.compute(distributed=False)['train/weighted_nvs_camera_loss'], 24.)

    def test_source_only_ablation_leaves_novel_loss_unchanged(self):
        batch, pred, mask = self.example()
        losses = nvs_flow_losses(pred, batch, source_distractor_mask=mask, group_mse_cfg=dict(camera_mode='separate', lambda_camera=2.), novel_weight=.5)
        self.assertEqual(losses['loss'].item(), 29.)
        self.assertEqual(losses['nvs_loss'].item(), 22.)

    def test_zero_camera_weight_has_zero_camera_gradient_without_novel_views(self):
        batch, pred, mask = self.example(novel=False)
        cfg = dict(camera_mode='separate', lambda_camera=0., novel_camera_mode='separate')
        losses = nvs_flow_losses(pred, batch, source_distractor_mask=mask, group_mse_cfg=cfg)
        self.assertEqual(losses['loss'].item(), 10.)
        self.assertEqual(losses['nvs_loss'].item(), 0.)
        losses['loss'].backward()
        self.assertEqual(pred.grad[:, :, 0].abs().sum().item(), 0.)

    def test_camera_weight_independent_of_patch_count_and_empty_clean_group(self):
        cfg = dict(camera_mode='separate')
        for count in [1, 9]:
            pred = torch.tensor([2.] + [3.]*count).reshape(1, 1, -1, 1).requires_grad_()
            mask = torch.tensor([False]+[True]*count).reshape(1, 1, -1)
            loss, parts = grouped_mse(pred, torch.zeros_like(pred), mask, mask.any(-1), cfg)
            self.assertEqual(loss.item(), 13.)
            self.assertEqual(parts['clean'].item(), 0.)
            loss.backward(); self.assertEqual(pred.grad[0, 0, 0, 0].item(), 4.)

    def test_per_sample_mean_not_pooled_by_token_count(self):
        pred = torch.tensor([[[[2.], [1.], [1.], [3.]]], [[[4.], [5.], [3.], [3.]]]])
        mask = torch.tensor([[[False,False,False,True]],[[False,False,True,True]]])
        loss, parts = grouped_mse(pred, torch.zeros_like(pred), mask, mask.any(-1), dict(camera_mode='separate'))
        self.assertEqual(parts['clean'].item(), 13.)
        self.assertEqual(parts['camera'].item(), 10.)
        self.assertEqual(loss.item(), 32.)

    def test_invalid_options_and_full_view_conflict(self):
        for cfg in [dict(camera_mode='bad'),dict(novel_camera_mode='bad'),dict(lambda_camera=-1),dict(lambda_camera=float('nan')),dict(lambda_novel_camera=float('inf')),dict(mode='flat',camera_mode='separate')]:
            with self.assertRaises(ValueError):camera_loss_options(cfg)
        batch, pred, mask = self.example()
        for cfg in [dict(camera_mode='separate'),dict(novel_camera_mode='separate')]:
            with self.assertRaises(ValueError):nvs_flow_losses(pred,batch,group_mse_cfg=cfg,full_view_loss=True)
        actual=nvs_flow_losses(pred,batch,full_view_loss=True)['loss']
        torch.testing.assert_close(actual,pred.square().mean())


if __name__ == '__main__':unittest.main()
