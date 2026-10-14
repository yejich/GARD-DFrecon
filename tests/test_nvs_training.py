import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import torch
from mvr.nvs_training import NVSGARD,prepare_nvs_flow,nvs_flow_losses,target_plucker_rays,NVSLossTracker
from gard.GARD import GARD

def options():return dict(in_channels=16,hidden_size=[32,32],depth=[2,2],num_heads=[2,2],mlp_ratio=2.,use_pos_embed=False)
class NVSTests(unittest.TestCase):
 def test_variable_source_multiple_novel_backward_and_pose_slicing(self):
  from mvr.nvs_training import nvs_training_step
  from mvr.nvs_camera import align_target_camera
  import numpy as np
  model=NVSGARD(**options()).train();model._nonreentrant_checkpoint=True
  # Loaded GARD weights have a nonzero output head; fresh construction zero-initializes it.
  for block in model.blocks:torch.nn.init.normal_(block.adaLN_modulation[-1].weight,std=.03)
  torch.nn.init.normal_(model.final_layer.linear.weight,std=.03)
  torch.nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight,std=.03)
  gt=np.repeat(np.eye(4)[None],4,axis=0);gt[:,0,3]=np.arange(4)
  for n in [2,3,2]:
   source=torch.randn(1,n,5,16);teacher=torch.randn(1,4,5,16)
   targets,scale,residual=align_target_camera(gt,np.linalg.inv(gt[:n])[:,:3])
   self.assertEqual(targets.shape,(4-n,4,4));self.assertAlmostEqual(scale,1.);self.assertAlmostEqual(residual,0.)
   rays=target_plucker_rays(torch.tensor(targets,dtype=torch.float32)[None],torch.eye(3)[None,None].repeat(1,4-n,1,1),(28,28))
   model.zero_grad(set_to_none=True)
   losses=nvs_training_step(model,source,teacher,rays,torch.tensor([.6]),(28,28))
   losses['loss'].backward()
   self.assertTrue(torch.isfinite(losses['loss']))
   self.assertGreater(model.nvs_camera_embedder.weight.grad.abs().sum().item(),0.)
  # Novel loss is an average: duplicating identical novel errors does not double it.
  for n in [2,3]:
   flow=prepare_nvs_flow(torch.zeros(1,n,5,16),torch.zeros(1,4,5,16),torch.ones(1),noise=torch.zeros(1,4,5,16))
   prediction=torch.zeros_like(flow.target_velocity);prediction[:,n:]=2.
   self.assertAlmostEqual(nvs_flow_losses(prediction,flow)['nvs_loss'].item(),4.)
 def test_separate_loss_values_weights_and_gradients(self):
  source=torch.zeros(1,4,3,2);teacher=torch.zeros(1,5,3,2)
  batch=prepare_nvs_flow(source,teacher,torch.ones(1),noise=torch.zeros_like(teacher))
  pred=torch.ones_like(teacher);pred[:,:4,0]=2;pred[:,4]=3;pred.requires_grad_()
  losses=nvs_flow_losses(pred,batch,novel_weight=.5)
  self.assertEqual(losses['restoration_loss'].item(),2.)
  self.assertEqual(losses['nvs_loss'].item(),9.)
  self.assertEqual(losses['loss'].item(),6.5)
  losses['loss'].backward();self.assertGreater(pred.grad[:,:,0].abs().sum().item(),0)
 def test_camera_is_in_legacy_clean_group(self):
  from mvr.grouped_mse import grouped_mse
  source=torch.zeros(2,3,5,2);teacher=torch.zeros(2,4,5,2)
  batch=prepare_nvs_flow(source,teacher,torch.ones(2),noise=torch.zeros_like(teacher))
  pred=torch.randn_like(teacher,requires_grad=True)
  mask=torch.zeros(2,3,5,dtype=torch.bool);mask[0,1,1:]=True;mask[1,2,2:]=True
  cfg=dict(mode='group2',lambda_clean=1.,lambda_distractor=1.)
  losses=nvs_flow_losses(pred,batch,source_distractor_mask=mask,group_mse_cfg=cfg)
  legacy,_=grouped_mse(pred[:,:3],batch.target_velocity[:,:3],mask,mask.any(-1),cfg)
  torch.testing.assert_close(losses['restoration_loss'],legacy)
  losses['loss'].backward();self.assertGreater(pred.grad[:,:,0].abs().sum().item(),0.)
  mask[:,:,0]=True
  with self.assertRaises(ValueError):nvs_flow_losses(pred,batch,source_distractor_mask=mask)
 def test_attention_alignment_gradients_and_geometry(self):
  from mvr.nvs_attention import geometry_target
  cfg=dict(use=True,lambda_coeff=1.,mvrm_layer_idx=1,distractor_aware=dict(use=False),da3_point_cloud=dict(use=True,vis_pc_temperature=.01,visibility_mask='cycle_consistency',vis_pc_cycle_threshold=1.5))
  depth=torch.ones(1,4,28,28);pose=torch.eye(4)[None,None].repeat(1,4,1,1)[:,:,:3];k=torch.eye(3)[None,None].repeat(1,4,1,1);k[:,:,0,0]=k[:,:,1,1]=14
  target=geometry_target(depth,pose,k,cfg)
  torch.testing.assert_close(target.sum(-1),torch.ones(1,16))
  from mvr.nvs_training import nvs_training_step
  model=NVSGARD(**options()).train();model._nonreentrant_checkpoint=True
  source=torch.randn(1,3,5,16);teacher=torch.randn(1,4,5,16);rays=torch.randn(1,1,4,6)
  losses=nvs_training_step(model,source,teacher,rays,torch.tensor([.7]),(28,28),alignment_target=target,alignment_cfg=cfg)
  self.assertGreater(losses['attention_alignment_loss'].item(),0)
  losses['attention_alignment_loss'].backward()
  self.assertGreater(model.blocks[1].attn.qkv.weight.grad.abs().sum().item(),0)
 def test_sample_weighted_logging_and_window_reset(self):
  tracker=NVSLossTracker()
  def losses(a,b):return dict(loss=torch.tensor(a+.5*b),restoration_loss=torch.tensor(a),nvs_loss=torch.tensor(b),weighted_nvs_loss=torch.tensor(.5*b),restoration_clean_loss=torch.tensor(a),restoration_distractor_loss=torch.tensor(0.),attention_alignment_loss=torch.tensor(0.),weighted_attention_alignment_loss=torch.tensor(0.))
  tracker.update(losses(1.,4.),1);tracker.update(losses(3.,8.),3)
  logger=Mock();wandb=Mock()
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'losses.jsonl';p=tracker.write(7,logger=logger,tracker=wandb,jsonl_path=path,distributed=False)
   self.assertEqual(p['train/restoration_loss'],2.5);self.assertEqual(p['train/nvs_loss'],7);self.assertEqual(p['train/weighted_nvs_loss'],3.5);self.assertEqual(p['train/loss'],6);self.assertEqual(p['train/samples'],4)
   self.assertEqual(json.loads(path.read_text())['step'],7);wandb.log.assert_called_once_with(p,step=7);logger.info.assert_called_once()
  self.assertIsNone(tracker.totals)
 def test_ddp_reduces_sums_and_counts_before_average(self):
  tracker=NVSLossTracker();tracker.update(dict(zip(tracker.keys,[torch.tensor(x) for x in [3.,1.,4.,2.,1.,4.,0.,0.]])),1)
  def add_remote(totals):totals.add_(torch.tensor([30.,6.,48.,24.,6.,48.,0.,0.,3.],dtype=torch.float64))
  with patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.all_reduce',side_effect=add_remote):
   p=tracker.compute(prefix='eval')
  self.assertEqual(p['eval/loss'],8.25);self.assertEqual(p['eval/restoration_loss'],1.75);self.assertEqual(p['eval/nvs_loss'],13);self.assertEqual(p['eval/samples'],4)
 def test_no_target_RGB_in_inference_endpoint(self):
  s=torch.randn(2,4,5,16);h=torch.randn(2,5,5,16);noise=torch.randn_like(h)
  a=prepare_nvs_flow(s,h,torch.ones(2),noise=noise);b=prepare_nvs_flow(s,h*10,torch.ones(2),noise=noise)
  torch.testing.assert_close(a.model_input[:,:4],2*s+.3*noise[:,:4]);torch.testing.assert_close(a.model_input[:,4:],noise[:,4:]);torch.testing.assert_close(a.model_input,b.model_input,rtol=0,atol=0)
  self.assertEqual(a.source_condition[:,4:].abs().sum().item(),0)
  pred=torch.randn_like(h,requires_grad=True);nvs_flow_losses(pred,a)['loss'].backward()
  self.assertGreater(pred.grad[:,:4].abs().sum().item(),0);self.assertGreater(pred.grad[:,4:].abs().sum().item(),0)
 def test_baseline_and_source_pose_isolation(self):
  torch.manual_seed(1);base=GARD(**options()).eval();torch.manual_seed(1);model=NVSGARD(**options()).eval()
  for block in base.blocks:torch.nn.init.normal_(block.adaLN_modulation[-1].weight,std=.03)
  torch.nn.init.normal_(base.final_layer.linear.weight,std=.03);torch.nn.init.normal_(base.final_layer.adaLN_modulation[-1].weight,std=.03);model.load_state_dict(base.state_dict(),strict=False)
  x=torch.randn(1,5,5,16);t=torch.ones(1);r=torch.randn(1,5,4,6);m=torch.tensor([[False,False,False,False,True]])
  torch.testing.assert_close(base(x,t,(28,28)),model(x,t,(28,28),nvs_rays=r,nvs_view_mask=m),rtol=0,atol=0)
  torch.nn.init.normal_(model.nvs_camera_embedder.weight,std=.1);torch.nn.init.constant_(model.nvs_camera_embedder.bias,.3)
  other=r.clone();other[:,:4]=100*torch.randn_like(other[:,:4]);a=model(x,t,(28,28),nvs_rays=r,nvs_view_mask=m);b=model(x,t,(28,28),nvs_rays=other,nvs_view_mask=m);torch.testing.assert_close(a,b,rtol=0,atol=0)
  a.square().mean().backward();self.assertGreater(model.nvs_camera_embedder.weight.grad.abs().sum().item(),0);self.assertIsNone(model.mask_head)
 def test_target_ray_geometry(self):
  pose=torch.eye(4)[None,None];k=torch.tensor([[14.,0,13.5],[0,14.,13.5],[0,0,1.]])[None,None];r=target_plucker_rays(pose,k,(28,28));self.assertEqual(r.shape,(1,1,4,6));torch.testing.assert_close(torch.linalg.norm(r[...,:3],dim=-1),torch.ones(1,1,4));torch.testing.assert_close(r[...,3:],torch.zeros(1,1,4,3));pose=pose.clone();pose[:,:,0,3]=1;other=target_plucker_rays(pose,k,(28,28));torch.testing.assert_close(r[...,:3],other[...,:3]);self.assertGreater(other[...,3:].abs().sum().item(),0)
if __name__=='__main__':unittest.main()
