import unittest
import torch
from mvr.grouped_mse import grouped_mse, token_groups

class GroupedMSETests(unittest.TestCase):
    def test_group_mean_and_weights(self):
        pred=torch.tensor([[[[3.],[1.],[1.],[1.]]]],requires_grad=True)
        mask=torch.tensor([[[True,False,False,False]]])
        loss,parts=grouped_mse(pred,torch.zeros_like(pred),mask,torch.tensor([[True]]),
                               {'mode':'group2','lambda_distractor':2.,'lambda_clean':.5})
        self.assertEqual(loss.item(),18.5)
        loss.backward()
        self.assertAlmostEqual(pred.grad[0,0,0,0].item(),12.)
        self.assertEqual(parts['clean'].item(),1.)

    def test_missing_group_is_zero_and_finite(self):
        p=torch.ones(2,1,5,3,requires_grad=True)
        m=torch.zeros(2,1,5,dtype=torch.bool)
        loss,parts=grouped_mse(p,torch.zeros_like(p),m,torch.zeros(2,1,dtype=torch.bool),{'mode':'group2'})
        self.assertEqual(loss.item(),1.)
        self.assertEqual(parts['distractor'].item(),0.)
        loss.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_reference_permutation_and_clean_switch(self):
        masks=torch.ones(2,3,14,28)
        flags=torch.tensor([[True,False,True],[False,True,False]])
        m,v=token_groups(masks,flags,torch.tensor([2,1]))
        self.assertEqual(v.tolist(),[[True,True,False],[True,False,False]])
        self.assertFalse(m[:,:,0].any())
        self.assertEqual(m[:,:,1].tolist(),v.tolist())

    def test_group3_and_view2_differ(self):
        p=torch.tensor([[[[1.],[1.]],[[2.],[3.]]]])
        mask=torch.tensor([[[False,False],[False,True]]]);v=torch.tensor([[False,True]])
        g,_=grouped_mse(p,torch.zeros_like(p),mask,v,{'mode':'group3','lambda_a':1,'lambda_b':2,'lambda_c':3})
        self.assertEqual(g.item(),36.)
        vloss,_=grouped_mse(p,torch.zeros_like(p),mask,v,{'mode':'view2'})
        self.assertEqual(vloss.item(),7.5)

if __name__=='__main__':unittest.main()
