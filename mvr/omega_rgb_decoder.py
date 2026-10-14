"""Omega feature contract and DA3 RGB decoder weight conversion."""
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]

class OmegaRGBEncoder(nn.Module):
    """Expose four concatenated frame/inter-frame features to the DA3 trainer.

    Trainer expects ImageNet-normalized images and removes one prefix token.
    Omega normalizes internally: undo input normalization, remove its 17 special
    tokens, then prepend a single camera token to match that trainer contract.
    """
    def __init__(self, checkpoint='ckpts/vggt_omega_1b_512.pt', layers=(4,11,17,23)):
        super().__init__()
        sys.path.insert(0,str(ROOT/'vggt-omega'))
        from vggt_omega.models import VGGTOmega
        model=VGGTOmega(enable_camera=False,enable_depth=False)
        state=torch.load(ROOT/checkpoint,map_location='cpu',mmap=True,weights_only=True)
        model.aggregator.load_state_dict({k.removeprefix('aggregator.'):v for k,v in state.items() if k.startswith('aggregator.')},strict=True)
        self.aggregator=model.aggregator
        self.layers=tuple(layers)
        self.aggregator.cached_layer_indices=set(self.layers)
        self.register_buffer('encoder_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('encoder_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        self.requires_grad_(False)
        self.eval()

    def train(self,mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self,images,mode='all'):
        assert mode=='all'
        b,v,c,h,w=images.shape
        assert h%16==0 and w%16==0
        rgb=images*self.encoder_std[None]+self.encoder_mean[None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs,start=self.aggregator(rgb)
        result={}
        for level,layer in enumerate(self.layers):
            x=outputs[layer]
            x=torch.cat([x[:,:,:1],x[:,:,start:]],dim=2)
            result[level]=x.reshape(b*v,-1,2048).float()
        return result


def convert_da3_decoder(state,target):
    """Preserve pretrained transformer; resize per-patch RGB prediction 14 -> 16.

    unpatchify orders output coordinates [patch_y,patch_x,rgb], not [rgb,y,x].
    Positional embeddings use the same bicubic align_corners=False interpolation
    that GeneralDecoder_Variable applies at inference.
    """
    result=dict(state)
    old=result['decoder_pos_embed'];new=target['decoder_pos_embed']
    if old.shape!=new.shape:
        a=int((old.shape[1]-1)**.5);b=int((new.shape[1]-1)**.5)
        patch=old[:,1:].reshape(1,a,a,-1).permute(0,3,1,2)
        patch=F.interpolate(patch,size=(b,b),mode='bicubic',align_corners=False)
        result['decoder_pos_embed']=torch.cat([old[:,:1],patch.permute(0,2,3,1).reshape(1,b*b,-1)],1)
    for key in ['decoder_pred.weight','decoder_pred.bias']:
        old=result[key];new=target[key]
        if old.shape==new.shape:continue
        a=int((old.shape[0]//3)**.5);b=int((new.shape[0]//3)**.5)
        tail=old.shape[1:]
        x=old.reshape(a,a,3,-1).permute(2,3,0,1).reshape(1,-1,a,a)
        x=F.interpolate(x,size=(b,b),mode='bicubic',align_corners=False)
        result[key]=x.reshape(3,-1,b,b).permute(2,3,0,1).reshape((b*b*3,)+tail)
    assert result.keys()==target.keys()
    for k in result:assert result[k].shape==target[k].shape,(k,result[k].shape,target[k].shape)
    return result
