# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

import types
import torch
from torch.nn import functional as F

def install(den,mode,hw):
 assert mode=='self_token';T=(hw[0]//14)*(hw[1]//14)+1
 masks={n:~torch.eye(n,device='cuda',dtype=torch.bool) for n in [T,4*T]}
 for n,a in masks.items():
  assert not a.diagonal().any() and (a.sum(-1)==n-1).all()
 audit={'mode':'self_token','scope':'Every query including CLS, every GARD local/global layer and head, all ODE calls; DA3 unchanged','layers':{},'tokens_per_view':T}
 def attach(module,idx):
  original=module.forward
  entry={'calls':0,'attention_type':'local' if idx%2==0 else 'global','snapshots':[]};audit['layers'][str(idx)]=entry
  def forward(self,x,pos=None,layer_idx=None,attn_type=None,analysis=None):
   B,N,C=x.shape;assert layer_idx==idx and attn_type==entry['attention_type'];assert N==(T if idx%2==0 else 4*T)
   q,k,v=self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
   q,k=self.q_norm(q),self.k_norm(k)
   if self.rope is not None and pos is not None:q,k=self.rope(q,pos),self.rope(k,pos)
   q,k=q.to(v.dtype),k.to(v.dtype)
   if entry['calls']==0:
    y0=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
    y0=self.proj_drop(self.proj(y0.transpose(1,2).reshape(B,N,C)))
    reference=original(x,pos=pos,layer_idx=layer_idx,attn_type=attn_type,analysis=analysis)
    err=float((y0-reference).abs().max());assert err==0.;entry['unmasked_forward_max_error']=err
   y=F.scaled_dot_product_attention(q,k,v,attn_mask=masks[N],dropout_p=0.)
   if entry['calls'] in [0,48]:
    rows=torch.tensor(sorted(set([0,1,N//2,N-1])),device=x.device)
    with torch.autocast('cuda',enabled=False):
     a=((q[:,:,rows].float()*self.scale)@k.float().transpose(-2,-1)).masked_fill(~masks[N][rows],float('-inf')).softmax(-1)
     assert torch.allclose(a.sum(-1),torch.ones_like(a.sum(-1)),atol=1e-5)
     diagonal=a[:,:,torch.arange(len(rows),device=x.device),rows]
     assert (diagonal==0).all()
    entry['snapshots'].append({'call':entry['calls']+1,'sampled_rows':rows.tolist(),'self_probability':0.})
   entry['calls']+=1
   return self.proj_drop(self.proj(y.transpose(1,2).reshape(B,N,C)))
  module.forward=types.MethodType(forward,module)
 for i,b in enumerate(den.blocks):attach(b.attn,i)
 assert len(audit['layers'])==14
 return audit
