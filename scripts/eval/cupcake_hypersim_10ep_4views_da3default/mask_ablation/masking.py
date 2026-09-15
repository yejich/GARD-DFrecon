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

def install(den, mode, hw):
 H,W=hw;T=(H//14)*(W//14)+1;N=4*T
 rows=[T+1+18*(W//14)+12,2*T+1+7*(W//14)+19]
 allowed=torch.ones((2,N),dtype=torch.bool,device='cuda')
 for j,row in enumerate(rows):
  if mode=='self_token':allowed[j,row]=False
  elif mode=='self_view':allowed[j,(j+1)*T:(j+2)*T]=False
  else:raise ValueError(mode)
 assert allowed.any(-1).all()
 assert (~allowed).sum().item()==(2 if mode=='self_token' else 2*T)
 audit={'mode':mode,'layer':9,'rows':rows,'tokens_per_view':T,'calls':0,'snapshots':[]}
 module=den.blocks[9].attn;original=module.forward
 def forward(self,x,pos=None,layer_idx=None,attn_type=None,analysis=None):
  assert layer_idx==9 and attn_type=='global' and x.shape[1]==N
  B,n,C=x.shape
  q,k,v=self.qkv(x).reshape(B,n,3,self.num_heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
  q,k=self.q_norm(q),self.k_norm(k)
  if self.rope is not None and pos is not None:q,k=self.rope(q,pos),self.rope(k,pos)
  q,k=q.to(v.dtype),k.to(v.dtype)
  y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
  if audit['calls']==0:
   reference=original(x,pos=pos,layer_idx=layer_idx,attn_type=attn_type,analysis=analysis)
   unmasked=self.proj_drop(self.proj(y.transpose(1,2).reshape(B,n,C)))
   err=float((reference-unmasked).abs().max());assert err==0.,err
   audit['unmasked_forward_max_error']=err
  y[:,:,rows,:]=F.scaled_dot_product_attention(q[:,:,rows,:],k,v,attn_mask=allowed,dropout_p=0.)
  if audit['calls'] in [0,24,48]:
   with torch.autocast('cuda',enabled=False):
    logits=(q[:,:,rows,:].float()*self.scale)@k.float().transpose(-2,-1)
    att=logits.masked_fill(~allowed,float('-inf')).softmax(-1)
    assert (att.masked_select((~allowed)[None,None])==0).all()
    assert torch.allclose(att.sum(-1),torch.ones_like(att.sum(-1)),atol=1e-5)
   audit['snapshots'].append({'call':audit['calls']+1,'blocked_mass':0.,'destination_mass':att[0].mean(0).reshape(2,4,T).sum(-1).cpu().tolist()})
  audit['calls']+=1
  return self.proj_drop(self.proj(y.transpose(1,2).reshape(B,n,C)))
 module.forward=types.MethodType(forward,module)
 return audit
