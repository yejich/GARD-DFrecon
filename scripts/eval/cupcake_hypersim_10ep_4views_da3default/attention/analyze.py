# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import sys,json,gc
ROOT=REPO_ROOT;FC=FEATURE_COMPLETION_ROOT
sys.path[:0]=[str(ROOT),str(ROOT/'Depth-Anything-3/src'),str(FC)]
import torch,numpy as np
from omegaconf import OmegaConf
from depth_anything_3.utils.io.input_processor import InputProcessor
from mvr.eval_models.da3_restore import MultiViewDA3RestoreModel
from mvr.eval_models.gard_completion import build_gard_denoiser
from RAE.src.stage2.transport import create_transport,Sampler
OUT=output_dir(__file__); meta=json.loads((OUT.parent/'results.json').read_text());order=[5,1,2,6]
queries=[dict(frame=1,view=1,x=175,y=259),dict(frame=2,view=2,x=273,y=105)]
@torch.no_grad()
def main():
 norm,_,_=InputProcessor()(meta['input_setting']['input_paths'],process_res=504,process_res_method='upper_bound_resize',sequential=True)
 H,W=norm.shape[-2:]; assert (H,W)==(336,504);ph,pw=H//14,W//14;tokens=ph*pw+1
 for q in queries:q.update(row=q['y']//14,col=q['x']//14,index=q['view']*tokens+1+(q['y']//14)*pw+q['x']//14)
 run=Path(meta['checkpoint']).parent.parent;cfg=OmegaConf.load(run/'config.yaml')
 print('Extracting same DA3 input features',flush=True)
 model=MultiViewDA3RestoreModel();base,features=model.model.forward(norm.unsqueeze(0).cuda(),export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.train,mode='train',export_rgb_feat_layers=True)
 model._check_ref(base);cond=features[('extract_feat',17)].clone().float();del model,base,features;gc.collect();torch.cuda.empty_cache()
 den=build_gard_denoiser(None);state=torch.load(meta['checkpoint'],map_location='cpu',mmap=True,weights_only=False);den.load_state_dict(state['ema']);del state;den.eval();den.requires_grad_(False)
 captures=[];current={'step':-1,'t':None};selected=[0,24,48]
 def track(module,args):current.update(step=current['step']+1,t=float(args[1][0]))
 def capture(module,args,kwargs):
  if current['step'] not in selected:return
  x=args[0];pos=kwargs.get('pos');B,N,C=x.shape
  assert B==1 and N==4*tokens and kwargs['attn_type']=='global'
  q,k,v=module.qkv(x).reshape(B,N,3,module.num_heads,module.head_dim).permute(2,0,3,1,4).unbind(0)
  q,k=module.q_norm(q),module.k_norm(k)
  if module.rope is not None and pos is not None:q,k=module.rope(q,pos),module.rope(k,pos)
  q,k=q.to(v.dtype),k.to(v.dtype)
  with torch.autocast('cuda',enabled=False):
   a=((q[:,:, [z['index'] for z in queries]].float()*module.scale)@k.float().transpose(-2,-1)).softmax(-1)
  assert torch.isfinite(a).all() and torch.allclose(a.sum(-1),torch.ones_like(a.sum(-1)),atol=1e-5)
  captures.append(dict(step=current['step'],t=current['t'],attention=a[0].cpu().numpy()))
  print('captured',current,flush=True)
 h1=den.register_forward_pre_hook(track);h2=den.blocks[9].attn.register_forward_pre_hook(capture,with_kwargs=True)
 torch.manual_seed(42);x=cond+.3*torch.randn_like(cond)
 sampler=Sampler(create_transport(**cfg.transport.params,time_dist_shift=meta['time_dist_shift'])).sample_ode(**cfg.sampler.params)
 print('Sampling with read-only query attention hooks',flush=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):final=sampler(x,den,mvrm_cfg=cfg.mvrm,model_img_size=(H,W),lq_latent=cond)[-1]
 assert torch.isfinite(final).all();assert len(captures)==3;h1.remove();h2.remove()
 np.savez_compressed(OUT/'attention.npz',attention=np.stack([c['attention'] for c in captures]),steps=np.array([c['step'] for c in captures]),times=np.array([c['t'] for c in captures]))
 report=dict(checkpoint=meta['checkpoint'],order=order,queries=queries,hw=[H,W],patch_hw=[ph,pw],layer=9,seed=42,steps=selected,times=[c['t'] for c in captures],heads=16,method='Read-only prehook; recompute only selected Q rows against all K after QK normalization and RoPE. Cast as SDPA then accumulate softmax in float32. Original attention forward and sampler unchanged.')
 (OUT/'metadata.json').write_text(json.dumps(report,indent=2));print('DONE',flush=True)
main()
