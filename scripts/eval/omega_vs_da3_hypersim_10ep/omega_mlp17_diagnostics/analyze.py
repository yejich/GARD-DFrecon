# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import sys,json
ROOT=REPO_ROOT;OUT=output_dir(__file__)
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src')]
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from mvr.omega_training import load_omega_encoder,build_normalizer,pair_dataset
from mvr.dataset.hypersim_pairs import pair_collate_fn
from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport,Sampler
@torch.inference_mode()
def main():
 run=next((ROOT/'result_train/omega_hypersim_pairs_ai001_eval/hypersim_pairs').glob('*group2'));cfg=OmegaConf.load(run/'config.yaml');cfg.stage_1.vggt_omega.ckpt=str(VGGT_OMEGA_CKPT);cfg.stage_1.vggt_omega.repo=str(VGGT_OMEGA_REPO);cfg.mvrm.analysis.vis_attn_map=False
 device=torch.device('cuda');encoder=load_omega_encoder(cfg,device);normalizer=build_normalizer(cfg,encoder,device,checkpoint_path=run/'checkpoints/latest.pt')
 den=instantiate_from_config(dict(target=cfg.stage_2.target,params=cfg.stage_2.params)).to(device).eval();state=torch.load(run/'checkpoints/latest.pt',map_location='cpu',weights_only=False,mmap=True);assert state['next_epoch']==10;den.load_state_dict(state['ema']);del state
 sampler=Sampler(create_transport(**cfg.transport.params,time_dist_shift=(cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base)**.5)).sample_ode(**cfg.sampler.params)
 ds=pair_dataset(cfg,'eval',encoder.patch_size);agg=encoder.model.aggregator;original=agg._run_inter_frame_attention_block;all_rows=[];depth=agg.depth;injection=encoder.layer
 for group in [0,26,53]:
  batch=pair_collate_fn([ds[group]]);lq=batch['lq_views'].cuda();hq=batch['hq_views'].cuda();H,W=lq.shape[-2:];refs={};dtypes={};rows=[];current={'mode':'clean'}
  stages=['MLP input','LayerNorm','Linear 1','GELU','Linear 2','LayerScale residual','Residual addition']
  features={}
  def capture_stage(x,stage):
   x=x.reshape(lq.shape[0],lq.shape[1],-1,x.shape[-1])[:,:,encoder.num_special:].float()
   kind=current['mode'];features[stage]=x
   if kind=='clean':refs[stage]=x.cpu()
   else:
    ref=refs[stage].to(x.device);nx=x.norm(dim=-1);nr=ref.norm(dim=-1)
    rows.append(dict(group=group,path=kind,stage=stage,name=stages[stage],
     cosine=float(F.cosine_similarity(x.flatten(),ref.flatten(),dim=0)),
     token_cosine=float(F.cosine_similarity(x,ref,dim=-1).mean()),
     rms_norm=float(nx.square().mean().sqrt()),clean_rms_norm=float(nr.square().mean().sqrt()),
     relative_l2=float((x-ref).norm()/ref.norm()),
     relative_token_norm_error=float(((nx-nr).abs()/nr.clamp_min(1e-8)).mean())))
   if stage==6:
    inp=features[0];res=features[5];norm_in=inp.norm(dim=-1);norm_res=res.norm(dim=-1)
    r=dict(group=group,path=kind,residual_to_input_rms=float(res.norm()/inp.norm()),
     residual_to_input_tokenmean=float((norm_res/norm_in.clamp_min(1e-8)).mean()),
     input_residual_cosine=float(F.cosine_similarity(inp,res,dim=-1).mean()),
     input_output_cosine=float(F.cosine_similarity(inp,x,dim=-1).mean()))
    if kind!='clean':
     cr=refs[5].to(x.device);ci=refs[0].to(x.device)
     # Local counterfactuals only: keep current input, replace residual norm or direction per token.
     norm_clean=cr.norm(dim=-1,keepdim=True);norm_current=res.norm(dim=-1,keepdim=True)
     for label,update in [('clean_residual_norm',res*norm_clean/norm_current.clamp_min(1e-8)),('clean_residual_direction',cr*norm_current/norm_clean.clamp_min(1e-8)),('clean_residual',cr)]:
      r[label+'_output_cosine']=float(F.cosine_similarity(inp+update,refs[6].to(x.device),dim=-1).mean())
    diagnostics.append(r);features.clear()
  diagnostics=[];handles=[];block=agg.frame_blocks[17]
  assert hasattr(block.mlp,'fc1') and isinstance(block.mlp.act,torch.nn.GELU)
  handles.append(block.norm2.register_forward_pre_hook(lambda m,a:capture_stage(a[0],0)))
  for module,stage in [(block.norm2,1),(block.mlp.fc1,2),(block.mlp.act,3),(block.mlp.fc2,4),(block.ls2,5),(block,6)]:
   handles.append(module.register_forward_hook(lambda m,a,o,stage=stage:capture_stage(o,stage)))
  # Capture complete token tensors after each frame/inter-frame pair, including register-only layers.
  with torch.autocast('cuda',dtype=torch.bfloat16):
   pred=encoder.forward_full(hq);del pred
   assert len(refs)==7
   current['mode']='input';pred=encoder.forward_full(lq,extract=True);cond=normalizer.normalize(pred['gard_feats'][injection]);del pred
   gen=torch.Generator(device=device).manual_seed(42+group);noise=torch.randn(cond.shape,generator=gen,device=device)
   x0=cond+cfg.mvrm.noise_lvl*noise if cfg.mvrm.noise_lvl is not None else noise
   restored=sampler(x0,den.forward,mvrm_cfg=cfg.mvrm,model_img_size=(H,W),lq_latent=cond)[-1];restored=normalizer.denormalize(restored.float())
   assert torch.isfinite(restored).all();current['mode']='gard';pred=encoder.forward_full(lq,restored=restored);del pred,restored,cond,x0,noise
  assert len(rows)==14
  (OUT/f'group_{group:02d}_residual.json').write_text(json.dumps(diagnostics,indent=2))
  all_rows.extend(rows);(OUT/f'group_{group:02d}.json').write_text(json.dumps(rows,indent=2));print('completed',group,'layers',depth,flush=True)
  refs.clear()
  for handle in handles:handle.remove()
 (OUT/'results.json').write_text(json.dumps({'rows':all_rows,'groups':[0,26,53],'target_block':17,'checkpoint':str(run/'checkpoints/latest.pt'),'measurement':'Patch tokens only; frame MLP internal read-only hooks; residual includes LayerScale; local residual counterfactuals are diagnostic only','injection_layer':injection,'precision':'BF16 autocast; FP32 cosine'},indent=2))
if __name__=='__main__':main()
