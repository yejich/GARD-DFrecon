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
  def compare(x,layer,kind):
   ref=refs[layer].to(x.device).float();x=x.float()
   token_cos=F.cosine_similarity(x,ref,dim=-1)
   ex=x.square().sum(-1);er=ref.square().sum(-1)
   totalx=ex.sum();totalr=er.sum()
   for name,sl in [('all',slice(None)),('camera',slice(0,1)),('register',slice(1,encoder.num_special)),('patch',slice(encoder.num_special,None))]:
    xx=x[:,:,sl];rr=ref[:,:,sl];sx=ex[:,:,sl];sr=er[:,:,sl]
    flat=F.cosine_similarity(xx.reshape(1,-1),rr.reshape(1,-1),dim=-1).item()
    rows.append({'group':group,'path':kind,'layer':layer,'region':name,'cosine':flat,
     'mean_token_cosine':float(token_cos[:,:,sl].mean()),'tokens':int(sx.numel()),
     'clean_energy_share':float(sr.sum()/totalr),'current_energy_share':float(sx.sum()/totalx),
     'clean_rms_token_norm':float(sr.mean().sqrt()),'current_rms_token_norm':float(sx.mean().sqrt()),
     'clean_max_token_norm':float(sr.max().sqrt()),'current_max_token_norm':float(sx.max().sqrt()),
     'attention_type':'global','stage':['input (after block 16)','frame attention + residual','frame MLP + residual','inter-frame attention + residual','inter-frame MLP + residual'][layer]})
  def capture_stage(x,stage):
   x=x.reshape(lq.shape[0],lq.shape[1],-1,x.shape[-1])
   if current['mode']=='clean':refs[stage]=x.float().cpu()
   else:compare(x,stage,current['mode'])
  handles=[]
  handles.append(agg.frame_blocks[17].register_forward_pre_hook(lambda m,a:capture_stage(a[0],0)))
  handles.append(agg.frame_blocks[17].norm2.register_forward_pre_hook(lambda m,a:capture_stage(a[0],1)))
  handles.append(agg.frame_blocks[17].register_forward_hook(lambda m,a,o:capture_stage(o,2)))
  handles.append(agg.inter_frame_blocks[17].norm2.register_forward_pre_hook(lambda m,a:capture_stage(a[0],3)))
  handles.append(agg.inter_frame_blocks[17].register_forward_hook(lambda m,a,o:capture_stage(o,4)))
  # Capture complete token tensors after each frame/inter-frame pair, including register-only layers.
  with torch.autocast('cuda',dtype=torch.bfloat16):
   pred=encoder.forward_full(hq);del pred
   assert len(refs)==5
   current['mode']='input';pred=encoder.forward_full(lq,extract=True);cond=normalizer.normalize(pred['gard_feats'][injection]);del pred
   gen=torch.Generator(device=device).manual_seed(42+group);noise=torch.randn(cond.shape,generator=gen,device=device)
   x0=cond+cfg.mvrm.noise_lvl*noise if cfg.mvrm.noise_lvl is not None else noise
   restored=sampler(x0,den.forward,mvrm_cfg=cfg.mvrm,model_img_size=(H,W),lq_latent=cond)[-1];restored=normalizer.denormalize(restored.float())
   assert torch.isfinite(restored).all();current['mode']='gard';pred=encoder.forward_full(lq,restored=restored);del pred,restored,cond,x0,noise
  assert len(rows)==40
  all_rows.extend(rows);(OUT/f'group_{group:02d}.json').write_text(json.dumps(rows,indent=2));print('completed',group,'layers',depth,flush=True)
  refs.clear()
  for handle in handles:handle.remove()
 (OUT/'results.json').write_text(json.dumps({'rows':all_rows,'groups':[0,26,53],'target_block':17,'checkpoint':str(run/'checkpoints/latest.pt'),'measurement':'Residual stream after residual addition, before next normalization; layer field is stage index 0-4; read-only hooks','injection_layer':injection,'precision':'BF16 autocast; FP32 cosine'},indent=2))
if __name__=='__main__':main()
