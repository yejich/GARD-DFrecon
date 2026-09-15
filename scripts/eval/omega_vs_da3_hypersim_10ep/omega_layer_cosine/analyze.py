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
   ref=refs[layer].to(x.device);sim=F.cosine_similarity(x.float().reshape(1,-1),ref.reshape(1,-1),dim=-1).item()
   rows.append({'group':group,'path':kind,'layer':layer,'cosine':sim,'shape':list(x.shape),'attention_type':agg.inter_frame_attention_types[layer]})
  def capture(*args,**kwargs):
   x=original(*args,**kwargs);layer=args[5] if len(args)>5 else kwargs['block_idx'];kind=current['mode']
   if kind=='clean':refs[layer]=x.float().cpu();dtypes[layer]=x.dtype
   elif kind=='gard' and layer==injection:pass
   else:compare(x,layer,kind)
   return x
  agg._run_inter_frame_attention_block=capture
  # Capture complete token tensors after each frame/inter-frame pair, including register-only layers.
  with torch.autocast('cuda',dtype=torch.bfloat16):
   pred=encoder.forward_full(hq);del pred
   assert len(refs)==depth
   current['mode']='input';pred=encoder.forward_full(lq,extract=True);cond=normalizer.normalize(pred['gard_feats'][injection]);del pred
   gen=torch.Generator(device=device).manual_seed(42+group);noise=torch.randn(cond.shape,generator=gen,device=device)
   x0=cond+cfg.mvrm.noise_lvl*noise if cfg.mvrm.noise_lvl is not None else noise
   restored=sampler(x0,den.forward,mvrm_cfg=cfg.mvrm,model_img_size=(H,W),lq_latent=cond)[-1];restored=normalizer.denormalize(restored.float())
   assert torch.isfinite(restored).all();compare(restored.to(dtypes[injection]),injection,'gard');current['mode']='gard';pred=encoder.forward_full(lq,restored=restored);del pred,restored,cond,x0,noise
  keyed={(r['path'],r['layer']):r['cosine'] for r in rows}
  for i in range(injection):assert abs(keyed['input',i]-keyed['gard',i])<1e-5
  assert len(rows)==2*depth
  all_rows.extend(rows);(OUT/f'group_{group:02d}.json').write_text(json.dumps(rows,indent=2));print('completed',group,'layers',depth,flush=True)
  refs.clear();agg._run_inter_frame_attention_block=original
 (OUT/'results.json').write_text(json.dumps({'rows':all_rows,'groups':[0,26,53],'depth':depth,'injection_layer':injection,'checkpoint':str(run/'checkpoints/latest.pt'),'feature':'Aggregator full token tensor after each frame/inter-frame pair; all views, camera/register/patch tokens and channels flattened per group','mask':'none','injection_point':'post-injection value at layer 3; original pre-injection block value skipped','attention_types':list(agg.inter_frame_attention_types),'precision':'same BF16 autocast as preceding Omega evaluation; cosine accumulated in FP32'},indent=2))
if __name__=='__main__':main()
