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
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src'),str(FEATURE_COMPLETION_ROOT)]
import torch,numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf
from mvr.dataset.hypersim_pairs import HypersimPairs
from mvr.eval_models.da3_restore import MultiViewDA3RestoreModel
from mvr.eval_models.gard_completion import build_gard_denoiser
from RAE.src.stage2.transport import create_transport,Sampler
@torch.no_grad()
def main():
 run=next((ROOT/'result_train/hypersim_pairs_ai001_eval/hypersim_pairs').glob('*group2'));cfg=OmegaConf.load(run/'config.yaml');ds=HypersimPairs(cfg.data.train.pairs.manifest,split='eval')
 model=MultiViewDA3RestoreModel();backbone=model.model.model.backbone.pretrained
 den=build_gard_denoiser(None);state=torch.load(run/'checkpoints/latest.pt',map_location='cpu',weights_only=False,mmap=True);assert state['next_epoch']==10;den.load_state_dict(state['ema']);del state;den.eval();den.requires_grad_(False)
 sampler=Sampler(create_transport(**cfg.transport.params,time_dist_shift=(cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base)**.5)).sample_ode(**cfg.sampler.params)
 original=backbone.process_attention;all_rows=[]
 for group in [0,26,53]:
  item=ds[group];H,W=item['pixel_masks'].shape[-2:]
  refs={};current={'mode':'clean'};local_rows=[]
  def compare(x,layer,kind):
   ref=refs[layer].to(x.device)
   value=F.cosine_similarity(x.float().reshape(1,-1),ref.float().reshape(1,-1),dim=-1).item()
   local_rows.append({'group':group,'path':kind,'layer':layer,'cosine':value,'shape':list(x.shape)})
  def capture(*args,**kwargs):
   x=original(*args,**kwargs);layer=kwargs['layer_idx'];kind=current['mode']
   if kind=='clean':refs[layer]=x.detach().float().cpu()
   elif kind=='gard' and layer==17:pass # injection follows the block, so score the injected tensor explicitly below
   else:compare(x,layer,kind)
   return x
  backbone.process_attention=capture
  clean=model._img_tensor([a.astype(np.float32)/255 for a in item['hq_views']]);lq=model._img_tensor([a.astype(np.float32)/255 for a in item['lq_views']])
  base,_=model.model.forward(clean,export_feat_layers=[],ref_view_strategy='first');model._check_ref(base);del base
  assert len(refs)==len(backbone.blocks)==40
  current['mode']='input'
  base,features=model.model.forward(lq,export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.train,mode='train');model._check_ref(base);cond=features[('extract_feat',17)].clone().float();del base,features
  torch.manual_seed(42+group);x=cond+float(cfg.mvrm.noise_lvl)*torch.randn_like(cond)
  with torch.autocast('cuda',dtype=torch.bfloat16):restored=sampler(x,den,mvrm_cfg=cfg.mvrm,model_img_size=(H,W),lq_latent=cond)[-1]
  assert torch.isfinite(restored).all();compare(restored,17,'gard');current['mode']='gard'
  result,_=model.model.forward(lq,export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.val,mvrm_result={('restored_latent',17):restored},mode='val');model._check_ref(result);del result,restored,x,cond,clean,lq
  # The restoration path must be identical to LQ before injection.
  keyed={(r['path'],r['layer']):r['cosine'] for r in local_rows}
  for i in range(17):assert abs(keyed['input',i]-keyed['gard',i])<1e-5
  all_rows.extend(local_rows);(OUT/f'group_{group:02d}.json').write_text(json.dumps(local_rows,indent=2));print('completed group',group,flush=True)
  refs.clear();backbone.process_attention=original
 (OUT/'results.json').write_text(json.dumps({'checkpoint':str(run/'checkpoints/latest.pt'),'rows':all_rows,'groups':[0,26,53],'reference_strategy':'first','feature':'DA3 block output x before final normalization; all views, all tokens including special/camera token, and channels flattened into one vector per group','injection_layer':17,'regions':'none','layer17_gard':'post injection; all other points post block','precision':'DA3 fp32; GARD autocast bf16; cosine accumulated float32'},indent=2))
if __name__=='__main__':main()
