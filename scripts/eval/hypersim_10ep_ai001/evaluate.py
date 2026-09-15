# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import sys,json
ROOT=REPO_ROOT
sys.path[:0]=[str(ROOT),str(ROOT/'Depth-Anything-3/src'),str(FEATURE_COMPLETION_ROOT)]
import torch,numpy as np
from PIL import Image,ImageDraw
from omegaconf import OmegaConf
from mvr.dataset.hypersim_pairs import HypersimPairs
from mvr.eval_models.da3_restore import MultiViewDA3RestoreModel
from mvr.eval_models.gard_completion import build_gard_denoiser
from mvr.eval_models.rgb_decode import build_rae_da3,decode_rgb_from_feats
OUT=output_dir(__file__)
run=next((ROOT/'result_train/hypersim_pairs_ai001_eval/hypersim_pairs').glob('*group2'))
cfg=OmegaConf.load(run/'config.yaml'); ds=HypersimPairs(cfg.data.train.pairs.manifest,split='eval')
assert not any(r['scene']=='ai_001_001' for r in ds.manifest['train'])
state=torch.load(run/'checkpoints/latest.pt',map_location='cpu',mmap=True,weights_only=False)
print('checkpoint',state['next_epoch'],state['step'],flush=True); assert state['next_epoch']==10
model=MultiViewDA3RestoreModel()
den=build_gard_denoiser(None); den.load_state_dict(state['ema']); den.eval(); den.requires_grad_(False)
del state
rae=build_rae_da3(); results=[]
@torch.no_grad()
def evaluate(i):
 item=ds[i]; clean=np.array(item['hq_views'])/255.; inp=np.array(item['lq_views'])/255.
 im=model._img_tensor([x.astype(np.float32) for x in inp]); H,W=inp.shape[1:3]
 base,features=model.model.forward(im,export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.train,mode='train',export_rgb_feat_layers=True)
 model._check_ref(base); cond=features[('extract_feat',17)].float()
 baseline=decode_rgb_from_feats(rae,base.feat,H,W)[0].permute(0,2,3,1).numpy()
 del base,features
 torch.manual_seed(42+i); x=cond+float(cfg.mvrm.noise_lvl)*torch.randn_like(cond); steps=50
 from RAE.src.stage2.transport import create_transport,Sampler
 shift=(cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base)**0.5
 sampler=Sampler(create_transport(**cfg.transport.params,time_dist_shift=shift)).sample_ode(**cfg.sampler.params)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  x=sampler(x,den,mvrm_cfg=cfg.mvrm,model_img_size=(H,W),lq_latent=cond)[-1]
 assert torch.isfinite(x).all()
 restored,_=model.model.forward(im,export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.val,mvrm_result={('restored_latent',17):x},mode='val',export_rgb_feat_layers=True)
 rgb=decode_rgb_from_feats(rae,restored.feat,H,W)[0].permute(0,2,3,1).numpy(); mask=item['pixel_masks']>0; metrics={}
 for name,pred in [('input',inp),('da3_decoded',baseline),('gard_ema',rgb)]:
  err=((pred-clean)**2).mean(-1)
  metrics[name]={key:float(-10*np.log10(max(err[m].mean(),1e-12))) for key,m in [('all',np.ones_like(mask)),('distractor',mask),('background',~mask)]}
 canvas=Image.new('RGB',(W*4,(H+28)*4),'white'); d=ImageDraw.Draw(canvas)
 for row in range(4):
  for col,(label,arr) in enumerate([('Input',inp),('DA3 decoded',baseline),('GARD 10ep EMA',rgb),('Clean GT',clean)]):
   canvas.paste(Image.fromarray(np.uint8(np.clip(arr[row],0,1)*255)),(col*W,row*(H+28)+28)); d.text((col*W+5,row*(H+28)+5),f'{label} | {item["hq_ids"][row]}',fill='black')
 canvas.save(OUT/f'group_{i:02d}.jpg')
 record=dict(group=i,ids=item['hq_ids'],distractor=item['distractor_views'],psnr_db=metrics); results.append(record)
 (OUT/'metrics.json').write_text(json.dumps(dict(checkpoint=str(run/'checkpoints/latest.pt'),epoch=10,ema=True,steps=50,noise_lvl=float(cfg.mvrm.noise_lvl),training_lq_cond=bool(cfg.mvrm.training_lq_cond),groups=results),indent=2)); print(json.dumps(record),flush=True)
for i in [0,26,53]: evaluate(i)
