# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import sys,json,gc
ROOT=REPO_ROOT; FC=FEATURE_COMPLETION_ROOT
sys.path[:0]=[str(ROOT),str(ROOT/'Depth-Anything-3/src'),str(FC)]
import torch,numpy as np
from PIL import Image,ImageDraw
from omegaconf import OmegaConf
from types import SimpleNamespace
from depth_anything_3.utils.io.input_processor import InputProcessor
from mvr.eval_models.da3_restore import MultiViewDA3RestoreModel
from mvr.eval_models.gard_completion import build_gard_denoiser
from mvr.eval_models.rgb_decode import build_rae_da3,decode_rgb_from_feats
from RAE.src.stage2.transport import create_transport,Sampler
import argparse
from masking import install
mode='self_token'
OUT=output_dir(__file__)
info=json.loads((FC/'outputs/Cupcake_k4_overfit_group2_260910/dataset_info.json').read_text())
info={k:info[k] for k in ['scene_dir','scene_id','clean_subdir','distractor_subdir','ext','target_hw']}
info.update(context_frames_requested=[1,2,5,6],distractor_frames_requested=[1,2],view_source_indices=[5,1,2,6],n_views=4)

run=next((ROOT/'result_train/hypersim_pairs_ai001_eval/hypersim_pairs').glob('*group2')); cfg=OmegaConf.load(run/'config.yaml')

def clear(): gc.collect();torch.cuda.empty_cache()
def cpu_feats(feats): return [(a.cpu(),b.cpu()) for a,b in feats]
def save_rgb(name,rgb):
 for j,f in enumerate(pair.view_source_indices): Image.fromarray(np.uint8(np.clip(rgb[j],0,1)*255)).save(OUT/f'{name}_{f:04d}.png')

@torch.no_grad()
def main():
 global pair
 order=info['view_source_indices']; scene=Path(info['scene_dir'])
 paths=[str(scene/('train' if f in [1,2] else 'val')/f'{f:04d}.JPG') for f in order]
 clean_paths=[str(scene/'val'/f'{f:04d}.JPG') for f in order]
 processor=InputProcessor()
 norm,_,_=processor(paths,process_res=504,process_res_method='upper_bound_resize',sequential=True)
 clean_norm,_,_=processor(clean_paths,process_res=504,process_res_method='upper_bound_resize',sequential=True)
 assert norm.ndim==4 and norm.shape==clean_norm.shape
 mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
 def pixels(t):return (t*std+mean).clamp(0,1).permute(0,2,3,1).numpy()
 H,W=norm.shape[-2:]
 pair=SimpleNamespace(view_source_indices=order,dist_views=list(pixels(norm)),clean_views=list(pixels(clean_norm)),corrupted_positions=[1,2],image_size_hw=(H,W))
 info.update(target_hw=[H,W],preprocessing='DA3 InputProcessor on original JPGs, process_res=504, upper_bound_resize',input_paths=paths,original_hw=[list(Image.open(x).size[::-1]) for x in paths])
 print('inputs',order,'original HW',info['original_hw'],'processed HW',(H,W),flush=True)
 model=MultiViewDA3RestoreModel(); im=norm.unsqueeze(0).cuda()
 base,features=model.model.forward(im,export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.train,mode='train',export_rgb_feat_layers=True)
 model._check_ref(base); cond=features[('extract_feat',17)].clone().float(); baseline_feats=cpu_feats(base.feat)
 del base,features; model.model.cpu();clear()
 print('Loading 10 epoch EMA and sampling',flush=True)
 den=build_gard_denoiser(None); state=torch.load(run/'checkpoints/latest.pt',map_location='cpu',mmap=True,weights_only=False)
 assert state['next_epoch']==10; den.load_state_dict(state['ema']); del state; den.eval();den.requires_grad_(False)
 audit=install(den,mode,(H,W))
 torch.manual_seed(42); x=cond+float(cfg.mvrm.noise_lvl)*torch.randn_like(cond)
 shift=(cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base)**.5
 sample=Sampler(create_transport(**cfg.transport.params,time_dist_shift=shift)).sample_ode(**cfg.sampler.params)
 with torch.autocast('cuda',dtype=torch.bfloat16): x=sample(x,den,mvrm_cfg=cfg.mvrm,model_img_size=(H,W),lq_latent=cond)[-1].clone()
 assert torch.isfinite(x).all()
 assert len(audit['layers'])==14 and all(v['calls']==49 for v in audit['layers'].values())
 (OUT/'mask_audit.json').write_text(json.dumps(audit,indent=2))
 del den,cond;clear()
 print('Restoring geometry and RGB features',flush=True)
 model.model.cuda()
 restored,_=model.model.forward(im,export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.val,mvrm_result={('restored_latent',17):x},mode='val',export_rgb_feat_layers=True)
 model._check_ref(restored); geom=model._geometry_from_output(restored); restored_feats=cpu_feats(restored.feat)
 del restored,model,im,x;clear()
 print('Decoding RGB',flush=True);rae=build_rae_da3()
 def decode(feats):
  parts=[]
  for j in range(len(pair.view_source_indices)):
   f=[(a[:,j:j+1].cuda(),b[:,j:j+1].cuda()) for a,b in feats]
   parts.append(decode_rgb_from_feats(rae,f,H,W)[0,0].permute(1,2,0).numpy())
  return np.stack(parts)
 baseline=decode(baseline_feats); rgb=decode(restored_feats)
 assert np.isfinite(rgb).all()
 inp=np.stack(pair.dist_views);clean=np.stack(pair.clean_views)
 save_rgb('gard',rgb);save_rgb('input',inp);save_rgb('clean',clean)
 np.savez_compressed(OUT/'completion.npz',**geom,images=rgb,view_idx=np.array(pair.view_source_indices),corrupted_positions=np.array(pair.corrupted_positions),checkpoint=str(run/'checkpoints/latest.pt'))
 metrics=[]
 for j,f in enumerate(pair.view_source_indices):
  rec={'frame':f,'input':str(Path(info['scene_dir'])/('train' if j in pair.corrupted_positions else 'val')/f'{f:04d}.JPG')}
  for name,a in [('input',inp),('da3_decoded',baseline),('gard',rgb)]:rec[name+'_psnr']=float(-10*np.log10(max(float(((a[j]-clean[j])**2).mean()),1e-12)))
  metrics.append(rec)
 def grid(rows,name):
  canvas=Image.new('RGB',(W*4,(H+30)*len(rows)),'white');d=ImageDraw.Draw(canvas)
  for r,j in enumerate(rows):
   for c,(label,a) in enumerate([('Input',inp),('DA3 decoded',baseline),('GARD Hypersim 10ep EMA',rgb),('Clean reference',clean)]):
    canvas.paste(Image.fromarray(np.uint8(np.clip(a[j],0,1)*255)),(c*W,r*(H+30)+30));d.text((c*W+8,r*(H+30)+8),f'{label} | frame {pair.view_source_indices[j]:04d}',fill='black')
  canvas.save(OUT/name)
 grid(pair.corrupted_positions,'distractor_comparison.jpg');grid([j for j in range(len(pair.view_source_indices)) if j not in pair.corrupted_positions],'clean_comparison.jpg')
 meta={'mask_mode':mode,'mask_layers':list(range(14)),'query_scope':'all queries including CLS','checkpoint':str(run/'checkpoints/latest.pt'),'epoch':10,'ema':True,'cfg':False,'sampler':OmegaConf.to_container(cfg.sampler),'noise_lvl':.3,'training_lq_cond':True,'seed':42,'time_dist_shift':shift,'input_setting':info,'metrics':metrics}
 (OUT/'results.json').write_text(json.dumps(meta,indent=2));print(json.dumps(metrics),flush=True);print('DONE',flush=True)
main()
