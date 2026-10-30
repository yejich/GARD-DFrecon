"""Frozen quantitative suite inference; resumable, no training or input selection."""
import sys, json, argparse, hashlib, gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src'),str(ROOT/'scripts/eval/layer_query_attention')]
import numpy as np
import torch
from PIL import Image
OUT=ROOT/'result_eval/20260923_quant_eval'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clear():gc.collect();torch.cuda.empty_cache()
def save(p,**a):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,**a);tmp.replace(p)
def groups(dataset):
 return [g for g in json.loads((OUT/'eval_set_inventory.json').read_text())['groups'] if (g['dataset']=='Hypersim')==(dataset=='hypersim')]
def feature_source(g,fid):
 base=Path(g['result_directory'])
 if fid==6010 and g['dataset']!='RobustNeRF':
  if g['dataset']=='Hypersim':return OUT/g['id']/'final_features_6010.pt'
  return ROOT/'result_eval/20260923_real_6k_10ep_d2'/g['id']/'final_features_010.pt'
 return base/f'final_features_{fid:03d}.pt'
def rgb_source(g,fid):
 base=Path(g['result_directory']);key='da3_baseline' if fid is None else f'gard_{fid:03d}'
 if fid==6010 and g['dataset']=='RealX3D':base=ROOT/'result_eval/20260923_real_6k_10ep_d2'/g['id'];key='gard_010'
 return base/'rgb'/f'{key}.npz'
def reference_paths(g):
 if g['dataset']=='Hypersim':return [str(Path(p).with_name('clean.png')) for p in g['input_paths']]
 f='realx3d_clean12_manifest.json' if g['dataset']=='RealX3D' else 'robustnerf_clean_neighbors_manifest.json'
 return next(x['clean_input_paths'] for x in json.loads((OUT/f).read_text())['groups'] if x['group_id']==g['id'])
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['hypersim','real'],required=True);ap.add_argument('--backend',choices=['da3','vgtw'],required=True);ap.add_argument('--groups',nargs='+');a=ap.parse_args()
 gs=groups(a.dataset);gs=[g for g in gs if not a.groups or g['id'] in a.groups]
 torch.manual_seed(42);torch.set_num_threads(4)
 if a.backend=='vgtw':
  sys.path[:0]=['/mnt/dataset1/yejichoi/VGTW','/mnt/dataset1/yejichoi/VGTW/scripts']
  from vgtw_common import load_model,run_model
  from vgtw.utils.load_fn import load_and_preprocess_images
  ck=Path('/mnt/dataset1/yejichoi/VGTW/vgtw_lora_fp32.pt');model=load_model(str(ck),'cuda')
  for g in gs:
   p=OUT/g['id']/'vgtw.npz'
   if p.exists():continue
   assert [digest(x) for x in g['input_paths']]==g['input_sha256']
   print('START',g['id'],'vgtw',flush=True);x=run_model(g['input_paths'],model,'cuda')
   # All suite images are landscape or square; native loader therefore has no crop/pad.
   assert all(Image.open(v).height<=Image.open(v).width for v in g['input_paths'])
   depth=x['depth'].squeeze(-1);conf=x['depth_conf'];conf=conf.squeeze(-1) if conf.ndim==4 else conf
   save(p,depth=depth,depth_conf=conf,extrinsics=x['extrinsic'],intrinsics=x['intrinsic'],world_points=x['world_points_from_depth'],distractor_valid_mask=x['distractor_valid_mask'],rgb=load_and_preprocess_images(g['input_paths']).permute(0,2,3,1).numpy())
   p.with_suffix('.json').write_text(json.dumps(dict(checkpoint=str(ck),input_sha256=g['input_sha256'],native_resolution=list(depth.shape[1:]),preprocessing='full-image resize, native 518; no crop/padding for these landscape/square inputs',confidence_percentile=50),indent=2))
   print('DONE',g['id'],'vgtw',flush=True)
  return
 import realx3d_d2 as c
 from omegaconf import OmegaConf
 from mvr.eval_models.rgb_decode import build_rae_da3,decode_rgb_from_feats
 cfg=OmegaConf.load(ROOT/'run_configs/eval/gard_da3_attention_model.yaml');cfg.mvrm.analysis.vis_attn_map=False;cfg.mvrm.val.analysis={'vis_attn_map':False}
 model=c.MultiViewDA3RestoreModel();model.model.cpu();clear()
 rae=build_rae_da3();rae.mae_decoder.cpu();rae.adapter.cpu();clear()
 suite=json.loads((ROOT/'run_configs/eval/robustnerf_4sets_6k_20k_d2_rgb_20260923.json').read_text());spec=suite['checkpoints'][0]
 mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
 for g in gs:
  folder=OUT/g['id'];folder.mkdir(parents=True,exist_ok=True)
  assert [digest(p) for p in g['input_paths']]==g['input_sha256']
  norm,_,_=c.InputProcessor()(g['input_paths'],process_res=504,process_res_method='upper_bound_resize',sequential=True)
  save(folder/'inputs.npz',rgb=(norm*std+mean).clamp(0,1).permute(0,2,3,1).numpy())
  fp=feature_source(g,6010)
  if not fp.exists():
   print('START',g['id'],'6K sampling',flush=True)
   cond=torch.load(Path(g['result_directory'])/'lq_features.pt',map_location='cuda',weights_only=True)
   state=torch.load(spec['path'],map_location='cpu',mmap=True,weights_only=False)
   den=c.GARD(**OmegaConf.to_container(cfg.stage_2.params,resolve=True)).eval().requires_grad_(False);den.load_state_dict(state['ema'],strict=True);del state;den.cuda()
   gen=torch.Generator(device='cuda').manual_seed(42);initial=cond+.3*torch.randn(cond.shape,device='cuda',generator=gen)
   shift=(cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base)**.5
   sampler=c.Sampler(c.create_transport(**cfg.transport.params,time_dist_shift=shift)).sample_ode(**cfg.sampler.params)
   with torch.autocast('cuda',dtype=torch.bfloat16):final=sampler(initial,den,mvrm_cfg=cfg.mvrm,model_img_size=tuple(norm.shape[-2:]),lq_latent=cond)[-1]
   assert torch.isfinite(final).all();torch.save(final.cpu(),fp)
   fp.with_suffix('.json').write_text(json.dumps(dict(checkpoint=spec,seed=42,noise=.3,input_sha256=g['input_sha256'],feature_sha256=digest(fp)),indent=2))
   del den,cond,initial,final,sampler;clear()
  for key,fid in [('da3',None),('gard_6k',6010),('gard_003',3),('gard_010',10),('gard_020',20),('gard_030',30),('clean_reference',None)]:
   dst=folder/f'{key}.npz'
   if dst.exists():continue
   print('START',g['id'],key,flush=True)
   paths=reference_paths(g) if key=='clean_reference' else g['input_paths']
   inp=c.InputProcessor()(paths,process_res=504,process_res_method='upper_bound_resize',sequential=True)[0] if key=='clean_reference' else norm
   h,w=inp.shape[-2:];kwargs={}
   if fid is not None:kwargs=dict(mvrm_cfg=cfg.mvrm.val,mode='val',mvrm_result={('restored_latent',17):torch.load(feature_source(g,fid),map_location='cuda',weights_only=True)})
   model.model.cuda()
   output,_=model.model.forward(inp[None].cuda(),export_feat_layers=[],ref_view_strategy='first',export_rgb_feat_layers=True,**kwargs)
   model._check_ref(output);geo=model._geometry_from_output(output)
   if 'sky' in output:geo['sky']=output['sky'][0].float().cpu().numpy()
   feats=[(x.cpu(),y.cpu()) for x,y in output.feat];del output
   # Same restored feature, independent native ray decoder with fixed RANSAC RNG.
   if key!='clean_reference':
    torch.manual_seed(42)
    ray,_=model.model.forward(inp[None].cuda(),export_feat_layers=[],ref_view_strategy='first',use_ray_pose=True,**kwargs)
    geo['ray_extrinsics']=ray.extrinsics[0].float().cpu().numpy();geo['ray_intrinsics']=ray.intrinsics[0].float().cpu().numpy();del ray
   del kwargs;model.model.cpu();clear()
   cached=rgb_source(g,fid)
   if key!='clean_reference' and cached.exists():rgb=np.load(cached)['rgb']
   else:
    rae.mae_decoder.cuda();rae.adapter.cuda();rgb=[]
    for vi in range(len(paths)):
     part=[(x[:,vi:vi+1].cuda(),y[:,vi:vi+1].cuda()) for x,y in feats]
     rgb.append(decode_rgb_from_feats(rae,part,h,w)[0,0].permute(1,2,0).numpy());del part
    rgb=np.stack(rgb);rae.mae_decoder.cpu();rae.adapter.cpu();clear()
   del feats
   save(dst,**geo,rgb=rgb,input_rgb=(inp*std+mean).clamp(0,1).permute(0,2,3,1).numpy())
   dst.with_suffix('.json').write_text(json.dumps(dict(paths=paths,input_sha256=[digest(p) for p in paths],feature=None if fid is None else str(feature_source(g,fid)),feature_sha256=None if fid is None else digest(feature_source(g,fid)),rgb_cached_from=str(cached) if key!='clean_reference' and cached.exists() else None,seed=42,reference_view='first'),indent=2))
   print('DONE',g['id'],key,flush=True)
 print('INFERENCE COMPLETE',a.dataset,a.backend,flush=True)
if __name__=='__main__':main()
