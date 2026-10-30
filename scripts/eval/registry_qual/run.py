"""Resumable epoch-checkpoint qualitative evaluation of the frozen eval registry."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src')]
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from depth_anything_3.utils.io.input_processor import InputProcessor
from mvr.eval_models.da3_restore import MultiViewDA3RestoreModel
from mvr.eval_models.rgb_decode import build_rae_da3,decode_rgb_from_feats
from mvr.fixed_suite_training_eval import load_groups
from mvr.fixed_pose_eval import pose_reference
from mvr.d2_mask_training import FixedT1D2GARD,fixed_t1_input
from gard.GARD import GARD
from mvr.nvs_training import NVSGARD
from stage2.transport import create_transport,Sampler
from scripts.eval.quant_eval.metrics import write_ply,centers

def clear():gc.collect();torch.cuda.empty_cache()
def output_name(name):
    dataset, group = name.split('/', 1)
    dataset = {
        'Hypersim': 'hypersim',
        'HypersimNegative': 'hypersimnegative',
        'OnTheGo': 'onthego',
        'RealX3D': 'realx3d',
        'RobustNeRF': 'robustnerf3d',
    }.get(dataset, dataset.lower())
    return f'{dataset}/{group}'
def grid(path,columns):
 v,h,w=columns[0][1].shape[:3];panel=Image.new('RGB',(w*len(columns),(h+24)*v),'white');draw=ImageDraw.Draw(panel)
 for ci,(label,a) in enumerate(columns):
  for i in range(v):
   panel.paste(Image.fromarray(np.uint8(np.clip(a[i],0,1)*255)),(ci*w,i*(h+24)+24));draw.text((ci*w+4,i*(h+24)+4),f'{label} | view{i}',fill='black')
 panel.save(path,quality=92)
def color_depth(d,lo,hi):return plt.get_cmap('turbo')(np.clip((d-lo)/max(hi-lo,1e-6),0,1))[...,:3]
def plot_pose(folder,base,gard,ref,filename="pose_comparison.png"):
 fig=plt.figure(figsize=(12,10));ax=fig.add_subplot(111,projection='3d')
 # Independent centered baseline normalization for qualitative shape comparison only.
 for label,e,c in [('DA3',base,'blue'),('GARD',gard,'red'),('Reference',ref,'green')]:
  p=centers(e);p-=p.mean(0);scale=np.median([np.linalg.norm(p[i]-p[j]) for i in range(4) for j in range(i)]);p/=max(scale,1e-8);ax.plot(*p.T,'o-',color=c,label=label)
  for i,pt in enumerate(p):ax.text(*pt,str(i))
 ax.legend(fontsize=12);ax.set_title('Camera centers: centered / own median baseline\nShape diagnostic; not metric alignment',fontsize=14);fig.savefig(folder/filename,dpi=240,bbox_inches='tight');plt.close(fig)

def cloud_outputs(folder,data):
 conf=data['depth_conf'];good=np.isfinite(conf)&np.isfinite(data['depth'])&(data['depth']>0)&np.isfinite(data['world_points']).all(-1)
 reports={}
 for level in [20,40,50]:
  threshold=float(np.percentile(conf[np.isfinite(conf)],level));keep=good&(conf>=threshold);p=data['world_points'][keep];c=data['rgb'][keep]
  write_ply(folder/f'decoded_rgb_colored_p{level}.ply',p,c)
  ids=np.linspace(0,len(p)-1,min(len(p),35000),dtype=int)
  fig=plt.figure(figsize=(8,7));ax=fig.add_subplot(111,projection='3d');ax.scatter(*p[ids].T,c=c[ids],s=.4)
  if len(p):
   lo,hi=np.percentile(p,[1,99],axis=0);mid=(lo+hi)/2;span=max(hi-lo)/2
   ax.set_xlim(mid[0]-span,mid[0]+span);ax.set_ylim(mid[1]-span,mid[1]+span);ax.set_zlim(mid[2]-span,mid[2]+span)
  ax.set_title(f'Decoded RGB cloud P{level} | {len(p)} points');fig.savefig(folder/f'point_cloud_p{level}.png',dpi=130);plt.close(fig)
  reports[str(level)]={'threshold':threshold,'retained_fraction':float(keep.mean()),'points':len(p)}
 return reports

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);ap.add_argument('--registry',default=str(ROOT/'datasets/eval/registry.json'));ap.add_argument('--limit',type=int);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--legacy-no-mask',action='store_true');ap.add_argument('--auto-nvs',action='store_true',help='instantiate NVSGARD when the checkpoint contains NVS parameters');args=ap.parse_args()
 torch.set_num_threads(4);torch.manual_seed(42);cfg=OmegaConf.load(args.config)
 cfg.mvrm.analysis.vis_attn_map=False;cfg.mvrm.val.analysis={'vis_attn_map':False}
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 rows=load_groups([args.registry]);rows=rows[:args.limit] if args.limit else rows;rows=rows[args.shard::args.shards]
 (out/'provenance.json').write_text(json.dumps({'checkpoint':args.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),'weights':'EMA','registry_sha256':hashlib.sha256(Path(args.registry).read_bytes()).hexdigest(),'groups':len(rows),'seed':42,'scope':'qualitative only; quantitative evaluation pending'},indent=2))
 model=MultiViewDA3RestoreModel(pretrained=cfg.stage_1.da3.ckpt,device='cpu');decoder=build_rae_da3(device='cpu')
 den_cls=GARD if args.legacy_no_mask else FixedT1D2GARD
 ckpt=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
 is_nvs=args.auto_nvs or any(k.startswith('nvs_camera_embedder.') for k in ckpt['ema'])
 if is_nvs:
  model_cfg=ckpt.get('config',{}).get('model',OmegaConf.to_container(cfg.stage_2.params,resolve=True))
  den=NVSGARD(**model_cfg).eval().requires_grad_(False)
 else:
  den=den_cls(**OmegaConf.to_container(cfg.stage_2.params,resolve=True)).eval().requires_grad_(False)
 if is_nvs:
  missing,unexpected=den.load_state_dict(ckpt['ema'],strict=False)
  if unexpected or set(missing)!={'destination_projector.weight'}:
   raise RuntimeError(f'Unexpected NVS checkpoint mismatch: missing={missing}, unexpected={unexpected}')
 else:
  den.load_state_dict(ckpt['ema'],strict=True)
 del ckpt
 mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
 def decode(feats,h,w):
  decoder.mae_decoder.cuda();decoder.adapter.cuda();rgb=[]
  for v in range(4):rgb.append(decode_rgb_from_feats(decoder,[(a[:,v:v+1].cuda(),b[:,v:v+1].cuda()) for a,b in feats],h,w)[0,0].permute(1,2,0).numpy())
  decoder.mae_decoder.cpu();decoder.adapter.cpu();clear();return np.stack(rgb)
 for name,g,protocol in rows:
  folder=out/output_name(name);folder.mkdir(parents=True,exist_ok=True)
  if (folder/'complete.json').exists() and (folder/'pose_ray_comparison.png').exists():continue
  print('START',name,flush=True)
  norm,_,_=InputProcessor()(g['input_paths'],process_res=protocol['process_res'],process_res_method=protocol['process_res_method'],sequential=True)
  h,w=norm.shape[-2:];assert [h,w]==g['image_hw'];inputrgb=(norm*std+mean).clamp(0,1).permute(0,2,3,1).numpy();image=norm[None].cuda();model.model.cuda()
  output,features=model.model.forward(image,export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.train,mode='train',export_rgb_feat_layers=True)
  base=model._geometry_from_output(output);cond=features[('extract_feat',17)].clone().float();feats=[(a.cpu(),b.cpu()) for a,b in output.feat];del output,features
  torch.manual_seed(42);ray,_=model.model.forward(image,export_feat_layers=[],ref_view_strategy='first',use_ray_pose=True)
  model._check_ref(ray);base['ray_extrinsics']=ray.extrinsics[0].float().cpu().numpy();base['ray_intrinsics']=ray.intrinsics[0].float().cpu().numpy();del ray;model.model.cpu();clear()
  base['rgb']=decode(feats,h,w);del feats
  den.cuda()
  if args.legacy_no_mask:
   prob=np.zeros((4,h,w),dtype=np.float32)
  else:
   torch.manual_seed(42);logits=den(fixed_t1_input(cond,cfg.mvrm.noise_lvl),cond.new_ones(1),(h,w),mask_only=True)['mask_logits']
   prob=F.interpolate(logits[0].float().sigmoid().reshape(4,1,h//14,w//14),size=(h,w),mode='nearest')[:,0].cpu().numpy();del logits
  gen=torch.Generator(device='cuda').manual_seed(42);initial=cond+cfg.mvrm.noise_lvl*torch.randn(cond.shape,device='cuda',generator=gen)
  shift=(cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base)**.5;sample=Sampler(create_transport(**cfg.transport.params,time_dist_shift=shift)).sample_ode(**cfg.sampler.params)
  with torch.autocast('cuda',dtype=torch.bfloat16):final=sample(initial,den,mvrm_cfg=cfg.mvrm,model_img_size=(h,w),lq_latent=cond)[-1]
  assert torch.isfinite(final).all();den.cpu();del initial,cond,sample;clear();model.model.cuda()
  output,_=model.model.forward(image,export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.val,mode='val',mvrm_result={('restored_latent',17):final},export_rgb_feat_layers=True)
  gard=model._geometry_from_output(output);feats=[(a.cpu(),b.cpu()) for a,b in output.feat];del output
  torch.manual_seed(42);ray,_=model.model.forward(image,export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.val,mode='val',mvrm_result={('restored_latent',17):final},use_ray_pose=True)
  model._check_ref(ray);gard['ray_extrinsics']=ray.extrinsics[0].float().cpu().numpy();gard['ray_intrinsics']=ray.intrinsics[0].float().cpu().numpy();del ray,final,image;model.model.cpu();clear();gard['rgb']=decode(feats,h,w);del feats
  np.savez_compressed(folder/'da3.npz',**base);np.savez_compressed(folder/'gard.npz',**gard,mask_probability=prob)
  columns=[('Input',inputrgb),('DA3 decoded',base['rgb']),('GARD decoded',gard['rgb'])]
  if g.get('clean_paths') or g['dataset']=='Hypersim':
   paths=g.get('clean_paths',[str(Path(x).with_name('clean.png')) for x in g['input_paths']]);clean,_,_=InputProcessor()(paths,process_res=protocol['process_res'],process_res_method=protocol['process_res_method'],sequential=True);columns.append(('Clean GT',(clean*std+mean).clamp(0,1).permute(0,2,3,1).numpy()))
  grid(folder/'rgb_comparison.jpg',columns)
  gt=np.zeros((4,h,w));known=g['mask_known_views']
  # Load object masks with explicit unknown-view display, never unknown-as-negative.
  for view,paths in g.get('mask_paths',{}).items():
   for path in paths:
    with Image.open(path) as im:gt[int(view)]=np.maximum(gt[int(view)],np.asarray(im.convert('L').resize((w,h),Image.Resampling.NEAREST))>0)
  target=np.repeat(gt[...,None],3,-1)
  for v in range(4):
   if not known[v]:target[v]=.5
   Image.fromarray(np.uint16(prob[v]*65535)).save(folder/f'mask_probability_view{v}.png')
   Image.fromarray(np.uint8(gard['rgb'][v]*255)).save(folder/f'decoded_view{v}.png')
  grid(folder/'mask_comparison.jpg',[('Input',inputrgb),('GT (gray=unknown)',target),('Mask probability',np.repeat(prob[...,None],3,-1)),('Mask >=0.5',np.repeat((prob>=.5)[...,None],3,-1))])
  valid=np.concatenate([base['depth'].ravel(),gard['depth'].ravel()]);lo,hi=np.percentile(valid[np.isfinite(valid)&(valid>0)],[2,98]);grid(folder/'depth_comparison.jpg',[('Input',inputrgb),('DA3 depth',color_depth(base['depth'],lo,hi)),('GARD depth',color_depth(gard['depth'],lo,hi))])
  ref,info=pose_reference(g);plot_pose(folder,base['extrinsics'],gard['extrinsics'],ref);np.savez_compressed(folder/'poses.npz',da3=base['extrinsics'],gard=gard['extrinsics'],reference=ref)
  plot_pose(folder,base['ray_extrinsics'],gard['ray_extrinsics'],ref,filename='pose_ray_comparison.png')
  np.savez_compressed(folder/'poses_ray.npz',da3=base['ray_extrinsics'],gard=gard['ray_extrinsics'],da3_intrinsics=base['ray_intrinsics'],gard_intrinsics=gard['ray_intrinsics'],reference=ref)
  cutoffs=cloud_outputs(folder,gard)
  (folder/'complete.json').write_text(json.dumps({'group':g,'pose_reference':info,'cutoffs':cutoffs,'mask_timestep':1,'depth_visual_scale':[float(lo),float(hi)],'quantitative_status':'pending'},indent=2));print('DONE',name,flush=True)
  links=[f'<h2>{n}</h2>'+''.join(f'<img width="100%" loading="lazy" src="{output_name(n)}/{file}">' for file in ['rgb_comparison.jpg','mask_comparison.jpg','depth_comparison.jpg','pose_comparison.png','pose_ray_comparison.png','point_cloud_p20.png','point_cloud_p40.png','point_cloud_p50.png']) for n,_,_ in rows if (out/output_name(n)/'complete.json').exists()]
  (out/'index.html').write_text('<meta charset="utf-8"><h1>GARD mask head epoch003 EMA qualitative</h1>'+''.join(links))
 print('ALL COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
