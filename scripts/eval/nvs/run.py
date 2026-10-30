"""Resumable V4+M1 EMA evaluation, source-only conditioning and frozen targets."""
import argparse,sys,json,hashlib,gc,time,csv,os,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src')]
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torch
from omegaconf import OmegaConf
from types import SimpleNamespace
from depth_anything_3.utils.io.input_processor import InputProcessor
from mvr.eval_models.da3_restore import MultiViewDA3RestoreModel
from mvr.eval_models.rgb_decode import build_rae_da3,decode_rgb_from_feats
from mvr.nvs_training import NVSGARD,target_plucker_rays
from mvr.nvs_camera import target_cameras_for_condition
from RAE.src.stage2.transport import create_transport, Sampler
from scripts.eval.nvs.prepare import freeze
from scripts.eval.nvs.metrics import score,safe

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n');q.replace(p)

def replace_with_clean_inputs(data):
 """Use the first four clean reference views as the model condition."""
 for item in data['groups']:
  clean=list(item.get('reference_paths') or [])[:4]
  if len(clean) != 4 or not all(Path(p).is_file() for p in clean):
   raise FileNotFoundError(f'Need four clean reference inputs for {item["name"]}: {clean}')
  g=dict(item['group'])
  g['input_paths']=clean
  g['input_sha256']=[digest(p) for p in clean]
  # Keep pose scoring aligned with the clean images when the suite stores
  # explicit reference poses. Hypersim derives them from frame_order below.
  if 'pose_reference_extrinsics' in g:
   g['pose_reference_extrinsics']=np.linalg.inv(np.asarray(item['reference_c2w'][:4],dtype=np.float64))[:,:3].tolist()
  if g.get('dataset','').startswith('Hypersim'):
   import re
   frames=[]
   for p in clean:
    m=re.search(r'frame\.(\d+)',p)
    if m: frames.append(int(m.group(1)))
   if len(frames)==4:g['frame_order']=frames
  item['group']=g
 return data

def load_tokenwise_mask(group, image_hw, device, allow_missing=False):
    """Build a [1,V,CLS+patches] oracle distractor mask from source-view GT masks."""
    h, w = image_hw
    paths = group.get('mask_paths') or {}
    if not paths:
        if not allow_missing:
            raise ValueError('Token-wise oracle inference requires source-view GT masks')
        # Visualization fallback for real suites without labels: all source
        # patches are treated as clean; the novel view remains full-grid.
        return torch.zeros((1, len(group['input_paths']) + 1, 1 + (h // 14) * (w // 14)), device=device, dtype=torch.bool)
    # The eval manifest stores four source paths; the sampler state also has
    # one appended novel-view grid, whose mask is intentionally all-false.
    views = len(group['input_paths']) + 1
    pixel = torch.zeros((views, 1, h, w), device=device, dtype=torch.float32)
    for view, candidates in paths.items():
        if len(candidates) != 1:
            raise ValueError(f'Expected one GT mask for view {view}, got {candidates}')
        with Image.open(candidates[0]) as im:
            arr = np.asarray(im.convert('L').resize((w, h), Image.Resampling.NEAREST), dtype=np.float32) / 255.
        pixel[int(view), 0] = torch.from_numpy(arr).to(device)
    # Conservative boundary handling: any non-zero GT occupancy marks the
    # whole patch as a distractor, so partial boundary overlap is not treated
    # as clean.
    patch = F.avg_pool2d(pixel, kernel_size=14, stride=14).flatten(1) > 0
    cls = torch.zeros((views, 1), device=device, dtype=torch.bool)
    return torch.cat([cls, patch], dim=1)[None]

def clean_update_indices(num_steps, clean_steps):
    if clean_steps < 1 or clean_steps > num_steps:
        raise ValueError('clean step count must be in [1, num_steps]')
    return set(np.rint(np.linspace(0, num_steps - 2, clean_steps)).astype(int).tolist())
def aggregate(out,items):
 rows=[]
 for item in items:
  p=out/item['name']/'metrics.json'
  if not p.exists():continue
  metrics=json.loads(p.read_text());g=item['group'];row=dict(name=item['name'],dataset=g['dataset'],scene=g.get('scene',g['id'][:10]),M_reference_kind=item['target_rgb_reference_kind'])
  for method,m in metrics['methods'].items():
   for block in ['rgb_V','rgb_M','depth_V','depth_M','pose_V']:
    for k,v in m.get(block,{}).items():
     if isinstance(v,(float,int)) and not isinstance(v,bool):row[f'{method}/{block}/{k}']=v
   for level,c in m['confidence_cutoffs'].items():
    for k,v in c['cloud'].items():
     if isinstance(v,(float,int)) and not isinstance(v,bool):row[f'{method}/cloud_P{level}/{k}']=v
  rows.append(row)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 def write(name,data):
  fields=list(dict.fromkeys(k for r in data for k in r))
  with (out/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
 write('per_group.csv',rows)
 perviews=[]
 for item in items:
  p=out/item['name']/'metrics.json'
  if not p.exists():continue
  m=json.loads(p.read_text())
  for method,x in m['methods'].items():
   for block in ['rgb_V','rgb_M']:
    for v in x.get(block,{}).get('per_view',[]):perviews.append(dict(name=item['name'],dataset=item['group']['dataset'],method=method,slot=block,**v))
 write('per_view.csv',perviews)
 scene_rows=[]
 for dataset,scene in sorted({(r['dataset'],r['scene']) for r in rows}):
  group=[r for r in rows if r['dataset']==dataset and r['scene']==scene];sr=dict(dataset=dataset,scene=scene,groups=len(group))
  for k in keys:
   values=[r[k] for r in group if isinstance(r.get(k),(int,float))];
   if values:sr[k]=float(np.mean(values))
  scene_rows.append(sr)
 write('per_scene.csv',scene_rows);dataset_rows=[]
 for dataset in sorted({r['dataset'] for r in rows}):
  scenes=[r for r in scene_rows if r['dataset']==dataset];dr=dict(dataset=dataset,scenes=len(scenes),groups=sum(r['groups'] for r in scenes))
  for k in keys:
   values=[r[k] for r in scenes if isinstance(r.get(k),(int,float))];
   if values:dr[k]=float(np.mean(values))
  dataset_rows.append(dr)
 write('per_dataset.csv',dataset_rows)
 dump(out/'summary.json',dict(completed_groups=len(rows),total_groups=len(items),aggregation='equal group means within scene, equal scene means within dataset',datasets=dataset_rows))
 links=[]
 for r in rows:
  links.append(f'<h2>{r["name"]}</h2>'+''.join(f'<img loading="lazy" style="max-width:100%" src="{r["name"]}/{file}">' for file in ['rgb_comparison.jpg','M_rgb_comparison.jpg','depth_comparison.jpg','M_depth_comparison.jpg','camera_comparison.png','gard_nvs_point_cloud_comparison_p20.png','gard_nvs_point_cloud_comparison_p40.png','gard_nvs_point_cloud_comparison_p50.png']))
 (out/'qualitative_index.html').write_text('<meta charset="utf-8"><h1>NVS EMA: V4+M1</h1><p>Restoration V metrics and novel M metrics are separate. OnTheGo M is a held-out capture, clean status not guaranteed. Geometry reference provenance is recorded per group. Baseline DA3 uses V only; GARD uses V4 plus a noise M slot.</p>'+''.join(links))

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--limit',type=int);ap.add_argument('--num-steps',type=int,default=50);ap.add_argument('--tokenwise-clean-steps',type=int);ap.add_argument('--tokenwise-distractor-steps',type=int);ap.add_argument('--allow-missing-masks',action='store_true');ap.add_argument('--clean-inputs',action='store_true',help='replace fixed-suite distractor inputs with four clean reference views');args=ap.parse_args()
 tokenwise=args.tokenwise_clean_steps is not None or args.tokenwise_distractor_steps is not None
 if tokenwise and (args.tokenwise_clean_steps is None or args.tokenwise_distractor_steps is None):
  ap.error('--tokenwise-clean-steps and --tokenwise-distractor-steps must be provided together')
 if tokenwise and args.tokenwise_distractor_steps != args.num_steps:
  ap.error('The first prototype keeps distractor tokens on the full global grid; set distractor steps equal to --num-steps')
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 # Prefer the camera/geometry paths recorded with the checkpoint.  The fixed
 # suite registry remains selected by GARD_EVAL_ROOT, but Hypersim assets must
 # not silently fall back to a machine-specific datasets/eval tree.
 checkpoint=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
 cfg=OmegaConf.create(checkpoint['config'])
 camera_cfg=cfg.get('camera',{})
 camera_root=camera_cfg.get('root')
 camera_parameters=camera_cfg.get('parameters')
 if camera_root:
  os.environ.setdefault('HYPERSIM_CAMERA_ROOT',str(camera_root))
  os.environ.setdefault('HYPERSIM_GEOMETRY_ROOT',str(camera_root))
 if camera_parameters:
  os.environ.setdefault('HYPERSIM_CAMERA_PARAMETERS',str(camera_parameters))
 plan=out/'eval_plan.json'
 if not plan.exists():freeze(plan)
 data=json.loads(plan.read_text())
 if args.clean_inputs:
  data=replace_with_clean_inputs(data)
 items=data['groups'][args.start:args.start+args.limit] if args.limit is not None else data['groups'][args.start:];dump(out/'status.json',dict(status='loading',groups=len(items),start=args.start,gpu=os.environ.get("CUDA_VISIBLE_DEVICES","unspecified"),clean_inputs=args.clean_inputs))
 checkpoint_epoch=int(checkpoint.get('epoch',0));den=NVSGARD(**OmegaConf.to_container(cfg.model,resolve=True)).eval().requires_grad_(False);den.load_state_dict(checkpoint['ema'],strict=True);den.cuda();del checkpoint
 nvs_branch=bool(cfg.get('nvs_branch',True))
 autocast_enabled=str(cfg.get('precision','fp32')).lower()=='bf16'
 eval_root=Path(os.environ.get('GARD_EVAL_ROOT',str(ROOT/'datasets/eval')))
 source_views=len(items[0]['group']['input_paths']) if items else 0
 camera_pose_mode=str(cfg.get('camera_pose_mode','da3_aligned')).lower()
 if camera_pose_mode not in ('da3_aligned','first_relative'):
  raise ValueError("checkpoint camera_pose_mode must be 'da3_aligned' or 'first_relative'")
 dump(out/'provenance.json',dict(checkpoint=args.checkpoint,checkpoint_sha256=digest(args.checkpoint),weights='EMA',epoch=checkpoint_epoch,registry_sha256=data['registry_sha256'],protocol_sha256=digest(eval_root/'evaluation_protocol.json'),eval_plan_sha256=digest(plan),clean_inputs=args.clean_inputs,clean_input_policy='first four reference_paths per frozen group' if args.clean_inputs else 'fixed input_paths',source_only_DA3=True,target_RGB_in_model_condition=False,V=source_views,M=1 if nvs_branch else 0,N=source_views+(1 if nvs_branch else 0),seed=42,sampler='repository Sampler.sample_ode(euler)',num_grid_points=args.num_steps,velocity_calls=args.num_steps-1,time_dist_shift=float(np.sqrt(1492992/4096)),noise_V=.3,noise_M=1.,initial_V='noise for flowmap=noise; 0.3*noise+z_lq for flowmap_from=noise_lq',model_condition='transport adds z_lq to x on every velocity call',camera_pose_mode=camera_pose_mode,source_camera_condition=False,mask_head=False,tokenwise_oracle=tokenwise,tokenwise_clean_steps=args.tokenwise_clean_steps,tokenwise_distractor_steps=args.tokenwise_distractor_steps,tokenwise_mask_semantics='source-view object GT when available; any non-zero patch occupancy; CLS clean; novel view full-grid',allow_missing_masks=args.allow_missing_masks))
 model=MultiViewDA3RestoreModel(pretrained=cfg.encoder,device='cuda');decoder=build_rae_da3(device='cpu')
 import lpips
 lp=lpips.LPIPS(net='alex',spatial=True).cuda().eval().requires_grad_(False)
 mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None];restorecfg=SimpleNamespace(restore_feat_layers=[17],concat_feat=False)
 def decode(output,h,w):
  decoder.mae_decoder.cuda();decoder.adapter.cuda();rgb=[]
  for i in range(output.feat[0][0].shape[1]):
   rgb.append(decode_rgb_from_feats(decoder,[(a[:,i:i+1],b[:,i:i+1]) for a,b in output.feat],h,w)[0,0].permute(1,2,0).numpy())
  decoder.mae_decoder.cpu();decoder.adapter.cpu();return np.stack(rgb)
 completed=0
 for item in items:
  name=item['name'];folder=out/name;folder.mkdir(parents=True,exist_ok=True)
  if (folder/'complete.json').exists():
   done=json.loads((folder/'complete.json').read_text())
   if done.get('metric_revision')!=2:
    base=dict(np.load(folder/'da3.npz'));gard=dict(np.load(folder/'gard_nvs.npz'));raw=dict(np.load(folder/'clean_reference_raw.npz'));previous=json.loads((folder/'metrics.json').read_text());result=score(folder,item,base,gard,raw,lp)
    for k in ['features_L17','M_camera_condition_alignment']:result[k]=previous[k]
    dump(folder/'metrics.json',result);done['metric_revision']=2;dump(folder/'complete.json',done);aggregate(out,items);print('RESCORED',name,flush=True)
   completed+=1;continue
  start=time.time();dump(out/'status.json',dict(status='running',group=name,completed=completed,total=len(items),gpu=os.environ.get("CUDA_VISIBLE_DEVICES","unspecified")));print('START',name,flush=True)
  try:
   g=item['group'];v=len(g['input_paths']);assert [digest(p) for p in g['input_paths']]==g['input_sha256'];assert digest(item['target_path'])==item['target_sha256']
   norm=InputProcessor()(g['input_paths'],process_res=504,process_res_method='upper_bound_resize',sequential=True)[0];h,w=norm.shape[-2:];assert [h,w]==g['image_hw'];image=norm[None].cuda();inputrgb=(norm*std+mean).clamp(0,1).permute(0,2,3,1).numpy()
   base_output,_=model.model.forward(image,export_feat_layers=[17],ref_view_strategy='first',export_rgb_feat_layers=True);model._check_ref(base_output);cond=base_output.aux['feat_layer_17'].float().clone();base=model._geometry_from_output(base_output);base['rgb']=decode(base_output,h,w);del base_output
   if nvs_branch:
    target_pose,scale,residual=target_cameras_for_condition(np.array(item['source_target_c2w']),base['extrinsics'],v,camera_pose_mode);rays=target_plucker_rays(torch.tensor(target_pose,dtype=torch.float32,device='cuda')[None],torch.tensor(item['source_target_intrinsics'][v:],dtype=torch.float32,device='cuda')[None],(h,w));fullrays=torch.cat([torch.zeros(1,v,rays.shape[2],6,device='cuda'),rays],1);mask=torch.tensor([[False]*v+[True]],device='cuda')
    condition=torch.cat([cond,torch.zeros_like(cond[:,:1])],1)
   else:
    # Restoration-only checkpoints were trained with exactly the source V
    # grids; do not append an untrained novel-view grid at inference.
    target_pose=scale=residual=rays=fullrays=mask=None
    condition=cond
   generator=torch.Generator(device='cuda').manual_seed(42);noise=torch.randn(condition.shape,generator=generator,device='cuda');state=noise.clone()
   flowmap_from=str(cfg.get('flowmap_from','noise_lq'))
   if flowmap_from == 'noise':
    # prepare_nvs_flow starts from pure noise; the transport sampler adds
    # source_lq to the model input on every velocity evaluation.
    state[:,:v]=noise[:,:v]
   elif flowmap_from == 'noise_lq':
    # prepare_nvs_flow starts from 0.3*noise + z_lq; the transport sampler
    # adds the second z_lq to the model input on every velocity evaluation.
    state[:,:v]=.3*noise[:,:v]+cond[:,:v]
   else:
    raise ValueError(f'Unsupported checkpoint flowmap_from={flowmap_from!r}')
   if tokenwise:
    distractor_mask=load_tokenwise_mask(g,(h,w),state.device,args.allow_missing_masks)
    if distractor_mask.shape != state.shape[:-1]:
     raise ValueError(f'Mask shape {distractor_mask.shape} does not match state {state.shape[:-1]}')
    clean_mask=~distractor_mask
    clean_indices=clean_update_indices(args.num_steps,args.tokenwise_clean_steps)
    full_mask=torch.zeros_like(distractor_mask)
    full_mask[:,:,0]=True
    full_mask[:,v:]=True
   else:
    distractor_mask=clean_mask=clean_indices=full_mask=None
   # Match training/fixed-suite inference exactly: integrate the latent state
   # x, while adding z_lq only inside each model velocity call.  The previous
   # implementation integrated x+z_lq itself, which changes the ODE path.
   transport_cfg=SimpleNamespace(
    lq_latent_cond='addition',
    val={'lq_cond_sampling':True,'analysis':None,
         'guidance':SimpleNamespace(use_cfg=False,cfg_scale=1.0)},
   )
   sampler=Sampler(create_transport(path_type='Linear',prediction='velocity',loss_weight=None,
                                     time_dist_type='logit-normal_0_1',
                                     time_dist_shift=float(np.sqrt(1492992/4096)))).sample_ode(
                                         sampling_method='euler',num_steps=args.num_steps,
                                         atol=1e-6,rtol=1e-3)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=autocast_enabled):
    final=sampler(state,den,mvrm_cfg=transport_cfg,model_img_size=(h,w),lq_latent=condition,
                  nvs_rays=fullrays,nvs_view_mask=mask)[-1]
   assert torch.isfinite(final).all();torch.save(final.cpu(),folder/'final_features_L17.pt');del state,noise,condition,sampler
   if nvs_branch:
    del rays,fullrays
   # Dummy mean-color M: target photograph never enters the inference backbone.
   dummy=torch.cat([image,torch.zeros_like(image[:,:1])],1) if nvs_branch else image
   output,_=model.model.forward(dummy,export_feat_layers=[],ref_view_strategy='first',mode='val',mvrm_cfg=restorecfg,mvrm_result={('restored_latent',17):final},export_rgb_feat_layers=True);model._check_ref(output);gard=model._geometry_from_output(output);gard['rgb']=decode(output,h,w);del output
   if completed==0 and nvs_branch:
    other=dummy.clone();other[:,v]=.25
    check,_=model.model.forward(other,export_feat_layers=[],ref_view_strategy='first',mode='val',mvrm_cfg=restorecfg,mvrm_result={('restored_latent',17):final});torch.testing.assert_close(check.depth[0].float().cpu(),torch.tensor(gard['depth']),rtol=0,atol=0);dump(out/'placeholder_independence.json',dict(exact_depth_match=True,target_dummy_values=[0.,.25],all5_views=True));del check,other
   target_norm=InputProcessor()([item['target_path']],process_res=504,process_res_method='upper_bound_resize',sequential=True)[0];assert target_norm.shape[-2:]==(h,w);gard['target_rgb']=(target_norm*std+mean).clamp(0,1)[0].permute(1,2,0).numpy();gard['target_valid']=np.load(item['target_valid_path']) if item['target_valid_path'] else np.ones((h,w),bool);gard['input_rgb']=inputrgb
   # Teacher/reference pass only AFTER sampling, never used by inference.
   rn=InputProcessor()(item['reference_paths'],process_res=504,process_res_method='upper_bound_resize',sequential=True)[0];assert rn.shape[-2:]==(h,w)
   reference,_=model.model.forward(rn[None].cuda(),export_feat_layers=[17],ref_view_strategy='first');model._check_ref(reference);raw=model._geometry_from_output(reference);raw['input_rgb']=(rn*std+mean).clamp(0,1).permute(0,2,3,1).numpy()
   if item['clean_source_paths'] is not None and len(item['reference_paths'])==v+1 and nvs_branch:
    z=reference.aux['feat_layer_17'].float();diff=final[:,:,1:]-z[:,:,1:];cos=torch.nn.functional.cosine_similarity(final[:,:,1:],z[:,:,1:],dim=-1);feature=dict(V_cosine=float(cos[:,:v].mean()),M_cosine=float(cos[:,v:].mean()),V_relative_L2=float(torch.linalg.vector_norm(diff[:,:v])/torch.linalg.vector_norm(z[:,:v,1:])),M_relative_L2=float(torch.linalg.vector_norm(diff[:,v:])/torch.linalg.vector_norm(z[:,v:,1:])),M_cosine_p1=float(torch.quantile(cos[:,v:].flatten(),.01)),M_cosine_p5=float(torch.quantile(cos[:,v:].flatten(),.05)))
   elif item['clean_source_paths'] is not None:
    z=reference.aux['feat_layer_17'].float();cos=torch.nn.functional.cosine_similarity(final[:,:v,1:],z[:,:v,1:],dim=-1);feature=dict(V_cosine=float(cos.mean()),V_cosine_p1=float(torch.quantile(cos.flatten(),.01)),V_cosine_p5=float(torch.quantile(cos.flatten(),.05)),M_cosine=None,M_relative_L2=None)
   else:feature=dict(status='N/A',reason='No all-clean paired teacher')
   del reference,final,cond,image,dummy,rn;gc.collect();torch.cuda.empty_cache()
   np.savez_compressed(folder/'da3.npz',**base);np.savez_compressed(folder/'gard_nvs.npz',**gard);np.savez_compressed(folder/'clean_reference_raw.npz',**raw)
   result=score(folder,item,base,gard,raw,lp);result['features_L17']=feature;result['M_camera_condition_alignment']=dict(scale=scale,source_camera_center_residual=residual) if nvs_branch else dict(status='N/A',reason='restoration-only checkpoint');dump(folder/'metrics.json',result);dump(folder/'complete.json',dict(name=name,seconds=time.time()-start,metric_revision=2,weights='EMA',epoch=checkpoint_epoch,V=v,M=1 if nvs_branch else 0,nvs_branch=nvs_branch));completed+=1
   del base,gard,raw;gc.collect();torch.cuda.empty_cache();aggregate(out,items);print('DONE',name,'seconds',round(time.time()-start,2),'M',result['methods']['gard_nvs'].get('rgb_M','N/A'),flush=True)
  except Exception as exc:
   dump(folder/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));dump(out/'status.json',dict(status='failed',group=name,completed=completed,error=str(exc)));raise
 aggregate(out,items);dump(out/'status.json',dict(status='complete',completed=completed,total=len(items),gpu=os.environ.get("CUDA_VISIBLE_DEVICES","unspecified")));print('ALL COMPLETE',completed,flush=True)
if __name__=='__main__':main()
