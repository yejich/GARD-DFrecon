"""Real Hypersim mixed restoration + held-out clean NVS fine-tuning."""
import argparse,contextlib,copy,json,logging,math,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src')]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from types import SimpleNamespace
from mvr.nvs_training import NVSGARD,NVSLossTracker,nvs_training_step,target_plucker_rays
from mvr.nvs_attention import geometry_target,clear_attention
from mvr.grouped_mse import token_groups,grouped_mse
from mvr.nvs_camera import HypersimCameraReferences,align_target_camera,align_target_camera_single
from mvr.dataset.hypersim_mixed import MixedHypersimPairs
from mvr.dataset.hypersim_nvs import MixedHypersimNVS,nvs_collate_fn,mixed_nvs_sampler
from mvr.dataset.phase_c_nvs import PhaseCNVS,PhaseCUpdateSampler,phase_c_collate
from depth_anything_3.api import DepthAnything3
from stage2.transport import create_transport

def save_json(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint-dir',type=Path);p.add_argument('--max-steps',type=int,default=0);p.add_argument('--resume',type=Path);p.add_argument('--until-epoch',type=int,default=0);args=p.parse_args()
 checkpoint_dir=args.checkpoint_dir or args.out
 checkpoint_dir.mkdir(parents=True,exist_ok=True)
 cfg=OmegaConf.load(args.config)
 cfg.nvs_branch = bool(cfg.get('nvs_branch', True))
 cfg.flowmap_from = str(cfg.get('flowmap_from', 'noise_lq'))
 if cfg.flowmap_from not in ('noise', 'noise_lq'):
  raise ValueError("flowmap_from must be 'noise' or 'noise_lq'")
 precision=cfg.get('precision','fp32')
 if cfg.get('mask_aware_attention',False):raise ValueError('Mask-aware attention is disabled in this experiment')
 if precision not in ('fp32','bf16'):raise ValueError(f'Unsupported precision: {precision}')
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local);device=torch.device('cuda',local)
 if world>1:dist.init_process_group('nccl')
 torch.set_num_threads(4);torch.manual_seed(cfg.seed+rank);np.random.seed(cfg.seed+rank);random.seed(cfg.seed+rank)
 args.out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,format=f'RANK{rank} %(asctime)s %(message)s');logger=logging.getLogger('nvs')
 run=None
 wandb_enabled=bool(cfg.wandb.get('enabled',True))
 if rank==0 and wandb_enabled:
  import wandb
  previous_run=json.loads((args.out/'wandb_run.json').read_text()).get('id') if args.resume and (args.out/'wandb_run.json').exists() else None
  run_name=cfg.wandb.get('name') or (args.out.parents[2].name if args.out.parent.name=='hypersim_pairs' and args.out.parent.parent.name=='training' else args.out.name)
  run=wandb.init(entity=cfg.wandb.entity,project=cfg.wandb.project,name=run_name,dir=str(args.out),job_type='train',config=OmegaConf.to_container(cfg,resolve=True),id=cfg.wandb.get('run_id') or previous_run,resume='allow')
  save_json(args.out/'wandb_run.json',dict(url=run.url,id=run.id,entity=run.entity));OmegaConf.save(cfg,args.out/'config.yaml')
  save_json(args.out/'status.json',dict(status='initializing',world_size=world))
 elif rank==0:
  OmegaConf.save(cfg,args.out/'config.yaml')
  save_json(args.out/'status.json',dict(status='initializing',world_size=world,wandb_enabled=False))
 # Preserve original+negative scene split and exactly-once negative variants.
 phase_c=bool(cfg.data.get('name','')=='phase_c')
 ds=PhaseCNVS(cfg.data.manifest,resolution=cfg.resolution,patch_size=14,seed=cfg.seed) if phase_c else MixedHypersimPairs(cfg.data.manifest,cfg.data.negative_manifest,original_root=cfg.data.root,negative_root=cfg.data.negative_root,scene_split_policy=cfg.data.scene_split,process_res=cfg.resolution)
 nvs_cfg=cfg.get('nvs',{})
 total_views=int(nvs_cfg.get('total_views',4));min_source=int(nvs_cfg.get('min_source_views',3));max_source=int(nvs_cfg.get('max_source_views',3))
 if phase_c:
  sampler=PhaseCUpdateSampler(ds,rank,world,int(cfg.get('per_gpu_batch_size',1)),cfg.accumulation,int(cfg.steps_per_epoch),cfg.seed)
  loader=DataLoader(ds,batch_sampler=sampler,num_workers=cfg.workers,collate_fn=phase_c_collate,pin_memory=True)
  cameras=None
 else:
  sampler=mixed_nvs_sampler(ds,int(cfg.get('per_gpu_batch_size',1)),rank,world,total_views=total_views,min_source_views=min_source,max_source_views=max_source,accumulation=cfg.accumulation,seed=cfg.seed,original_samples=cfg.data.original_samples,sampling_mode=cfg.data.get('sampling_mode','mixed'))
  loader=DataLoader(MixedHypersimNVS(ds,total_views),batch_sampler=sampler,num_workers=cfg.workers,collate_fn=nvs_collate_fn,pin_memory=True)
  cameras=HypersimCameraReferences(cfg.camera.root,cfg.camera.parameters)
 if rank==0:
  if phase_c: save_json(args.out/'data_audit.json',dict(phase_c=True,groups=len(ds.groups),dataset_groups={k:len(v) for k,v in ds.by_dataset.items()},source_views=[1,2,3],novel_views=[3,2,1]))
  else: save_json(args.out/'data_audit.json',dict(**sampler.summary,source_views=list(range(min_source,max_source+1)),novel_views=[total_views-n for n in range(min_source,max_source+1)],held_out=ds.manifest['mixed_sampling']['held_out_scenes'],original_eligible_anchors=sum(map(len,sampler.original_pools[max_source].values()))))
 model=NVSGARD(**OmegaConf.to_container(cfg.model,resolve=True));model._nonreentrant_checkpoint=True
 ckpt=torch.load(args.resume or cfg.checkpoint,map_location='cpu',weights_only=False,mmap=True);weights=ckpt.get('model') if args.resume else ckpt.get('ema',ckpt.get('model',ckpt))
 # New NVS/cross-view loss modules may be absent when initializing old GARD.
 expected=set(model.state_dict());filtered={k:v for k,v in weights.items() if not k.startswith('mask_head.')}
 missing=expected-set(filtered);extra=set(filtered)-expected
 allowed={k for k in expected if k.startswith(('nvs_camera_embedder.','destination_projector.'))}
 if extra or (missing and (args.resume or missing!=allowed)):raise ValueError(f'Invalid checkpoint keys: missing={sorted(missing)}, extra={sorted(extra)}')
 model.load_state_dict(filtered,strict=not bool(missing));del filtered,weights
 if cfg.distractor_loss.get('mode','none') != 'infoNCE':
  # The destination head is only used by the InfoNCE ablation. Freeze it in
  # push_only/none so DDP does not wait for a gradient that cannot exist.
  model.destination_projector.requires_grad_(False)
 if not cfg.nvs_branch:
  # The camera embedder is only used for novel-view ray conditioning. In the
  # restoration-only branch it must be excluded from DDP reductions.
  model.nvs_camera_embedder.requires_grad_(False)
 model=model.to(device).train();ema=copy.deepcopy(model).eval().requires_grad_(False)
 if args.resume:ema.load_state_dict(ckpt['ema'],strict=True)
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,betas=(.9,.95),weight_decay=0,foreach=False)
 step=0;start_epoch=0;resume_batch=0
 if args.resume:
  optimizer.load_state_dict(ckpt['optimizer']);step=ckpt['step'];start_epoch=ckpt['epoch'];resume_batch=ckpt['next_batch']
  resume_rng=ckpt['rng'][rank]
 else:resume_rng=None
 del ckpt
 denoiser=DDP(model,device_ids=[local],gradient_as_bucket_view=True) if world>1 else model
 encoder=DepthAnything3.from_pretrained(cfg.encoder).to(device).eval().requires_grad_(False)
 mean=torch.tensor([.485,.456,.406],device=device)[None,None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,None,:,None,None]
 transport=create_transport(path_type='Linear',prediction='velocity',loss_weight=None,time_dist_type='logit-normal_0_1',time_dist_shift=math.sqrt(1492992/4096))
 featcfg=SimpleNamespace(extract_feat_layers=[17],concat_feat=False,break_and_return_feat=True)
 meter=NVSLossTracker();source_counts=({k:0 for k in ds.by_dataset} if phase_c else {'original':0,'negative':0});view_counts={str(n):0 for n in range(min_source,max_source+1)};micro=0;started=time.time()
 def extract(batch):
  v=batch['lq_views'].to(device,non_blocking=True);hqs=torch.cat([batch['hq_views'],batch['nvs_hq_views']],1).to(device,non_blocking=True)
  source_views=v.shape[1]
  if hqs.shape[1]!=total_views or not min_source<=source_views<=max_source:raise ValueError('Invalid N/M batch layout')
  source,_=encoder.forward((v-mean)/std,export_feat_layers=[17],ref_view_strategy='first')
  if source.ref_b_idx is not None and not torch.all(source.ref_b_idx==0):raise RuntimeError('Source reference changed')
  lq=source.aux['feat_layer_17'].float().clone()
  pose=source.extrinsics.float().cpu().numpy()
  hq_prediction,_=encoder.forward((hqs-mean)/std,export_feat_layers=[17],ref_view_strategy='first')
  if hq_prediction.ref_b_idx is not None and not torch.all(hq_prediction.ref_b_idx==0):raise RuntimeError('Teacher reference changed')
  teacher=hq_prediction.aux['feat_layer_17'].float().clone();rays=[];scales=[];residuals=[]
  alignment_target=None;alignment_aux=None
  geometry_contrastive=cfg.get('geometry_contrastive_loss',{})
  need_geometry_aux=bool(geometry_contrastive.get('use',False))
  if cfg.attention_alignment.use or need_geometry_aux:
   if cfg.attention_alignment.distractor_aware.use or need_geometry_aux:
    alignment_target,neg_l2,visibility=geometry_target(hq_prediction.depth,hq_prediction.extrinsics,hq_prediction.intrinsics,cfg.attention_alignment,return_aux=True)
    alignment_aux=(alignment_target,neg_l2,visibility)
   else:
    alignment_target=geometry_target(hq_prediction.depth,hq_prediction.extrinsics,hq_prediction.intrinsics,cfg.attention_alignment)
  for i,records in enumerate(batch.get('camera_records',[])):
   gt,k=cameras.cameras(records,v.shape[-2:]);target,scale,residual=align_target_camera(gt,pose[i]);scales.append(scale);residuals.append(residual)
   rays.append(target_plucker_rays(torch.tensor(target,device=device,dtype=torch.float32)[None],torch.tensor(k[source_views:],device=device,dtype=torch.float32)[None],v.shape[-2:]))
  if phase_c:
   for i in range(v.shape[0]):
    gt=batch['camera_c2w'][i].numpy()
    if source_views==1:
     target,scale,residual=align_target_camera_single(gt,pose[i],scale=1.0)
    else:
     target,scale,residual=align_target_camera(gt,pose[i])
    scales.append(scale);residuals.append(residual)
    k=batch['camera_k'][i,source_views:].to(device=device,dtype=torch.float32)[None]
    rays.append(target_plucker_rays(torch.tensor(target,device=device,dtype=torch.float32)[None],k,v.shape[-2:]))
  return lq,teacher,torch.cat(rays,0),v.shape[-2:],scales,residuals,alignment_target,alignment_aux
 def checkpoint(epoch,next_batch):
  state=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state());states=[None]*world
  if world>1:dist.all_gather_object(states,state)
  else:states=[state]
  if rank==0:
   tmp=checkpoint_dir/'latest.tmp';torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),step=step,epoch=epoch,next_batch=next_batch,rng=states,config=OmegaConf.to_container(cfg,resolve=True)),tmp);tmp.replace(checkpoint_dir/'latest.pt')
  if world>1:dist.barrier()
 logger.info('Loaded real models/data; parameters=%s samples/epoch=%s',sum(p.numel() for p in model.parameters()),len(ds))
 if not args.resume:
  # One actual variable-N, fixed-V data batch is reused ONLY for the memory preflight.
  first=next(iter(loader));lq,hq,rays,size,_,_,alignment_target,alignment_aux=extract(first)
  first_mask,first_flags=token_groups(first['pixel_masks'].to(device),first['distractor_views'].to(device),None)
  optimizer.zero_grad(set_to_none=True)
  # Allocate Adam state at step0, without changing model weights, for comparable peaks.
  for group in optimizer.param_groups:
   for parameter in group['params']:
    optimizer.state[parameter]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(parameter),exp_avg_sq=torch.zeros_like(parameter))
  peaks={}
  optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
   loss=nvs_training_step(denoiser,lq,hq,rays,torch.tensor([.9],device=device),size,flowmap_from=cfg.flowmap_from,nvs_branch=cfg.nvs_branch,novel_weight=cfg.novel_weight,source_distractor_mask=first_mask,group_mse_cfg=cfg.group_mse,alignment_target=alignment_target,alignment_aux=alignment_aux,alignment_cfg=cfg.attention_alignment,distractor_suppression_cfg=cfg.distractor_loss,geometry_contrastive_cfg=cfg.get('geometry_contrastive_loss',{}),full_view_loss=bool(cfg.get('full_view_loss',False)))['loss']
  loss.backward();clear_attention(denoiser,cfg.attention_alignment);torch.cuda.synchronize()
  peaks['V4_peak_allocated_GiB']=torch.cuda.max_memory_allocated(device)/1024**3
  peaks['V4_peak_reserved_GiB']=torch.cuda.max_memory_reserved(device)/1024**3
  optimizer.zero_grad(set_to_none=True);del first,lq,hq,rays,loss,alignment_target,alignment_aux;torch.cuda.empty_cache()
  reports=[None]*world
  if world>1:dist.all_gather_object(reports,dict(rank=rank,**peaks))
  else:reports=[dict(rank=rank,**peaks)]
  if rank==0:save_json(args.out/'memory_preflight.json',dict(reports=reports,precision=precision,note='Real variable-N, fixed-V batch with group2 loss and configured attention alignment. Adam state allocated at step0; memory probes do not update model weights. Gradient checkpointing.'));logger.info('MEMORY %s',reports)

 for epoch in range(start_epoch,cfg.epochs):
  sampler.set_epoch(epoch);optimizer.zero_grad(set_to_none=True)
  for batch_idx,batch in enumerate(loader):
   if epoch==start_epoch and batch_idx<resume_batch:continue
   if resume_rng is not None:
    torch.set_rng_state(resume_rng['torch']);torch.cuda.set_rng_state(resume_rng['cuda']);random.setstate(resume_rng['python']);np.random.set_state(resume_rng['numpy']);resume_rng=None
   lq,hq,rays,size,scales,residuals,alignment_target,alignment_aux=extract(batch);t,_,_=transport.sample(hq)
   source_mask,_=token_groups(batch['pixel_masks'].to(device),batch['distractor_views'].to(device),None)
   sync=(micro+1)%cfg.accumulation==0
   with (denoiser.no_sync() if world>1 and not sync else contextlib.nullcontext()):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
     losses=nvs_training_step(denoiser,lq,hq,rays,t,size,noise_lvl=.3,flowmap_from=cfg.flowmap_from,nvs_branch=cfg.nvs_branch,novel_weight=cfg.novel_weight,source_distractor_mask=source_mask,group_mse_cfg=cfg.group_mse,alignment_target=alignment_target,alignment_aux=alignment_aux,alignment_cfg=cfg.attention_alignment,distractor_suppression_cfg=cfg.distractor_loss,geometry_contrastive_cfg=cfg.get('geometry_contrastive_loss',{}),full_view_loss=bool(cfg.get('full_view_loss',False)),loss_tracker=meter)
     if not torch.isfinite(losses['loss']):raise RuntimeError('Nonfinite training loss')
     (losses['loss']/cfg.accumulation).backward()
     clear_attention(denoiser,cfg.attention_alignment)
   source_counts[batch['data_source'][0]]+=1;view_counts[str(lq.shape[1])]+=1;micro+=1
   if not sync:continue
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
   with torch.no_grad():
    for a,b in zip(ema.parameters(),model.parameters()):a.mul_(cfg.ema_decay).add_(b,alpha=1-cfg.ema_decay)
   # Linear warmup and decay in optimizer-update units.
   total_steps=cfg.epochs*len(loader)//cfg.accumulation;warmup=cfg.warmup_epochs*len(loader)//cfg.accumulation
   ratio=1. if step<=warmup else max(.1,1-(step-warmup)/max(total_steps-warmup,1))
   for group in optimizer.param_groups:group['lr']=cfg.lr*ratio
   if step%cfg.log_interval==0 or step==1:
    payload=meter.write(step,logger=logger,tracker=None,jsonl_path=args.out/'losses.jsonl',distributed=world>1)
    memory=dict(rank=rank,allocated_GiB=torch.cuda.memory_allocated()/1024**3,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3)
    memories=[None]*world
    if world>1:dist.all_gather_object(memories,memory)
    else:memories=[memory]
    if rank==0:
     extra={'train/epoch':epoch,'train/lr':optimizer.param_groups[0]['lr'],'train/grad_norm':float(norm),'train/elapsed_seconds':time.time()-started,'train/source_views':lq.shape[1],'train/novel_views':hq.shape[1]-lq.shape[1]}
     for n,count in view_counts.items():extra[f'train/source_N{n}_groups_rank0']=count
     for m in memories:extra[f'gpu{m["rank"]+2}/peak_allocated_GiB']=m['peak_allocated_GiB'];extra[f'gpu{m["rank"]+2}/peak_reserved_GiB']=m['peak_reserved_GiB']
     if run is not None:run.log({**payload,**extra},step=step)
     save_json(args.out/'status.json',dict(status='running',step=step,epoch=epoch,source_counts_rank0=source_counts,source_view_counts_rank0=view_counts,memory=memories,wandb_url=run.url if run is not None else None,wandb_enabled=wandb_enabled))
   if step==1 or step%cfg.checkpoint_interval==0:checkpoint(epoch,batch_idx+1)
   if args.max_steps and step>=args.max_steps:
    checkpoint(epoch,batch_idx+1)
    if rank==0:
     save_json(args.out/'status.json',dict(status='complete',step=step,max_steps=args.max_steps,wandb_enabled=wandb_enabled))
     if run is not None:run.finish()
    if world>1:dist.destroy_process_group()
    return
  checkpoint(epoch+1,0)
  if rank==0 and epoch+1 in set(cfg.save_ema_epochs)|set(cfg.get('eval_epochs',[])):
   tmp=checkpoint_dir/f'epoch_{epoch+1:03d}.tmp';torch.save(dict(ema=ema.state_dict(),epoch=epoch+1,step=step,config=OmegaConf.to_container(cfg,resolve=True)),tmp);tmp.replace(checkpoint_dir/f'epoch_{epoch+1:03d}.pt')
  if world>1:dist.barrier()
  if args.until_epoch and epoch+1>=args.until_epoch:
   if rank==0:
    save_json(args.out/'status.json',dict(status='epoch_boundary',step=step,completed_epochs=epoch+1,reason='release GPUs for scheduled evaluation',wandb_enabled=wandb_enabled))
    if run is not None:run.finish()
   if world>1:dist.destroy_process_group()
   return
 if rank==0:
  save_json(args.out/'status.json',dict(status='complete',step=step,epochs=cfg.epochs,wandb_enabled=wandb_enabled))
  if run is not None:run.finish()
 if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
