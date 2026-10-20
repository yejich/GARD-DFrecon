"""Portable nine-experiment trainer, offline logs, EMA visualization, exact resume."""
import argparse,contextlib,copy,json,logging,math,os,random,sys,time
from pathlib import Path
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src')]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from depth_anything_3.api import DepthAnything3
from gard.GARD import GARD
from mvr.grouped_mse import grouped_mse
from mvr.nvs_attention import clear_attention
from mvr.handoff.data import ExperimentPairs,UpdateSampler,collate
from mvr.handoff.model import PixelRayGARD,flow_batch
from mvr.handoff.engine import Engine
from mvr.handoff import alignment

def write(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--resume');p.add_argument('--max-steps',type=int,default=0);p.add_argument('--until-epoch',type=int,default=0);p.add_argument('--eval-now',action='store_true');p.add_argument('--smoke-matrix',action='store_true');a=p.parse_args()
    cfg=OmegaConf.load(a.config);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(hours=4))
    if cfg.global_batch!=world*cfg.microbatch*cfg.accumulation:raise ValueError('World size differs from prepared config; prepare again for GPU count')
    torch.set_num_threads(4);torch.manual_seed(cfg.seed+rank);np.random.seed(cfg.seed+rank);random.seed(cfg.seed+rank)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    handlers=[logging.StreamHandler(sys.stdout)]
    if rank==0:handlers.append(logging.FileHandler(out/'train.log'))
    logging.basicConfig(level=logging.INFO,format=f'RANK{rank} %(asctime)s %(message)s',handlers=handlers,force=True);log=logging.getLogger('handoff')
    log.info('EXPERIMENT=%s dataset=%s nvs=%s flow=%s camera=%s distractor_aware=%s global=%d micro=%d accum=%d world=%d',cfg.experiment,cfg.dataset,cfg.nvs,cfg.flow,cfg.camera_mode,cfg.attention_alignment.distractor_aware.use,cfg.global_batch,cfg.microbatch,cfg.accumulation,world)
    ds=ExperimentPairs(cfg);sampler=UpdateSampler(ds,rank,world,cfg.microbatch,cfg.accumulation)
    loader=DataLoader(ds,batch_sampler=sampler,collate_fn=collate,num_workers=cfg.workers,pin_memory=True)
    if rank==0:
        write(out/'data_audit.json',{'heldout_scenes':ds.heldout,'eligible_anchors':[sum(map(len,p.values())) for p in ds.pools],'sampling':'scene-balanced replacement, fixed optimizer-update budget','negative_probability':cfg.negative_probability,'steps_per_epoch':cfg.steps_per_epoch})
    model=(PixelRayGARD if cfg.nvs else GARD)(**OmegaConf.to_container(cfg.model,resolve=True))
    model._nonreentrant_checkpoint=True
    for block in model.blocks:
        block.attn.retain_attention_head_mean=True
        block.attn.checkpoint_attention_heads=os.environ.get('HANDOFF_ATTENTION_INNER_CHECKPOINT','false').lower()=='true'
        block.attn.attention_head_chunk_size=int(os.environ.get('HANDOFF_ATTENTION_HEAD_CHUNK_SIZE','16'))
    state=torch.load(a.resume or cfg.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    weights=state['model'] if a.resume else state.get('ema',state.get('model',state))
    weights={k:v for k,v in weights.items() if not k.startswith('mask_head.')}
    expected=set(model.state_dict());missing=expected-set(weights);extra=set(weights)-expected
    allowed={k for k in expected if k.startswith('camera_patch.')}
    if extra or missing!=(set() if a.resume else allowed):raise ValueError(f'Checkpoint mismatch missing={missing} extra={extra}')
    model.load_state_dict(weights,strict=bool(a.resume));del weights
    if rank==0:
        write(out/'runtime.json',dict(world_size=world,attention_inner_checkpoint=model.blocks[0].attn.checkpoint_attention_heads,attention_head_chunk_size=model.blocks[0].attn.attention_head_chunk_size,torch=torch.__version__,gpu=torch.cuda.get_device_name(local)))
    model=model.to(device).train();ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,betas=(.9,.95),weight_decay=0,foreach=False)
    step,epoch0,batch0=0,0,0;resume_rng=None
    if a.resume:
        for key in ['experiment','flow','camera_mode','dataset','global_batch','microbatch','accumulation','steps_per_epoch']:
            if state['config'][key]!=cfg[key]:raise ValueError(f'Resume config mismatch: {key}')
        if len(state['rng'])!=world:raise ValueError('Exact resume requires original world size')
        ema.load_state_dict(state['ema']);optimizer.load_state_dict(state['optimizer']);step,epoch0,batch0=state['step'],state['epoch'],state['next_batch'];resume_rng=state['rng'][rank]
    del state
    ddp=DDP(model,device_ids=[local],gradient_as_bucket_view=True) if world>1 else model
    encoder=DepthAnything3.from_pretrained(cfg.encoder).to(device).eval().requires_grad_(False);engine=Engine(encoder,cfg,device)
    from stage2.transport import create_transport
    transport=create_transport(path_type='Linear',prediction='velocity',loss_weight=None,time_dist_type='logit-normal_0_1',time_dist_shift=math.sqrt(1492992/4096))
    def checkpoint(epoch,next_batch,weight_epoch=None):
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state());states=[None]*world
        if world>1:dist.all_gather_object(states,rng)
        else:states=[rng]
        if rank==0:
            latest=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),step=step,epoch=epoch,next_batch=next_batch,rng=states,config=OmegaConf.to_container(cfg,resolve=True))
            tmp=out/'latest.tmp';torch.save(latest,tmp);tmp.replace(out/'latest.pt')
            if weight_epoch is not None:
                tmp=out/'weights.tmp';torch.save(dict(ema=ema.state_dict(),epoch=weight_epoch,step=step,config=OmegaConf.to_container(cfg,resolve=True)),tmp);tmp.replace(out/f'epoch_{weight_epoch:03d}.pt')
            log.info('CHECKPOINT latest.pt epoch=%d step=%d next_batch=%d EMA_epoch=%s',epoch,step,next_batch,weight_epoch)
        if world>1:dist.barrier()
    def evaluate(epoch):
        # All ranks enter matching barriers; rank0 logs/images are offline artifacts.
        if world>1:dist.barrier()
        if rank==0:
            from scripts.train.handoff.evaluate import evaluate
            with torch.random.fork_rng(devices=[local]):evaluate(engine,ema,cfg,out/'eval_visualization'/f'epoch_{epoch:03d}',log)
        if world>1:dist.barrier()
    if a.smoke_matrix:
        specs=OmegaConf.load(ROOT/'run_configs/JIHYE/handoff/20261013_experiments.yaml').experiments
        reports=[]
        for name,spec in specs.items():
            if bool(spec.nvs)!=bool(cfg.nvs):continue
            cfg.dataset=spec.dataset;cfg.flow=spec.flow;cfg.camera_mode=spec.camera;cfg.attention_alignment.distractor_aware.use=spec.distractor_aware
            # Deterministic real original/negative examples; no synthetic random tensors.
            source=0 if spec.dataset=='transient' else 1
            scene=sorted(ds.pools[source])[rank%len(ds.pools[source])]
            anchor=ds.pools[source][scene][0]
            batch=collate([ds[(source,anchor,42+rank)] for _ in range(cfg.microbatch)])
            ex=engine.extract(batch);t=torch.full((cfg.microbatch,),.9,device=device)
            x,velocity=flow_batch(ex['lq'],ex['teacher'],t,cfg.flow,ex['source_views'],ex['pseudo'])
            analysis=OmegaConf.create({'vis_attn_map':True,'mvrm_attn_map':{'extract_idx':[cfg.attention_alignment.mvrm_layer_idx]}})
            kwargs={} if not cfg.nvs else dict(camera_map=ex['camera_map'],camera_views=ex['camera_views'])
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg.precision=='bf16'):
                prediction=ddp(x,t,ex['image_size'],analysis=analysis,**kwargs)
                src=ex['source_views'];rest,_=grouped_mse(prediction[:,:src],velocity[:,:src],ex['mask'][:,:src],ex['flags'][:,:src],cfg.group_mse)
                novel=(prediction[:,src:].float()-velocity[:,src:]).square().mean() if cfg.nvs else rest.new_zeros(())
                align=alignment.loss(ddp,ex['targets'],ex['valid'],cfg.attention_alignment,ex['transient'])
                loss=rest+cfg.novel_weight*novel+cfg.attention_alignment.lambda_coeff*align
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite smoke loss')
            loss.backward();clear_attention(ddp,cfg.attention_alignment)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
            reports.append(dict(experiment=name,loss=float(loss),grad_norm=float(norm),rank=rank,peak_GiB=torch.cuda.max_memory_allocated()/1024**3))
            log.info('MATRIX PASS %s',json.dumps(reports[-1]));del ex,x,velocity,prediction,loss
        if rank==0:write(out/'matrix_smoke.json',reports)
        if a.eval_now:evaluate(0)
        if world>1:dist.destroy_process_group()
        return
    if a.eval_now:evaluate(epoch0)
    meter={};count=0;last=time.monotonic();started=last
    for epoch in range(epoch0,min(cfg.epochs,a.until_epoch or cfg.epochs)):
        sampler.set_epoch(epoch);optimizer.zero_grad(set_to_none=True)
        for bi,batch in enumerate(loader):
            if epoch==epoch0 and bi<batch0:continue
            if resume_rng is not None:
                torch.set_rng_state(resume_rng['torch']);torch.cuda.set_rng_state(resume_rng['cuda']);random.setstate(resume_rng['python']);np.random.set_state(resume_rng['numpy']);resume_rng=None
            ex=engine.extract(batch);t,_,_=transport.sample(ex['teacher'])
            x,velocity=flow_batch(ex['lq'],ex['teacher'],t,cfg.flow,ex['source_views'],ex['pseudo'])
            sync=(bi+1)%cfg.accumulation==0
            with ddp.no_sync() if world>1 and not sync else contextlib.nullcontext():
                analysis=OmegaConf.create({'vis_attn_map':True,'mvrm_attn_map':{'extract_idx':[cfg.attention_alignment.mvrm_layer_idx]}}) if cfg.attention_alignment.use else None
                kwargs={} if not cfg.nvs else dict(camera_map=ex['camera_map'],camera_views=ex['camera_views'])
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg.precision=='bf16'):
                    prediction=ddp(x,t,ex['image_size'],analysis=analysis,**kwargs)
                    src=ex['source_views'];rest,parts=grouped_mse(prediction[:,:src],velocity[:,:src],ex['mask'][:,:src],ex['flags'][:,:src],cfg.group_mse)
                    novel=(prediction[:,src:].float()-velocity[:,src:]).square().mean() if cfg.nvs else rest.new_zeros(())
                    align=alignment.loss(ddp,ex['targets'],ex['valid'],cfg.attention_alignment,ex['transient']) if analysis is not None else rest.new_zeros(())
                    loss=rest+cfg.novel_weight*novel+cfg.attention_alignment.lambda_coeff*align
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                (loss/cfg.accumulation).backward();clear_attention(ddp,cfg.attention_alignment)
            for key,value in dict(loss=loss,restoration=rest,clean=parts['clean'],distractor=parts['distractor'],novel=novel,attention=align).items():meter[key]=meter.get(key,0)+float(value.detach())
            count+=1;del ex,prediction,loss,x,velocity
            if not sync:continue
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            warmup=cfg.warmup_epochs*cfg.steps_per_epoch;total=cfg.epochs*cfg.steps_per_epoch
            ratio=(min((step+1)/max(warmup,1),1.) if cfg.get('warmup_from_zero',False) else 1.) if step<warmup else max(.1,1-(step-warmup)/max(total-warmup,1))
            for group in optimizer.param_groups:group['lr']=cfg.lr*ratio
            optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
            with torch.no_grad():
                for e,m in zip(ema.parameters(),model.parameters()):e.mul_(cfg.ema_decay).add_(m,alpha=1-cfg.ema_decay)
            if step==1 or step%cfg.log_interval==0:
                keys=sorted(meter);values=torch.tensor([meter[k]/count for k in keys],device=device)
                if world>1:dist.all_reduce(values);values/=world
                if rank==0:
                    now=time.monotonic();stats=dict(epoch=epoch+1,step=step,**dict(zip(keys,values.tolist())),lr=optimizer.param_groups[0]['lr'],grad_norm=float(norm),seconds_since_start=now-started,peak_GiB=torch.cuda.max_memory_allocated()/1024**3)
                    log.info('TRAIN %s',json.dumps(stats));
                    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
                    write(out/'status.json',dict(status='training',**stats))
                meter={};count=0
            if step%cfg.checkpoint_interval==0 and bi+1<len(loader):checkpoint(epoch,bi+1)
            if a.max_steps and step>=a.max_steps:
                checkpoint(epoch,bi+1);log.info('SMOKE COMPLETE step=%d',step)
                if world>1:dist.destroy_process_group()
                return
        checkpoint(epoch+1,0,epoch+1 if epoch+1 in cfg.save_epochs else None)
        if epoch+1 in cfg.eval_epochs:evaluate(epoch+1)
    if rank==0:write(out/'status.json',dict(status='complete',step=step,epochs=cfg.epochs))
    if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
