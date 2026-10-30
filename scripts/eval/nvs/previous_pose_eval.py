"""Pose-only explicit scope: prior attention-aligned GARD, frozen V4 registry."""
import sys,json,csv,gc,time,hashlib,os,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src')]
import numpy as np,torch
from omegaconf import OmegaConf
from gard.GARD import GARD
from mvr.eval_models.da3_restore import MultiViewDA3RestoreModel
from mvr.fixed_suite_training_eval import load_groups
from mvr.fixed_pose_eval import pose_reference
from scripts.eval.quant_eval.metrics import pose_metrics
from scripts.eval.registry_qual.run import plot_pose
from scripts.eval.nvs.metrics import safe
from depth_anything_3.utils.io.input_processor import InputProcessor
from stage2.transport import create_transport,Sampler

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(safe(d),indent=2,allow_nan=False)+'\n');q.replace(p)
def csvwrite(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def summarize(out,epochs,items):
 rows=[];pairs=[];views=[]
 for epoch in epochs:
  for name,g,protocol in items:
   p=out/f'epoch_{epoch:03d}'/name/'metrics.json'
   if not p.exists():continue
   m=json.loads(p.read_text())
   for method,branches in m['methods'].items():
    for branch,metrics in branches.items():
     row=dict(epoch=epoch,dataset=g['dataset'],scene=g.get('scene',g['id'][:10]),group=g['id'],method=method,branch=branch,**{k:metrics.get(k) for k in ['rre_median','rte_median','auc_5','auc_10','auc_20','auc_30','ate_rmse_sim3','ate_degenerate']});rows.append(row)
     for pair in metrics['pairs']:pairs.append(dict(epoch=epoch,dataset=g['dataset'],group=g['id'],method=method,branch=branch,**pair))
     for v in range(4):
      ps=[pair for pair in metrics['pairs'] if v in [pair['i'],pair['j']]];views.append(dict(epoch=epoch,dataset=g['dataset'],group=g['id'],method=method,branch=branch,view=v,path=g['input_paths'][v],rre_mean=float(np.mean([x['rre'] for x in ps])),rte_mean=float(np.mean([x['rte'] for x in ps]))))
 csvwrite(out/'per_group.csv',rows);csvwrite(out/'per_pair.csv',pairs);csvwrite(out/'per_view.csv',views);summary=[];scenes=[]
 for epoch,dataset,scene,method,branch in sorted({(r['epoch'],r['dataset'],r['scene'],r['method'],r['branch']) for r in rows}):
  selected=[r for r in rows if (r['epoch'],r['dataset'],r['scene'],r['method'],r['branch'])==(epoch,dataset,scene,method,branch)];rr=dict(epoch=epoch,dataset=dataset,scene=scene,method=method,branch=branch,groups=len(selected))
  for k in ['rre_median','rte_median','auc_5','auc_10','auc_20','auc_30']:rr[k]=float(np.mean([r[k] for r in selected]))
  scenes.append(rr)
 for epoch,dataset,method,branch in sorted({(r['epoch'],r['dataset'],r['method'],r['branch']) for r in scenes}):
  selected=[r for r in scenes if (r['epoch'],r['dataset'],r['method'],r['branch'])==(epoch,dataset,method,branch)];rr=dict(epoch=epoch,dataset=dataset,method=method,branch=branch,scenes=len(selected),groups=sum(r['groups'] for r in selected))
  for k in ['rre_median','rte_median','auc_5','auc_10','auc_20','auc_30']:rr[k]=float(np.mean([r[k] for r in selected]))
  summary.append(rr)
 csvwrite(out/'per_scene.csv',scenes);csvwrite(out/'per_dataset.csv',summary);dump(out/'summary.json',dict(scope='pose head and ray pose only, frozen V4',epochs=epochs,completed_epoch_group_pairs=len(rows)//4,total_epoch_group_pairs=len(items)*len(epochs),aggregation='equal group means within scene, equal scene means within dataset',records=summary))
 links=[]
 for epoch in epochs:
  for name,_,_ in items:
   folder=out/f'epoch_{epoch:03d}'/name
   if (folder/'complete.json').exists():links.append(f'<h2>Epoch {epoch}: {name}</h2><img style="max-width:48%" loading="lazy" src="epoch_{epoch:03d}/{name}/camera_comparison.png"><img style="max-width:48%" loading="lazy" src="epoch_{epoch:03d}/{name}/ray_comparison.png">')
 (out/'index.html').write_text('<meta charset="utf-8"><h1>Previous attention-aligned GARD: pose + ray</h1><p>Frozen input V4, reference first, seed42. Pose-only requested scope; these outputs are not a full RGB/depth/cloud evaluation. Baseline and GARD compared to measured Hypersim GT or provided SfM cameras.</p>'+''.join(links))

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--epochs',type=int,nargs='+',default=[3,10]);ap.add_argument('--checkpoint30');ap.add_argument('--checkpoint20');ap.add_argument('--run',default='da3_negative_attn_true_20260925_000427_160229884');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);items=load_groups([ROOT/'datasets/eval/registry.json']);torch.set_num_threads(4);torch.manual_seed(42)
 run=ROOT/'result_train'/a.run;cfg=OmegaConf.load(run/'config.yaml');assert cfg.mvrm.loss.attn_align.use;cfg.mvrm.analysis.vis_attn_map=False;cfg.mvrm.val.analysis={'vis_attn_map':False};OmegaConf.save(cfg,out/'inference_config.yaml');checks={}
 for epoch in a.epochs:
  checks[epoch]=Path(a.checkpoint20) if epoch==20 and a.checkpoint20 else Path(a.checkpoint30) if epoch==30 and a.checkpoint30 else next(run.glob(f'training/**/checkpoints/epoch_{epoch:03d}.pt'))
 dump(out/'provenance.json',dict(scope='explicit pose + ray only',registry_sha256=sha(ROOT/'datasets/eval/registry.json'),config_sha256=sha(run/'config.yaml'),groups=65,V=4,M=0,reference_view='first',seed=42,steps=50,euler_updates=49,source_training_attention_alignment=bool(cfg.mvrm.loss.attn_align.use),source_training_distractor_aware=bool(cfg.mvrm.loss.attn_align.distractor_aware.use),weights='EMA',checkpoints={str(e):dict(path=str(p),sha256=sha(p)) for e,p in checks.items()},epoch_interpretation={str(e):('Original10 plus continuation local10' if e==20 and a.checkpoint20 else 'Original run epoch') for e in checks}))
 encoder=MultiViewDA3RestoreModel(device='cuda');den=GARD(**OmegaConf.to_container(cfg.stage_2.params,resolve=True)).eval().requires_grad_(False).cuda();count=0;shift=(cfg.misc.time_dist_shift_dim/cfg.misc.time_dist_shift_base)**.5;sample=Sampler(create_transport(**cfg.transport.params,time_dist_shift=shift)).sample_ode(**cfg.sampler.params)
 for epoch,path in checks.items():
  checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False);den.load_state_dict(checkpoint['ema'],strict=True);del checkpoint
  for name,g,protocol in items:
   folder=out/f'epoch_{epoch:03d}'/name;folder.mkdir(parents=True,exist_ok=True)
   if (folder/'complete.json').exists():count+=1;continue
   dump(out/'status.json',dict(status='running',epoch=epoch,group=name,completed=count,total=len(items)*len(checks),gpu=6));print('START',epoch,name,flush=True);started=time.time()
   norm=InputProcessor()(g['input_paths'],process_res=504,process_res_method='upper_bound_resize',sequential=True)[0];h,w=norm.shape[-2:];assert [h,w]==g['image_hw'];image=norm[None].cuda();cache=out/'source_cache'/name;cache.mkdir(parents=True,exist_ok=True);ref,info=pose_reference(g)
   if not (cache/'poses.npz').exists():
    base,_=encoder.model.forward(image,export_feat_layers=[17],ref_view_strategy='first');encoder._check_ref(base);cond=base.aux['feat_layer_17'].float().clone();torch.save(cond.cpu(),cache/'source_L17.pt');bp=base.extrinsics[0].float().cpu().numpy();bk=base.intrinsics[0].float().cpu().numpy();del base
    torch.manual_seed(42);ray,_=encoder.model.forward(image,export_feat_layers=[],ref_view_strategy='first',use_ray_pose=True);encoder._check_ref(ray);np.savez_compressed(cache/'poses.npz',camera=bp,camera_intrinsics=bk,ray=ray.extrinsics[0].float().cpu().numpy(),ray_intrinsics=ray.intrinsics[0].float().cpu().numpy());del ray;dump(cache/'provenance.json',dict(input_sha256=g['input_sha256'],input_paths=g['input_paths'],reference_view='first'))
   else:
    assert json.loads((cache/'provenance.json').read_text())['input_sha256']==g['input_sha256'];cond=torch.load(cache/'source_L17.pt',map_location='cuda',weights_only=True)
   generator=torch.Generator(device='cuda').manual_seed(42);initial=cond+.3*torch.randn(cond.shape,generator=generator,device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):final=sample(initial,den,mvrm_cfg=cfg.mvrm,model_img_size=(h,w),lq_latent=cond)[-1]
   del cond,initial;assert torch.isfinite(final).all();data=dict(np.load(cache/'poses.npz'));metrics=dict(methods={'da3':{},'gard':{}},reference=info,epoch=epoch,input_sha256=g['input_sha256'],input_paths=g['input_paths'],ate_units='meter' if info['kind']=='hypersim_gt' else 'arbitrary_SfM_units; do not compare across scenes')
   for branch,rayflag in [('camera',False),('ray',True)]:
    torch.manual_seed(42);prediction,_=encoder.model.forward(image,export_feat_layers=[],ref_view_strategy='first',mvrm_cfg=cfg.mvrm.val,mode='val',mvrm_result={('restored_latent',17):final},use_ray_pose=rayflag);encoder._check_ref(prediction);pose=prediction.extrinsics[0].float().cpu().numpy();k=prediction.intrinsics[0].float().cpu().numpy();np.savez_compressed(folder/('poses_ray.npz' if rayflag else 'poses.npz'),da3=data[branch],gard=pose,reference=ref,da3_intrinsics=data[branch+'_intrinsics'],gard_intrinsics=k)
    metrics['methods']['da3'][branch]=pose_metrics(data[branch],ref);metrics['methods']['gard'][branch]=pose_metrics(pose,ref);plot_pose(folder,data[branch],pose,ref,filename='ray_comparison.png' if rayflag else 'camera_comparison.png');del prediction
   dump(folder/'metrics.json',metrics);dump(folder/'complete.json',dict(epoch=epoch,name=name,seconds=time.time()-started));count+=1;del image,final;gc.collect();torch.cuda.empty_cache();summarize(out,a.epochs,items);print('DONE',epoch,name,'seconds',time.time()-started,flush=True)
 summarize(out,a.epochs,items);dump(out/'status.json',dict(status='complete_requested_available_checkpoints',epochs=a.epochs,completed=count,total=len(items)*len(checks),gpu=6));print('COMPLETE',count,flush=True)
if __name__=='__main__':main()
