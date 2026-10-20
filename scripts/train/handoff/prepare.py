"""Resolve recipient-local paths, batch settings and the nine experiment configs."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from omegaconf import OmegaConf

def prepare(args):
    cfg=OmegaConf.load(ROOT/'run_configs/JIHYE/handoff/20261013_experiments.yaml')
    spec=cfg.experiments[args.experiment];del cfg.experiments
    cfg.update(dict(dataset=spec.dataset,nvs=spec.nvs,flow=spec.flow,camera_mode=spec.camera,experiment=args.experiment))
    cfg.attention_alignment.distractor_aware.use=spec.distractor_aware
    data=Path(args.data).resolve();cfg.data=dict(manifest=str(data/'manifests/original_full.json'),negative_manifest=str(data/'manifests/negative_full.json'),root=str(data/'hypersim_pairs'),negative_root=str(data/'hypersim_negative'),scene_split=str(data/'manifests/scene_split_policy.json'))
    cfg.camera=dict(root=str(data/'camera_reference/hypersim'),parameters=str(data/'camera_reference/metadata_camera_parameters.csv'))
    cfg.checkpoint=str(Path(args.checkpoint).resolve());cfg.global_batch=args.global_batch;cfg.microbatch=args.microbatch
    if args.global_batch%(args.gpus*args.microbatch):raise ValueError('global batch must divide GPUs × microbatch exactly')
    cfg.accumulation=args.global_batch//(args.gpus*args.microbatch)
    if min(args.gpus,args.microbatch,cfg.accumulation)<1:raise ValueError('Batch settings must be positive')
    for key in ['epochs','steps_per_epoch','workers']:
        value=getattr(args,key)
        if value is not None:cfg[key]=value
    if args.eval_epochs is not None:cfg.eval_epochs=[int(x) for x in args.eval_epochs.split(',') if x]
    if args.save_epochs is not None:cfg.save_epochs=[int(x) for x in args.save_epochs.split(',') if x]
    if args.eval_limit is not None:cfg.eval_limit=args.eval_limit
    cfg.eval_manifest=str(Path(args.eval_manifest).resolve()) if args.eval_manifest else str(data/'eval_visualization/manifest.json')
    for p in [cfg.data.manifest,cfg.data.negative_manifest,cfg.data.scene_split,cfg.camera.parameters,cfg.checkpoint]:
        if not Path(p).is_file():raise FileNotFoundError(p)
    for p in [cfg.data.root,cfg.data.negative_root,cfg.camera.root]:
        if not Path(p).is_dir():raise FileNotFoundError(p)
    # Native packaged eval groups are preferred; do not silently substitute them.
    if not Path(cfg.eval_manifest).is_file():
        raise FileNotFoundError(f'{cfg.eval_manifest}: provide --eval-manifest, or run validate.py --create-eval with the data root first')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True);cfg.run_name=out.name
    OmegaConf.save(cfg,out/'config.yaml')
    print(f'PREPARED {args.experiment}: global={cfg.global_batch} GPUs={args.gpus} micro={cfg.microbatch} accum={cfg.accumulation}',flush=True)
    print(f'CONFIG {out/"config.yaml"}\nEVAL {cfg.eval_manifest}',flush=True)
    return cfg

def parser():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--gpus',type=int,required=True);p.add_argument('--global-batch',type=int,default=32);p.add_argument('--microbatch',type=int,default=1);p.add_argument('--epochs',type=int);p.add_argument('--steps-per-epoch',type=int);p.add_argument('--workers',type=int);p.add_argument('--eval-manifest');p.add_argument('--eval-epochs');p.add_argument('--save-epochs');p.add_argument('--eval-limit',type=int);return p
if __name__=='__main__':prepare(parser().parse_args())
