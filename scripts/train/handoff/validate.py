"""CPU dataset and camera preflight; optional deterministic eval generation."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src')]
from omegaconf import OmegaConf
from mvr.handoff.data import ExperimentPairs,UpdateSampler,collate,freeze_visualization,read_eval_manifest
from mvr.nvs_camera import HypersimCameraReferences

def main():
 p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--data');p.add_argument('--create-eval',action='store_true');p.add_argument('--all-files',action='store_true');a=p.parse_args()
 if a.config:cfg=OmegaConf.load(a.config)
 else:
  cfg=OmegaConf.load(ROOT/'run_configs/JIHYE/handoff/20261013_experiments.yaml');r=Path(a.data).resolve();cfg.data=dict(manifest=str(r/'manifests/original_full.json'),negative_manifest=str(r/'manifests/negative_full.json'),root=str(r/'hypersim_pairs'),negative_root=str(r/'hypersim_negative'),scene_split=str(r/'manifests/scene_split_policy.json'));cfg.camera=dict(root=str(r/'camera_reference/hypersim'),parameters=str(r/'camera_reference/metadata_camera_parameters.csv'));cfg.dataset='mixed';cfg.eval_manifest=str(r/'eval_visualization/manifest.json')
 if a.create_eval:
  if Path(cfg.eval_manifest).exists():raise FileExistsError('Frozen eval manifest already exists; no overwrite')
  freeze_visualization(cfg,cfg.eval_manifest)
 ds=ExperimentPairs(cfg);cameras=HypersimCameraReferences(cfg.camera.root,cfg.camera.parameters)
 sampler=UpdateSampler(ds,0,1,1,1)
 for index,item in enumerate(sampler):
  sample=ds[item[0]];batch=collate([sample]);cameras.cameras(sample['camera_records'],batch['lq_views'].shape[-2:])
  if index>=7:break
 if a.all_files:
  for dataset in ds.datasets:
   for row in dataset.records:
    for name in ['clean.png','distractor.png','mask_object.png','mask_shadow.png']:
     path=dataset.root/row['directory']/name
     if not path.is_file():raise FileNotFoundError(path)
 groups=read_eval_manifest(cfg.eval_manifest)
 train_dirs={str((dataset.root/r['directory']).resolve()) for dataset in ds.datasets for r in dataset.records}
 for g in groups:
  for p in g['input_paths']+g['clean_paths']:
   if not Path(p).is_file():raise FileNotFoundError(p)
   if str(Path(p).parent.resolve()) in train_dirs:raise ValueError('Train/eval image overlap')
  if g.get('input_sha256')!=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in g['input_paths']]:raise ValueError('Eval hash mismatch')
  if any(r['scene'] not in ds.heldout for r in g['camera_records']):raise ValueError('Eval scene overlap')
  cameras.cameras(g['camera_records'],(378,504))
 print(json.dumps(dict(status='PASS',train_records=[len(d.records) for d in ds.datasets],heldout=ds.heldout,eval_groups=len(groups),sampling='8 real CPU batches and camera refs checked',all_files=a.all_files),indent=2))
if __name__=='__main__':main()
