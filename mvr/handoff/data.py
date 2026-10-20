"""Portable manifests, equal update budgets and explicit held-out visualization."""
import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
from mvr.dataset.hypersim_pairs import HypersimPairs,pair_collate_fn
from mvr.dataset.hypersim_mixed import filter_scenes

class ExperimentPairs(Dataset):
    def __init__(self,cfg):
        self.cfg=cfg
        self.datasets=[]
        held=set(json.loads(Path(cfg.data.scene_split).read_text())['eval_scenes'])
        for manifest,root in [(cfg.data.manifest,cfg.data.root),(cfg.data.negative_manifest,cfg.data.negative_root)]:
            ds=HypersimPairs(manifest,root=root,process_res=cfg.resolution,include_object_masks=True)
            held.update(ds.manifest.get('eval_scenes',[]))
            self.datasets.append(ds)
        for ds in self.datasets:ds.records,_=filter_scenes(ds.records,held)
        self.heldout=sorted(held)
        self.pools=[{} for _ in self.datasets]
        for source,ds in enumerate(self.datasets):
            for i,r in enumerate(ds.records):
                if len(r['candidates'])>=3:self.pools[source].setdefault(r['scene'],[]).append(i)
        if not self.pools[0] or (cfg.dataset=='mixed' and not self.pools[1]):raise ValueError('No eligible 4-view train groups')
    def __len__(self):return self.cfg.steps_per_epoch*self.cfg.global_batch
    def __getitem__(self,item):
        source,anchor,seed=item;ds=self.datasets[source]
        sample=ds[(anchor,4,seed)]
        sample['camera_records']=[{k:ds.records[i][k] for k in ['scene','camera','frame']} for i in sample['frame_ids']]
        sample['data_source']='original' if source==0 else 'negative'
        return sample

class UpdateSampler(Sampler):
    def __init__(self,ds,rank,world,micro,accum):self.ds,self.rank,self.world,self.micro,self.accum=ds,rank,world,micro,accum;self.epoch=0
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return self.ds.cfg.steps_per_epoch*self.accum
    def __iter__(self):
        rng=random.Random(self.ds.cfg.seed+self.epoch*1000003)
        for _ in range(len(self)):
            global_rows=[]
            for _ in range(self.world*self.micro):
                source=int(self.ds.cfg.dataset=='mixed' and rng.random()<self.ds.cfg.negative_probability)
                pool=self.ds.pools[source];scene=rng.choice(sorted(pool));anchor=rng.choice(pool[scene])
                global_rows.append((source,anchor,rng.getrandbits(63)))
            yield global_rows[self.rank*self.micro:(self.rank+1)*self.micro]

def collate(samples):
    out=pair_collate_fn(samples)
    out['camera_records']=[s['camera_records'] for s in samples]
    out['data_source']=[s['data_source'] for s in samples]
    return out


def read_eval_manifest(path):
    data=json.loads(Path(path).read_text())
    base=Path(path).resolve().parent
    groups=data['groups']
    for g in groups:
        g['input_paths']=[str((base/p).resolve()) if not Path(p).is_absolute() else p for p in g['input_paths']]
        if g.get('clean_paths'):g['clean_paths']=[str((base/p).resolve()) if not Path(p).is_absolute() else p for p in g['clean_paths']]
    return groups


def freeze_visualization(cfg,path,groups_per_scene=2):
    """Deterministic held-out synthetic defaults; explicit manifest can override."""
    out=[];held=set(json.loads(Path(cfg.data.scene_split).read_text())['eval_scenes'])
    for kind,manifest,root in [('original',cfg.data.manifest,cfg.data.root),('negative',cfg.data.negative_manifest,cfg.data.negative_root)]:
        m=json.loads(Path(manifest).read_text());records=m['eval'];selected={}
        for i,r in enumerate(records):
            if r['scene'] not in held:raise ValueError('Eval contains training scene')
            if len(r['candidates'])<3 or selected.get(r['scene'],0)>=groups_per_scene:continue
            ids=[i]+r['candidates'][:3];rows=[records[j] for j in ids]
            if len({x['directory'] for x in rows})!=4:continue
            index=selected.get(r['scene'],0);selected[r['scene']]=index+1
            flags=[False,True,True,False]
            inputs=[str(Path(root)/x['directory']/('distractor.png' if f else 'clean.png')) for x,f in zip(rows,flags)]
            out.append(dict(id=f'{kind}_{r["scene"]}_{index:02d}',dataset='Hypersim' if kind=='original' else 'HypersimNegative',input_paths=inputs,
                input_sha256=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs],
                clean_paths=[str(Path(root)/x['directory']/'clean.png') for x in rows],
                camera_records=[{k:x[k] for k in ['scene','camera','frame']} for x in rows],
                mask_paths={str(v):[str(Path(root)/x['directory']/'mask_object.png')] for v,x in enumerate(rows) if flags[v]},clutter_views=[1,2],
                note='For NVS: first 3 are sources; last clean view is held-out novel. Baseline receives all 4 views. Compare source RGB on first3 separately.'))
    if not out:raise ValueError('No held-out eval visualization groups')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps({'schema_version':'handoff_visualization_v1','groups':out,'selection':'first two eligible groups per held-out scene/type; no prediction-based selection'},indent=2)+'\n')
    return out
