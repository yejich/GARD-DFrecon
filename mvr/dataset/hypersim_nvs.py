"""Add held-out clean supervision without encoding novel RGB in the source."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .hypersim_pairs import pair_collate_fn

class HypersimNVS(Dataset):
 def __init__(self,pairs,novel_views=1):
  if novel_views<1:raise ValueError('Need novel views')
  self.pairs=pairs;self.novel_views=novel_views
 def __len__(self):return len(self.pairs)
 def __getitem__(self,item):
  sample=dict(self.pairs[item]);ids=sample['frame_ids'];anchor=self.pairs.records[ids[0]]
  pool=[i for i in anchor['candidates'] if i not in ids and self.pairs.records[i]['scene']==anchor['scene']]
  if len(pool)<self.novel_views:raise ValueError('Not enough distinct covisible clean targets')
  seed=int(item[2]) if isinstance(item,(tuple,list)) else self.pairs.manifest['seed']+int(item)
  chosen=random.Random(seed^0x4e5653).sample(pool,self.novel_views)
  sample['nvs_frame_ids']=chosen;sample['nvs_hq_views']=[self.pairs.image(self.pairs.records[i],'clean.png') for i in chosen]
  sample['nvs_directories']=[str(self.pairs.root/self.pairs.records[i]['directory']) for i in chosen]
  sample['source_directories']=[str(self.pairs.root/self.pairs.records[i]['directory']) for i in ids]
  return sample

def nvs_collate_fn(samples):
 out=pair_collate_fn(samples)
 novel=np.stack([np.stack(s['nvs_hq_views']) for s in samples]).astype(np.float32)/255
 out['nvs_hq_views']=torch.from_numpy(novel).permute(0,1,4,2,3)
 for key in ['nvs_frame_ids','nvs_directories','source_directories']:out[key]=[s[key] for s in samples]
 return out

class MixedHypersimNVS(Dataset):
 """Hold out M=total_views-N clean views, preserving mixed epoch ownership."""
 def __init__(self,pairs,total_views=4):
  if total_views!=4:raise ValueError("Mixed NVS currently requires V=4")
  self.pairs=pairs;self.total_views=total_views
  self.negative_members={i:g['indices'] for g in pairs.negative_groups for i in g['indices']}
 def __len__(self):return len(self.pairs)
 def __getitem__(self,item):
  sample=dict(self.pairs[item]);source,anchor,views,seed=item;ds=self.pairs.datasets[source]
  offset=len(self.pairs.datasets[0]) if source else 0
  ids=[i-offset for i in sample['frame_ids']]
  candidates=self.negative_members[anchor] if source else ds.records[anchor]['candidates']
  pool=[i for i in candidates if i not in ids]
  novel_views=self.total_views-views
  if novel_views<1 or len(pool)<novel_views:raise ValueError('Not enough distinct clean M views for fixed V')
  chosen=random.Random(seed^0x4e5653).sample(pool,novel_views)
  sample['nvs_frame_ids']=[i+offset for i in chosen];sample['nvs_hq_views']=[ds.image(ds.records[i],'clean.png') for i in chosen]
  sample['nvs_directories']=[str(ds.root/ds.records[i]['directory']) for i in chosen]
  sample['source_directories']=[str(ds.root/ds.records[i]['directory']) for i in ids]
  sample['camera_records']=[{k:ds.records[i][k] for k in ['scene','camera','frame']} for i in ids+chosen]
  return sample

# The trainer uses these records only to place M in the source DA3 gauge;
# source camera matrices are never passed as GARD tokens.
_original_nvs_collate_fn=nvs_collate_fn
def nvs_collate_fn(samples):
 out=_original_nvs_collate_fn(samples)
 if 'camera_records' in samples[0]:out['camera_records']=[s['camera_records'] for s in samples]
 if 'data_source' in samples[0]:out['data_source']=[s['data_source'] for s in samples]
 return out


def mixed_nvs_sampler(pairs,batch_size,rank,world_size,*,total_views=4,min_source_views=2,max_source_views=3,**kwargs):
 """Same N across ranks; every original anchor must support all V clean views."""
 from .hypersim_mixed import MixedPairBatchSampler
 if total_views!=4 or not 2<=min_source_views<=max_source_views<total_views:
  raise ValueError('Require V=4 and 2 <= min_source_views <= max_source_views <= 3')
 sampler=MixedPairBatchSampler(pairs,batch_size,rank,world_size,min_views=min_source_views,max_views=max_source_views,**kwargs)
 if sampler.sampling_mode == 'mixed':
  for n,pool in sampler.original_pools.items():
   eligible={scene:[i for i in ids if len(pairs.datasets[0].records[i]['candidates'])>=total_views-1] for scene,ids in pool.items()}
   sampler.original_pools[n]={scene:ids for scene,ids in eligible.items() if ids}
   if not sampler.original_pools[n]:raise ValueError(f'No original anchors support V={total_views} for N={n}')
 if any(len(set(group['indices']))<total_views for group in pairs.negative_groups):
  raise ValueError('Every negative variant must provide four distinct views')
 sampler.summary.update(total_views=total_views,min_source_views=min_source_views,max_source_views=max_source_views,
                        min_novel_views=total_views-max_source_views,max_novel_views=total_views-min_source_views)
 return sampler
