"""Append held-out clean supervision; keep source LQ RGB V-only."""
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
 """Hold out one covisible clean M while preserving mixed epoch ownership."""
 def __init__(self,pairs):
  self.pairs=pairs
  self.negative_members={i:g['indices'] for g in pairs.negative_groups for i in g['indices']}
 def __len__(self):return len(self.pairs)
 def __getitem__(self,item):
  sample=dict(self.pairs[item]);source,anchor,views,seed=item;ds=self.pairs.datasets[source]
  offset=len(self.pairs.datasets[0]) if source else 0
  ids=[i-offset for i in sample['frame_ids']]
  candidates=self.negative_members[anchor] if source else ds.records[anchor]['candidates']
  pool=[i for i in candidates if i not in ids]
  if not pool:raise ValueError('No distinct clean M remaining')
  chosen=random.Random(seed^0x4e5653).choice(pool)
  sample['nvs_frame_ids']=[chosen+offset];sample['nvs_hq_views']=[ds.image(ds.records[chosen],'clean.png')]
  sample['nvs_directories']=[str(ds.root/ds.records[chosen]['directory'])]
  sample['source_directories']=[str(ds.root/ds.records[i]['directory']) for i in ids]
  sample['camera_records']=[{k:ds.records[i][k] for k in ['scene','camera','frame']} for i in ids+[chosen]]
  return sample

# The trainer uses these records only to place M in the source DA3 gauge;
# source camera matrices are never passed as GARD tokens.
_original_nvs_collate_fn=nvs_collate_fn
def nvs_collate_fn(samples):
 out=_original_nvs_collate_fn(samples)
 if 'camera_records' in samples[0]:out['camera_records']=[s['camera_records'] for s in samples]
 if 'data_source' in samples[0]:out['data_source']=[s['data_source'] for s in samples]
 return out
