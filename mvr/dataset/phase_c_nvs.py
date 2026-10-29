"""Dataset and sampler for Phase-C clean multi-dataset NVS pretraining."""
import h5py, json, random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _resize(img, resolution=504, patch=14):
    h,w=img.shape[:2]; scale=resolution/max(h,w); nh=max(patch,int(round(h*scale))//patch*patch); nw=max(patch,int(round(w*scale))//patch*patch)
    img=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    return img,nh/h,nw/w


def _read(path):
    p=str(path)
    if p.endswith('.hdf5'):
        with h5py.File(p,'r') as f: x=f['dataset'][:].astype(np.float32)
        if x.max()<=1.5:
            bright=.3*x[...,0]+.59*x[...,1]+.11*x[...,2]; valid=bright>1e-4; cur=np.percentile(bright[valid],90) if valid.any() else 1.; scale=np.power(.8,2.2)/max(cur,1e-4); x=np.power(np.maximum(scale*x,0),1/2.2)
        return np.clip(x*255,0,255).astype(np.uint8)
    x=cv2.imread(p,cv2.IMREAD_COLOR)
    if x is None: raise FileNotFoundError(p)
    return cv2.cvtColor(x,cv2.COLOR_BGR2RGB)


class PhaseCNVS(Dataset):
    def __init__(self, manifest, resolution=504, patch_size=14, seed=42):
        self.manifest=json.loads(Path(manifest).read_text()); self.groups=[g for g in self.manifest['groups'] if g['split']=='train']; self.resolution=resolution; self.patch=patch_size; self.seed=seed
        self.by_dataset={n:[i for i,g in enumerate(self.groups) if g['dataset']==n] for n in ('hypersim','tartanair','dl3dv')}
        if any(not v for v in self.by_dataset.values()): raise ValueError({k:len(v) for k,v in self.by_dataset.items()})
    def __len__(self): return len(self.groups)
    def _one(self,g,idx):
        v=g['views'][idx]; img=_read(v['path']); img,sx,sy=_resize(img,self.resolution,self.patch); k=np.asarray(v['K'],np.float32); k[0]*=sx; k[1]*=sy
        return torch.from_numpy(img).permute(2,0,1).float()/255.,np.asarray(v['c2w'],np.float32),k
    def __getitem__(self,key):
        if isinstance(key,tuple): gi,n,seed=map(int,key)
        else: gi,n,seed=int(key),2,self.seed+int(key)
        g=self.groups[gi]; rng=random.Random(seed); views=g['views']; anchor=rng.randrange(len(views)); radius=max(3, min(len(views)//2, 20*n)); lo=max(0,anchor-radius); hi=min(len(views),anchor+radius+1); pool=list(range(lo,hi));
        if len(pool)<4: pool=list(range(len(views)))
        chosen=[anchor]+rng.sample([i for i in pool if i!=anchor],3); chosen=sorted(chosen)
        rng.shuffle(chosen); src=chosen[:n]; tgt=chosen[n:]
        source=[]; target=[]; poses=[]; ks=[]
        for i in chosen:
            x,p,k=self._one(g,i); poses.append(p);ks.append(k); (source if i in src else target).append(x)
        # Preserve explicit source-first / target-last ordering for the model.
        order=src+tgt; source=[self._one(g,i)[0] for i in src]; target=[self._one(g,i)[0] for i in tgt]
        h,w=source[0].shape[-2:]
        return dict(lq_views=torch.stack(source),hq_views=torch.stack(source),nvs_hq_views=torch.stack(target),camera_c2w=torch.from_numpy(np.stack([np.asarray(g['views'][i]['c2w'],np.float32) for i in order])),camera_k=torch.from_numpy(np.stack([self._one(g,i)[2] for i in order])),pixel_masks=torch.zeros(n,h,w),distractor_views=torch.zeros(n,dtype=torch.bool),data_source=g['dataset'],scene=g['scene'],source_views=n)


class PhaseCUpdateSampler(Sampler):
    def __init__(self,ds,rank,world,micro,accum,steps_per_epoch,seed=42): self.ds,self.rank,self.world,self.micro,self.accum,self.steps,self.seed=ds,rank,world,micro,accum,steps_per_epoch,seed; self.epoch=0
    def set_epoch(self,e): self.epoch=e
    def __len__(self): return self.steps*self.accum
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch*1000003); names=('hypersim','tartanair','dl3dv')
        for _ in range(len(self)):
            rows=[]
            for _ in range(self.world*self.micro):
                name=names[rng.randrange(3)]; gi=rng.choice(self.ds.by_dataset[name]); n=rng.choice((1,2,3)); rows.append((gi,n,rng.getrandbits(63)))
            yield rows[self.rank*self.micro:(self.rank+1)*self.micro]


def phase_c_collate(samples):
    if len(samples)!=1: raise ValueError('Phase C variable-N loader currently requires microbatch=1')
    s=samples[0]
    out={k:(v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k,v in s.items()}
    return out
