"""RGB-only multiview sequences for frozen-Omega decoder fine-tuning."""
import bisect
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from cut3r_data.base.easy_dataset import EasyDataset
from cut3r_data.utils.image import ImgNorm

class RGBSequenceDataset(EasyDataset):
    def __init__(self, manifest, domain, split, resolution, num_views=4):
        self.sequences = json.loads(Path(manifest).read_text())[domain][split]
        self.domain, self.split = domain, split
        self._resolutions = [tuple(r) for r in resolution]
        self.num_views = num_views
        assert self.sequences and all(len(s['images']) >= num_views for s in self.sequences)
        self.ends = np.cumsum([len(s['images']) for s in self.sequences]).tolist()
    def __len__(self): return self.ends[-1]
    def __getitem__(self, index):
        idx, aspect, views = index if isinstance(index, tuple) else (index, 0, self.num_views)
        sequence_idx = bisect.bisect_right(self.ends, int(idx))
        seq = self.sequences[sequence_idx]
        anchor = int(idx) - (self.ends[sequence_idx-1] if sequence_idx else 0)
        # Nearby distinct frames from the same scene/camera or RealX3D train/val sequence.
        candidates = sorted((j for j in range(len(seq['images'])) if j != anchor), key=lambda j: abs(j-anchor))[:20]
        seed = int(idx) + 100000 * aspect if self.split == 'val' else int(torch.randint(0, 2**32, ()).item())
        rng = np.random.default_rng(seed)
        selected = [anchor] + rng.choice(candidates, views-1, replace=False).tolist()
        result=[]
        for j in selected:
            path=seq['images'][j]
            with Image.open(path) as im:
                im=ImageOps.fit(im.convert('RGB'),self._resolutions[aspect],method=Image.Resampling.LANCZOS)
                tensor=ImgNorm(im)
            result.append({'img':tensor,'domain':self.domain,'scene':seq['scene'],'path':path})
        return result
