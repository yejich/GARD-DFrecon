"""Save t=1 masks in encoder reference-first order, with token-grid metadata."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def save_mask_grid(folder, images, pixel_masks, flags, reference_indices, logits, metadata=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if images.shape[0]!=1:raise ValueError('Save one group at a time')
    v,h,w=images.shape[1],images.shape[-2],images.shape[-1]
    ref=0 if reference_indices is None else int(torch.as_tensor(reference_indices).reshape(-1)[0])
    order=[ref]+[i for i in range(v) if i!=ref]
    images=images[0,order].detach().cpu()
    target=pixel_masks[0,order].detach().cpu().float()*flags[0,order].detach().cpu()[:,None,None]
    probability=logits[0].detach().float().cpu().sigmoid()
    if probability.ndim==2:
        probability=probability.reshape(v,h//14,w//14)
        probability=F.interpolate(probability[:,None],size=(h,w),mode='nearest')[:,0]
    binary=probability>=.5
    panel=Image.new('RGB',(w*4,(h+24)*v),'white');draw=ImageDraw.Draw(panel)
    for i in range(v):
        rgb=Image.fromarray(images[i].permute(1,2,0).clamp(0,1).mul(255).byte().numpy())
        masks=[Image.fromarray(array.mul(255).byte().numpy(),'L') for array in [target[i],probability[i],binary[i].float()]]
        probability_u16=Image.fromarray(np.round(probability[i].numpy()*65535).astype(np.uint16))
        probability_u16.save(folder/f'view_{i}_probability.png')
        masks[0].save(folder/f'view_{i}_gt.png');masks[2].save(folder/f'view_{i}_prediction.png')
        for j,(label,im) in enumerate(zip(['Input RGB','GT object mask','Prediction probability','Prediction >= 0.5'],[rgb,*masks])):
            panel.paste(im.convert('RGB'),(j*w,i*(h+24)+24))
            draw.text((j*w+4,i*(h+24)+4),f'{label} | original view {order[i]}',fill='black')
    panel.save(folder/'grid.jpg',quality=92)
    info=dict(timestep=1,reference_index=ref,view_order=order,threshold=.5,
              patch_display='nearest-neighbor enlargement of token logits',**(metadata or {}))
    (folder/'metadata.json').write_text(json.dumps(info,indent=2))
    return str(folder/'grid.jpg')
