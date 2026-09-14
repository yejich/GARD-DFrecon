"""Per-sample means for token/view groups; independent of their sizes."""
import torch
import torch.nn.functional as F


def token_groups(pixel_masks, distractor_views, reference_indices, patch_size=14):
    """Follow DA3's per-sample reference-first permutation; CLS is unmasked."""
    b, v, h, w = pixel_masks.shape
    mask = F.avg_pool2d(pixel_masks.float().reshape(b*v,1,h,w), patch_size, patch_size)
    mask = (mask.flatten(1) >= .5).reshape(b,v,-1)
    mask &= distractor_views[...,None]
    mask = torch.cat([torch.zeros(b,v,1,dtype=torch.bool,device=mask.device), mask],dim=-1)
    if reference_indices is None:
        reference_indices = torch.zeros(b,dtype=torch.long,device=mask.device)
    reference_indices = torch.as_tensor(reference_indices,device=mask.device).reshape(-1)
    if reference_indices.numel() != b:
        raise ValueError('One reference index is required per sample')
    orders=[]
    for ref in reference_indices.tolist():
        orders.append([ref]+[i for i in range(v) if i != ref])
    order=torch.tensor(orders,device=mask.device)
    mask=torch.gather(mask,1,order[...,None].expand_as(mask))
    views=torch.gather(distractor_views,1,order)
    return mask,views


def grouped_mse(prediction, target, mask, distractor_views, cfg):
    errors=(prediction.float()-target.float()).square().mean(-1)  # B,V,T
    if mask.shape != errors.shape:
        raise ValueError(f'Token mask {mask.shape} does not match features {errors.shape}')
    mode=cfg.get('mode','group2')
    def mean(selected):
        count=selected.sum(dim=(1,2))
        values=(errors*selected).sum(dim=(1,2))/count.clamp_min(1)
        return values, count
    terms={}
    if mode=='flat':
        return errors.mean(), {'flat':errors.mean()}
    if mode=='group2':
        groups={'distractor':mask,'clean':~mask}
        weights={'distractor':cfg.get('lambda_distractor',1.),'clean':cfg.get('lambda_clean',1.)}
    elif mode=='view2':
        selected=distractor_views[...,None].expand_as(mask)
        groups={'distractor':selected,'clean':~selected}
        weights={'distractor':cfg.get('lambda_distractor',1.),'clean':cfg.get('lambda_clean',1.)}
    elif mode=='group3':
        selected=distractor_views[...,None].expand_as(mask)
        groups={'clean_view':~selected,'same_view_clean':selected&~mask,'distractor':mask}
        weights={'clean_view':cfg.get('lambda_a',1.),'same_view_clean':cfg.get('lambda_b',1.),'distractor':cfg.get('lambda_c',1.)}
    else:
        raise ValueError(mode)
    total=errors.sum()*0.
    for name,selected in groups.items():
        values,counts=mean(selected)
        terms[name]=values.mean()
        terms[name+'_tokens']=counts.float().mean()
        total=total+float(weights[name])*values.mean()
    return total,terms
