"""Frozen DA3 feature extraction with the existing early-return API."""
import torch
from omegaconf import OmegaConf

@torch.no_grad()
def encode_training_features(encoder, images, train_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=True))
    if len(cfg.extract_feat_layers) != 1:
        raise ValueError('Feature-only training requires one extraction layer')
    cfg.break_and_return_feat = True
    output, features, reference_indices = encoder(
        image=images, export_feat_layers=[], mvrm_cfg=cfg, mode='train')
    if output is not None:
        raise RuntimeError('Expected DA3 feature-only early return')
    latent = features[('extract_feat', cfg.extract_feat_layers[0])].detach().float()
    return latent, reference_indices
