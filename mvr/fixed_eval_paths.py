"""Resolve relocated fixed-suite assets without changing the frozen sample selection."""
import hashlib
from pathlib import Path


def resolve_asset_path(value):
    path = Path(value)
    if path.exists():
        return path
    relocated = str(path)
    if relocated.startswith('/mnt/dataset1/'):
        relocated = relocated.replace('/mnt/dataset1/', '/mnt/nvme1n1/', 1)
    relocated = relocated.replace('/hypersim_001_pairs/scenes_v2/', '/hypersim_001_pairs/scenes_001/')
    relocated = relocated.replace('/hypersim_pairs/scenes_v2/', '/hypersim_pairs/scenes_001/')
    relocated = relocated.replace('/datasets/hypersim_negative_007_010/', '/datasets/hypersim_negative/')
    relocated = relocated.replace(
        '/mnt/nvme1n1/yejichoi/hypersim-synthesis/workspace/hypersim/ml-hypersim/evermotion_dataset/scenes/',
        '/mnt/nvme1n1/MV_Restoration/hypersim/data/')
    candidate = Path(relocated)
    return candidate if candidate.exists() else path


def resolve_asset_paths(value):
    """Return a resolved copy, keeping original manifests intact for provenance."""
    if isinstance(value, dict):
        return {k: resolve_asset_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_asset_paths(v) for v in value]
    if isinstance(value, str) and value.startswith('/'):
        return str(resolve_asset_path(value))
    return value


def verify_frozen_assets(data):
    """Check both original and relocated files against the existing hashes."""
    for row in data['groups']:
        group = row.get('group', row)
        pairs = list(zip(group['input_paths'], group['input_sha256'], strict=True))
        if 'reference_paths' in row:
            pairs.extend(zip(row['reference_paths'], row['reference_sha256'], strict=True))
        if 'target_path' in row:
            pairs.append((row['target_path'], row['target_sha256']))
        for filename, expected in pairs:
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Fixed eval asset changed: {filename}')
