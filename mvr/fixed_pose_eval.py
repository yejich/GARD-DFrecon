"""Camera references in fixed input order; Hypersim GT or COLMAP SfM poses."""
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scripts.eval.quant_eval.metrics import pose_metrics

ROOT = Path(__file__).resolve().parents[1]


def resolve_reference_path(value):
    path = Path(value)
    if not path.exists() and str(path).startswith('/mnt/dataset1/'):
        relocated = Path('/mnt/nvme1n1') / path.relative_to('/mnt/dataset1')
        if relocated.exists():
            return relocated
    if not path.exists():
        raise FileNotFoundError(f'Pose reference unavailable: {path}')
    return path


def pose_reference(group):
    if 'pose_reference_extrinsics' in group:
        extrinsics = np.asarray(group['pose_reference_extrinsics'], dtype=np.float64)
        if extrinsics.shape != (len(group['input_paths']), 3, 4) or not np.isfinite(extrinsics).all():
            raise ValueError(f'Invalid stored reference poses for {group["id"]}')
        return extrinsics, dict(group['pose_reference_info'])
    paths = group['input_paths']
    if group['dataset'].startswith('Hypersim'):
        scene = re.search(r'ai_\d{3}_\d{3}', paths[0]).group()
        camera = re.search(r'cam_\d{2}', paths[0]).group()
        cfg = json.loads((ROOT/'run_configs/eval/d2_fixed_4view_2clutter_rgb.json').read_text())['correspondence']
        cfg.update(group.get('pose_correspondence', {}))
        with open(resolve_reference_path(cfg['hypersim_camera_parameters'])) as stream:
            row = next(r for r in csv.DictReader(stream) if r['scene_name'] == scene)
        scale = float(row['settings_units_info_meters_scale'])
        detail = resolve_reference_path(cfg['hypersim_data_root'])/scene/'_detail'/camera
        arrays = []
        for name in ('frame_indices', 'orientations', 'positions'):
            with h5py.File(detail/f'camera_keyframe_{name}.hdf5') as stream:
                arrays.append(stream['dataset'][:])
        ids, rotations, centers = arrays
        ids = ids.reshape(-1).tolist()
        extrinsics = []
        for frame in group['frame_order']:
            index = ids.index(frame)
            rotation = np.diag([1., -1., -1.]) @ rotations[index].T
            extrinsics.append(np.c_[rotation, -rotation @ (centers[index]*scale)])
        return np.stack(extrinsics), {'kind': 'hypersim_gt', 'unit': 'meter', 'source': str(detail)}
    from depth_anything_3.utils.read_write_model import read_images_binary
    candidates = [p/'sparse/0/images.bin' for p in Path(paths[0]).parents]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(f'COLMAP reference unavailable for {group["id"]}')
    images = {im.name: im for im in read_images_binary(source).values()}
    extrinsics = []
    for path in paths:
        im = images[Path(path).name]
        extrinsics.append(np.c_[im.qvec2rotmat(), im.tvec])
    return np.stack(extrinsics), {'kind': 'colmap_sfm', 'unit': 'arbitrary_colmap_scale', 'source': str(source)}


def score_poses(baseline, restored, reference, reference_info):
    results, scalars = {}, {}
    for name, prediction in [('da3', baseline), ('gard', restored)]:
        if not np.isfinite(prediction).all():
            raise ValueError(f'Nonfinite {name} camera prediction')
        result = pose_metrics(prediction, reference)
        results[name] = result
        for key in ('rre_median', 'rte_median', 'auc_5', 'auc_10', 'auc_20', 'auc_30'):
            if np.isfinite(result[key]):
                scalars[f'pose/{reference_info["kind"]}/{name}/{key}'] = result[key]
        # Do not present arbitrary COLMAP units as metric position accuracy.
        if reference_info['kind'] == 'hypersim_gt' and not result['ate_degenerate']:
            scalars[f'pose/hypersim_gt/{name}/ate_rmse_sim3_m'] = result['ate_rmse_sim3']
    return scalars, results
