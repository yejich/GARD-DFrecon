"""CPU-only, rerunnable comparison of matching registry qualitative evaluations."""
import argparse
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def grid(path, columns):
    views, height, width = columns[0][1].shape[:3]
    panel = Image.new('RGB', (width * len(columns), (height + 24) * views), 'white')
    draw = ImageDraw.Draw(panel)
    for col, (label, data) in enumerate(columns):
        for view in range(views):
            panel.paste(Image.fromarray(np.uint8(np.clip(data[view], 0, 1) * 255)),
                        (col * width, view * (height + 24) + 24))
            draw.text((col * width + 4, view * (height + 24) + 4),
                      f'{label} | view {view}', fill='black')
    panel.save(path, quality=92)


def panel_column(path, column, shape):
    """Recover the already processed pixels, avoiding a second resize/crop policy."""
    views, height, width = shape[:3]
    with Image.open(path) as image:
        if image.height != views * (height + 24) or image.width % width:
            raise ValueError(f'Unexpected source panel dimensions: {path}')
        if image.width < (column + 1) * width:
            return None
        return np.stack([np.asarray(image.crop((column * width, v * (height + 24) + 24,
                        (column + 1) * width, (v + 1) * (height + 24))), dtype=float) / 255
                         for v in range(views)])


def plot_poses(path, series):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    reference = next((e for label, e in series if label == 'Reference' and e is not None), None)
    anchor = reference if reference is not None else series[0][1]
    anchor = np.asarray(anchor, dtype=float)
    target = -np.einsum('vji,vj->vi', anchor[:, :3, :3], anchor[:, :3, 3])
    target -= target.mean(0)
    target_scale = max(float(np.median([np.linalg.norm(target[i] - target[j])
                       for i in range(len(target)) for j in range(i)])), 1e-8)
    target /= target_scale
    for label, extrinsics in series:
        if extrinsics is None or np.asarray(extrinsics).size == 0:
            continue
        e = np.asarray(extrinsics)
        points = -np.einsum('vji,vj->vi', e[:, :3, :3], e[:, :3, 3])
        points -= points.mean(0)
        distances = [np.linalg.norm(points[i] - points[j]) for i in range(len(points)) for j in range(i)]
        points /= max(float(np.median(distances)), 1e-8)
        # Proper rotation only: never allow a reflection to hide a pose error.
        u, _, vt = np.linalg.svd(target.T @ points)
        signs = np.ones(3)
        signs[-1] = np.linalg.det(u @ vt)
        rotation = (u * signs) @ vt
        points = points @ rotation.T
        ax.plot(*points.T, 'o-', label=label)
        for i, point in enumerate(points):
            ax.text(*point, str(i))
    ax.legend()
    ax.set_title('Camera centers: rotation aligned to reference (DA3 if absent)\nCentered / median baseline; shape diagnostic only')
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for epoch in ('003', '010', '020'):
        parser.add_argument('--epoch' + epoch, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = [getattr(args, 'epoch' + e).resolve() for e in ('003', '010', '020')]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = sorted({p.parent.relative_to(root) for root in roots for p in root.rglob('complete.json')})
    sections, status = [], []
    for name in names:
        folders = [root / name for root in roots]
        missing = [str(i) for i, folder in zip((3, 10, 20), folders) if not (folder / 'complete.json').exists()]
        if missing:
            sections.append(f'<h2>{html.escape(str(name))}</h2><p>Pending epochs: {", ".join(missing)}</p>')
            status.append({'group': str(name), 'pending_epochs': missing})
            continue
        metadata = [json.loads((f / 'complete.json').read_text()) for f in folders]
        groups = [m['group'] for m in metadata]
        if any(g != groups[0] for g in groups[1:]):
            raise ValueError(f'Group identity/input metadata mismatch: {name}')
        with np.load(folders[0] / 'da3.npz') as f:
            base = {k: f[k] for k in ('rgb', 'depth', 'extrinsics')}
        gard = []
        for folder in folders:
            with np.load(folder / 'gard.npz') as f:
                gard.append({k: f[k] for k in ('rgb', 'depth', 'mask_probability')})
            with np.load(folder / 'da3.npz') as f:
                if not np.allclose(f['rgb'], base['rgb'], atol=1e-5) or not np.allclose(f['depth'], base['depth'], atol=1e-5):
                    raise ValueError(f'DA3 baseline differs between epochs: {name}')
        shape = base['rgb'].shape
        source = folders[0] / 'rgb_comparison.jpg'
        inputs = panel_column(source, 0, shape)
        clean = panel_column(source, 3, shape)
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        labels = ('Epoch 3', 'Epoch 10', 'Epoch 20')
        columns = [('Input', inputs), ('DA3 decoded', base['rgb'])] + [(label, data['rgb']) for label, data in zip(labels, gard)]
        if clean is not None:
            columns.append(('Clean GT (per view)', clean))
        grid(target / 'rgb_comparison.jpg', columns)
        gt = panel_column(folders[0] / 'mask_comparison.jpg', 1, shape)
        grid(target / 'mask_comparison.jpg', [('Input', inputs), ('GT (gray=unknown)', gt)] +
             [(label, np.repeat(data['mask_probability'][..., None], 3, -1)) for label, data in zip(labels, gard)])
        depths = [base['depth']] + [data['depth'] for data in gard]
        valid = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in depths])
        lo, hi = np.percentile(valid, [2, 98])
        colored = [plt.get_cmap('turbo')(np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1))[..., :3] for d in depths]
        grid(target / 'depth_comparison.jpg', [('Input', inputs)] + list(zip(('DA3',) + labels, colored)))
        for pose_file, output_file in [('poses.npz', 'pose_comparison.png'), ('poses_ray.npz', 'pose_ray_comparison.png')]:
            series = []
            for i, folder in enumerate(folders):
                with np.load(folder / pose_file, allow_pickle=False) as poses:
                    if i == 0:
                        series.append(('DA3', poses['da3']))
                        try:
                            reference = poses['reference']
                        except ValueError:
                            reference = None  # Saved None is an object array; never unpickle it.
                        if reference is not None:
                            series.append(('Reference', reference))
                    series.append((labels[i], poses['gard']))
            plot_poses(target / output_file, series)
        record = {'group': str(name), 'input_paths': groups[0]['input_paths'], 'depth_scale': [float(lo), float(hi)],
                  'source_directories': [str(f) for f in folders], 'input_clean_source': 'processed original JPEG panel'}
        (target / 'comparison.json').write_text(json.dumps(record, indent=2))
        rel = name.as_posix()
        section = f'<h2>{html.escape(str(name))}</h2>'
        for filename in ('rgb_comparison.jpg', 'mask_comparison.jpg', 'depth_comparison.jpg', 'pose_comparison.png', 'pose_ray_comparison.png'):
            section += f'<img loading="lazy" src="{html.escape(rel)}/{filename}">'
        for label, folder in zip(labels, folders):
            section += f'<p>{label}: '
            for path in [folder / 'complete.json', folder / 'rgb_comparison.jpg'] + sorted(folder.glob('*.ply')):
                href = html.escape(os.path.relpath(path, out))
                section += f'<a href="{href}">{html.escape(path.name)}</a> '
            section += '</p>'
        sections.append(section)
        status.append(record)
    (out / 'index.html').write_text('<meta charset="utf-8"><style>body{font-family:sans-serif}img{max-width:100%;display:block;margin-bottom:16px}</style><h1>Mask head EMA: epochs 3 / 10 / 20</h1><p>Depth uses one 2–98 percentile scale across all four models per group. Camera plots compare normalized shape after proper rotation alignment to the reference (DA3 if absent); they are not absolute position accuracy. Input and per-view clean are recovered from the original processed JPEG panel.</p>' + ''.join(sections))
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    print(f'Compared {sum("pending_epochs" not in r for r in status)}/{len(names)} groups: {out / "index.html"}')


if __name__ == '__main__':
    main()
