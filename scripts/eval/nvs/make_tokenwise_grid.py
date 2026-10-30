"""Make a compact input/mask/step comparison grid for one fixed NVS group."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def fit(im, size):
    return im.convert('RGB').resize(size, Image.Resampling.LANCZOS)


def mask_panel(path, size):
    canvas = Image.new('RGB', size, 'white')
    if path is None:
        return canvas
    mask = Image.open(path).convert('L').resize(size, Image.Resampling.NEAREST)
    base = Image.new('RGB', size, 'black')
    white = Image.new('RGB', size, 'white')
    canvas = Image.composite(white, base, mask)
    return canvas


def psnr_by_view(result_dir):
    metrics = json.loads((result_dir / 'metrics.json').read_text())
    method = metrics['methods']['gard_nvs']
    if 'per_view' in method['rgb_V'] and 'per_view' in method['rgb_M']:
        values = {x['view']: x['psnr'] for x in method['rgb_V']['per_view']}
        values[4] = method['rgb_M']['per_view'][0]['psnr']
        return values
    # Older RobustNeRF statue metrics omit per-view RGB scores. Recompute
    # against the frozen clean reference stored beside the result.
    reference = np.load(result_dir / 'reference.npz')['rgb'].astype(np.float32)
    values = {}
    for view in range(5):
        pred = np.asarray(Image.open(result_dir / f'decoded_view{view}.png').convert('RGB'), dtype=np.float32) / 255.
        mse = float(np.mean((pred - reference[view]) ** 2))
        values[view] = float(-10 * np.log10(max(mse, 1e-12)))
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--group', default='RobustNeRF/android_set01')
    ap.add_argument('--mixed', type=Path, required=True)
    ap.add_argument('--steps', type=Path, nargs=3, required=True)
    ap.add_argument('--thumb-width', type=int, default=504)
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    item = next(x for x in plan['groups'] if x['name'] == args.group)
    group = item['group']
    paths = [Path(p) for p in group['input_paths']] + [Path(item['target_path'])]
    masks = {int(k): Path(v[0]) for k, v in (group.get('mask_paths') or {}).items()}
    result_dirs = list(args.steps) + [args.mixed]
    psnr = [psnr_by_view(p) for p in result_dirs]
    labels = ['Input', 'Mask', 'Steps 001', 'Steps 005', 'Steps 050', 'Mixed\nclean005/dist050']
    rows, cols = 5, len(labels)
    tw = args.thumb_width
    first = Image.open(result_dirs[0] / 'decoded_view0.png')
    th = round(first.height * tw / first.width)
    label_h = 86
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    header_font = ImageFont.truetype(font_path, 20)
    row_font = ImageFont.truetype(font_path, 24)
    psnr_font = ImageFont.truetype(font_path, 18)
    canvas = Image.new('RGB', (cols * tw, label_h + rows * (th + label_h)), 'white')
    draw = ImageDraw.Draw(canvas)
    for v in range(rows):
        y = label_h + v * (th + label_h)
        draw.text((10, y + 12), f'Input | view {v}', fill='black', font=header_font)
        draw.text((tw + 10, y + 12), f'Mask | view {v}', fill='black', font=header_font)
        canvas.paste(fit(Image.open(paths[v]), (tw, th)), (0, y + label_h))
        canvas.paste(mask_panel(masks.get(v) if v < 4 else None, (tw, th)), (tw, y + label_h))
        for c, result_dir in enumerate(result_dirs, start=2):
            image = Image.open(result_dir / f'decoded_view{v}.png')
            title = f"{labels[c]}\nPSNR {psnr[c - 2][v]:.2f} dB"
            draw.multiline_text((c * tw + 10, y + 7), title, fill='black', font=header_font, spacing=1)
            canvas.paste(fit(image, (tw, th)), (c * tw, y + label_h))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    metadata = dict(group=item['name'], rows='views 0..4', columns=labels,
                    input_paths=[str(p) for p in paths], mask_paths={str(k): str(v) for k, v in masks.items()},
                    result_dirs=[str(p) for p in result_dirs])
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
