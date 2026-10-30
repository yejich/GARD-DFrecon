"""Paired report for the GT-mask token-wise NVS sampler prototype."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'result_eval/20261016_nvs_epoch020_steps050'
BASE5 = ROOT / 'result_eval/20261016_nvs_epoch020_steps005'
TOKEN = ROOT / 'result_eval/20261019_nvs_epoch020_tokenwise_clean005_dist050'

FIELDS = {
    'rgb_M_psnr': ('rgb_M', 'psnr', 1),
    'rgb_M_ssim': ('rgb_M', 'ssim', 1),
    'rgb_M_lpips': ('rgb_M', 'lpips_alex', -1),
    'depth_M_absrel': ('depth_M', 'absrel', -1),
    'depth_M_delta1': ('depth_M', 'delta1', 1),
    'depth_V_absrel': ('depth_V', 'absrel', -1),
    'pose_V_rre': ('pose_V', 'rre_median', -1),
    'pose_V_rte': ('pose_V', 'rte_median', -1),
}
FEATURES = ('V_cosine', 'M_cosine', 'V_relative_L2', 'M_relative_L2')

def metric(d, block, key):
    return d['methods']['gard_nvs'][block][key]

def main():
    rows = []
    for path in sorted(TOKEN.rglob('metrics.json')):
        rel = path.relative_to(TOKEN)
        base_path = BASE / rel
        base5_path = BASE5 / rel
        if not base_path.exists() or not base5_path.exists():
            continue
        token = json.loads(path.read_text())
        base = json.loads(base_path.read_text())
        base5 = json.loads(base5_path.read_text())
        row = {'group': str(rel.parent), 'dataset': rel.parts[0]}
        for name, (block, key, direction) in FIELDS.items():
            row[name + '_tokenwise'] = metric(token, block, key)
            row[name + '_baseline50'] = metric(base, block, key)
            row[name + '_baseline5'] = metric(base5, block, key)
            row[name + '_delta_vs50'] = direction * (row[name + '_tokenwise'] - row[name + '_baseline50'])
            row[name + '_delta_vs5'] = direction * (row[name + '_tokenwise'] - row[name + '_baseline5'])
        for key in FEATURES:
            a = token.get('features_L17', {}).get(key)
            b = base.get('features_L17', {}).get(key)
            c = base5.get('features_L17', {}).get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(c, (int, float)):
                row[key + '_tokenwise'] = a
                row[key + '_baseline50'] = b
                row[key + '_baseline5'] = c
                row[key + '_delta_vs50'] = a - b
                row[key + '_delta_vs5'] = a - c
        rows.append(row)

    fields = list(rows[0]) if rows else ['group']
    with (TOKEN / 'paired_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    def mean(name):
        x = [r[name] for r in rows if isinstance(r.get(name), (int, float))]
        return float(np.mean(x)) if x else None
    lines = [
        '# GT-mask token-wise NVS prototype', '',
        'Token-wise oracle run: source object-mask tokens use 5 Euler updates across the full t=1→0 interval; source distractor tokens, CLS tokens, and the novel view use the full 50-update grid. This is paired against same-group global 5-step and global 50-step runs.', '',
        f'Completed groups: **{len(rows)}** (Hypersim + HypersimNegative only; groups without GT masks were excluded).', '',
        '| metric | token-wise | global 5 | global 50 | Δ vs global 5 | Δ vs global 50 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name in FIELDS:
        lines.append(f'| {name} | {mean(name + "_tokenwise"):.6f} | {mean(name + "_baseline5"):.6f} | {mean(name + "_baseline50"):.6f} | {mean(name + "_delta_vs5"):+.6f} | {mean(name + "_delta_vs50"):+.6f} |')
    lines += ['', '## Feature diagnostics', '', '| metric | token-wise | global 5 | global 50 | Δ vs global 5 | Δ vs global 50 |', '|---|---:|---:|---:|---:|---:|']
    for name in FEATURES:
        if name + '_tokenwise' in (rows[0] if rows else {}):
            lines.append(f'| {name} | {mean(name + "_tokenwise"):.6f} | {mean(name + "_baseline5"):.6f} | {mean(name + "_baseline50"):.6f} | {mean(name + "_delta_vs5"):+.6f} | {mean(name + "_delta_vs50"):+.6f} |')
    lines += ['', '## Interpretation', '',
              '- The denoiser still runs on the full token tensor at every global time point; this prototype tests quality impact of token-wise state updates, not wall-clock acceleration.',
              '- GT masks are object-only masks. Mask pixels are resized to the processed resolution and converted to patch tokens with 14×14 average pooling and a 0.5 occupancy threshold.',
              '- The 17-group result is an oracle synthetic result and does not establish behavior on real data without GT masks.', '']
    (TOKEN / 'paired_comparison.md').write_text('\n'.join(lines))
    print(json.dumps({'groups': len(rows), 'csv': str(TOKEN / 'paired_comparison.csv'), 'report': str(TOKEN / 'paired_comparison.md')}))

if __name__ == '__main__':
    main()
