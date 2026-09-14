"""Build a training snapshot from downloaded clean/distractor pair collections."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PAIR_FILES = ('clean.png', 'distractor.png', 'mask_object.png', 'mask_shadow.png')


def collections(root):
    """Accept one scene collection or its parent, excluding the legacy `scenes`."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if any(root.glob('ai_*_*/cam_*')):
        return [root]
    found = sorted(p for p in root.iterdir() if p.is_dir()
                   and (p.name == 'scenes_v2' or re.fullmatch(r'scenes_\d{3}', p.name)))
    if not found:
        raise ValueError(f'No scene collections under {root}; point at scenes_v2, scenes_002, etc. or their parent')
    return found


def fingerprint(data):
    """Paths may move to another server; relative identities and groups must not."""
    stable = {k: data[k] for k in ('train', 'eval', 'eval_groups', 'eval_scene',
                                  'minimum_covisibility', 'distractor_probability', 'seed', 'eval_views')}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def collect_manifest(root, threshold=.25, probability=.7, eval_scene='ai_001_001', seed=42):
    import random
    root = Path(root).expanduser().resolve()
    if not 0 <= threshold <= 1 or not 0 <= probability <= 1:
        raise ValueError('Probability and covisibility must be in [0, 1]')
    sources = collections(root)
    splits = {'train': [], 'eval': []}
    seen = {}
    skipped = 0
    for source in sources:
        for folder in sorted(source.glob('ai_*_*/cam_*/frame.*')):
            if not folder.is_dir() or not re.fullmatch(r'frame\.\d+', folder.name):
                continue
            if not all((folder / name).is_file() for name in PAIR_FILES):
                skipped += 1
                continue
            scene, camera, frame = folder.relative_to(source).parts
            key = (scene, camera, int(frame.split('.')[1]))
            if key in seen:
                raise ValueError(f'Duplicate source frame {key}: {seen[key]} and {folder}; select one version')
            seen[key] = str(folder)
            splits['eval' if scene == eval_scene else 'train'].append(
                dict(directory=str(folder.relative_to(root)), scene=scene, camera=camera, frame=key[2]))
    if not splits['train'] or not splits['eval']:
        raise ValueError(f'Need train pairs and held-out {eval_scene} pairs. Include scenes_v2 along with new collections.')
    # Source folder order must not alter the held-out view ordering.
    for records in splits.values():
        records.sort(key=lambda r: (r['scene'], r['camera'], r['frame']))
        lookup = {(r['scene'], r['camera'], r['frame']): i for i, r in enumerate(records)}
        cache = {}
        for record in records:
            trajectory = (root / record['directory']).parent
            if trajectory not in cache:
                path = trajectory / 'clean_multiview.json'
                if not path.exists():
                    raise FileNotFoundError(f'{path}: include clean_multiview.json when downloading/transferring pairs; raw Hypersim alone is not sufficient')
                cache[trajectory] = json.loads(path.read_text())
            try:
                candidates = cache[trajectory][str(record['frame'])]['eligible_candidates']
            except KeyError as error:
                raise ValueError(f'Missing covisibility for {trajectory}/frame.{record["frame"]:04d}') from error
            record['candidates'] = sorted({lookup[(record['scene'], record['camera'], c['frame_id'])]
                for c in candidates if c['covisibility'] >= threshold
                and (record['scene'], record['camera'], c['frame_id']) in lookup
                and c['frame_id'] != record['frame']})
    rng = random.Random(seed)
    groups = []
    for i, record in enumerate(splits['eval']):
        if len(record['candidates']) >= 3:
            indices = [i] + rng.sample(record['candidates'], 3)
            groups.append(dict(indices=indices, distractor=[rng.random() < probability for _ in indices]))
    if not groups:
        raise ValueError('No held-out anchor has three overlapping downloaded neighbors for fixed 4-view evaluation')
    if not any(len(r['candidates']) >= 3 for r in splits['train']):
        raise ValueError('No training anchor supports the default maximum of four views')
    data = dict(schema_version=2, root=str(root), eval_scene=eval_scene,
                collections=[str(p.relative_to(root)) for p in sources],
                minimum_covisibility=threshold,
                overlap_definition='fraction of clean anchor GT points visible in candidate; same camera trajectory',
                distractor_probability=probability, seed=seed, eval_views=4, eval_groups=groups, **splits)
    data['fingerprint'] = fingerprint(data)
    summary = dict(train_pairs=len(splits['train']), eval_pairs=len(splits['eval']),
                   eval_groups=len(groups), skipped_incomplete_pairs=skipped,
                   collections=data['collections'], fingerprint=data['fingerprint'],
                   split_scene_counts={k: len({r['scene'] for r in v}) for k, v in splits.items()},
                   eligible_anchors={k: {n: sum(len(r['candidates']) >= n-1 for r in v) for n in range(1,5)} for k,v in splits.items()})
    return data, summary


def build_manifest(root, output=None, threshold=.25, probability=.7, eval_scene='ai_001_001', seed=42,
                   output_dir=None, print_summary=True):
    data, summary = collect_manifest(root, threshold, probability, eval_scene, seed)
    if output_dir:
        output = Path(output_dir) / f"groups_{data['fingerprint'][:16]}.json"
    output = Path(output or 'data/hypersim_pairs/groups.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write; workers load one snapshot, never a file still being scanned.
    import os, tempfile
    for path, content in [(output, data), (output.with_suffix('.summary.json'), summary)]:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
            json.dump(content, f, indent=2)
            temporary = f.name
        os.replace(temporary, path)
    if print_summary:
        print(json.dumps(summary, indent=2), file=sys.stderr)
    return output.resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    out = ap.add_mutually_exclusive_group()
    out.add_argument('--output')
    out.add_argument('--output-dir')
    ap.add_argument('--threshold', type=float, default=.25)
    ap.add_argument('--probability', type=float, default=.7)
    args = ap.parse_args()
    print(build_manifest(args.root, args.output, threshold=args.threshold, probability=args.probability,
                         output_dir=args.output_dir))

if __name__ == '__main__':
    main()
