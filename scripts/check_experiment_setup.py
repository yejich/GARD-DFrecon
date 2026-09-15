"""Read-only audit of migrated experiment files and local dependencies."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mvr.experiment_paths import (
    FEATURE_COMPLETION_ROOT, SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO,
    VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT,
)


def main():
    required = [ROOT / name for name in (
        'ckpts/gard_denoiser.pt', 'ckpts/mae_adapter_giant.pt',
        'data/hypersim_pairs/groups.json',
        'data/omega_latent_stats/inter3_hypersim_pairs.pt',
        'datasets/hypersim_pairs',
    )]
    required.extend(sorted((ROOT / 'result_train').glob('**/checkpoints/latest.pt')))
    missing = []
    for path in required:
        ok = path.exists()
        print(f"{'OK' if ok else 'MISSING'}: {path.relative_to(ROOT)}")
        if not ok:
            missing.append(str(path))
        if path.is_symlink():
            print(f'  storage: {path.resolve()}')
    manifest = ROOT / 'data/hypersim_pairs/groups.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        print(f"Saved split: train={len(data['train'])}, eval={len(data['eval'])}, groups={len(data['eval_groups'])}")
    print('\nExternal experiment dependencies (needed for the corresponding experiment):')
    for path in (FEATURE_COMPLETION_ROOT, SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO,
                 VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT):
        print(f"{'OK' if path.exists() else 'MISSING'}: {path}")
    return int(bool(missing))


if __name__ == '__main__':
    raise SystemExit(main())
