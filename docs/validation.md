# Export validation — 2026-09-14

This export is based on upstream GARD `de483e0` plus the local Hypersim
10-epoch training changes. Validation used the existing Python 3.10 GARD
environment; a fresh `uv sync` and a new full GPU training run were not performed
as part of publishing this code.

- Nine tests passed, including real clean/distractor image loading, fixed eval
  identity, DDP view-count synchronization, gradient accumulation alignment,
  Group2 weighted gradients, empty groups, and reference-view mask permutation.
- All 27,292 required image/mask files exist for the 6,743 train / 80 eval pairs.
- The published manifest preserves all records, candidate edges, fixed eval
  groups, switches and seeds from the original; only its root path is portable.
- Trainer imports and `--help` succeeded from this independent checkout with its
  own vendored Depth Anything 3 source on PYTHONPATH.
- Bash syntax checked. Training configuration retains the original numerical
  settings; machine-specific paths are replaced with relative paths/overrides.
- Original experiment: 10 completed epochs, 33,680 rank-local microbatch steps,
  4,210 optimizer updates. Reported fixed eval flow loss is from that run.
- Fine-tuned checkpoints, images, training/evaluation output directories,
  20-epoch extension work and local W&B credentials are excluded.
