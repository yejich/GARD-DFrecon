# Hypersim clean–distractor GARD fine-tuning

Set `HYPERSIM_PAIRS_ROOT` to the transferred `scenes_v2` directory. The original fixed group manifest is included; images are not.

Run from the repository root:

```bash
bash run_scripts/train/train_GARD_hypersim_pairs.sh
```

The script defaults to visible GPUs **0,1** (override `CUDA_VISIBLE_DEVICES`; the original run used physical GPUs 6,7), two DDP ranks, one multiview group per GPU, and gradient accumulation over eight microbatches. `global_batch_size: 16` is the **effective** batch in this trainer: 2 GPUs × 1 group × 8 accumulation steps. Changing only the GPU count without adjusting this arithmetic will change microbatch size.

- Frozen DA3-GIANT-1.1; GARD initialized strictly from `ckpts/gard_denoiser.pt` EMA weights, fresh optimizer.
- BF16 autocast, original GARD feature flow loss and attention alignment loss, learning rate 2e-5, 10 epochs, one warmup epoch.
- Train: 6,743 saved pairs from 41 scenes with successful synthesis; held out: all 80 successful pairs from `ai_001_001`. The five scenes with zero successful synthesis are absent.
- Train N is uniform 1–4 per microbatch, synchronized across DDP ranks. Scene is uniformly sampled among scenes having a suitable anchor; anchor is random and may recur. N−1 candidates are sampled without replacement from successful pairs in the same camera trajectory with cached clean anchor-to-candidate GT covisibility ≥25%. This is directional overlap and does not impose candidate-to-candidate overlap or minimum baseline.
- Each selected input independently uses distractor.png with p=0.7, otherwise clean.png. Every HQ target uses clean.png. No additional blur. All-clean/all-distractor groups are permitted.
- Long edge 504, patch-aligned 378×504 images. Training uses full images rather than crops.
- `manifests/hypersim_001_groups.json` snapshots eligible candidates and 80 fixed four-view eval groups with fixed clean/distractor switches. No eval scene enters training candidates. Existing backbone/GARD pretraining may include this scene; this split is a fine-tuning holdout, not a verified unseen-pretraining benchmark.
- Each epoch evaluates fixed-seed flow velocity loss using EMA weights on all 80 groups. This is a training diagnostic, not rendered RGB PSNR or downstream geometry evaluation.
- Each completed epoch atomically replaces `checkpoints/latest.pt` under `result_train/hypersim_pairs_ai001_eval/`. It includes raw/EMA weights, Adam state, scheduler and each rank's RNG. Resume at epoch boundaries with `--resume /absolute/path/to/latest.pt`; retain the same manifest/config and world size. Full checkpoints and the temporary atomic replacement require substantial disk space.

Validation commands:

```bash
.venv/bin/python -m unittest discover -s tests -p test_hypersim_pairs.py
# Eight N=4 microbatches, one optimizer update, then two fixed eval groups:
bash run_scripts/train/train_GARD_hypersim_pairs.sh --max-steps 1 --fixed-views 4
# Dynamic N=1–4, same update/eval check:
bash run_scripts/train/train_GARD_hypersim_pairs.sh --max-steps 1
```

Smoke runs stop and do not write training checkpoints. A successful smoke run is not the full 10-epoch training run. CUDA visibility remaps the two selected physical GPUs to cuda:0/1 in logs.

## Group2 loss (current default)

The flow velocity MSE is averaged separately over (1) distractor tokens and (2) all other tokens, then combined as `lambda_distractor * mse_distractor + lambda_clean * mse_clean`. Both weights default to 1.0, matching the earlier overfit experiment; each group's contribution is independent of its token count. Configure `mvrm.loss.group_mse` in the YAML. The existing attention alignment loss is added unchanged.

Distractor tokens use the union of object and shadow masks, resized with nearest-neighbor interpolation to the input resolution, then 14×14 average pooling with coverage ≥0.5. Clean-selected input views have an empty mask. CLS belongs to the clean group, including on corrupted views. Masks follow DA3's reference-first permutation separately for each sample. Missing groups contribute zero, without NaNs. Each sample's group means are computed before the batch average. This grouping weights **tokens**, not whole clean/distractor views. Epoch evaluation uses the same weighted MSE (without attention alignment) and fixed groups/noise.

Group metrics appear in the training log as `train_group/distractor`, `train_group/clean` and their token counts. Their configured weighted sum is `train/loss_transport`. These are feature flow-velocity errors, not RGB pixel MSE. The group2 run has a separate experiment suffix from the earlier flat-MSE smoke runs.

## W&B logging

Use an existing W&B login or supply `WANDB_API_KEY` through your shell environment. Only rank 0 opens/logs a run. Each invocation creates a new W&B run, including checkpoint resumes (checkpoint training state is still restored).

```bash
bash run_scripts/train/train_GARD_hypersim_pairs.sh \
  --wandb --wandb-project cross-view-feature-completion \
  --wandb-run-name "Hypersim_group2_p07_gpu67"
```

Optional: `--wandb-entity TEAM_OR_USER`. CLI arguments are forwarded by the bash script. Monitor `train_group/distractor`, `train_group/clean`, `train/loss_transport`, `train_attn/loss_attn`, `epoch/loss`, and `eval/group_mse`. Run configuration is uploaded with the API-key field removed; credentials are never intentionally logged.
