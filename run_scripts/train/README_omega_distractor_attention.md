# Optional distractor-aware attention alignment (Omega)

The default Omega YAML sets `mvrm.loss.attn_align.distractor_aware.use: false`.
With the option absent or disabled, the original attention loss is unchanged.
This option is implemented in the Omega trainer only; the DA3 trainer is unchanged.

To compare against the baseline, copy the training YAML to a separate experiment
config and enable:

```yaml
mvrm:
  loss:
    attn_align:
      distractor_aware:
        use: true
        cross_view_only_for_distractor: true
        max_correspondence_distance: 0.03
        balance_query_groups: true
        lambda_distractor: 1.0
        lambda_clean: 1.0
```

Run the separate config with `CONFIG=/absolute/path/to/config.yaml` before the
usual training bash command. Give it a different `log.tracker.wandb.msg` or
`log.result_root_dir` so its logs/checkpoints do not overwrite the baseline.
An already running process keeps the config and code it loaded at startup.

Behavior:

- HQ Omega depth/pose still define the target correspondence distribution.
- Uses the same object+shadow patch mask as group2 MSE: at least 50% pixel
  coverage in a 16x16 patch, and only views actually selected as distractor.
- Distractor queries remain supervised; distractor keys are excluded for all queries.
- If `cross_view_only_for_distractor` is true, distractor queries cannot use any
  same-view key. Clean queries retain same-view/self correspondences.
- Keeps the configured HQ cycle-consistency constraint. Nearest matches use raw
  distances in this mode to avoid softmax underflow.
- Keys farther than `max_correspondence_distance` are excluded. This is Euclidean
  distance between HQ patch-mean 3D points in **Omega output coordinates**, not a
  guaranteed metric distance. The initial 0.03 is three times the default target
  temperature; it is a tunable starting value, not a measured visibility threshold.
- Only the target is masked and renormalized. Predicted attention is not masked
  or renormalized, including its probability mass on special tokens.
- Rows with no admissible correspondence contribute zero alignment loss; MSE
  still supervises these tokens. All-distractor groups have zero alignment loss.
- With `balance_query_groups: true`, compute distractor and clean valid-query
  means per sample, multiply by their respective lambdas, then average samples.
  Empty groups contribute zero. With false, average all valid queries together
  and ignore the two group lambdas.
- `attn_align.lambda_coeff` still scales the combined attention loss.
- No distractor masks are required for inference.

Tests:

```bash
PYTHONPATH=.:RAE/src:src .venv/bin/python -m unittest discover \
  -s tests -p test_omega_distractor_attention.py -v
```
