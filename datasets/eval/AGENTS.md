# Default evaluation policy

When the user requests eval without specifying a different suite, use all groups in `datasets/eval/registry.json` and the authoritative `datasets/eval/evaluation_protocol.json` contract. Preserve the frozen input paths, hashes, view order, and reference-first setting.

Required outputs: quantitative and qualitative depth, pose, point-cloud results; input-view decoded RGB PSNR, SSIM, LPIPS and image comparisons. Color point clouds with decoded RGB. Evaluate confidence percentile cutoffs P20/P40/P50, dropping the bottom corresponding percentile across each group's input-view confidence values. Pose and input-view RGB metrics are independent of these cutoffs.

Record GT versus pseudo-GT, registration, scale units, coverage, and missing-reference reasons. On-the-go has no paired clean input RGB target: report these restoration RGB metrics as N/A, keep qualitative outputs, and label held-out render metrics separately. Do not claim full protocol completion from the legacy partial pose/RGB runner. Keep mask outputs additionally when a mask head is available.

Entire ai_001_007 is held out from training, including all unselected negative groups.
