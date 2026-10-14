# Hypersim synthetic training data

This repository distributes **data only**. Training code and configuration are maintained in [GARD-DFrecon](https://github.com/yejich/GARD-DFrecon). The latest data inventory is [the October 13 release](releases/20261013_full_synthetic_v4_n3_m1/README.md); the September 20K snapshot is historical.

## Dataset settings

| Setting / extracted folder | Meaning | Train pairs | Eval pairs | Total pairs |
|---|---|---:|---:|---:|
| `transient_only/` | Original clean scene paired with transient distractors | 42,006 | 1,057 | 43,063 |
| `transient_static/pairs/` | Clean scene containing inserted static A, paired with additional transient B | 35,928 | 728 | 36,656 |
| `real_nvs/` | Real unpaired data for NVS supervision | — | — | Not uploaded |

Counts are paired camera images, not multiview groups. `transient_only` contains the previous **20,965 pairs plus 22,098 additional pairs**. The previous snapshot's `scenes_v2` **6,823 pairs are already part of the 20,965**, never an extra dataset. Collection names (`scenes_v2`, `scenes_002`, etc.) are generation folders, not the three training settings.

`transient_static` includes generation added in collections 1–6 and 7. Static A is present in both paired images; masks label transient B. Standalone static clean resources are provided under `transient_static/clean_resources/` and are not additional paired samples.

## Download

Install `huggingface_hub` and `requests`, then download the small extraction script:

```bash
hf download cyjcyj91/Hypersim-Distractor \
  releases/20261013_full_synthetic_v4_n3_m1/download_and_extract.py \
  --repo-type dataset --local-dir ./hf_metadata
python ./hf_metadata/releases/20261013_full_synthetic_v4_n3_m1/download_and_extract.py \
  --destination /data/hypersim_synthetic_full
```

The script resolves one immutable commit for the entire download, verifies archive SHA256, safely extracts one archive at a time, removes staging TARs, and supports resume. It creates:

```text
hypersim_synthetic_full/
  transient_only/<collection>/<scene>/<camera>/frame.XXXX/
  transient_static/
    pairs/<collection>/<scene>/<camera>/group_XXXXXX/variant_00/
    clean_resources/<collection>/<scene>/<camera>/group_XXXXXX/variant_00/
  camera_reference/
  manifests/
    transient_only.json
    transient_static.json
    scene_split_policy.json
```

`real_nvs/` is reserved for future real data and is not created or populated by this release. Existing legacy extraction roots are moved to the new layout before resume; conflicting old/new roots cause an explicit error. Files originally downloaded from the old release may still include its old documentation or training directory; those are not used by the new instructions.

## Inventory, splits and snapshots

The inventory is frozen at approximately **2026-10-13 19:34 KST**. Later synthesis is excluded, including subsequently generated scenes_008 negative samples. Six standalone scenes_008 static groups completed before the cutoff are included as clean resources.

Seven scenes remain held out: `ai_001_001`, `ai_001_006`, `ai_001_007`, `ai_001_008`, `ai_001_009`, `ai_001_010`, `ai_003_006`. Manifests contain relative sample directories and within-split candidate indices. For V=4, N=3, M=1, 41,900 transient-only training anchors have at least three eligible other views; each transient+static variant provides four views. M RGB is teacher supervision, not an input image.

Archive blobs retain their historical names and internal paths to avoid copying hundreds of GB. The downloader maps them into the layout above. `archive_index.json` identifies all **401** required archives, including reused September archives; reused files are not counted twice. Versioned `summary.json` is authoritative for current counts. Root `manifest.jsonl` and September receipts describe only the historical snapshot and must not be interpreted as the expanded inventory.

The older September snapshot contains **20,965 pairs (20,885 train, 80 eval), 165 archives, 42.22 GB decimal**. Its original snapshot remains accessible at revision `a935b4e251faeaf1e96723a28463d44e8913c834`. The legacy `scenes` generation is excluded to avoid duplicate versions of `_001` data.

Review `LICENSE_NOTES.md` and `attribution/objaverse_assets.json` for Hypersim and individual Objaverse attribution. Pair files include clean/distractor RGB, object/shadow masks and provenance metadata. Camera references are supplied for NVS rays. Absolute producer paths in provenance are informational.
