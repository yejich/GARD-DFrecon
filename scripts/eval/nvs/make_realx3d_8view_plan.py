"""Build an 8-source-view RealX3D plan using distinct clean cameras.

The first four source views remain the frozen mixed inputs. Four additional
clean views are selected from the group's clean-reference pool, excluding the
held-out target and all original source frame ids.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PLAN = ROOT / "result_eval/20261026_nvs_v4_n2-3_nvsfalse_flownoise_dattnfalse_aligntrue_15/eval_plan.json"
MANIFEST = ROOT / "result_eval/20261002_realx3d_multigroup/mixed/manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def c2w(frame):
    e = np.asarray(frame["extrinsics"], dtype=np.float64)
    return np.linalg.inv(np.concatenate([e, np.array([[0., 0., 0., 1.]])], axis=0))


def main(out_path):
    source = json.loads(SOURCE_PLAN.read_text())
    manifest = {x["id"]: x for x in json.loads(MANIFEST.read_text())["groups"]}
    rows = []
    for item in source["groups"]:
        if item["group"]["dataset"] != "RealX3D":
            continue
        old = item["group"]
        sg = manifest[old["id"]]
        input_indices = list(sg["input_indices"])
        target_index = int(sg["target_index"])
        original_ids = set(input_indices + [target_index])
        original_clean_paths = [str(Path(p)) for p in (item.get("clean_source_paths") or [])]
        paired_clean_paths = set(original_clean_paths)
        candidates = [int(i) for i in sg["clean_reference_indices"]
                      if int(i) not in original_ids
                      and str(Path(sg["frames"][str(i)]["path"])) not in paired_clean_paths]
        if len(candidates) < 4:
            raise RuntimeError(f"{old['id']} has fewer than four distinct clean views")
        extra_indices = candidates[:4]
        source_indices = input_indices + extra_indices + [target_index]
        frames = sg["frames"]

        source_paths = [str(Path(frames[str(i)]["path"])) for i in input_indices]
        extra_paths = [str(Path(frames[str(i)]["path"])) for i in extra_indices]
        target_path = str(Path(frames[str(target_index)]["path"]))
        all_source_paths = source_paths + extra_paths
        if not all(Path(p).is_file() for p in all_source_paths + [target_path]):
            missing = [p for p in all_source_paths + [target_path] if not Path(p).is_file()]
            raise FileNotFoundError(missing)

        c2ws = [c2w(frames[str(i)]) for i in source_indices]
        intrinsics = [frames[str(i)]["intrinsics"] for i in source_indices]
        original_clean_valid = [str(Path(p)) for p in (item.get("clean_source_valid_paths") or [])]
        clean_paths = original_clean_paths + extra_paths
        clean_valid = original_clean_valid + [str(Path(frames[str(i)]["valid_path"])) for i in extra_indices]
        if not all(Path(p).is_file() for p in clean_paths + [target_path]):
            missing = [p for p in clean_paths + [target_path] if not Path(p).is_file()]
            raise FileNotFoundError(missing)
        g = dict(old)
        g["input_paths"] = all_source_paths
        g["input_sha256"] = [sha(p) for p in all_source_paths]
        g["input_indices_8view"] = input_indices + extra_indices
        g["extra_clean_indices"] = extra_indices
        g["pose_reference_extrinsics"] = np.linalg.inv(np.stack(c2ws[:8]))[:, :3].tolist()
        g["evaluation_scope"] = "realx3d_8_source_views_mixed_plus_four_distinct_clean"

        row = dict(item)
        row["group"] = g
        row["target_path"] = target_path
        row["target_sha256"] = sha(target_path)
        row["target_valid_path"] = str(Path(frames[str(target_index)]["valid_path"]))
        row["source_target_c2w"] = np.stack(c2ws).tolist()
        row["source_target_intrinsics"] = intrinsics
        row["source_target_distortion"] = []
        row["clean_source_paths"] = clean_paths
        row["clean_source_valid_paths"] = clean_valid
        row["reference_paths"] = clean_paths + [target_path]
        row["reference_sha256"] = [sha(p) for p in row["reference_paths"]]
        row["reference_c2w"] = np.stack(c2ws).tolist()
        row["reference_intrinsics"] = intrinsics
        row["reference_distortion"] = []
        row["selection"] = "four distinct clean cameras from clean_reference_indices, excluding mixed inputs and held-out target"
        rows.append(row)

    out = dict(source, V=8, M=1, N=9, groups=rows,
               protocol="RealX3D: four frozen mixed inputs plus four distinct clean source cameras")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {out_path} with {len(rows)} groups and V=8, M=1")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "result_eval/20261030_realx3d_8view_intermediate/eval_plan.json"))
