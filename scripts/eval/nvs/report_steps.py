"""Aggregate NVS inference-step ablations and build a qualitative HTML index."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "result_eval/20261016_nvs_epoch020_steps_report"
STEPS = [1, 5, 10, 20, 50]
RUNS = {s: ROOT / f"result_eval/20261016_nvs_epoch020_steps{s:03d}" for s in STEPS}

def val(d, *keys):
    for key in keys:
        if not isinstance(d, dict): return None
        d = d.get(key)
    return d if isinstance(d, (int, float)) and np.isfinite(d) else None

def flatten_metrics(step, path):
    try:
        m = json.loads(path.read_text())
        method = m["methods"]["gard_nvs"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    complete = path.parent / "complete.json"
    timing = json.loads(complete.read_text()) if complete.exists() else {}
    row = {
        "step": step, "group": m.get("id", path.parent.name),
        "dataset": m.get("id", "").split("/", 1)[0],
        "seconds": timing.get("seconds"),
        "rgb_M_psnr": val(method, "rgb_M", "psnr"),
        "rgb_M_ssim": val(method, "rgb_M", "ssim"),
        "rgb_M_lpips": val(method, "rgb_M", "lpips_alex"),
        "depth_V_absrel": val(method, "depth_V", "absrel"),
        "depth_V_delta1": val(method, "depth_V", "delta1"),
        "depth_M_absrel": val(method, "depth_M", "absrel"),
        "depth_M_delta1": val(method, "depth_M", "delta1"),
        "pose_V_rre": val(method, "pose_V", "rre_median"),
        "pose_V_rte": val(method, "pose_V", "rte_median"),
        "V_cosine": val(m, "features_L17", "V_cosine"),
        "M_cosine": val(m, "features_L17", "M_cosine"),
    }
    return row

def mean_rows(rows, keys):
    out = {}
    for key in keys:
        xs = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        out[key] = float(np.mean(xs)) if xs else None
    return out

def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for step, run in RUNS.items():
        for path in run.rglob("metrics.json"):
            row = flatten_metrics(step, path)
            if row: rows.append(row)
    write_csv(OUT / "per_group.csv", rows)
    numeric = ["seconds", "rgb_M_psnr", "rgb_M_ssim", "rgb_M_lpips", "depth_V_absrel", "depth_V_delta1", "depth_M_absrel", "depth_M_delta1", "pose_V_rre", "pose_V_rte", "V_cosine", "M_cosine"]
    summary = []
    for step in STEPS:
        for dataset in sorted({r["dataset"] for r in rows}):
            selected = [r for r in rows if r["step"] == step and r["dataset"] == dataset]
            summary.append({"step": step, "dataset": dataset, "groups": len(selected), **mean_rows(selected, numeric)})
        selected = [r for r in rows if r["step"] == step]
        summary.append({"step": step, "dataset": "ALL", "groups": len(selected), **mean_rows(selected, numeric)})
    write_csv(OUT / "summary.csv", summary)

    md = ["# GARD NVS inference-step ablation", "", "Checkpoint: epoch_020 EMA; official registry groups; 65 groups expected.", "", "| step | groups | sec/group | M PSNR ↑ | M SSIM ↑ | M LPIPS ↓ | V depth AbsRel ↓ | M depth AbsRel ↓ | V pose RRE ↓ | V pose RTE ↓ |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r["dataset"] != "ALL": continue
        def f(k): return "-" if r.get(k) is None else f"{r[k]:.4f}"
        md.append(f"| {r['step']} | {r['groups']}/65 | {f('seconds')} | {f('rgb_M_psnr')} | {f('rgb_M_ssim')} | {f('rgb_M_lpips')} | {f('depth_V_absrel')} | {f('depth_M_absrel')} | {f('pose_V_rre')} | {f('pose_V_rte')} |")
    md += ["", "Detailed files: `per_group.csv`, `summary.csv`. The HTML index links identical groups across all completed step settings.", ""]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")

    groups = sorted({r["group"] for r in rows})
    html = ["<!doctype html><meta charset='utf-8'><title>GARD NVS inference steps</title>", "<style>body{font-family:sans-serif;max-width:1900px;margin:20px auto} table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px;vertical-align:top} img{width:260px;display:block} .missing{width:260px;height:170px;background:#eee;text-align:center;padding-top:20px;box-sizing:border-box}</style>", "<h1>GARD NVS inference-step comparison</h1><p>Each row is one frozen eval group. Columns show restored source views and novel-view RGB/depth for completed runs.</p>", "<table><tr><th>group</th>" + "".join(f"<th>{s} steps</th>" for s in STEPS) + "</tr>"]
    for group in groups:
        html.append(f"<tr><th>{group}</th>")
        for step in STEPS:
            run = RUNS[step]; folder = run / group
            imgs = []
            for name in ("rgb_comparison.jpg", "M_rgb_comparison.jpg", "M_depth_comparison.jpg"):
                p = folder / name
                if p.exists(): imgs.append(f"<img src='../{run.relative_to(ROOT)}/{group}/{name}' loading='lazy'>")
            html.append("<td>" + ("".join(imgs) if imgs else "<div class='missing'>pending</div>") + "</td>")
        html.append("</tr>")
    html.append("</table>")
    (OUT / "qualitative.html").write_text("\n".join(html), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": str(OUT / "summary.csv"), "qualitative": str(OUT / "qualitative.html")}))

if __name__ == "__main__": main()
