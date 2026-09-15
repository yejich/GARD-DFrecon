# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parent;data=json.loads((ROOT/'results.json').read_text());rows=data['rows'];groups=data['groups'];layers=np.arange(data["depth"])
series={path:np.array([[next(r['cosine'] for r in rows if r['path']==path and r['group']==g and r['layer']==i) for i in layers] for g in groups]) for path in ['input','gard']}
labels={'input':'Clean vs distractor','gard':'Clean vs restoration'};colors={'input':'#ea580c','gard':'#2563eb'}
fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True,sharey=True)
for ax,group in zip(axes.flat,[None]+groups):
 for path in series:
  values=series[path].mean(0) if group is None else series[path][groups.index(group)]
  ax.plot(layers,values,label=labels[path],color=colors[path],linewidth=2,marker='o',markersize=3)
 ax.axvline(data["injection_layer"],color='gray',linestyle='--',label='GARD injection (3)');ax.set_title('Mean of 3 groups' if group is None else f'Group {group:02d}');ax.set_xlabel('Index i: after frame_blocks[i] + inter_frame_blocks[i] (0-based)');ax.set_ylabel('Cosine similarity');ax.grid(alpha=.25);ax.legend(fontsize=9);ax.set_xticks(layers);ax.tick_params(axis='x',labelsize=8)
fig.suptitle('Whole-feature cosine similarity to clean Omega features\nAll views, tokens (including special/camera) and channels flattened per group; no region masks',fontsize=13);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(ROOT/'layer_cosine.png',dpi=170);fig.savefig(ROOT/'layer_cosine.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5))
for path in series:ax.plot(layers,series[path].mean(0),label=labels[path],color=colors[path],linewidth=2,marker='o',markersize=3)
ax.axvline(data["injection_layer"],color='gray',linestyle='--',label='GARD injection');ax.set(xlabel='Index i: after frame_blocks[i] + inter_frame_blocks[i] (0-based)',ylabel='Whole-feature cosine similarity',title='Clean feature similarity — mean of groups 00, 26, 53');ax.set_xticks(layers);ax.tick_params(axis='x',labelsize=8);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(ROOT/'mean_cosine.png',dpi=180);plt.close(fig)
plot=go.Figure()
for path in series:plot.add_trace(go.Scatter(x=layers,y=series[path].mean(0),name=labels[path],mode='lines+markers',line=dict(color=colors[path])))
plot.add_vline(x=data["injection_layer"],line_dash='dash');plot.update_layout(title='Whole-feature cosine similarity (3-group mean)',xaxis_title='Omega aggregator block index',yaxis_title='Cosine similarity');plot.update_xaxes(tickmode='array',tickvals=layers,ticktext=[str(i) for i in layers]);plot.write_html(ROOT/'mean_cosine.html',include_plotlyjs=True)
with (ROOT/'layer_cosine.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['layer','clean_distractor_mean','clean_restoration_mean']);writer.writerows(zip(layers,series['input'].mean(0),series['gard'].mean(0)))
(ROOT/'README.md').write_text("""# VGGT-Omega whole-feature layer cosine similarity

Fixed held-out groups 00,26,53. Each point is the full aggregator token tensor after one frame/inter-frame block pair. It includes all views, camera/register tokens and patch tokens; all dimensions are flattened into one vector per group. No masks or per-token averaging. Curves average three group cosine values.

Clean vs distractor compares the clean and corrupted-input passes. Clean vs restoration uses the completed 10-epoch Omega GARD EMA checkpoint, saved normalization, 50 ODE time points and seed 42+group, matching the prior depth evaluation. At index 3 the restored value is measured after injection (and cast to the actual aggregator token dtype); indices 0-2 are verified equal between input/restored passes. Later layers reflect propagation. Register-only attention blocks still report the full tensor, not only the updated special tokens.

This is the Omega aggregator, not its DINOv3 image encoder and not the GARD denoiser. One Omega index comprises a frame/inter-frame pair and does not correspond one-to-one with a DA3 block index. No view reordering occurs in this Omega implementation. Native inference uses BF16 autocast, cosine FP32. Whole-feature cosine measures direction and is weighted by feature norms; it does not directly imply pose/depth improvement.

Results: mean_cosine.png / layer_cosine.png / layer_cosine.pdf / mean_cosine.html / layer_cosine.csv / results.json.
""")
for i in sorted(set([0,3,4,data['depth']-1])):print(i,series['input'].mean(0)[i],series['gard'].mean(0)[i])
