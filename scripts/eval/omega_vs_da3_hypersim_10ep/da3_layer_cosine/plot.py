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
ROOT=output_dir(__file__);data=json.loads((ROOT/'results.json').read_text());rows=data['rows'];groups=data['groups'];layers=np.arange(40)
series={path:np.array([[next(r['cosine'] for r in rows if r['path']==path and r['group']==g and r['layer']==i) for i in layers] for g in groups]) for path in ['input','gard']}
labels={'input':'Clean vs distractor','gard':'Clean vs restoration'};colors={'input':'#ea580c','gard':'#2563eb'}
fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True,sharey=True)
for ax,group in zip(axes.flat,[None]+groups):
 for path in series:
  values=series[path].mean(0) if group is None else series[path][groups.index(group)]
  ax.plot(layers,values,label=labels[path],color=colors[path],linewidth=2)
 ax.axvline(17,color='gray',linestyle='--',label='GARD injection (17)');ax.set_title('Mean of 3 groups' if group is None else f'Group {group:02d}');ax.set_xlabel('DA3 block index (0-based)');ax.set_ylabel('Cosine similarity');ax.grid(alpha=.25);ax.legend(fontsize=9);ax.set_xticks([0,5,10,15,17,20,25,30,35,39])
fig.suptitle('Whole-feature cosine similarity to clean DA3 features\nAll views, tokens (including special/camera) and channels flattened per group; no region masks',fontsize=13);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(ROOT/'layer_cosine.png',dpi=170);fig.savefig(ROOT/'layer_cosine.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5))
for path in series:ax.plot(layers,series[path].mean(0),label=labels[path],color=colors[path],linewidth=2)
ax.axvline(17,color='gray',linestyle='--',label='GARD injection');ax.set(xlabel='DA3 block index (0-based)',ylabel='Whole-feature cosine similarity',title='Clean feature similarity — mean of groups 00, 26, 53');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(ROOT/'mean_cosine.png',dpi=180);plt.close(fig)
plot=go.Figure()
for path in series:plot.add_trace(go.Scatter(x=layers,y=series[path].mean(0),name=labels[path],mode='lines+markers',line=dict(color=colors[path])))
plot.add_vline(x=17,line_dash='dash');plot.update_layout(title='Whole-feature cosine similarity (3-group mean)',xaxis_title='DA3 block index',yaxis_title='Cosine similarity');plot.write_html(ROOT/'mean_cosine.html',include_plotlyjs=True)
with (ROOT/'layer_cosine.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['layer','clean_distractor_mean','clean_restoration_mean']);writer.writerows(zip(layers,series['input'].mean(0),series['gard'].mean(0)))
(ROOT/'README.md').write_text('''# DA3 layer-wise whole-feature cosine similarity

Fixed held-out groups 00,26,53; same input switches and first-reference strategy as the preceding depth/pose comparison. Each layer compares the raw DA3 block-output x tensor, before final output normalization, across all four views, every token including the special/camera token, and all channels flattened into a single vector. No region masks, no per-token averaging. Curves are the arithmetic mean of three group-level cosine values, with separate group curves provided.

Clean vs distractor uses the original corrupted-input forward pass. Clean vs restoration uses the same input and the 10-epoch Group2 GARD EMA, 50 ODE time points, noise level from saved config, seed 42+group. At block 17 the plotted restoration value is the injected output after GARD, not the pre-injection block output. Blocks 0–16 are verified equal between the two paths; blocks 18–39 reflect propagation of the injected feature. Block indices are zero-based. Special token is overwritten by the DA3 camera token at block 13; its semantics are not identical across all layers.

Cosine compares direction, not feature magnitude, and this whole-feature measure weights tokens through their feature norms. It is not a direct pose/depth accuracy metric. Same spatial token correspondence and unchanged view order are used. Native DA3 feature capture is float32; GARD sampling uses BF16 autocast; cosine is computed in float32. No model source code or training checkpoint was changed; capture wraps process_attention read-only and injects restoration through the existing supported path.
''')
for i in [0,12,13,17,18,25,39]:print(i,series['input'].mean(0)[i],series['gard'].mean(0)[i])
