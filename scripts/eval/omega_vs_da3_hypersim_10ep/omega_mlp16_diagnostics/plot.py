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
P=output_dir(__file__)
summ=[]
for layer in [16,17]:
 data=json.loads((P.parent/f'omega_mlp{layer}_diagnostics/results.json').read_text())
 assert data['target_block']==layer
 rows=data['rows']
 for stage in [0,5,6]:
  for path in ['input','gard']:
   subset=[r for r in rows if r['stage']==stage and r['path']==path]
   assert len(subset)==3
   summ.append(dict(block=layer,stage=stage,path=path,cosine=float(np.mean([r['token_cosine'] for r in subset])),flat_cosine=float(np.mean([r['cosine'] for r in subset])),groups={str(r['group']):r['token_cosine'] for r in subset}))
(P/'comparison_summary.json').write_text(json.dumps(summ,indent=2))
with (P/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['block','stage','path','cosine','flat_cosine']);w.writeheader();w.writerows({k:v for k,v in r.items() if k!='groups'} for r in summ)
fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,layer in zip(axs,[16,17]):
 for path,color,label in [('input','#ed6a00','Clean vs distractor'),('gard','#3268dc','Clean vs restoration')]:
  vals=[r['cosine'] for r in summ if r['block']==layer and r['path']==path]
  ax.plot(range(3),vals,'o-',color=color,label=label)
  other=[r['cosine'] for r in summ if r['block']==layer and r['path']!=path]
  for k,v in enumerate(vals):
   ax.annotate(f'{v:.4f}',(k,v),xytext=(0,10 if v>=other[k] else -18),textcoords='offset points',ha='center',color=color)
 ax.set_xticks(range(3),['Input x\n(after frame attention)','Residual r(x)\n(after LayerScale)','Output x + r(x)']);ax.set_title(f'Aggregator block {layer} (0-based)');ax.grid(alpha=.2);ax.legend(fontsize=9)
axs[0].set_ylabel('Mean patch-token cosine');axs[0].set_ylim(min(r['cosine'] for r in summ)-.022,max(r['cosine'] for r in summ)+.022)
fig.suptitle('Frame MLP: input, residual and output | groups 00, 26, 53 mean\nEach point compares matching clean and distractor/restoration features')
fig.tight_layout(rect=(0,0,1,.9));fig.savefig(P/'blocks16_17_comparison.png',dpi=180);plt.close(fig)
(P/'README.md').write_text('Same three groups, checkpoint, seeds, sampler and read-only hooks as omega_mlp17_diagnostics, targeting frame_blocks[16]. Compare stage 0 (frame-attention output / MLP input), stage 5 (LayerScale residual), stage 6 (residual addition). Patch tokens only, no spatial masks. FP32 cosine, BF16 inference. comparison_summary.json stores per-group and mean cosine for blocks 16 and 17. No inference or training settings modified.\n')
print(json.dumps(summ,indent=2))
