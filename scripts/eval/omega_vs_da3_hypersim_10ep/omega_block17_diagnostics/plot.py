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
rows=json.loads((P/'results.json').read_text())['rows']
labels=['Input\n(block 16 output)','Frame attention\n+ residual','Frame MLP\n+ residual','Inter-frame attention\n+ residual','Inter-frame MLP\n+ residual']
summary=[]
for region in ['all','camera','register','patch']:
 for stage in range(5):
  r={'stage':stage,'name':labels[stage].replace('\n',' '),'region':region}
  for path in ['input','gard']:
   subset=[x for x in rows if x['region']==region and x['layer']==stage and x['path']==path]
   for metric in ['cosine','mean_token_cosine','current_rms_token_norm','clean_rms_token_norm']:
    r[path+'_'+metric]=float(np.mean([x[metric] for x in subset]))
  summary.append(r)
(P/'summary.json').write_text(json.dumps(summary,indent=2))
with (P/'metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def plot(regions,out,group=None):
 fig,axs=plt.subplots(2,len(regions),figsize=(7*len(regions),8),squeeze=False)
 for j,region in enumerate(regions):
  for i,(metric,title) in enumerate([('cosine','Flattened-vector cosine'),('mean_token_cosine','Mean per-token cosine')]):
   ax=axs[i,j]
   for path,color,label in [('input','#ed6a00','Clean vs distractor'),('gard','#3268dc','Clean vs restoration')]:
    y=[np.mean([r[metric] for r in rows if r['path']==path and r['region']==region and r['layer']==s and (group is None or r['group']==group)]) for s in range(5)]
    ax.plot(range(5),y,'o-',color=color,label=label)
    if len(regions)==1:
     for k,v in enumerate(y):ax.annotate(f'{v:.4f}',(k,v),xytext=(0,(10 if path=='gard' else -18) if k<2 else (10 if path=='input' else -18)),textcoords='offset points',ha='center',color=color,fontsize=9)
   ax.margins(y=.18);ax.set_xticks(range(5),labels,fontsize=8);ax.grid(alpha=.2);ax.set_title(region.capitalize()+' tokens — '+title);ax.legend(fontsize=9)
 fig.suptitle('VGGT-Omega aggregator block 17 (0-based) | '+('groups 00, 26, 53 mean' if group is None else f'group {group:02d}')+'\nResidual stream measured after addition, before next normalization',fontsize=12)
 fig.tight_layout(rect=(0,0,1,.93));fig.savefig(P/out,dpi=180);plt.close(fig)
plot(['patch'],'patch_stages.png')
plot(['all','camera','register','patch'],'all_token_stages.png')
for g in [0,26,53]:plot(['patch'],f'group_{g:02d}_patch.png',g)
# Verify the new read-only hooks reproduce prior pair-end measurements.
old=json.loads((P.parent/'omega_token_diagnostics/results.json').read_text())['rows']
errors=[]
for r in rows:
 if r['layer'] not in (0,4):continue
 layer=16 if r['layer']==0 else 17
 prev=next(o for o in old if o['group']==r['group'] and o['path']==r['path'] and o['region']==r['region'] and o['layer']==layer)
 errors.append(abs(prev['mean_token_cosine']-r['mean_token_cosine']))
assert max(errors)<1e-5,max(errors)
(P/'validation.json').write_text(json.dumps({'max_difference_from_previous_endpoints':max(errors),'endpoint_comparisons':len(errors)},indent=2))
(P/'README.md').write_text('# Omega block 17 diagnostics\n\nThree fixed evaluation groups: 00, 26, 53. Same 10-epoch EMA checkpoint, seed 42 + group, sampler and normalization as prior evaluation. GARD injected after block 3.\n\nFive stages: block input, frame attention + residual, frame MLP + residual, inter-frame attention + residual, inter-frame MLP + residual. Attention intermediates captured by norm2 pre-hooks; no forward modifications. Patch tokens only in patch_stages.png; no spatial masks. layer in metrics.csv denotes stage index, not backbone block. All comparisons against clean features at the same stage.\n\nFP32 cosine with BF16 model autocast. Endpoint agreement checked against prior block16/block17 results in validation.json.\n')
print(json.dumps([r for r in summary if r['region']=='patch'],indent=2))
