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
diags=sum([json.loads(p.read_text()) for p in sorted(P.glob('group_*_residual.json'))],[])
def mean(data):
 return {k:float(np.mean([r[k] for r in data])) for k,v in data[0].items() if isinstance(v,(int,float)) and k not in ('stage','group')}
summary=[dict(stage=s,name=next(r['name'] for r in rows if r['stage']==s),path=path,**mean([r for r in rows if r['stage']==s and r['path']==path])) for s in range(7) for path in ['input','gard']]
dr=[dict(path=path,**mean([r for r in diags if r['path']==path])) for path in ['clean','input','gard']]
(P/'summary.json').write_text(json.dumps(dict(stages=summary,residuals=dr),indent=2))
for filename,data in [('metrics.csv',rows),('residual_metrics.csv',diags)]:
 with (P/filename).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
labels=['MLP input','LayerNorm','Linear 1','GELU','Linear 2','LayerScale\nresidual','Residual\naddition']
fig,axs=plt.subplots(1,3,figsize=(18,5))
for ax,metric,title in zip(axs,['token_cosine','relative_l2','rms_norm'],['Mean patch-token cosine vs clean','Relative L2 error vs clean','RMS patch-token norm']):
 for path,color,label in [('input','#ed6a00','Distractor'),('gard','#3268dc','Restoration')]:
  ax.plot(range(7),[r[metric] for r in summary if r['path']==path],'o-',color=color,label=label)
 if metric=='rms_norm':
  ax.plot(range(7),[r['clean_rms_norm'] for r in summary if r['path']=='input'],'o--',color='gray',label='Clean');ax.set_yscale('log')
 ax.set_xticks(range(7),labels,rotation=28,ha='right');ax.set_title(title);ax.grid(alpha=.2);ax.legend()
fig.suptitle('VGGT-Omega block 17 frame MLP | groups 00, 26, 53 mean\nInternal stages have different feature spaces; cosine at each stage uses the matching clean reference')
fig.tight_layout(rect=(0,0,1,.9));fig.savefig(P/'mlp_stages.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,5))
x=np.arange(3)
axs[0].bar(x,[r['residual_to_input_rms'] for r in dr],color=['gray','#ed6a00','#3268dc']);axs[0].set_xticks(x,['Clean','Distractor','Restoration']);axs[0].set_ylabel('||LayerScale residual|| / ||MLP input||');axs[0].set_title('Residual magnitude relative to input')
labels=['Actual output','Clean residual\nnorm only','Clean residual\ndirection only','Full clean\nresidual']
for path,color,label in [('input','#ed6a00','Distractor'),('gard','#3268dc','Restoration')]:
 r=next(r for r in dr if r['path']==path)
 y=[next(r['token_cosine'] for r in summary if r['path']==path and r['stage']==6)]+[r[k+'_output_cosine'] for k in ['clean_residual_norm','clean_residual_direction','clean_residual']]
 axs[1].plot(range(4),y,'o-',color=color,label=label)
 for k,v in enumerate(y):axs[1].annotate(f'{v:.4f}',(k,v),xytext=(0,(10 if path=='input' else -17) if k<2 else (10 if path=='gard' else -17)),textcoords='offset points',ha='center',color=color)
axs[1].set_xticks(range(4),labels);axs[1].set_ylabel('Mean patch-token cosine vs clean output');axs[1].set_title('Local diagnostic: replace residual norm or direction');axs[1].legend();axs[1].margins(y=.2)
for ax in axs:ax.grid(axis='y',alpha=.2)
fig.suptitle('Block 17 frame MLP residual | counterfactuals use clean features, not deployable restoration')
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(P/'residual_diagnostics.png',dpi=170);plt.close(fig)
old=json.loads((P.parent/'omega_block17_diagnostics/results.json').read_text())['rows']
errs=[]
for r in rows:
 if r['stage'] not in (0,6):continue
 o=next(o for o in old if o['region']=='patch' and o['group']==r['group'] and o['path']==r['path'] and o['layer']==(1 if r['stage']==0 else 2))
 errs.append(abs(o['mean_token_cosine']-r['token_cosine']))
assert max(errs)<1e-5,max(errs)
(P/'validation.json').write_text(json.dumps({'max_endpoint_difference':max(errs),'comparisons':len(errs)},indent=2))
(P/'README.md').write_text('Frame MLP in VGGT-Omega aggregator block 17 (0-based). Same fixed groups 00/26/53, checkpoint, seeds and sampler as prior diagnostic. Patch tokens only, no spatial masks. Read-only hooks on norm2, fc1, GELU, fc2, ls2 and block output. Residual is LayerScale(MLP(LayerNorm(input))). Metrics FP32; model BF16 autocast.\n\nResidual counterfactuals hold each current input fixed and replace each token residual norm/direction with the matching clean residual. They are local diagnostics using clean oracle information, not model evaluation or a proposed inference method. All internal cosine comparisons use matching clean stages; feature dimension and basis vary across stages. RMS ratios use pooled norm across all patch tokens within each group, then average groups. Endpoints checked against previous diagnostic.\n')
print(json.dumps({'stages':summary,'residuals':dr},indent=2))
