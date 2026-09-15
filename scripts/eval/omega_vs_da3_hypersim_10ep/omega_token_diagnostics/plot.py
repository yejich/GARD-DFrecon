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
ROOT=Path(__file__).resolve().parent;data=json.loads((ROOT/'results.json').read_text());rows=data['rows'];layers=np.arange(data['depth']);groups=data['groups'];regions=['all','camera','register','patch'];paths=['input','gard'];colors={'input':'#ea580c','gard':'#2563eb'};labels={'input':'Clean vs distractor','gard':'Clean vs restoration'}
def values(path,region,key,group=None):
 return np.array([np.mean([r[key] for r in rows if r['path']==path and r['region']==region and r['layer']==i and (group is None or r['group']==group)]) for i in layers])
def axes_style(ax):
 ax.set_xticks(layers);ax.tick_params(axis='x',labelsize=7);ax.axvline(3,ls='--',color='gray',lw=1);ax.axvspan(16.7,17.3,color='gray',alpha=.12);ax.grid(alpha=.2);ax.set_xlabel('Aggregator index i (0-based)')
for group in [None]+groups:
 fig,axes=plt.subplots(2,4,figsize=(21,9),sharex=True)
 for c,region in enumerate(regions):
  for row,key in enumerate(['cosine','mean_token_cosine']):
   ax=axes[row,c]
   for path in paths:ax.plot(layers,values(path,region,key,group),color=colors[path],label=labels[path],marker='o',ms=3)
   ax.set_title(region.title()+' tokens');ax.set_ylabel('Flattened-vector cosine' if row==0 else 'Mean per-token cosine');axes_style(ax);ax.legend(fontsize=8)
 fig.suptitle(('Three-group mean' if group is None else f'Group {group:02d}')+' | camera / register / patch contributions\nDashed: GARD injection at 3. Shaded: block 17. Each point is after frame + inter-frame blocks.',fontsize=14);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(ROOT/('cosine_comparison.png' if group is None else f'group_{group:02d}_cosine.png'),dpi=150)
 if group is None:fig.savefig(ROOT/'cosine_comparison.pdf')
 plt.close(fig)
fig,ax=plt.subplots(figsize=(13,5))
for path in paths:
 for key,style,desc in [('cosine','-','flattened vector'),('mean_token_cosine','--','mean per token')]:ax.plot(layers,values(path,'all',key),color=colors[path],ls=style,label=labels[path]+' / '+desc,lw=2)
axes_style(ax);ax.set_ylabel('Cosine similarity');ax.set_title('All tokens: flattened-vector vs mean per-token cosine');ax.legend();fig.tight_layout();fig.savefig(ROOT/'all_tokens_cosine.png',dpi=170);plt.close(fig)
regioncolors={'camera':'#dc2626','register':'#7c3aed','patch':'#059669'}
fig,axes=plt.subplots(1,3,figsize=(19,5),sharey=True)
for ax,(path,key,title) in zip(axes,[('input','clean_energy_share','Clean'),('input','current_energy_share','Distractor'),('gard','current_energy_share','Restoration')]):
 for region in regions[1:]:ax.plot(layers,100*values(path,region,key),label=region,color=regioncolors[region],marker='o',ms=3)
 axes_style(ax);ax.set_title(title);ax.set_ylim(0,100);ax.set_ylabel('Share of total squared feature norm (%)');ax.legend()
fig.suptitle('Token energy share: sum(||token||^2) / sum(||all tokens||^2)',fontsize=14);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(ROOT/'energy_share.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(19,5))
for ax,region in zip(axes,regions[1:]):
 for path,key,label,col in [('input','clean_rms_token_norm','Clean','#111827'),('input','current_rms_token_norm','Distractor',colors['input']),('gard','current_rms_token_norm','Restoration',colors['gard'])]:ax.plot(layers,values(path,region,key),label=label,color=col,marker='o',ms=3)
 axes_style(ax);ax.set_title(region.title());ax.set_ylabel('RMS token L2 norm (log scale)');ax.set_yscale('log');ax.legend()
fig.tight_layout();fig.savefig(ROOT/'token_norm.png',dpi=150);plt.close(fig)
with (ROOT/'per_group_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for layer in [3,16,17,18,22,23]:
 for region in regions:
  summary.append(dict(layer=layer,region=region,clean_energy_share=float(values('input',region,'clean_energy_share')[layer]),input_flat=float(values('input',region,'cosine')[layer]),restored_flat=float(values('gard',region,'cosine')[layer]),input_tokenmean=float(values('input',region,'mean_token_cosine')[layer]),restored_tokenmean=float(values('gard',region,'mean_token_cosine')[layer])))
(ROOT/'selected_layers.json').write_text(json.dumps(summary,indent=2))
(ROOT/'README.md').write_text('''# Whole-feature metric diagnostics by token type

Same three groups 00,26,53, same saved 10-epoch Omega GARD EMA/normalization, seed 42+group and 50 sampler time points. Full aggregator tensors after frame+inter-frame block i, i=0..23; injection at i=3 is scored after the injected feature is cast to the actual tensor dtype. No spatial clean/distractor region masks.

Each group contains 4 views; each view has 1 camera + 16 register + 768 patch tokens. Token-mean cosine weights tokens equally. Flattened-vector cosine retains norm weighting. Energy share is sum of squared L2 norms for a token class divided by the full tensor energy; it is not a token count percentage or an exact additive decomposition of cosine. RMS token norm is sqrt(mean squared L2 norm), not mean L2 norm. Clean energy fractions in the three-group plots are arithmetic means of per-group fractions.

cosine_comparison.png compares flattened and token-mean cosine for all/camera/register/patch. energy_share.png and token_norm.png expose token-class norm changes. group_XX_cosine.png gives individual groups, and CSV/JSON preserve exact values. These diagnostics test norm dominance; they do not establish a causal explanation of attention or downstream depth/pose accuracy.
''')
(ROOT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Omega token diagnostics</title><style>body{font:16px sans-serif}img{width:100%}</style>'+''.join(f'<h2>{name}</h2><img src="{name}.png">' for name in ['all_tokens_cosine','cosine_comparison','energy_share','token_norm']))
print(json.dumps([r for r in summary if r['layer'] in [16,17,23]],indent=2))
