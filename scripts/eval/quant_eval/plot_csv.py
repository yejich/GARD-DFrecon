"""Publication-friendly static plots of exported CSVs; no inference or rescoring."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'result_eval/20260923_quant_eval';DEST=OUT/'plots';DEST.mkdir(exist_ok=True)
def read(name):
 with (OUT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
summary=read('dataset_summary.csv');groups=read('all_groups_metrics.csv')
methods=['da3','gard_6k','gard_003','gard_010','gard_020','gard_030','vgtw']
labels=['DA3','6K\n10ep','20K\n3ep','20K\n10ep','20K\n20ep','20K\n30ep','VGTW']
colors=['#7A869A','#E69F00','#A8DADC','#57ACCA','#287BA5','#164B76','#BA5370']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
def value(row,key):
 try:return float(row[key])
 except (KeyError,ValueError,TypeError):return np.nan
index={(r['dataset'],r['method']):r for r in summary};export=[]
def bar(ax,ds,key,title,ylabel='',fraction=False):
 vals=[value(index[(ds,m)],key) for m in methods];ax.bar(range(7),np.nan_to_num(vals,nan=0),color=colors,width=.7,zorder=2)
 finite=[v for v in vals if np.isfinite(v)];peak=max(finite,default=1)
 pts=[value(r,key) for r in groups if r['dataset']==ds];peak=max([peak]+[v for v in pts if np.isfinite(v)])
 ymax=1.10 if fraction else max(peak*1.23,1e-6);ax.set_ylim(0,ymax)
 for i,m in enumerate(methods):
  dots=[value(r,key) for r in groups if r['dataset']==ds and r['method']==m];dots=[v for v in dots if np.isfinite(v)]
  ax.scatter(i+np.linspace(-.17,.17,len(dots)),dots,s=13,facecolors='white',edgecolors='#263445',linewidth=.6,zorder=3,alpha=.9)
  if not np.isfinite(vals[i]):ax.text(i,.035*ymax,'N/A',ha='center',fontsize=9,color='#89929E')
  else:
   v=vals[i];text=f'{v:.4f}' if peak<.1 else (f'{v:.3f}' if peak<=1.1 else f'{v:.2f}')
   ax.text(i,v+.035*ymax,text,ha='center',va='bottom',fontsize=8.5,color='#263445',zorder=4,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.4))
 ax.set_xticks(range(7),labels);ax.tick_params(axis='x',length=0,labelsize=9);ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel(ylabel);ax.grid(axis='x',visible=False)
 if fraction:ax.set_yticks(np.linspace(0,1,6))
def layout(ds,title,rows,cols):
 fig,axes=plt.subplots(rows,cols,figsize=(5.35*cols,3.55*rows),squeeze=False)
 n=sum(r['dataset']==ds and r['method']=='da3' for r in groups)
 ref='Actual clean GT | distance unit: meters' if ds=='Hypersim' else 'Clean DA3 pseudo-GT | distance unit: normalized clean-camera baseline'
 fig.suptitle(f'{ds}  /  {title}',x=.055,y=.988,ha='left',fontsize=21,fontweight='bold',color='#16324F')
 fig.text(.055,.953,f'{ref}  |  {n} groups  |  Bars: equal-group mean; dots: individual groups',fontsize=11,color='#526171')
 fig.subplots_adjust(left=.065,right=.985,bottom=.07,top=.90,hspace=.58,wspace=.28)
 fig.text(.055,.019,'Lower is better: AbsRel, RMSE, pose error, cloud distance, LPIPS. Higher is better: delta1, AUC, F-score, PSNR, SSIM.\n6K / 20K labels denote GARD checkpoints. VGTW has no DA3 ray-pose branch or restored RGB decoder. Real metrics measure DA3-reference agreement.',fontsize=9,color='#657384')
 return fig,axes.ravel()
def save(fig,name,pdf):
 fig.savefig(DEST/f'{name}.png',dpi=180);fig.savefig(DEST/f'{name}.pdf');pdf.savefig(fig);plt.close(fig);export.append(name);print('SAVED',name,flush=True)
with PdfPages(DEST/'quantitative_comparison_all.pdf') as pdf:
 for ds in ['Hypersim','RealX3D','RobustNeRF']:
  unit='m' if ds=='Hypersim' else 'normalized units'
  fig,axs=layout(ds,'Overall metrics',4,3)
  specs=[('depth_all_absrel','Depth: AbsRel ↓','',False),('depth_all_rmse','Depth: RMSE ↓',unit,False),('depth_all_delta1','Depth: delta1 ↑','fraction',True),('pose_camera_auc_20','Camera-token pose: AUC@20° ↑','fraction',True),('pose_ray_auc_20','Ray pose: AUC@20° ↑','fraction',True),('pose_camera_rte_median','Camera-token pose: translation error ↓','mean of group medians (degrees)',False),('cloud_all_accuracy','Point cloud: Accuracy ↓',unit,False),('cloud_all_completeness','Point cloud: Completeness ↓',unit,False),('cloud_all_fscore_0.02',f'Point cloud: F-score @ {"2 cm" if ds=="Hypersim" else "0.02 normalized units"} ↑','fraction',True),('rgb_vs_clean_all_psnr','Decoded RGB vs clean RGB: PSNR ↑','dB',False),('rgb_vs_clean_all_ssim','Decoded RGB vs clean RGB: SSIM ↑','',True),('rgb_vs_clean_all_lpips_spatial','Decoded RGB vs clean RGB: LPIPS ↓','',False)]
  for ax,(key,title,yl,frac) in zip(axs,specs):bar(ax,ds,key,title,yl,frac)
  save(fig,ds.lower()+'_overview',pdf)
  fig,axs=layout(ds,'Pose branches and cloud detail',3,3)
  specs=[('pose_camera_rre_median','Camera-token rotation error ↓','degrees',False),('pose_ray_rre_median','Ray rotation error ↓','degrees',False),('pose_ray_rte_median','Ray translation-direction error ↓','degrees',False),('pose_camera_auc_5','Camera-token AUC@5° ↑','fraction',True),('pose_ray_auc_5','Ray AUC@5° ↑','fraction',True),('cloud_all_chamfer_l1','Point cloud: Chamfer-L1 ↓',unit,False),('cloud_clutter_views_fscore_0.02','Clutter-source cloud: F-score@0.02 ↑','fraction',True),('retained_fraction','Prediction pixels retained','fraction; native filters differ',True),('cloud_all_retained_reference_pixel_fraction','Retained on valid reference support','fraction; denominator differs from left',True)]
  for ax,(key,title,yl,frac) in zip(axs,specs):bar(ax,ds,key,title,yl,frac)
  save(fig,ds.lower()+'_pose_cloud_detail',pdf)
  fig,axs=layout(ds,'Restoration regions and RGB references',3,3)
  reg='clutter_views' if ds=='RealX3D' else 'distractor';regtitle='Clutter views' if reg=='clutter_views' else 'Distractor pixels'
  specs=[(f'depth_{reg}_absrel',f'{regtitle}: depth AbsRel ↓','',False),('depth_clean_views_absrel','Clean input views: depth AbsRel ↓','',False),(f'rgb_vs_clean_{reg}_psnr',f'{regtitle}: RGB PSNR ↑','dB',False),(f'rgb_vs_clean_{reg}_ssim',f'{regtitle}: RGB SSIM ↑','',True),(f'rgb_vs_clean_{reg}_lpips_spatial',f'{regtitle}: RGB LPIPS ↓','',False),('rgb_vs_clean_clean_views_psnr','Clean input views: RGB PSNR ↑','dB',False),('rgb_vs_clean_decoded_all_psnr','Vs clean-decoded RGB: PSNR ↑','dB',False),('rgb_vs_clean_decoded_all_ssim','Vs clean-decoded RGB: SSIM ↑','',True),('rgb_vs_clean_decoded_all_lpips_spatial','Vs clean-decoded RGB: LPIPS ↓','',False)]
  for ax,(key,title,yl,frac) in zip(axs,specs):bar(ax,ds,key,title,yl,frac)
  save(fig,ds.lower()+'_regions_rgb',pdf)
text='# Quantitative comparison plots\n\n[All plots in one PDF](quantitative_comparison_all.pdf)\n\nBars are dataset-level equal-group means; dots are individual groups, not confidence intervals. Metrics with different reference types or units are shown in separate figures. Blank CSV cells are N/A, never zero scores.\n\n'
for ds in ['hypersim','realx3d','robustnerf']:
 text+=f'## {ds}\n\n'
 for suffix,label in [('overview','Overall metrics'),('pose_cloud_detail','Pose and cloud details'),('regions_rgb','Regions and RGB references')]:
  n=f'{ds}_{suffix}';text+=f'- {label}: [PNG]({n}.png) / [PDF]({n}.pdf)\n'
text+='\nRealX3D region panels use whole clutter views because verified dense distractor masks are unavailable. Retention indicates output coverage, not an independent accuracy score. Reference support and prediction retention have different denominators. Source: ../dataset_summary.csv and ../all_groups_metrics.csv.\n'
(DEST/'README.md').write_text(text)
(DEST/'manifest.json').write_text(json.dumps({'source_csv':['dataset_summary.csv','all_groups_metrics.csv'],'figures':export,'combined_pdf':'quantitative_comparison_all.pdf','group_counts':{'Hypersim':5,'RealX3D':4,'RobustNeRF':4}},indent=2))
p=OUT/'README.md';s=p.read_text();marker='\n<!-- PLOTS -->';s=s.split(marker)[0]+marker+'\n## Graphs\n\n[Graph index](plots/README.md) / [All graphs PDF](plots/quantitative_comparison_all.pdf)\n';p.write_text(s)
