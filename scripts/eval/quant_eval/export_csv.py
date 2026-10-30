"""Export existing verified JSON metrics; no inference or rescoring."""
import csv,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'result_eval/20260923_quant_eval'
LABELS={'da3':'DA3 baseline','gard_6k':'GARD 6K / 10ep','gard_003':'GARD 20K / 3ep','gard_010':'GARD 20K / 10ep','gard_020':'GARD 20K / 20ep','gard_030':'GARD 20K / 30ep','vgtw':'VGTW'}
def write(name,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 print(name,len(rows),'rows')
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def scores(m):
 out={}
 for reg,vals in m['depth']['regions'].items():
  for k in ['absrel','rmse','delta1']:out[f'depth_{reg}_{k}']=vals.get(k)
 for branch in ['pose_camera','pose_ray']:
  for k in ['rre_median','rte_median','auc_5','auc_10','auc_20','auc_30','ate_rmse_sim3']:out[f'{branch}_{k}']=m.get(branch,{}).get(k)
 for reg,vals in m['cloud'].items():
  for k in ['accuracy','completeness','chamfer_l1',*[f'{n}_{t}' for t in ['0.01','0.02','0.05'] for n in ['precision','recall','fscore']],'retained_reference_pixel_fraction']:out[f'cloud_{reg}_{k}']=vals.get(k)
 for task in ['rgb_vs_clean','rgb_vs_clean_decoded']:
  for reg,vals in m.get(task,{}).items():
   for k in ['psnr','ssim','lpips_spatial']:out[f'{task}_{reg}_{k}']=vals.get(k)
 out['retained_fraction']=m['retained_fraction']
 return out
inventory=json.loads((OUT/'eval_set_inventory.json').read_text());rows=[];decoder=[];regions=[];pairs=[]
for g in inventory['groups']:
 record=json.loads((OUT/g['id']/'metrics.json').read_text());ref=record['reference']
 base=dict(group=g['id'],dataset=g['dataset'],reference_type='actual_GT' if g['dataset']=='Hypersim' else 'DA3_pseudo_GT',geometry_distance_unit=ref['unit'],pose_ate_unit=ref.get('pose_unit',ref['unit']))
 for method,m in record['methods'].items():
  identity={**base,'method':method,'method_label':LABELS[method]};rows.append({**identity,**scores(m)})
  for task,vals in [('depth',m['depth']['regions']),('cloud',m['cloud']),('rgb_vs_clean',m.get('rgb_vs_clean',{})),('rgb_vs_clean_decoded',m.get('rgb_vs_clean_decoded',{}))]:
   for region,measurements in vals.items():regions.append({**identity,'task':task,'region':region,**measurements})
  for branch in ['pose_camera','pose_ray']:
   for pair in m.get(branch,{}).get('pairs',[]):pairs.append({**identity,'branch':branch,**pair})
 for region,vals in record['clean_decoder_vs_clean_rgb'].items():decoder.append({**base,'region':region,**vals})
assert len(rows)==91
write('all_groups_metrics.csv',rows);write('all_regions_metrics.csv',regions);write('all_pose_pairs.csv',pairs);write('clean_decoder_metrics.csv',decoder)
summary=[]
meta_keys=set(base)|{'method','method_label'}
metric_keys=[k for k in dict.fromkeys(k for r in rows for k in r) if k not in meta_keys]
for dataset in ['Hypersim','RealX3D','RobustNeRF']:
 for method,label in LABELS.items():
  selected=[r for r in rows if r['dataset']==dataset and r['method']==method]
  row={k:selected[0][k] for k in ['dataset','reference_type','geometry_distance_unit','pose_ate_unit']};row.update(method=method,method_label=label,group_count=len(selected))
  for key in metric_keys:
   values=[r.get(key) for r in selected if finite(r.get(key))];row[key]=statistics.mean(values) if values else None;row[key+'_n_groups']=len(values)
  summary.append(row)
write('dataset_summary.csv',summary)
(OUT/'CSV_GUIDE.md').write_text('''# CSV results

- dataset_summary.csv: 21 rows, dataset × method means. Primary comparison table.
- all_groups_metrics.csv: 91 rows, group × method; all headline/region metrics in columns.
- all_regions_metrics.csv: detailed group/method/task/region measurements with support counts.
- all_pose_pairs.csv: individual camera-pair errors for camera-head and ray-head pose.
- clean_decoder_metrics.csv: clean-decoded RGB versus actual clean RGB, separately from restoration.

UTF-8 with BOM for spreadsheet compatibility. Missing/unsupported values are blank, never zero; VGTW has no native DA3 ray branch or restored RGB decoder. Dataset means are equal group means with a valid group count next to every metric; no combined GT/pseudo-GT or mixed-unit distance mean. AUC, delta1 and F-score are fractions in [0,1]; rotation/translation-direction errors are degrees; PSNR is dB. Real distance units are normalized, not meters. Threshold suffixes 0.01/0.02/0.05 use each row's geometry_distance_unit. RobustNeRF includes two sets each for crab2/yoda. RGB SSIM and spatial LPIPS are region-map means. Full execution details: EXECUTED_PROTOCOL.md.
''')
p=OUT/'README.md';s=p.read_text();marker='\n<!-- CSV-EXPORTS -->';s=s.split(marker)[0]+marker+'\n## CSV tables\n\n- [Dataset summary](dataset_summary.csv)\n- [All group metrics](all_groups_metrics.csv)\n- [Region metrics](all_regions_metrics.csv)\n- [Pose pairs](all_pose_pairs.csv)\n- [Clean decoder metrics](clean_decoder_metrics.csv)\n- [Column interpretation](CSV_GUIDE.md)\n';p.write_text(s)
