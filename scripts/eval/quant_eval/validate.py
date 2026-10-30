"""Audit completed quantitative outputs and write a compact result index."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'result_eval/20260923_quant_eval'
groups=json.loads((OUT/'eval_set_inventory.json').read_text())['groups'];methods=['da3','gard_6k','gard_003','gard_010','gard_020','gard_030','vgtw'];results=[]
for g in groups:
 folder=OUT/g['id'];record={'group':g['id'],'dataset':g['dataset'],'checks':{},'errors':[]}
 try:
  assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in g['input_paths']]==g['input_sha256'],'input hash mismatch'
  m=json.loads((folder/'metrics.json').read_text());assert set(m['methods'])==set(methods)
  assert all((folder/p).is_file() for p in ['reference.npz','reference.ply','rgb_comparison.jpg','depth_comparison.jpg'])
  for method in methods:
   with np.load(folder/f'{method}.npz') as x:
    assert x['depth'].shape[0]==4
    for k in ['depth','depth_conf','extrinsics','intrinsics','world_points','rgb']:assert np.isfinite(x[k]).all(),f'{method}: nonfinite {k}'
    if method!='vgtw':assert 'ray_extrinsics' in x and np.isfinite(x['ray_extrinsics']).all()
   sm=m['methods'][method]
   assert len(sm['pose_camera']['pairs'])==6 and 0<=sm['pose_camera']['auc_20']<=1
   if method!='vgtw':assert len(sm['pose_ray']['pairs'])==6 and 'rgb_vs_clean_decoded' in sm
   assert sm['depth']['regions']['all']['invalid_pixels']==0
   assert (folder/(f'{method}_decoded_rgb.ply' if method!='vgtw' else 'vgtw_input_rgb.ply')).is_file()
  record['checks']={'input_hashes':True,'finite_predictions':True,'methods':7,'camera_pose_pairs':42,'ray_pose_pairs':36,'colored_clouds':7,'reference_pixels':m['reference_region_pixels']}
  if g['dataset']!='Hypersim':record['reference_quality']={k:m['reference'][k] for k in ['registration_diagnostic_not_used_for_geometry','reference_coverage','rgb_registration']}
  record['status']='passed'
 except Exception as e:record['status']='failed_or_incomplete';record['errors'].append(str(e))
 results.append(record)
status={'status':'passed' if all(r['status']=='passed' for r in results) else 'incomplete','groups':results,'completed':sum(r['status']=='passed' for r in results),'expected':len(groups)}
(OUT/'validation.json').write_text(json.dumps(status,indent=2));print(json.dumps({k:v for k,v in status.items() if k!='groups'}))
text='# Quantitative evaluation\n\n'+f"Validation: {status['completed']}/{status['expected']} groups passed. Execution device: GPU 5 (initial Hypersim inference used GPUs 0/1 before user migration).\n\n"
text+='- [Hypersim results](hypersim_RESULTS.md) / [CSV](hypersim_metrics.csv) / [regions](hypersim_regions.csv) / [pose pairs](hypersim_pose_pairs.csv)\n- [Real results](real_RESULTS.md) / [CSV](real_metrics.csv) / [regions](real_regions.csv) / [pose pairs](real_pose_pairs.csv)\n- [Executed protocol](EXECUTED_PROTOCOL.md) / [validation](validation.json) / [pipeline status](pipeline_status.json)\n\n'
text+='Hypersim uses actual clean GT. Real geometry uses the agreed clean DA3 pseudo-GT (RealX3D 12 views; crab2 20/19; yoda 14/9). Real geometry distances use median clean-DA3 evaluation-camera baseline units; they are not meters. RGB reference uses same-number clean frames and records camera warp displacement/coverage. DA3 teacher affinity and COLMAP reference errors limit real-score interpretation.\n\nCamera and ray pose scores are separate. Point clouds use the camera-head geometry and decoded RGB for DA3/GARD; VGTW uses native camera geometry and input RGB with confidence percentile 50 plus its predicted mask.\n\n'
text+='| Group | Depth comparison | RGB comparison | Metrics |\n|---|---|---|---|\n'
for g in groups:
 name=g['id'];text+=f'| {name} | [image]({name}/depth_comparison.jpg) | [image]({name}/rgb_comparison.jpg) | [JSON]({name}/metrics.json) |\n'
text+='\nColored point clouds: each group/da3_decoded_rgb.ply, gard_6k_decoded_rgb.ply, gard_003/010/020/030_decoded_rgb.ply, vgtw_input_rgb.ply, reference.ply.\n'
(OUT/'README.md').write_text(text)
