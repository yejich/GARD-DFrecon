"""Finalize already-scored Hypersim pose SO(3) and VGTW half-pixel resize metrics."""
from score import *
from references import stackresize
for g in json.loads((OUT/'eval_set_inventory.json').read_text())['groups']:
 if g['dataset']!='Hypersim':continue
 folder=OUT/g['id'];rp=folder/'metrics.json'
 if not rp.exists():continue
 report=json.loads(rp.read_text());ref=dict(np.load(folder/'reference.npz'));regs=regions(g,ref);ref['alignment_mask']=ref['valid']&~ref['union_mask'];depthcols=[('reference',ref['depth'])]
 for name in METHODS:
  data=dict(np.load(folder/f'{name}.npz'));m=report['methods'][name];m['pose_camera']=pose_metrics(data['extrinsics'],ref['extrinsics'])
  if 'ray_extrinsics' in data:m['pose_ray']=pose_metrics(data['ray_extrinsics'],ref['extrinsics'])
  d=stackresize(data['depth'],g['image_hw'],True);depth,aligned_d=depth_scores(d,ref,regs);depthcols.append((name,aligned_d))
  if name!='vgtw':continue
  m['depth']=depth;points=stackresize(data['world_points'],g['image_hw'],True);rgb=stackresize(data['rgb'],g['image_hw']);keep,thr=confidence_keep(data,True);keep=stackresize(keep.astype(np.uint8),g['image_hw'],True)>0;fit=ref['alignment_mask']&np.isfinite(points).all(-1);sim=fit_sim(points[fit],ref['points'][fit],robust=True);aligned=transform(points,sim)
  m['cloud_alignment']={'scale':sim[0],'rotation':sim[1].tolist(),'translation':sim[2].tolist(),**sim[3]};m['retained_fraction']=float(keep.mean());m['retained_per_view']=[float(x.mean()) for x in keep]
  for region in ['all','clutter_views']:
   support=regs[region];sel=keep&support;pp,cc=voxel(aligned[sel],rgb[sel],report['cloud_voxel_size']);qq,_=voxel(ref['points'][support],None,report['cloud_voxel_size']);m['cloud'][region]=cloud_metrics(pp,qq,report['cloud_thresholds']);m['cloud'][region]['retained_reference_pixel_fraction']=float(sel.sum()/max(support.sum(),1))
   if region=='all':write_ply(folder/'vgtw_input_rgb.ply',pp,cc)
 report['finalization']='All pose rotations projected to SO(3); resized VGTW arrays use half-pixel nearest mapping (INTER_NEAREST_EXACT).';rp.write_text(json.dumps(safe(report),indent=2,allow_nan=False));lo,hi=np.percentile(ref['depth'][ref['valid']],[2,98]);grid(folder/'depth_comparison.jpg',[(n,depthcolor(d,lo,hi)) for n,d in depthcols]);print('FINALIZED',g['id'],flush=True)
summarize('hypersim')
