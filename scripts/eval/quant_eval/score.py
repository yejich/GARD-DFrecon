"""Fixed support depth/pose/cloud/RGB scoring; no model selection or threshold tuning."""
import sys,json,argparse,csv
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from skimage.metrics import structural_similarity
from metrics import *
from references import hypersim,real,resize,stackresize
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'result_eval/20260923_quant_eval'
METHODS=['da3','gard_6k','gard_003','gard_010','gard_020','gard_030','vgtw']
def safe(x):
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,list):return [safe(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
 return x

def regions(g,ref):
 valid=ref['valid'];cl=np.zeros_like(valid);cl[g['clutter_views']]=True;d=ref['object_mask'];u=ref['union_mask']
 out={'all':valid,'clutter_views':valid&cl,'clean_views':valid&~cl}
 if bool(ref['mask_known']):out.update(distractor=valid&d,background=valid&cl&~u,object_shadow=valid&u)
 return out

def depth_scores(pred,ref,regs):
 gt=ref['depth'];valid=ref['valid'];good=np.isfinite(pred)&(pred>0);fit=ref['alignment_mask']&good
 if not fit.any():return {'status':'no_alignment_support'},pred
 scale=float(np.median(gt[fit]/pred[fit]));d=pred*scale;out={'scale':scale,'regions':{}}
 for name,s in regs.items():
  n=int(s.sum());bad=int((s&~good).sum())
  if not n:out['regions'][name]={'pixels':0};continue
  ratio=np.maximum(d[s]/gt[s],gt[s]/np.maximum(d[s],1e-12));delta=float(np.sum((ratio<1.25)&good[s])/n)
  # Invalid predictions explicitly fail region error instead of silently dropping pixels.
  out['regions'][name]=dict(pixels=n,invalid_pixels=bad,absrel=None if bad else float(np.mean(abs(d[s]-gt[s])/gt[s])),rmse=None if bad else float(np.sqrt(np.mean((d[s]-gt[s])**2))),delta1=delta,status='invalid_prediction' if bad else 'ok')
 return out,d
@torch.no_grad()
def rgb_scores(pred,target,regs,lp,device):
 pred=np.clip(pred,0,1);target=np.clip(target,0,1);ss=[];ls=[]
 for a,b in zip(pred,target):
  _,s=structural_similarity(a,b,data_range=1.,channel_axis=-1,full=True);ss.append(s.mean(-1))
  at=torch.from_numpy(a.transpose(2,0,1).copy())[None].to(device)*2-1;bt=torch.from_numpy(b.transpose(2,0,1).copy())[None].to(device)*2-1
  ls.append(lp(at,bt)[0,0].cpu().numpy())
 ss=np.stack(ss);ls=np.stack(ls);err=((pred-target)**2).mean(-1);out={}
 for n,s in regs.items():
  if not s.any():out[n]={'pixels':0};continue
  mse=float(err[s].mean());out[n]=dict(pixels=int(s.sum()),psnr=120. if mse<1e-12 else float(-10*np.log10(mse)),ssim=float(ss[s].mean()),lpips_spatial=float(ls[s].mean()))
 return out

def grid(path,cols):
 h,w=cols[0][1].shape[1:3];tw=252;th=round(h*tw/w);out=Image.new('RGB',(tw*len(cols),4*(th+25)),'white');dr=ImageDraw.Draw(out)
 for i,(name,arr) in enumerate(cols):
  for v in range(4):out.paste(Image.fromarray(np.uint8(np.clip(arr[v],0,1)*255)).resize((tw,th)),(i*tw,v*(th+25)+25));dr.text((i*tw+3,v*(th+25)+5),name+' v'+str(v),fill='black')
 out.save(path,quality=92)
def depthcolor(d,lo,hi):
 return np.stack([cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip((np.nan_to_num(x,nan=hi)-lo)/(hi-lo),0,1)*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)/255 for x in d])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['hypersim','real'],required=True);ap.add_argument('--groups',nargs='+');ap.add_argument('--device',default='cuda');a=ap.parse_args();selftest();torch.set_num_threads(4)
 import lpips
 lp=lpips.LPIPS(net='alex',spatial=True).to(a.device).eval()
 groups=[g for g in json.loads((OUT/'eval_set_inventory.json').read_text())['groups'] if (g['dataset']=='Hypersim')==(a.dataset=='hypersim') and (not a.groups or g['id'] in a.groups)]
 for g in groups:
  folder=OUT/g['id'];hw=tuple(g['image_hw']);print('SCORE',g['id'],flush=True)
  ref,meta=hypersim(g,hw) if a.dataset=='hypersim' else real(g,hw)
  if 'decoded_rgb' not in ref:ref['decoded_rgb']=np.load(folder/'clean_reference.npz')['rgb']
  np.savez_compressed(folder/'reference.npz',**ref);(folder/'reference.json').write_text(json.dumps(meta,indent=2))
  regs=regions(g,ref)
  ref['alignment_mask']=ref['valid']&~ref['union_mask'] if bool(ref['mask_known']) else regs['clean_views']
  voxel_size=.005 if a.dataset=='hypersim' else .002;thresholds=[.01,.02,.05]
  report=dict(group=g['id'],dataset=g['dataset'],reference=meta,cloud_voxel_size=voxel_size,cloud_thresholds=thresholds,methods={},reference_region_pixels={n:int(s.sum()) for n,s in regs.items()},rgb_protocol='PSNR over region MSE; SSIM local-map mean; AlexNet spatial LPIPS map region mean (receptive fields include neighboring pixels)')
  colors=[('input',np.load(folder/'inputs.npz')['rgb']),('clean ref',ref['rgb']),('clean decoded',ref['decoded_rgb'])];depthcols=[('reference',ref['depth'])]
  reference_points,reference_colors=voxel(ref['points'][ref['valid']],ref['rgb'][ref['valid']],voxel_size);write_ply(folder/'reference.ply',reference_points,reference_colors)
  # RGB reference validity erosion avoids SSIM windows crossing holes; LPIPS full receptive field remains a caveat for rendered real references.
  rgbregs=regs
  if a.dataset=='real':
   rgbvalid=np.stack([cv2.erode(x.astype(np.uint8),np.ones((7,7),np.uint8))>0 for x in ref['rgb_valid']]);rgbref=dict(ref);rgbref['valid']=rgbvalid;rgbregs=regions(g,rgbref)
  report['clean_decoder_vs_clean_rgb']=rgb_scores(ref['decoded_rgb'],ref['rgb'],rgbregs,lp,a.device)
  for name in METHODS:
   data=dict(np.load(folder/f'{name}.npz'));d=stackresize(data['depth'],hw,True);points=stackresize(data['world_points'],hw,True);rgb=stackresize(data['rgb'],hw);keep,thr=confidence_keep(data,name=='vgtw');keep=stackresize(keep.astype(np.uint8),hw,True)>0
   m={'pose_camera':pose_metrics(data['extrinsics'],ref['extrinsics'])}
   if 'ray_extrinsics' in data:m['pose_ray']=pose_metrics(data['ray_extrinsics'],ref['extrinsics'])
   m['depth'],aligned_depth=depth_scores(d,ref,regs);depthcols.append((name,aligned_depth))
   fit=ref['alignment_mask']&np.isfinite(points).all(-1)
   sim=fit_sim(points[fit],ref['points'][fit],robust=True);aligned=transform(points,sim)
   m['cloud_alignment']={'scale':sim[0],'rotation':sim[1].tolist(),'translation':sim[2].tolist(),**sim[3]};m['confidence_threshold']=thr;m['retained_fraction']=float(keep.mean());m['retained_per_view']=[float(x.mean()) for x in keep];m['cloud']={}
   for region in ['all','clutter_views']:
    support=regs[region];predsel=keep&support;pp,cc=voxel(aligned[predsel],rgb[predsel],voxel_size);qq,_=voxel(ref['points'][support],None,voxel_size)
    m['cloud'][region]=cloud_metrics(pp,qq,thresholds);m['cloud'][region]['retained_reference_pixel_fraction']=float(predsel.sum()/max(support.sum(),1))
    if region=='all':write_ply(folder/f'{name}_decoded_rgb.ply' if name!='vgtw' else folder/'vgtw_input_rgb.ply',pp,cc)
   if name!='vgtw':
    m['rgb_vs_clean']=rgb_scores(rgb,ref['rgb'],rgbregs,lp,a.device);m['rgb_vs_clean_decoded']=rgb_scores(rgb,ref['decoded_rgb'],rgbregs,lp,a.device);colors.append((name,rgb))
   else:m['rgb_status']='VGTW has no restored RGB decoder; cloud uses input RGB'
   report['methods'][name]=m;print('SCORED',g['id'],name,flush=True)
  (folder/'metrics.json').write_text(json.dumps(safe(report),indent=2,allow_nan=False))
  grid(folder/'rgb_comparison.jpg',colors);lo,hi=np.percentile(ref['depth'][ref['valid']],[2,98]);grid(folder/'depth_comparison.jpg',[(n,depthcolor(d,lo,hi)) for n,d in depthcols])
  print('GROUP COMPLETE',g['id'],flush=True)
 summarize(a.dataset)
def summarize(dataset):
 allgroups=json.loads((OUT/'eval_set_inventory.json').read_text())['groups'];records=[];rows=[]
 for g in allgroups:
  if (g['dataset']=='Hypersim')!=(dataset=='hypersim'):continue
  p=OUT/g['id']/'metrics.json'
  if not p.exists():continue
  record=json.loads(p.read_text());records.append(record)
  for method,m in record['methods'].items():
   row=dict(group=g['id'],dataset=g['dataset'],method=method,absrel=m['depth']['regions']['all']['absrel'],rmse=m['depth']['regions']['all']['rmse'],delta1=m['depth']['regions']['all']['delta1'],pose_auc20=m['pose_camera']['auc_20'],ray_auc20=m.get('pose_ray',{}).get('auc_20'),cloud_accuracy=m['cloud']['all'].get('accuracy'),cloud_completeness=m['cloud']['all'].get('completeness'),cloud_fscore02=m['cloud']['all'].get('fscore_0.02'),retained=m['retained_fraction'],rgb_psnr=m.get('rgb_vs_clean',{}).get('all',{}).get('psnr'),rgb_lpips=m.get('rgb_vs_clean',{}).get('all',{}).get('lpips_spatial'))
   rows.append(row)
 if not rows:return
 with (OUT/f'{dataset}_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/f'{dataset}_metrics.json').write_text(json.dumps(records,indent=2))
 details=[];pairs=[]
 for record in records:
  for method,m in record['methods'].items():
   base=dict(group=record['group'],dataset=record['dataset'],method=method)
   for region,values in m['depth']['regions'].items():details.append(dict(**base,task='depth',region=region,**values))
   for region,values in m['cloud'].items():details.append(dict(**base,task='cloud',region=region,**values))
   for task in ['rgb_vs_clean','rgb_vs_clean_decoded']:
    for region,values in m.get(task,{}).items():details.append(dict(**base,task=task,region=region,**values))
   for branch in ['pose_camera','pose_ray']:
    for pair in m.get(branch,{}).get('pairs',[]):pairs.append(dict(**base,branch=branch,**pair))
 for label,items in [('regions',details),('pose_pairs',pairs)]:
  keys=list(dict.fromkeys(k for x in items for k in x))
  with (OUT/f'{dataset}_{label}.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(items)

 text=f'# {dataset}: {len(records)} groups measured\n\nFull metrics: {dataset}_metrics.csv and each group/metrics.json. Cloud colors use decoded RGB for DA3/GARD; VGTW uses input RGB.\n\n'
 for ds in sorted(set(r['dataset'] for r in rows)):
  text+=f'## {ds} — equal group means\n\n|Method|AbsRel ↓|δ1 ↑|Camera AUC20 ↑|Ray AUC20 ↑|Cloud F@0.02 ↑|RGB PSNR ↑|\n|---|---:|---:|---:|---:|---:|---:|\n'
  for method in METHODS:
   subset=[r for r in rows if r['method']==method and r['dataset']==ds];vals=[]
   for k in ['absrel','delta1','pose_auc20','ray_auc20','cloud_fscore02','rgb_psnr']:
    v=[r[k] for r in subset if r[k] is not None];vals.append(f'{np.mean(v):.4f}' if v else 'N/A')
   text+='|'+method+'|'+'|'.join(vals)+'|\n'
 text+='\nReal scores measure DA3 pseudo-GT agreement, not independent measured GT. Real cloud/depth distance unit = median clean-DA3 evaluation-camera baseline; real pose ATE unit = median COLMAP evaluation-camera baseline; Hypersim unit = meter. Pose AUC uses exact empirical CDF area. RGB real references use corresponding clean frames warped to evaluation cameras; their coverage and displacement are reported. Spatial LPIPS includes neighboring context.\n'
 (OUT/f'{dataset}_RESULTS.md').write_text(text)
if __name__=='__main__':main()
