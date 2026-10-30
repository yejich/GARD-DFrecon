"""Score the matched five-group E10/E20 comparison outputs and build grids."""
import csv, json, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/eval/quant_eval')]
from metrics import centers, fit_sim, transform, pose_metrics, cloud_metrics, voxel
from references import hypersim

OUT = ROOT / 'result_eval/20261026_three_config_visualization'
CONFIGS = {
    'no_nvs_da_off': 'NVS-off_DA-off',
    'no_nvs_da_on': 'NVS-off_DA-on',
    'nvs_da_off': 'NVS-on_DA-off',
}
EPOCHS = ('epoch010', 'epoch020')

def resize(a, hw):
    return cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)

def resolve_dataset_path(p):
    p = Path(p)
    if not p.exists() and str(p).startswith('/mnt/dataset1/'):
        q = Path('/mnt/nvme1n1') / p.relative_to('/mnt/dataset1')
        if q.exists(): return q
    return p

def rgb_grid(path, arrays, labels):
    h, w = arrays[0].shape[1:3]; tw = 280; th = round(h * tw / w)
    out = Image.new('RGB', (tw * len(arrays), 4 * (th + 24)), 'white'); d = ImageDraw.Draw(out)
    for ci, (a, label) in enumerate(zip(arrays, labels)):
        for vi in range(4):
            im = Image.fromarray(np.uint8(np.clip(a[vi], 0, 1) * 255)).resize((tw, th))
            out.paste(im, (ci * tw, vi * (th + 24) + 24)); d.text((ci * tw + 3, vi * (th + 24) + 4), f'{label} v{vi}', fill='black')
    out.save(path, quality=92)

def depth_grid(path, arrays, labels):
    valid = np.concatenate([x[np.isfinite(x) & (x > 0)] for x in arrays])
    lo, hi = np.percentile(valid, [2, 98]); colored = []
    for a in arrays:
        colored.append(np.stack([cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip((np.nan_to_num(x, nan=hi)-lo)/max(hi-lo, 1e-8), 0, 1)*255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)/255 for x in a]))
    rgb_grid(path, colored, labels)

def copy_grid(path, source_paths, labels):
    imgs=[]
    for p in source_paths:
        with Image.open(p) as im: imgs.append(np.asarray(im.convert('RGB'))/255.)
    h,w=imgs[0].shape[:2]; tw=720; th=round(h*tw/w); out=Image.new('RGB',(tw*len(imgs),th),'white'); d=ImageDraw.Draw(out)
    for i,(im,label) in enumerate(zip(imgs,labels)):
        out.paste(Image.fromarray(np.uint8(im*255)).resize((tw,th)),(i*tw,0)); d.text((i*tw+4,4),label,fill='white')
    out.save(path,quality=92)

def main():
    registry = json.loads((ROOT/'datasets/eval/outdomain_visualization.json').read_text())
    groups = registry['groups'][:5]
    lp = __import__('lpips').LPIPS(net='alex', spatial=True).cuda().eval()
    rows=[]
    for epoch in EPOCHS:
        for gi, g in enumerate(groups):
            gid=g['id']; hw=tuple(g['image_hw'])
            # The frozen registry predates this server relocation.
            for p in g['input_paths']:
                if not Path(p).exists() and str(p).startswith('/mnt/dataset1/'):
                    q=resolve_dataset_path(p)
                    if q.exists():
                        g['input_paths'][g['input_paths'].index(p)]=str(q)
            ref,meta=hypersim(g,hw)
            clean=np.stack([resize(np.asarray(Image.open(Path(p).with_name('clean.png')).convert('RGB')).astype(np.float32)/255.,hw) for p in g['input_paths']])
            method_data=[]; method_labels=[]; depth_data=[]; pose_paths=[]; cloud_paths=[]
            for key,label in CONFIGS.items():
                folder=OUT/key/epoch/'hypersim'/gid
                if not (folder/'complete.json').exists(): raise FileNotFoundError(folder/'complete.json')
                da=np.load(folder/'da3.npz'); gd=np.load(folder/'gard.npz')
                pred={k:gd[k] for k in gd.files}; method_data.append(pred); method_labels.append(label)
                d=pred['depth']; valid=ref['valid'] & ~ref['union_mask'] & np.isfinite(d) & (d>0)
                scale=float(np.median(ref['depth'][valid]/d[valid])) if valid.any() else 1.
                aligned=d*scale; depth_data.append(aligned)
                dm=np.maximum(aligned/ref['depth'],ref['depth']/np.maximum(aligned,1e-12)); delta=float(np.mean((dm[ref['valid']]<1.25)))
                rgb=np.clip(pred['rgb'],0,1); err=((rgb-clean)**2).mean(-1); ss=[]; ls=[]
                for a,b in zip(rgb,clean):
                    _,s=structural_similarity(a,b,data_range=1.,channel_axis=-1,full=True); ss.append(s.mean(-1))
                    at=torch.from_numpy(a.transpose(2,0,1).copy())[None].cuda()*2-1; bt=torch.from_numpy(b.transpose(2,0,1).copy())[None].cuda()*2-1
                    ls.append(lp(at,bt)[0,0].detach().cpu().numpy())
                ss=np.stack(ss); ls=np.stack(ls); allpix=np.ones_like(ref['valid'],bool)
                msk=allpix
                mse=float(err[msk].mean()); rgb_psnr=float(-10*np.log10(max(mse,1e-12)))
                # Main pose metric: ray-pose head, not the depth-head extrinsics.
                pose=pose_metrics(pred['ray_extrinsics'],ref['extrinsics'])
                keep=np.isfinite(pred['depth_conf']) & (pred['depth_conf']>=np.percentile(pred['depth_conf'][np.isfinite(pred['depth_conf'])],40)) & (pred['depth']>0) & np.isfinite(pred['world_points']).all(-1)
                fit=ref['valid'] & np.isfinite(pred['world_points']).all(-1); sim=fit_sim(pred['world_points'][fit],ref['points'][fit],robust=True); aligned_points=transform(pred['world_points'],sim)
                pp,_=voxel(aligned_points[keep],rgb[keep],.005); qq,_=voxel(ref['points'][ref['valid']],None,.005); cloud=cloud_metrics(pp,qq,[.01,.02,.05])
                rows.append({'epoch':epoch[-3:],'group':gid,'config':label,'depth_absrel':float(np.mean(abs(aligned[ref['valid']]-ref['depth'][ref['valid']])/ref['depth'][ref['valid']])), 'depth_rmse':float(np.sqrt(np.mean((aligned[ref['valid']]-ref['depth'][ref['valid']])**2))),'depth_delta1':delta,'pose_source':'ray_extrinsics','pose_rre_median':pose['rre_median'],'pose_rte_median':pose['rte_median'],'pose_auc20':pose['auc_20'],'cloud_accuracy':cloud.get('accuracy'),'cloud_completeness':cloud.get('completeness'),'cloud_chamfer_l1':cloud.get('chamfer_l1'),'cloud_fscore_002':cloud.get('fscore_0.02'),'rgb_psnr':rgb_psnr,'rgb_ssim':float(ss[msk].mean()),'rgb_lpips':float(np.stack(ls)[msk].mean())})
                pose_paths.append(folder/'pose_ray_comparison.png'); cloud_paths.append(folder/'point_cloud_p50.png')
            base=OUT/'grids'/f'{gid}_epoch{epoch[-3:]}'; base.mkdir(parents=True,exist_ok=True)
            rgb_grid(base/'rgb_grid.jpg',[x['rgb'] for x in method_data],method_labels)
            depth_grid(base/'depth_grid.jpg',depth_data,method_labels)
            copy_grid(base/'pose_grid.jpg',pose_paths,method_labels)
            copy_grid(base/'pointcloud_grid.jpg',cloud_paths,method_labels)
    keys=list(rows[0]);
    with (OUT/'metrics_per_group.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    agg=[]
    metric_keys=[k for k in keys if k not in ('epoch','group','config')]
    for epoch in ('010','020'):
      for config in CONFIGS.values():
        sel=[r for r in rows if r['epoch']==epoch and r['config']==config]; agg.append({'epoch':epoch,'config':config,**{k:float(np.mean([r[k] for r in sel])) for k in metric_keys}})
    with (OUT/'metrics_mean.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(agg[0])); w.writeheader(); w.writerows(agg)
    (OUT/'README.md').write_text('# Three-config E10/E20 visualization comparison\n\nFive groups: '+', '.join(g['id'] for g in groups)+'\n\nGeometry metrics are depth AbsRel/RMSE/delta1, pose median RRE/RTE/AUC20, and point-cloud accuracy/completeness/chamfer/F-score. RGB metrics are decoded input-view RGB against corresponding clean RGB: PSNR/SSIM/AlexNet spatial LPIPS. Grids are under `grids/`; per-group and mean tables are CSV.\n')
    print('WROTE',len(rows),'rows',flush=True)
if __name__=='__main__': main()
