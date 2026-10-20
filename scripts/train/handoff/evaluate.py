"""Offline metrics and images on frozen eval_visualization groups."""
import csv,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from skimage.metrics import structural_similarity
from mvr.handoff.data import read_eval_manifest
from scripts.eval.quant_eval.metrics import pose_metrics,fit_sim,transform,cloud_metrics,write_ply,centers
from mvr.eval_models.rgb_decode import build_rae_da3,decode_rgb_from_feats


def load_batch(g,engine):
    h,w=378,504
    with Image.open(g['input_paths'][0]) as im:
        iw,ih=im.size;s=engine.cfg.resolution/max(iw,ih);w=max(14,round(iw*s/14)*14);h=max(14,round(ih*s/14)*14)
    def images(paths):
        rows=[]
        for path in paths:
            with Image.open(path) as im:a=np.asarray(im.convert('RGB'))
            rows.append(cv2.resize(a,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)/255)
        return torch.tensor(np.stack(rows)).permute(0,3,1,2)[None]
    if len(g['input_paths'])!=4 or len(g['camera_records'])!=4:raise ValueError('Portable eval requires four views and four camera records')
    lq=images(g['input_paths']);hq=images(g['clean_paths']);masks=np.zeros((1,4,h,w),np.float32)
    for key,paths in g.get('mask_paths',{}).items():
        for path in paths:
            with Image.open(path) as im:a=np.asarray(im.convert('L'))>0
            masks[0,int(key)]=np.maximum(masks[0,int(key)],cv2.resize(a.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST))
    flags=torch.tensor([[i in g['clutter_views'] for i in range(4)]])
    return dict(lq_views=lq,hq_views=hq,pixel_masks=torch.tensor(masks),object_pixel_masks=torch.tensor(masks),distractor_views=flags,camera_records=[g['camera_records']])

def panel(path,columns):
    v,h,w=columns[0][1].shape[:3];canvas=Image.new('RGB',(w*len(columns),(h+24)*v),'white');draw=ImageDraw.Draw(canvas)
    for c,(label,arr) in enumerate(columns):
        for i in range(v):
            canvas.paste(Image.fromarray(np.uint8(np.clip(arr[i],0,1)*255)),(c*w,i*(h+24)+24));draw.text((c*w+4,i*(h+24)+4),f'{label} view{i}',fill='black')
    canvas.save(path,quality=92)

def arrays(output):
    from depth_anything_3.utils.geometry import unproject_depth,affine_inverse,as_homogeneous
    depth=output.depth.float();ext=output.extrinsics.float();k=output.intrinsics.float()
    points=unproject_depth(depth[...,None],k,affine_inverse(as_homogeneous(ext)))
    return {key:value[0].cpu().numpy() for key,value in dict(depth=depth,depth_conf=output.depth_conf.float(),extrinsics=ext,intrinsics=k,points=points).items()}

def pose_sim(pred,ref):
    # Orientation-based rotation plus positive scale and translation; works with 3 cameras.
    a=np.mean(ref[:,:3,:3].transpose(0,2,1)@pred[:,:3,:3],axis=0)
    u,_,vt=np.linalg.svd(a);sign=np.ones(3);sign[-1]=np.linalg.det(u@vt);rotation=(u*sign)@vt
    x=centers(pred)@rotation.T;y=centers(ref);xc=x-x.mean(0);yc=y-y.mean(0)
    scale=float(np.sum(xc*yc)/max(np.sum(xc**2),1e-12))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid pose alignment scale')
    return scale,rotation,y.mean(0)-scale*x.mean(0)

@torch.inference_mode()
def evaluate(engine,ema,cfg,out,log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lpips import LPIPS
    device=engine.device;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    groups=read_eval_manifest(cfg.eval_manifest);groups=groups[:cfg.eval_limit] if cfg.eval_limit else groups
    decoder=build_rae_da3(device='cpu');lp=LPIPS(net='alex').to(device).eval().requires_grad_(False)
    summary=[];log.info('EVAL START groups=%d manifest=%s',len(groups),cfg.eval_manifest)
    def decode(output,size):
        # EMA is only used during sampling; offload it during RGB decoding.
        ema.cpu();decoder.mae_decoder.to(device);decoder.adapter.to(device);h,w=size;result=[]
        for v in range(output.feat[0][0].shape[1]):result.append(decode_rgb_from_feats(decoder,[(a[:,v:v+1],b[:,v:v+1]) for a,b in output.feat],h,w)[0,0].permute(1,2,0).numpy())
        decoder.mae_decoder.cpu();decoder.adapter.cpu();ema.to(device);return np.stack(result)
    for g in groups:
        try:
            folder=out/g['id'];folder.mkdir(parents=True,exist_ok=True)
            if g.get('input_sha256'):
                actual=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in g['input_paths']]
                if actual!=g['input_sha256']:raise ValueError('Frozen eval input changed: '+g['id'])
            batch=load_batch(g,engine);ex=engine.extract(batch,supervise=False);latent=engine.sample(ema,ex,cfg.eval_num_steps)
            pred_output=engine.decode_geometry(batch,latent);pred=arrays(pred_output);rgb=decode(pred_output,ex['image_size']);del pred_output,latent
            total=4 if cfg.nvs else 3
            clean_output=engine.encode(batch['hq_views'].to(device));ref={k:v[:total] for k,v in arrays(clean_output).items()};del clean_output
            gt=batch['hq_views'][0,:total].permute(0,2,3,1).numpy();inp=batch['lq_views'][0,:total].permute(0,2,3,1).numpy()
            scores=[]
            for i in range(total):
                p=torch.from_numpy(rgb[i]).permute(2,0,1)[None].to(device);q=torch.from_numpy(gt[i]).permute(2,0,1)[None].to(device)
                scores.append(dict(view=i,slot='novel' if cfg.nvs and i==3 else 'source',psnr=float(-10*np.log10(max(np.mean((rgb[i]-gt[i])**2),1e-12))),ssim=float(structural_similarity(rgb[i],gt[i],channel_axis=-1,data_range=1.)),lpips=float(lp(2*p-1,2*q-1).item())))
                Image.fromarray(np.uint8(rgb[i]*255)).save(folder/f'decoded_view{i}.png')
            gt_c2w,k=engine.cameras.cameras(g['camera_records'][:total],ex['image_size']);reference_pose=np.linalg.inv(gt_c2w)[:,:3]
            pose=pose_metrics(pred['extrinsics'],reference_pose)
            pose_source3=pose_metrics(pred['extrinsics'][:3],reference_pose[:3])
            # Joint-clean DA3 reference is explicitly pseudo-GT, including static assets.
            sim=pose_sim(pred['extrinsics'],ref['extrinsics'])
            scaled=pred['depth']*sim[0];valid=np.isfinite(scaled)&np.isfinite(ref['depth'])&(scaled>0)&(ref['depth']>0)
            ratio=np.maximum(scaled[valid]/ref['depth'][valid],ref['depth'][valid]/scaled[valid])
            depth=dict(absrel=float(np.mean(abs(scaled[valid]-ref['depth'][valid])/ref['depth'][valid])),rmse=float(np.sqrt(np.mean((scaled[valid]-ref['depth'][valid])**2))),delta1=float(np.mean(ratio<1.25)),reference='joint clean DA3 pseudo-GT',alignment='single camera-center Sim3, reused across cutoffs')
            panel(folder/'rgb_comparison.jpg',[('Input',inp),('Decoded',rgb),('Clean RGB',gt)])
            lo,hi=np.percentile(ref['depth'][np.isfinite(ref['depth'])&(ref['depth']>0)],[2,98]);color=lambda a:plt.get_cmap('turbo')(np.clip((a-lo)/max(hi-lo,1e-6),0,1))[...,:3]
            panel(folder/'depth_comparison.jpg',[('Clean DA3 pseudo-GT',color(ref['depth'])),('Predicted aligned depth',color(scaled))])
            fig=plt.figure();ax=fig.add_subplot(111,projection='3d')
            for name,e in [('Prediction',pred['extrinsics']),('GT',reference_pose)]:
                pp=centers(e);pp-=pp.mean(0);pp/=max(np.linalg.norm(pp,axis=-1).mean(),1e-8);ax.plot(*pp.T,'o-',label=name)
            ax.legend();ax.set_title('Camera-center shape, centered/unit radius (no rotation fit)');fig.savefig(folder/'pose_comparison.png');plt.close(fig)
            conf=pred['depth_conf'];basevalid=np.isfinite(conf)&np.isfinite(pred['points']).all(-1)&(pred['depth']>0);cutoffs={}
            world=transform(pred['points'],sim);refgood=np.isfinite(ref['points']).all(-1)&np.isfinite(ref['depth'])&(ref['depth']>0);refcloud=ref['points'][refgood];refcloud=refcloud[::max(1,len(refcloud)//50000)]
            for percentile in [20,40,50]:
                tau=float(np.percentile(conf[np.isfinite(conf)],percentile));keep=basevalid&(conf>=tau);points=world[keep];colors=rgb[keep]
                write_ply(folder/f'decoded_rgb_p{percentile}.ply',points,colors)
                stride=max(1,len(points)//50000);metrics=cloud_metrics(points[::stride],refcloud,[.01,.02,.05])
                source_points=world[:3][keep[:3]];source_ref=ref['points'][:3][refgood[:3]];source_ref=source_ref[::max(1,len(source_ref)//50000)]
                source_metrics=cloud_metrics(source_points[::max(1,len(source_points)//50000)],source_ref,[.01,.02,.05])
                cutoffs[str(percentile)]=dict(cloud_source3=source_metrics,threshold=tau,retained_fraction=float(keep.mean()),cloud=metrics,unit='clean DA3 native gauge; not meters',reference='joint clean DA3 pseudo-GT',subsampling_stride=stride)
                fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='3d');ax.scatter(*points[::stride].T,c=colors[::stride],s=.3);ax.set_title(f'Decoded RGB P{percentile}');fig.savefig(folder/f'cloud_p{percentile}.png',dpi=120);plt.close(fig)
            np.savez_compressed(folder/'prediction.npz',**pred,rgb=rgb);np.savez_compressed(folder/'reference.npz',**ref,rgb=gt)
            source=scores[:3]  # Same observed first3 across baseline and NVS.
            result=dict(id=g['id'],rgb_per_view=scores,rgb_first3={key:float(np.mean([s[key] for s in source])) for key in ['psnr','ssim','lpips']},pose=pose,pose_source3=pose_source3,depth=depth,cutoffs=cutoffs,
                comparison_caveat='Both methods receive identical first3 source images. Only NVS predicts the held-out fourth view.',group=g)
            (folder/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');summary.append(result)
            log.info('EVAL GROUP %s RGB(first3)=%s pose_RRE=%.4f pose_RTE=%.4f depth=%s images=%s',g['id'],json.dumps(result['rgb_first3']),pose['rre_median'],pose['rte_median'],json.dumps(depth),folder/'rgb_comparison.jpg')
            for row in scores:
                log.info('EVAL RGB %s %s view=%d psnr=%.6f ssim=%.6f lpips=%.6f image=%s',g['id'],row['slot'],row['view'],row['psnr'],row['ssim'],row['lpips'],folder/f'decoded_view{row["view"]}.png')
            log.info('EVAL POSE SOURCE3 %s RRE=%.6f RTE=%.6f AUC5=%.6f AUC20=%.6f',g['id'],pose_source3['rre_median'],pose_source3['rte_median'],pose_source3['auc_5'],pose_source3['auc_20'])
            log.info('EVAL ARTIFACTS %s depth=%s pose=%s cloud_P20=%s cloud_P40=%s cloud_P50=%s',g['id'],folder/'depth_comparison.jpg',folder/'pose_comparison.png',folder/'cloud_p20.png',folder/'cloud_p40.png',folder/'cloud_p50.png')
            for level,value in cutoffs.items():log.info('EVAL CLOUD %s P%s %s',g['id'],level,json.dumps(value))
        except Exception as error:
            failure=dict(id=g['id'],status='failed',error=repr(error))
            folder=out/g['id'];folder.mkdir(parents=True,exist_ok=True)
            (folder/'error.json').write_text(json.dumps(failure,indent=2)+'\n')
            summary.append(failure);log.exception('EVAL FAILED group=%s',g['id'])
            if cfg.get('eval_fail_fast',False):raise
            torch.cuda.empty_cache()
    flat=[]
    for r in summary:
        for s in r.get('rgb_per_view',[]):flat.append(dict(group=r['id'],**s))
    if flat:
        with (out/'per_view.csv').open('w') as f:writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'index.html').write_text('<meta charset="utf-8"><h1>Offline evaluation</h1>'+''.join(f'<h2>{r["id"]}</h2>'+''.join(f'<img loading="lazy" style="max-width:100%" src="{r["id"]}/{file}">' for file in ['rgb_comparison.jpg','depth_comparison.jpg','pose_comparison.png','cloud_p20.png','cloud_p40.png','cloud_p50.png']) for r in summary if r.get('status')!='failed'))
    failed=sum(r.get('status')=='failed' for r in summary)
    (out/'status.json').write_text(json.dumps(dict(status='complete' if not failed else 'partial_failure',groups=len(summary),failed=failed),indent=2)+'\n')
    log.info('EVAL COMPLETE groups=%d failed=%d index=%s',len(summary),failed,out/'index.html')
    del decoder,lp;torch.cuda.empty_cache()
