"""Fixed-source, fixed-noise target-camera intervention on RealX3D g00."""
import argparse, gc, hashlib, html, json, math, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src')]
import numpy as np
from PIL import Image,ImageDraw

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def relocated(path):return str(path).replace('/mnt/dataset1/','/mnt/nvme1n1/')
def ext4(e):
    e=np.asarray(e,dtype=float)
    return np.concatenate([e,np.broadcast_to([0.,0.,0.,1.],(*e.shape[:-2],1,4))],-2)
def prepare(a):
    original=json.loads(a.plan.read_text());groups=[]
    for item in original['groups']:
        if not (item['name'].startswith('RealX3D/') and item['name'].endswith('__g00')):continue
        g=item['group'];sm=json.loads(Path(g['source_manifest']).read_text());sg=next(x for x in sm['groups'] if x['id']==g['id'])
        src=np.array(item['source_target_c2w'])[:4];orig=np.array(item['source_target_c2w'])[4];k=np.array(item['source_target_intrinsics'])[4]
        exclusion={relocated(x['path']) for x in sg['paired_clean_frames']}|{item['target_path']}
        candidates=[]
        for idx,f in sg['frames'].items():
            path=relocated(f['path'])
            if '/val/' not in path or path in exclusion or not np.allclose(f['intrinsics'],k):continue
            pose=np.linalg.inv(ext4(f['extrinsics']));candidates.append((idx,f,path,pose))
        def distance(x,y):
            angle=np.arccos(np.clip((np.trace(x[:3,:3].T@y[:3,:3])-1)/2,-1,1))
            baseline=np.median([np.linalg.norm(x[:3,3]-s[:3,3]) for s in src])
            return np.linalg.norm(x[:3,3]-y[:3,3])/max(baseline,1e-6)+angle
        chosen=[];anchors=[orig]
        for _ in range(2):
            c=max(candidates,key=lambda c:min(distance(c[3],p) for p in anchors));chosen.append(c);anchors.append(c[3]);candidates.remove(c)
        cases=[dict(id='original',label='Original target',c2w=orig.tolist(),gt_path=item['target_path'],valid_path=item['target_valid_path'])]
        for n,(idx,f,path,pose) in enumerate(chosen,1):
            cases.append(dict(id=f'clean_pose_{n}',label=f'Clean camera {idx}',c2w=pose.tolist(),gt_path=path,valid_path=relocated(f['valid_path'])))
        angle=np.deg2rad(15);rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        yaw=orig.copy();yaw[:3,:3]=yaw[:3,:3]@rotation
        cases.append(dict(id='yaw_plus15',label='Original + camera-local yaw 15 deg (no GT)',c2w=yaw.tolist(),gt_path=None,valid_path=None))
        for c in cases:
            if c['gt_path']:c['gt_sha256']=sha(c['gt_path'])
        groups.append(dict(item=item,cases=cases))
    if len(groups)!=8:raise ValueError(f'Expected 8 g00 groups, got {len(groups)}')
    a.output.mkdir(parents=True,exist_ok=True)
    dump(a.output/'plan.json',dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_size=a.checkpoint.stat().st_size,
        original_plan=str(a.plan.resolve()),original_plan_sha256=sha(a.plan),V=4,M=1,seed=42,
        num_grid_points=50,velocity_calls=49,weights='EMA',epoch=10,
        intervention='Only target camera extrinsics change. Source, intrinsics, role mask, noise and sampler fixed.',groups=groups))
    print('Prepared',len(groups),'groups /',sum(len(g['cases']) for g in groups),'cases',flush=True)

def rgb_save(path,rgb):Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(path)
def render_group(folder,row):
    records=[]
    for c in row['cases']:
        p=folder/c['id']/'metrics.json'
        if p.exists():records.append((c,json.loads(p.read_text())))
    if not records:return
    first=Image.open(folder/records[0][0]['id']/'novel.png');w,h=first.size;first.close()
    panel=Image.new('RGB',(w*3,(h+50)*len(records)),'white');draw=ImageDraw.Draw(panel)
    for i,(case,record) in enumerate(records):
        top=i*(h+50);dest=folder/case['id']
        for col,name in enumerate(['reference.png','novel.png','difference_vs_original.png']):
            if (dest/name).exists():
                with Image.open(dest/name) as im:panel.paste(im,(col*w,top+50))
            draw.text((col*w+4,top+4),['Requested clean RGB (gray = no GT)','Generated novel RGB','Change vs original (absolute x4)'][col],fill='black')
        text=case['label']
        if 'rgb' in record:text+=f" | PSNR={record['rgb']['psnr']:.2f}"
        text+=f" | pose RRE={record['camera_head']['target_pair_rre_median_deg']:.2f} deg"
        draw.text((4,top+24),text,fill='black')
    panel.save(folder/'comparison.jpg',quality=94)
    images=[]
    for c,_ in records:
        with Image.open(folder/c['id']/'novel.png') as im:images.append(im.copy())
    images[0].save(folder/'novel_sweep.gif',save_all=True,append_images=images[1:],duration=900,loop=0)
    sweep=Image.new('RGB',(w*len(records),(h+48)*2),'white');sd=ImageDraw.Draw(sweep);paired=[]
    for i,(case,record) in enumerate(records):
        with Image.open(folder/case['id']/'reference.png') as ref, Image.open(folder/case['id']/'novel.png') as gen:
            sweep.paste(ref,(i*w,48));sweep.paste(gen,(i*w,h+96))
            sd.text((i*w+4,4),case['id']+' | Clean GT',fill='black')
            if 'nearest_source_viewing_angle_deg' in case:sd.text((i*w+4,24),f"Nearest source angle {case['nearest_source_viewing_angle_deg']:.1f} deg",fill='black')
            sd.text((i*w+4,h+52),'Generated | '+case['id'],fill='black')
            pair=Image.new('RGB',(w*2,h+48),'white');pair.paste(ref,(0,48));pair.paste(gen,(w,48));pd=ImageDraw.Draw(pair)
            pd.text((4,4),'Clean GT | '+case['id'],fill='black');pd.text((w+4,4),'Generated | same source and noise',fill='black');paired.append(pair)
    sweep.save(folder/'pose_sweep_grid.jpg',quality=94)
    paired[0].save(folder/'matched_sweep.gif',save_all=True,append_images=paired[1:],duration=1200,loop=0)


def aggregate_unlocked(a):
    plan=json.loads((a.output/'plan.json').read_text());sections=[];allrecords=[]
    for row in plan['groups']:
        name=row['item']['name'];folder=a.output/name;render_group(folder,row)
        records=[]
        for c in row['cases']:
            p=folder/c['id']/'metrics.json'
            if p.exists():records.append(dict(group=name,**json.loads(p.read_text())))
        allrecords.extend(records)
        section=f'<h2>{html.escape(name)} ({len(records)}/{len(row["cases"])})</h2>'
        for filename in ['source_views.jpg','pose_sweep_grid.jpg','matched_sweep.gif','comparison.jpg','camera_response.png']:
            if (folder/filename).exists():section+=f'<p>{filename}</p><img loading="lazy" src="{name}/{filename}">'
        sections.append(section)
    summary=dict(completed_cases=len(allrecords),total_cases=sum(len(r['cases']) for r in plan['groups']),groups=len(plan['groups']),records=allrecords)
    dump(a.output/'summary.json',summary)
    (a.output/'index.html').write_text('<meta charset="utf-8"><style>body{font-family:sans-serif}img{max-width:100%}</style><h1>NVS epoch 10: RealX3D g00 target camera sweep</h1><p>EMA. Source V4 and noise seed 42 fixed within each group. Only target extrinsics change; intrinsics and sampling fixed. '+html.escape(plan.get('description','3 photographed cameras + 15 degree camera-local yaw (no GT).'))+' RGB difference alone does not establish pose control. Camera plots anchor predicted source cameras to the source-only DA3 frame. Relative pose errors use source-target pairs.</p>'+''.join(sections))
    print('REPORT',len(allrecords),'/',summary['total_cases'],flush=True)

def aggregate(a):
    import fcntl
    with (a.output/'report.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        aggregate_unlocked(a)

def worker(a):
    import torch
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from mvr.nvs_training import NVSGARD,target_plucker_rays
    from mvr.nvs_camera import align_target_camera
    from mvr.eval_models.da3_restore import MultiViewDA3RestoreModel
    from mvr.eval_models.rgb_decode import build_rae_da3,decode_rgb_from_feats
    from scripts.eval.quant_eval.metrics import pose_metrics
    from skimage.metrics import structural_similarity
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=json.loads((a.output/'plan.json').read_text());torch.set_num_threads(4);torch.manual_seed(42)
    ck=torch.load(plan['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    if int(ck['epoch'])!=10:raise ValueError('Expected epoch10')
    cfg=OmegaConf.create(ck['config']);den=NVSGARD(**OmegaConf.to_container(cfg.model,resolve=True)).eval().requires_grad_(False)
    den.load_state_dict(ck['ema'],strict=True);den.cuda();del ck
    model=MultiViewDA3RestoreModel(pretrained=cfg.encoder,device='cuda');decoder=build_rae_da3(device='cpu')
    mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
    restorecfg=SimpleNamespace(restore_feat_layers=[17],concat_feat=False)
    def decode(output,h,w):
        decoder.mae_decoder.cuda();decoder.adapter.cuda();result=[]
        for i in range(5):result.append(decode_rgb_from_feats(decoder,[(x[:,i:i+1],y[:,i:i+1]) for x,y in output.feat],h,w)[0,0].permute(1,2,0).numpy())
        decoder.mae_decoder.cpu();decoder.adapter.cpu();return np.stack(result)
    def pose_scores(pred,requested):
        m=pose_metrics(pred,np.linalg.inv(requested)[:,:3]);pairs=[p for p in m['pairs'] if p['j']==4]
        return dict(target_pair_rre_median_deg=float(np.median([p['rre'] for p in pairs])),
                    target_pair_rte_median_deg=float(np.median([p['rte'] for p in pairs])))
    with torch.inference_mode():
      for index,row in enumerate(plan['groups']):
        if index%2!=a.shard:continue
        item=row['item'];g=item['group'];folder=a.output/item['name'];folder.mkdir(parents=True,exist_ok=True)
        if all((folder/c['id']/'complete.json').exists() for c in row['cases']):continue
        assert [sha(p) for p in g['input_paths']]==g['input_sha256']
        norm=InputProcessor()(g['input_paths'],process_res=504,process_res_method='upper_bound_resize',sequential=True)[0]
        h,w=norm.shape[-2:];image=norm[None].cuda();source_rgb=(norm*std+mean).clamp(0,1).permute(0,2,3,1).numpy()
        rgb_save(folder/'source_views.jpg',np.concatenate(list(source_rgb),axis=1))
        source,_=model.model.forward(image,export_feat_layers=[17],ref_view_strategy='first');model._check_ref(source)
        cond=source.aux['feat_layer_17'].float().clone();sourcepose=source.extrinsics[0].float().cpu().numpy();del source
        sourcegt=np.array(item['source_target_c2w'])[:4];k=np.array(item['source_target_intrinsics'])[4:]
        condition=torch.cat([cond,torch.zeros_like(cond[:,:1])],1)
        noise=torch.randn(condition.shape,generator=torch.Generator(device='cuda').manual_seed(42),device='cuda')
        start=noise.clone();start[:,:4]=cond+.3*noise[:,:4]
        mask=torch.tensor([[False]*4+[True]],device='cuda');placeholder=torch.cat([image,torch.zeros_like(image[:,:1])],1)
        initial_hash=hashlib.sha256(start.cpu().numpy().tobytes()).hexdigest()
        dump(folder/'fixed_inputs.json',dict(source_paths=g['input_paths'],source_sha256=g['input_sha256'],
             source_only_extrinsics=sourcepose.tolist(),initial_state_sha256=initial_hash,seed=42,target_intrinsics=k.tolist()))
        original_rgb=None;original_feature=None;plot_rows=[]
        for case in row['cases']:
          dest=folder/case['id'];dest.mkdir(exist_ok=True)
          if (dest/'complete.json').exists():
            if case['id']=='original':
                assert json.loads((dest/'metrics.json').read_text())['initial_state_sha256']==initial_hash
                original_rgb=np.load(dest/'outputs.npz')['rgb'];original_feature=torch.from_numpy(np.load(dest/'outputs.npz')['novel_latent']).cuda().float()
            saved=np.load(dest/'poses.npz');plot_rows.append((case,saved['requested_da3_c2w'],saved['camera_head_target_da3_c2w'],saved['ray_target_da3_c2w']))
            continue
          started=time.time();print('START',item['name'],case['id'],flush=True)
          requested=np.concatenate([sourcegt,np.array(case['c2w'])[None]],0)
          target,scale,residual=align_target_camera(requested,sourcepose)
          rays=target_plucker_rays(torch.tensor(target,dtype=torch.float32,device='cuda')[None],torch.tensor(k,dtype=torch.float32,device='cuda')[None],(h,w))
          full=torch.cat([torch.zeros(1,4,rays.shape[2],6,device='cuda'),rays],1)
          state=start.clone();shift=math.sqrt(1492992/4096);times=1-torch.linspace(0,1,50,device='cuda');times=shift*times/(1+(shift-1)*times)
          for ti in range(49):
            with torch.autocast('cuda',dtype=torch.bfloat16):velocity=den(state+condition,times[ti:ti+1],(h,w),nvs_rays=full,nvs_view_mask=mask)
            state=state+(times[ti+1]-times[ti])*velocity.float()
          torch.manual_seed(42)
          output,_=model.model.forward(placeholder,export_feat_layers=[],ref_view_strategy='first',mode='val',mvrm_cfg=restorecfg,mvrm_result={('restored_latent',17):state},export_rgb_feat_layers=True)
          model._check_ref(output);rgb=decode(output,h,w);head=output.extrinsics[0].float().cpu().numpy();depth=output.depth[0,4].float().cpu().numpy();del output
          torch.manual_seed(42)
          ray_output,_=model.model.forward(placeholder,export_feat_layers=[],ref_view_strategy='first',mode='val',mvrm_cfg=restorecfg,mvrm_result={('restored_latent',17):state},use_ray_pose=True)
          ray=ray_output.extrinsics[0].float().cpu().numpy();del ray_output
          head_target,_,_=align_target_camera(np.linalg.inv(ext4(head)),sourcepose)
          ray_target,_,_=align_target_camera(np.linalg.inv(ext4(ray)),sourcepose)
          novel_feature=state[:,4:].float()
          if case['id']=='original':original_rgb=rgb.copy();original_feature=novel_feature.clone()
          record=dict(case=case['id'],label=case['label'],seconds=time.time()-started,
             initial_state_sha256=initial_hash,camera_head=pose_scores(head,requested),ray=pose_scores(ray,requested),
             novel_rgb_mae_vs_original=float(np.abs(rgb[4]-original_rgb[4]).mean()),
             source_rgb_mae_vs_original=float(np.abs(rgb[:4]-original_rgb[:4]).mean()),
             novel_feature_relative_change_vs_original=float(torch.linalg.vector_norm(novel_feature-original_feature)/torch.linalg.vector_norm(original_feature)),
             camera_alignment_scale=scale,source_camera_center_residual=residual)
          rgb_save(dest/'novel.png',rgb[4]);rgb_save(dest/'source_outputs.jpg',np.concatenate(list(rgb[:4]),axis=1))
          rgb_save(dest/'difference_vs_original.png',np.abs(rgb[4]-original_rgb[4])*4)
          if case['gt_path']:
            assert sha(case['gt_path'])==case['gt_sha256']
            targetnorm=InputProcessor()([case['gt_path']],process_res=504,process_res_method='upper_bound_resize',sequential=True)[0]
            gt=(targetnorm*std+mean).clamp(0,1)[0].permute(1,2,0).numpy();valid=np.load(case['valid_path']).astype(bool)
            assert gt.shape==rgb[4].shape and valid.shape==(h,w)
            mse=float(((rgb[4]-gt)**2)[valid].mean());_,ssim_map=structural_similarity(gt,rgb[4],channel_axis=-1,data_range=1.,full=True)
            record['rgb']=dict(psnr=float(-10*np.log10(max(mse,1e-12))),ssim=float(ssim_map.mean(-1)[valid].mean()),valid_pixels=int(valid.sum()))
            rgb_save(dest/'reference.png',gt)
          else:rgb_save(dest/'reference.png',np.full((h,w,3),.5))
          np.savez_compressed(dest/'outputs.npz',rgb=rgb,novel_latent=novel_feature.cpu().numpy(),target_depth=depth)
          np.savez_compressed(dest/'poses.npz',camera_head=head,ray=ray,requested_sfm_c2w=requested,
              requested_da3_c2w=target[0],camera_head_target_da3_c2w=head_target[0],ray_target_da3_c2w=ray_target[0],target_intrinsics=k[0])
          dump(dest/'metrics.json',record);dump(dest/'complete.json',dict(epoch=10,weights='EMA',seed=42,seconds=record['seconds']))
          plot_rows.append((case,target[0],head_target[0],ray_target[0]));render_group(folder,row)
          print('DONE',item['name'],case['id'],json.dumps(record),flush=True)
          del state,velocity,rays,full,novel_feature,rgb;gc.collect();torch.cuda.empty_cache()
        fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
        from scripts.eval.quant_eval.metrics import centers
        points=centers(sourcepose);ax.scatter(*points.T,c='gray',label='Fixed source DA3 centers')
        for ci,(case,req,head,ray) in enumerate(plot_rows):
            color=f'C{ci}';ax.scatter(*req[:3,3],marker='*',s=100,c=color,label=case['id']+' requested')
            ax.scatter(*head[:3,3],marker='o',c=color);ax.scatter(*ray[:3,3],marker='x',c=color)
            for cam,style in [(req,'-'),(head,'--'),(ray,':')]:
                p=cam[:3,3];end=p+cam[:3,2]*.15;ax.plot(*np.stack([p,end]).T,style,c=color)
        ax.legend(fontsize=7);ax.set_title('Requested * / camera head o / ray x\nEach prediction anchored using its source cameras; arrows = viewing direction')
        fig.savefig(folder/'camera_response.png',dpi=140);plt.close(fig)
        dump(folder/'complete.json',dict(cases=len(row['cases']),epoch=10,weights='EMA'))
        del cond,condition,start,noise,image,placeholder,original_feature,original_rgb;gc.collect();torch.cuda.empty_cache()
        aggregate(a)
    dump(a.output/f'worker{a.shard}_complete.json',dict(complete=True));print('WORKER COMPLETE',a.shard,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','report']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='worker':worker(a)
    else:aggregate(a)
if __name__=='__main__':main()
