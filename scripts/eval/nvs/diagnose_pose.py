"""Fixed-seed camera intervention, trained V count and camera/ray decoder audit."""
import sys,json,time,gc,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'RAE/src'),str(ROOT/'Depth-Anything-3/src')]
import numpy as np,torch
from types import SimpleNamespace
from omegaconf import OmegaConf
from mvr.nvs_training import NVSGARD,target_plucker_rays
from mvr.nvs_camera import align_target_camera
from mvr.eval_models.da3_restore import MultiViewDA3RestoreModel
from depth_anything_3.utils.io.input_processor import InputProcessor
from scripts.eval.quant_eval.metrics import pose_metrics
from scripts.eval.nvs.metrics import safe
out=ROOT/'result_eval/20261008_nvs_epoch003_v4_m1';dest=out/'pose_diagnosis';dest.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--latest',action='store_true');args=ap.parse_args();filename='latest_interventions.json' if args.latest else 'interventions.json'
item=json.loads((out/'eval_plan.json').read_text())['groups'][0];g=item['group'];torch.set_num_threads(3);torch.manual_seed(42)
ck=torch.load(ROOT/'result_train/20261007_nvs_v3_m1_211817'/('latest.pt' if args.latest else 'epoch_003.pt'),map_location='cpu',mmap=True,weights_only=False);checkpoint_epoch=ck['epoch'];checkpoint_step=ck['step'];model=NVSGARD(**ck['config']['model']).eval().requires_grad_(False);model.load_state_dict(ck['ema'],strict=True);model.cuda();del ck
encoder=MultiViewDA3RestoreModel(device='cuda');cfg=SimpleNamespace(restore_feat_layers=[17],concat_feat=False);extractcfg=SimpleNamespace(extract_feat_layers=[17],concat_feat=False,break_and_return_feat=True)
reports=[]
@torch.inference_mode()
def run():
 for v in [4,3]:
  paths=g['input_paths'][:v];norm=InputProcessor()(paths,process_res=504,process_res_method='upper_bound_resize',sequential=True)[0];h,w=norm.shape[-2:];image=norm[None].cuda();output,_=encoder.model.forward(image,export_feat_layers=[17],ref_view_strategy='first');cond=output.aux['feat_layer_17'].float().clone();sourcepose=output.extrinsics[0].float().cpu().numpy();del output
  idx=list(range(v))+[4];gt=np.array(item['source_target_c2w'])[idx];k=np.array(item['source_target_intrinsics'])[4:];target,scale,residual=align_target_camera(gt,sourcepose);reference=np.linalg.inv(gt)[:,:3]
  clean=InputProcessor()(item['clean_source_paths'][:v]+[item['target_path']],process_res=504,process_res_method='upper_bound_resize',sequential=True)[0][None].cuda();_,feat,_=encoder.model.forward(clean,export_feat_layers=[],ref_view_strategy='first',mode='train',mvrm_cfg=extractcfg);teacher=feat[('extract_feat',17)].float().clone();del feat
  placeholder=torch.cat([image,torch.zeros_like(image[:,:1])],1);condition=torch.cat([cond,torch.zeros_like(cond[:,:1])],1);generator=torch.Generator(device='cuda').manual_seed(42);noise=torch.randn(condition.shape,generator=generator,device='cuda');start=noise.clone();start[:,:v]=cond+.3*noise[:,:v];baseline=None
  cases=['actual','zero_rays','yaw90'] if v==4 and not args.latest else ['actual']
  for case in cases:
   print('START',v,case,flush=True);pose=target.copy();gtrequested=gt.copy()
   if case=='yaw90':
    rot=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]);pose[:,:3,:3]=pose[:,:3,:3]@rot;gtrequested[-1,:3,:3]=gtrequested[-1,:3,:3]@rot
   rays=target_plucker_rays(torch.tensor(pose,dtype=torch.float32,device='cuda')[None],torch.tensor(k,dtype=torch.float32,device='cuda')[None],(h,w))
   if case=='zero_rays':rays.zero_()
   full=torch.cat([torch.zeros(1,v,rays.shape[2],6,device='cuda'),rays],1);mask=torch.tensor([[False]*v+[True]],device='cuda');state=start.clone();shift=np.sqrt(1492992/4096);ts=1-torch.linspace(0,1,50,device='cuda');ts=shift*ts/(1+(shift-1)*ts)
   # Compare condition magnitude to source encoder token embeddings at t1.
   patchembed=model.s_embedder((start+condition)[:,:,1:].reshape(v+1,h//14,w//14,1536).permute(0,3,1,2));cameraembed=model.nvs_camera_embedder(torch.cat([rays,torch.ones_like(rays[...,:1])],-1));condition_ratio=float(cameraembed.square().mean().sqrt()/patchembed.square().mean().sqrt())
   for i in range(49):
    with torch.autocast('cuda',dtype=torch.bfloat16):velocity=model(state+condition,ts[i:i+1],(h,w),nvs_rays=full,nvs_view_mask=mask)
    state+=(ts[i+1]-ts[i])*velocity.float()
   m=state[:,v:,1:];tm=teacher[:,v:,1:];cls=state[:,v:,:1];tcls=teacher[:,v:,:1];record=dict(V=v,case=case,M_patch_cosine_to_teacher=float(torch.nn.functional.cosine_similarity(m,tm,dim=-1).mean()),M_patch_relative_L2=float(torch.linalg.vector_norm(m-tm)/torch.linalg.vector_norm(tm)),M_CLS_cosine_to_teacher=float(torch.nn.functional.cosine_similarity(cls,tcls,dim=-1).mean()),M_CLS_relative_L2=float(torch.linalg.vector_norm(cls-tcls)/torch.linalg.vector_norm(tcls)),camera_condition_to_patch_embed_RMS_ratio=condition_ratio)
   if case=='actual':baseline=state.clone()
   else:record['M_relative_feature_change_vs_actual']=float(torch.linalg.vector_norm(m-baseline[:,v:,1:])/torch.linalg.vector_norm(baseline[:,v:,1:]))
   for kind in ['camera_head','ray']:
    torch.manual_seed(42);prediction,_=encoder.model.forward(placeholder,export_feat_layers=[],ref_view_strategy='first',mode='val',mvrm_cfg=cfg,mvrm_result={('restored_latent',17):state},use_ray_pose=kind=='ray');poses=prediction.extrinsics[0].float().cpu().numpy();details=pose_metrics(poses,reference);pairs=[x for x in details['pairs'] if x['j']==v];req=pose_metrics(poses,np.linalg.inv(gtrequested)[:,:3]);requested=[x for x in req['pairs'] if x['j']==v];record[kind]=dict(M_rre_to_original=float(np.median([x['rre'] for x in pairs])),M_rte_to_original=float(np.median([x['rte'] for x in pairs])),M_rre_to_requested=float(np.median([x['rre'] for x in requested])))
    np.save(dest/f'{"latest_" if args.latest else ""}V{v}_{case}_{kind}_poses.npy',poses);del prediction
   reports.append(record);(dest/filename).write_text(json.dumps(safe(dict(group=g['id'],epoch=checkpoint_epoch,step=checkpoint_step,seed=42,reports=reports)),indent=2)+'\n');print('DONE',record,flush=True);del state,velocity;gc.collect();torch.cuda.empty_cache()
  # Oracle L17 camera/ray check: does the teacher recover the target pose?
  for kind in ['camera_head','ray']:
   torch.manual_seed(42);prediction,_=encoder.model.forward(placeholder,export_feat_layers=[],ref_view_strategy='first',mode='val',mvrm_cfg=cfg,mvrm_result={('restored_latent',17):teacher},use_ray_pose=kind=='ray');poses=prediction.extrinsics[0].float().cpu().numpy();details=pose_metrics(poses,reference);pairs=[x for x in details['pairs'] if x['j']==v];reports.append(dict(V=v,case='teacher_oracle',decoder=kind,M_rre=float(np.median([x['rre'] for x in pairs])),M_rte=float(np.median([x['rte'] for x in pairs]))));del prediction
  (dest/filename).write_text(json.dumps(safe(dict(group=g['id'],epoch=checkpoint_epoch,step=checkpoint_step,seed=42,reports=reports)),indent=2)+'\n');del baseline,cond,teacher,image,placeholder,condition,noise,start,clean;gc.collect();torch.cuda.empty_cache()
run();print('COMPLETE',flush=True)
