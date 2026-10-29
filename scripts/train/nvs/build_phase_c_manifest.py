"""Build a compact scene manifest for Phase-C clean NVS pretraining.

The manifest stores scene/trajectory groups, image paths, c2w poses and native
intrinsics.  The training loader samples four views from one group and splits
them into N=1..3 source views plus M=4-N target views.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import h5py
import numpy as np


def split_for(name, val_fraction=0.05):
    value=int(hashlib.sha1(name.encode()).hexdigest()[:8],16)/2**32
    return 'val' if value < val_fraction else 'train'


def hypersim(root, camera_csv, max_scenes=0):
    root=Path(root); rows={r['scene_name']:r for r in csv.DictReader(open(camera_csv))}; out=[]
    seen=0
    for scene_dir in sorted((root/'data').glob('ai_*')):
        if max_scenes and seen>=max_scenes: break
        if not (scene_dir/'_detail').is_dir() or scene_dir.name not in rows: continue
        for cam_dir in sorted((scene_dir/'_detail').glob('cam_*')):
            pos=cam_dir/'camera_keyframe_positions.hdf5'; ori=cam_dir/'camera_keyframe_orientations.hdf5'; ids=cam_dir/'camera_keyframe_frame_indices.hdf5'
            image_dir=scene_dir/'images'/f'scene_{cam_dir.name}_final_hdf5'
            if not (pos.is_file() and ori.is_file() and ids.is_file() and image_dir.is_dir()): continue
            with h5py.File(pos) as f: centers=f['dataset'][:]
            with h5py.File(ori) as f: rotations=f['dataset'][:]
            with h5py.File(ids) as f: frame_ids=f['dataset'][:].reshape(-1).astype(int)
            scale=float(rows[scene_dir.name]['settings_units_info_meters_scale'])
            views=[]
            for frame,center,rot in zip(frame_ids,centers,rotations):
                path=image_dir/f'frame.{frame:04d}.color.hdf5'
                if not path.is_file(): continue
                c2w=np.eye(4,dtype=np.float32); c2w[:3,:3]=rot@np.diag([1.,-1.,-1.]); c2w[:3,3]=center*scale
                with h5py.File(path) as f: h,w=f['dataset'].shape[:2]
                k=np.eye(3,dtype=np.float32); k[0,0]=float(rows[scene_dir.name]['M_proj_00'])*w/2; k[1,1]=float(rows[scene_dir.name]['M_proj_11'])*h/2
                k[0,2]=(1-float(rows[scene_dir.name]['M_proj_02']))*w/2-.5; k[1,2]=(1+float(rows[scene_dir.name]['M_proj_12']))*h/2-.5
                views.append(dict(path=str(path),frame=int(frame),c2w=c2w.tolist(),K=k.tolist(),size=[int(h),int(w)]))
            if len(views)>=4:
                out.append(dict(dataset='hypersim',scene=scene_dir.name+'/'+cam_dir.name,split=split_for(scene_dir.name),views=views)); seen+=1
                break
    return out


def tartanair(root, max_scenes=0):
    root=Path(root); out=[]
    for seq in sorted((root/'data').glob('*/Easy/*')):
        if max_scenes and len(out)>=max_scenes: break
        images=sorted((seq/'image_left').glob('*_left.png')); pose_path=seq/'pose_left.txt'
        if len(images)<4 or not pose_path.is_file(): continue
        poses=np.loadtxt(pose_path).astype(np.float32)
        if poses.ndim!=2 or poses.shape[1]!=7: continue
        views=[]
        for p in images:
            frame=int(p.name.split('_')[0])
            if frame>=len(poses): continue
            x,y,z,qx,qy,qz,qw=poses[frame]; n=np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw); qx,qy,qz,qw=[v/n for v in (qx,qy,qz,qw)]
            r=np.array([[1-2*qy*qy-2*qz*qz,2*qx*qy-2*qz*qw,2*qx*qz+2*qy*qw],[2*qx*qy+2*qz*qw,1-2*qx*qx-2*qz*qz,2*qy*qz-2*qx*qw],[2*qx*qz-2*qy*qw,2*qy*qz+2*qx*qw,1-2*qx*qx-2*qy*qy]],np.float32)
            c2w=np.eye(4,dtype=np.float32); c2w[:3,:3]=r; c2w[:3,3]=[x,y,z]
            k=np.array([[320.,0,320.],[0,320.,240.],[0,0,1]],np.float32)
            views.append(dict(path=str(p),frame=frame,c2w=c2w.tolist(),K=k.tolist(),size=[480,640]))
        if len(views)>=4: out.append(dict(dataset='tartanair',scene=str(seq.relative_to(root/'data')),split=split_for(str(seq)),views=views))
    return out


def dl3dv(root, max_scenes=0):
    root=Path(root); out=[]
    for scene_dir in sorted(root.glob('*/*')):
        if max_scenes and len(out)>=max_scenes: break
        tf=scene_dir/'transforms.json'
        if not tf.is_file(): continue
        meta=json.loads(tf.read_text()); frames=[]
        for row in meta.get('frames',[]):
            rel=row.get('file_path',''); p=scene_dir/rel
            if not p.is_file():
                for repl in ('images_8','images_4','images_2','images_1'):
                    q=scene_dir/'images_8'/Path(rel).name
                    if q.is_file(): p=q; break
            if not p.is_file(): continue
            c2w=np.asarray(row['transform_matrix'],np.float32); h=int(meta['h']); w=int(meta['w'])
            k=np.array([[meta['fl_x'],0,meta['cx']],[0,meta['fl_y'],meta['cy']],[0,0,1]],np.float32)
            frames.append(dict(path=str(p),frame=len(frames),c2w=c2w.tolist(),K=k.tolist(),size=[h,w]))
        if len(frames)>=4: out.append(dict(dataset='dl3dv',scene=str(scene_dir.relative_to(root)),split=split_for(str(scene_dir.relative_to(root))),views=frames))
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hypersim-root',required=True); ap.add_argument('--hypersim-camera-csv',required=True); ap.add_argument('--tartanair-root',required=True); ap.add_argument('--dl3dv-root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--max-scenes',type=int,default=0); a=ap.parse_args()
    groups=hypersim(a.hypersim_root,a.hypersim_camera_csv,a.max_scenes)+tartanair(a.tartanair_root,a.max_scenes)+dl3dv(a.dl3dv_root,a.max_scenes)
    payload=dict(schema='phase_c_nvs_v1',total_views=4,source_counts=[1,2,3],groups=groups)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(payload)+'\n')
    from collections import Counter
    print(json.dumps(dict(groups=len(groups),datasets=Counter(g['dataset'] for g in groups),splits=Counter(g['split'] for g in groups))))

if __name__=='__main__': main()
