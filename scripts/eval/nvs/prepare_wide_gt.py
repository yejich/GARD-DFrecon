"""Select wide photographed clean cameras from full RealX3D captures."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.eval.nvs.camera_sweep import dump,sha,ext4

def angle(a,b):return float(np.degrees(np.arccos(np.clip(np.dot(a[:3,2],b[:3,2]),-1,1))))
def main():
    previous=ROOT/'result_eval/20261012_nvs_epoch010_realx3d_g00_camera_sweep'
    out=ROOT/'result_eval/20261012_nvs_epoch010_realx3d_g00_wide_gt';out.mkdir(exist_ok=True)
    plan=json.loads((previous/'plan.json').read_text())
    spec=importlib.util.spec_from_file_location('realx_preprocessor',ROOT/'scripts/eval/realx3d_multigroup_20261002/prepare.py')
    process=importlib.util.module_from_spec(spec);spec.loader.exec_module(process);process.OUT=out;process.cache={}
    data=Path('/mnt/nvme1n1/RealX3D/data_4/dynamic')
    for row in plan['groups']:
        item=row['item'];src=np.array(item['source_target_c2w'])[:4];original=row['cases'][0]
        source_manifest=json.loads(Path(item['group']['source_manifest']).read_text())
        group=next(g for g in source_manifest['groups'] if g['id']==item['group']['id'])
        excluded={Path(f['source']).name for f in group['paired_clean_frames']}
        scene=next(p for p in data.iterdir() if p.name.lower()==group['scene'].lower())
        candidates=[]
        for split in ['val','test']:
            d=json.loads((scene/f'transforms_{split}.json').read_text())
            for frame in d['frames']:
                raw=scene/frame['file_path']
                if (split=='val' and raw.name in excluded) or str(raw)==group['frames']['4']['source'].replace('/mnt/dataset1/','/mnt/nvme1n1/'):continue
                pose=np.array(frame['transform_matrix'])@np.diag([1.,-1.,-1.,1.])
                cam={k:v for k,v in d.items() if k!='frames'};cam.update(frame)
                cam={**{k:cam.get(k,0) for k in ['w','h','fl_x','fl_y','cx','cy','k1','k2','k3','k4','p1','p2','is_fisheye']},'extrinsics':np.linalg.inv(pose)[:3].tolist()}
                candidates.append(dict(raw=raw,pose=pose,cam=cam,nearest=min(angle(pose,s) for s in src)))
        maximum=max(c['nearest'] for c in candidates);selected=[]
        for fraction in [1.,.65,.35]:
            pool=[c for c in candidates if all(c['raw']!=s['raw'] for s in selected) and all(angle(c['pose'],s['pose'])>=7 for s in selected)]
            if not pool:pool=[c for c in candidates if all(c['raw']!=s['raw'] for s in selected)]
            choice=min(pool,key=lambda c:abs(c['nearest']-maximum*fraction));selected.append(choice)
        selected.sort(key=lambda c:c['nearest'])
        cases=[original]
        for i,c in enumerate(selected,1):
            f=process.process(str(c['raw']),c['cam'])
            if not np.allclose(f['intrinsics'],item['source_target_intrinsics'][4],atol=1e-6):raise ValueError('K changed')
            cases.append(dict(id=f'wide_gt_{i}',label=f"GT {c['raw'].parent.name}/{c['raw'].name} | nearest source angle {c['nearest']:.1f} deg",
                c2w=c['pose'].tolist(),gt_path=f['path'],valid_path=f['valid_path'],gt_sha256=f['sha256'],
                raw_gt_path=str(c['raw']),nearest_source_viewing_angle_deg=c['nearest']))
        row['cases']=cases
        row['selection']=dict(clean_candidates=len(candidates),full_capture_search=True,maximum_nearest_source_angle_deg=maximum)
        print(item['name'],[(c['id'],round(c.get('nearest_source_viewing_angle_deg',0),1)) for c in cases],flush=True)
    plan['description']='All targets have photographed clean GT. Search full val+test captures; select medium/wide/widest viewing angles relative to fixed sources. No synthetic pose.'
    plan['previous_sweep']=str(previous);plan['intervention']='Target extrinsics only; same source V4, fixed K, seed42 and epoch010 EMA.'
    dump(out/'plan.json',plan)
    # Reuse the unchanged original camera result; verify fixed-state hash in worker.
    import shutil
    for row in plan['groups']:
        dest=out/row['item']['name'];dest.mkdir(parents=True,exist_ok=True)
        shutil.copytree(previous/row['item']['name']/'original',dest/'original',dirs_exist_ok=True)
    print('Prepared',out,flush=True)

if __name__=='__main__':main()
