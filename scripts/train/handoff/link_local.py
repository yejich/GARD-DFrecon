"""Create a recipient-shaped view of producer-local data without copying images."""
import argparse,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--original',required=True);p.add_argument('--negative',nargs='+',required=True);p.add_argument('--camera-root',required=True);p.add_argument('--camera-parameters',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
def link(source,dest):
 source=Path(source).resolve()
 if not source.exists():raise FileNotFoundError(source)
 if dest.exists() or dest.is_symlink():
  if dest.resolve()!=source:raise FileExistsError(dest)
 else:dest.symlink_to(source,target_is_directory=source.is_dir())
(out/'manifests').mkdir(exist_ok=True)
for name in ['original_full.json','negative_full.json','scene_split_policy.json']:
 shutil.copy2(Path(a.release)/'manifests'/name,out/'manifests'/name)
link(a.original,out/'hypersim_pairs');(out/'hypersim_negative').mkdir(exist_ok=True)
for root in a.negative:
 for collection in Path(root).glob('scenes*'):
  if collection.is_dir():link(collection,out/'hypersim_negative'/collection.name)
(out/'camera_reference').mkdir(exist_ok=True);link(a.camera_root,out/'camera_reference/hypersim');link(a.camera_parameters,out/'camera_reference/metadata_camera_parameters.csv')
print('Local data view:',out)
