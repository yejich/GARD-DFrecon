"""Download the pinned handoff, verify archive SHA256, and extract with bounded staging."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
import requests
from huggingface_hub import HfApi,hf_hub_url,snapshot_download
REPO='cyjcyj91/Hypersim-Distractor'
VERSION='releases/20261013_full_synthetic_v4_n3_m1'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
 return h.hexdigest()
def remap_member(name):
 path=Path(name)
 if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe TAR member')
 roots={'hypersim_pairs':'transient_only','hypersim_negative':'transient_static/pairs','hypersim_static':'transient_static/clean_resources'}
 return str(Path(roots.get(path.parts[0],path.parts[0])).joinpath(*path.parts[1:]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--destination',type=Path,required=True);ap.add_argument('--revision',default='main');ap.add_argument('--keep-archives',action='store_true');a=ap.parse_args()
 dest=a.destination.expanduser().resolve();dest.mkdir(parents=True,exist_ok=True)
 revision=HfApi().repo_info(REPO,repo_type='dataset',revision=a.revision).sha
 metadata=Path(snapshot_download(REPO,repo_type='dataset',revision=revision,allow_patterns=[VERSION+'/*'],ignore_patterns=[VERSION+'/archives/*']))/VERSION
 summary=json.loads((metadata/'summary.json').read_text())
 if summary.get('status')!='complete':raise RuntimeError('Release has not completed publication')
 for name in ['manifests','attribution']:
  shutil.copytree(metadata/name,dest/name,dirs_exist_ok=True)
 for name in ['summary.json','archive_index.json','LICENSE_NOTES.md','README.md']:shutil.copy2(metadata/name,dest/name)
 # Move an earlier extraction before reusing its archive completion state.
 for old,new in [('hypersim_pairs','transient_only'),('hypersim_negative','transient_static/pairs'),('hypersim_static','transient_static/clean_resources')]:
  previous=dest/old;current=dest/new
  if previous.exists():
   if current.exists():raise RuntimeError(f'Both legacy and new dataset roots exist: {previous}, {current}')
   current.parent.mkdir(parents=True,exist_ok=True);previous.rename(current)
 stage=dest/'.download_staging';stage.mkdir(exist_ok=True);state_path=dest/'.extraction_state.json'
 state=json.loads(state_path.read_text()) if state_path.exists() else {}
 archives=json.loads((metadata/'archive_index.json').read_text())
 for i,row in enumerate(archives):
  if state.get(row['path'])==row['sha256']:continue
  print(f'[{i+1}/{len(archives)}] {row["path"]}',flush=True)
  archive=stage/Path(row['path']).name
  if not archive.exists() or sha(archive)!=row['sha256']:
   if shutil.disk_usage(dest).free<row['bytes']*2+1024**3:raise RuntimeError('Insufficient extraction space')
   response=requests.get(hf_hub_url(REPO,row['path'],repo_type='dataset',revision=revision),stream=True,timeout=(30,180));response.raise_for_status()
   with archive.open('wb') as f:
    for data in response.iter_content(8*1024**2):f.write(data)
   if archive.stat().st_size!=row['bytes'] or sha(archive)!=row['sha256']:raise RuntimeError('Archive checksum mismatch')
  with tarfile.open(archive,'r:') as tar:
   for member in tar:
    target=(dest/remap_member(member.name)).resolve()
    if not member.isfile() or not target.is_relative_to(dest):raise ValueError('Unsafe TAR member')
    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.extracting')
    with tar.extractfile(member) as src,temp.open('wb') as out:shutil.copyfileobj(src,out)
    if target.exists() and target.name!='clean_multiview.json':
     if sha(target)!=sha(temp):raise ValueError(f'Conflicting existing file: {target}')
     temp.unlink()
    else:temp.replace(target)
  state[row['path']]=row['sha256'];tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
  if not a.keep_archives:archive.unlink()
 print('Complete. Data roots: transient_only/, transient_static/pairs/. Training code: https://github.com/yejich/GARD-DFrecon')
if __name__=='__main__':main()
