"""Validate the shared fixed manifest and required pair files without loading models."""
import argparse,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST=ROOT/'manifests/hypersim_001_groups.json'
PAIR_FILES=('clean.png','distractor.png','mask_object.png','mask_shadow.png')

def validate(manifest,root):
 m=json.loads(Path(manifest).read_text());root=Path(root);missing=[];pairs=[]
 for split in ('train','eval'):
  records=m[split]
  for r in records:
   folder=Path(r['directory'])
   if folder.is_absolute() or '..' in folder.parts:raise ValueError('Manifest paths must be relative')
   if (r['scene']==m['eval_scene'])!=(split=='eval'):raise ValueError('Held-out scene leaks between splits')
   for i in r['candidates']:
    c=records[i]
    if (c['scene'],c['camera'])!=(r['scene'],r['camera']) or c['frame']==r['frame']:raise ValueError('Invalid covisibility candidate')
   for name in PAIR_FILES:
    p=folder/name;pairs.append(p)
    if not (root/p).is_file():missing.append(str(p))
 for group in m['eval_groups']:
  if len(group['indices'])!=m['eval_views'] or len(group['distractor'])!=m['eval_views']:raise ValueError('Invalid fixed evaluation group')
  if len(set(group['indices']))!=m['eval_views']:raise ValueError('Repeated evaluation view')
  for i in group['indices']:
   if not 0<=i<len(m['eval']):raise ValueError('Evaluation index out of range')
 if missing:raise FileNotFoundError(f'{len(missing)} required files missing under {root}; first: {missing[:5]}')
 return m,pairs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default=os.environ.get('HYPERSIM_PAIRS_ROOT'));ap.add_argument('--manifest',default=str(DEFAULT_MANIFEST));args=ap.parse_args()
 if not args.root:ap.error('Set --root or HYPERSIM_PAIRS_ROOT')
 m,files=validate(args.manifest,args.root)
 print(json.dumps(dict(train_pairs=len(m['train']),eval_pairs=len(m['eval']),fixed_eval_groups=len(m['eval_groups']),required_files=len(files)),indent=2))
if __name__=='__main__':main()
