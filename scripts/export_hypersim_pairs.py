"""Create an uncompressed training-only archive, separate from Git/model weights."""
import argparse,os,tarfile
from pathlib import Path
from check_hypersim_pairs import validate,DEFAULT_MANIFEST

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default=os.environ.get('HYPERSIM_PAIRS_ROOT'));ap.add_argument('--manifest',default=str(DEFAULT_MANIFEST));ap.add_argument('--output',required=True);args=ap.parse_args()
 if not args.root:ap.error('Set --root or HYPERSIM_PAIRS_ROOT')
 _,files=validate(args.manifest,args.root);out=Path(args.output)
 if out.exists():raise FileExistsError(out)
 out.parent.mkdir(parents=True,exist_ok=True)
 size=sum((Path(args.root)/p).stat().st_size for p in files)
 print(f'Packing {len(files)} image/mask files ({size/1e9:.2f} GB) into {out}',flush=True)
 with tarfile.open(out,'x') as archive:
  for p in files:archive.add(Path(args.root)/p,arcname=str(Path('scenes_v2')/p),recursive=False)
 print('Done. Transfer this archive separately; do not add it to Git.')
if __name__=='__main__':main()
