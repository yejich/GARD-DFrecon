"""Download required frozen models once while online; never embeds tokens."""
import argparse,json
from pathlib import Path
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
models={name:snapshot_download(name) for name in ['depth-anything/DA3-GIANT-1.1','depth-anything/DA3-Base']}
# Torchvision AlexNet weights used by LPIPS must also be cached before going offline.
import lpips
metric=lpips.LPIPS(net='alex')
Path(a.out).write_text(json.dumps({'status':'ready','models':models,'lpips':'alex cached'},indent=2)+'\n')
print('DA3 Giant/Base and LPIPS ready for offline training/evaluation')
