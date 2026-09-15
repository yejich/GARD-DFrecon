# Bootstrap direct execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in _Path(__file__).resolve().parents if (p / "mvr").is_dir() and (p / "run_configs").is_dir())
_sys.path[:0] = [str(_root), str(_root / "RAE/src"), str(_root / "src"), str(_root / "Depth-Anything-3/src")]
from mvr.experiment_paths import (REPO_ROOT, output_dir, FEATURE_COMPLETION_ROOT,
    SYNTHETIC_DISTRACTOR_ROOT, VGGT_REPO, VGGT_OMEGA_REPO, VGGT_OMEGA_CKPT, VGTW_REPO, VGTW_CKPT)

from pathlib import Path
import sys
ROOT=REPO_ROOT;OUT=output_dir(__file__)
sys.path[:0]=[str(ROOT),str(ROOT/'Depth-Anything-3/src'),str(FEATURE_COMPLETION_ROOT)]
import numpy as np,torch
from PIL import Image
from mvr.eval_models.da3_restore import MultiViewDA3RestoreModel
model=MultiViewDA3RestoreModel()
with torch.inference_mode():
 for g in [0,26,53]:
  source=ROOT/f'result_eval/hypersim_10ep_ai001/five_methods/group_{g:02d}'
  images=[np.asarray(Image.open(source/f'clean_{v}.png')).astype(np.float32)/255 for v in range(4)]
  pred=model.geometry_plain(images)
  np.savez_compressed(OUT/f'group_{g:02d}/da3_clean.npz',**pred)
  print('saved',g,flush=True)
