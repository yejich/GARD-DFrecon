"""Paths shared by the migrated evaluation experiments."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def external_path(variable, default):
    return Path(os.environ.get(variable, str(default))).expanduser()

FEATURE_COMPLETION_ROOT = external_path("FEATURE_COMPLETION_ROOT", REPO_ROOT.parent / "distractor-free/feature_completion")
SYNTHETIC_DISTRACTOR_ROOT = external_path("SYNTHETIC_DISTRACTOR_ROOT", REPO_ROOT.parent / "synthetic_distractor")
VGGT_REPO = external_path("VGGT_REPO", "/mnt/dataset1/jaeeun/vggt")
VGGT_OMEGA_REPO = external_path("VGGT_OMEGA_REPO", REPO_ROOT / "vggt-omega")
VGGT_OMEGA_CKPT = external_path("VGGT_OMEGA_CKPT", REPO_ROOT / "ckpts/vggt_omega_1b_512.pt")
VGTW_REPO = external_path("VGTW_REPO", REPO_ROOT.parent / "VGTW")
VGTW_CKPT = external_path("VGTW_CKPT", VGTW_REPO / "vgtw_lora_fp32.pt")

def output_dir(script):
    relative = Path(script).resolve().relative_to(REPO_ROOT / "scripts/eval").parent
    folder = REPO_ROOT / "result_eval" / relative
    folder.mkdir(parents=True, exist_ok=True)
    return folder
