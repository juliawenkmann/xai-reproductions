import numpy as np
from pathlib import Path
from datetime import datetime
from urllib.request import urlretrieve
import torch

SCRIPT_PATH = Path(__file__).resolve()


def find_repo_root(path: Path) -> Path:
    """Walk upward from `path` until a .git directory is found."""
    for parent in path.parents:
        if (parent / ".git").is_dir():
            return parent
    return path.parents[-1]


REPO_ROOT = find_repo_root(SCRIPT_PATH)
OUT_DIR = REPO_ROOT / "reproductions" / "concept_algebra" / "out"
WEIGHTS_DIR = REPO_ROOT / "models" / "concept_algebra_gan"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Pretrained PyTorch generator (CelebA 64x64) — hosted on Hugging Face
# Model card: https://huggingface.co/hussamalafandi/DCGAN_CelebA
HF_GENERATOR_URL = (
    "https://huggingface.co/hussamalafandi/DCGAN_CelebA/resolve/main/generator.pth"
)
HF_LOCAL_WEIGHTS = WEIGHTS_DIR / "celeba_generator_hf.pth"

# ------------------------------ Utilities ------------------------------
def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_celebA_weights(local_path: Path = HF_LOCAL_WEIGHTS) -> Path:
    """Download CelebA DCGAN generator weights if not present."""
    if local_path.exists():
        return local_path
    print(f"[INFO] Downloading pretrained CelebA generator -> {local_path}")
    try:
        urlretrieve(HF_GENERATOR_URL, local_path.as_posix())
    except Exception as e:
        raise RuntimeError(
            "Failed to download pretrained weights. "
            f"Try again later or provide --weights. Error: {e}"
        )
    return local_path


def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """
    Map a CHW float tensor in [0,1] to HxWxC uint8 for display/saving.
    """
    t = t.clamp(0, 1)
    arr = (t.detach().cpu().numpy() * 255.0).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))  # HWC


def stack_rows(batch: torch.Tensor) -> np.ndarray:
    """
    Vertically stack a batch of images (N, C, H, W) into a single H*N x W x C image.
    Assumes values already in [0,1].
    """
    imgs = []
    for i in range(batch.size(0)):
        imgs.append(to_uint8_image(batch[i]))
    return np.vstack(imgs)


def make_column_grid(A, B, C, Y) -> np.ndarray:
    """
    Given four batches of images (A,B,C,Y) each (N,3,64,64) in [0,1],
    produce one big tiled image with 4 columns and N rows.
    """
    colA = stack_rows(A)
    colB = stack_rows(B)
    colC = stack_rows(C)
    colY = stack_rows(Y)
    return np.hstack([colA, colB, colC, colY])  # H_total x (4*W) x C
