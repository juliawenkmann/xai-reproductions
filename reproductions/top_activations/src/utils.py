import hashlib
import io
import os
from pathlib import Path
import random
import requests
import torch
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x
    
# ---------------------------
# Utilities
# ---------------------------
def pick_device(spec: str) -> str:
    if spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if spec == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return spec

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def dir_has_images(path: Path, suffixes: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> bool:
    if not path.exists():
        return False
    for suf in suffixes:
        if any(path.glob(f"*{suf}")):
            return True
    return False

def list_image_files(path: Path, suffixes: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[str]:
    files: List[str] = []
    for suf in suffixes:
        files.extend(str(p) for p in sorted(path.glob(f"*{suf}")))
    return files

def md5_bytes(data: bytes) -> str:
    h = hashlib.md5(); h.update(data); return h.hexdigest()

def md5_file(path: str, chunk: int = 1 << 16) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

# ---------------------------
# Models
# ---------------------------
def load_torchvision_model(name: str, device: str):
    name = name.lower()
    if name == "googlenet":
        from torchvision.models import googlenet, GoogLeNet_Weights
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        model = googlenet(weights=weights)
        preprocess = weights.transforms()
        categories = weights.meta.get("categories", None)
        hook_default = "inception5b"
    elif name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        preprocess = weights.transforms()
        categories = weights.meta.get("categories", None)
        hook_default = "layer4"
    elif name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        preprocess = weights.transforms()
        categories = weights.meta.get("categories", None)
        hook_default = "features.6"
    else:
        raise ValueError(f"Unsupported model: {name}")
    model.eval().to(device)
    return model, preprocess, categories, hook_default

def get_module_by_path(root: nn.Module, dotted_path: str) -> nn.Module:
    mod: nn.Module = root
    for part in dotted_path.split("."):
        if part.isdigit():
            mod = list(mod.children())[int(part)]
        else:
            mod = getattr(mod, part)
    return mod

# ---------------------------
# Sources
# ---------------------------
def list_images_recursive(root: Path) -> List[str]:
    if root.is_file():
        return [str(root)]
    paths = [str(p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths

def save_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def write_label_index(out_dir: Path, mapping: Dict[str, str]) -> None:
    if not mapping:
        return
    idx_path = out_dir / "_labels.csv"
    with open(idx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for path_str, label in mapping.items():
            w.writerow([Path(path_str).name, label])

def read_label_index(out_dir: Path) -> Dict[str, str]:
    idx_path = out_dir / "_labels.csv"
    if not idx_path.exists():
        return {}
    mapping: Dict[str, str] = {}
    with open(idx_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename")
            label = row.get("label", "")
            if filename:
                mapping[str(out_dir / filename)] = label
    return mapping

def materialize_dataset(ds, out_dir: Path, target_n: int) -> Tuple[List[str], Dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    path2label: Dict[str, str] = {}
    seen_hash = set()
    names = getattr(ds, "classes", None) or getattr(ds, "categories", None)

    for i in tqdm(range(len(ds)), desc="materialize", unit="img"):
        if target_n and len(saved_paths) >= target_n:
            break
        img, target = ds[i]
        try:
            data = save_jpeg_bytes(img, quality=92)
            h = md5_bytes(data)
            if h in seen_hash:
                continue
            seen_hash.add(h)
            p = out_dir / f"{h}.jpg"
            if not p.exists():
                with open(p, "wb") as f:
                    f.write(data)
            saved_paths.append(str(p))
            if isinstance(names, (list, tuple)) and 0 <= int(target) < len(names):
                path2label[str(p)] = str(names[int(target)])
            else:
                path2label[str(p)] = str(int(target))
        except Exception:
            continue
    write_label_index(out_dir, path2label)
    return saved_paths, path2label

def collect_source(args) -> Tuple[List[str], Dict[str, str]]:
    root = Path(args.data_root); root.mkdir(parents=True, exist_ok=True)

    if args.source == "picsum":
        out = root / "picsum"; out.mkdir(parents=True, exist_ok=True)
        if not args.force_download and dir_has_images(out, (".jpg",)):
            cached = list_image_files(out, (".jpg",))
            if cached:
                print(f"[cache] Reusing {len(cached)} cached Picsum images from {out}")
                return cached, {}
        random.seed(args.seed)
        saved, seen = [], set()
        with tqdm(total=args.num_images, desc="picsum", unit="img") as bar:
            attempts = 0
            while len(saved) < args.num_images and attempts < args.max_attempts:
                attempts += 1
                seed = random.randint(0, 10_000_000)
                url = f"https://picsum.photos/seed/{seed}/{args.image_size}/{args.image_size}"
                try:
                    r = requests.get(url, timeout=(5, 20))
                    if r.status_code == 200 and r.content:
                        h = md5_bytes(r.content)
                        if h in seen: continue
                        seen.add(h)
                        p = out / f"{h}.jpg"
                        if not p.exists():
                            with open(p, "wb") as f:
                                f.write(r.content)
                        saved.append(str(p)); bar.update(1)
                except Exception:
                    continue
        return saved, {}

    if args.source == "folder":
        if not args.images:
            raise RuntimeError("--images is required for --source folder")
        paths = list_images_recursive(Path(args.images))
        if not paths:
            raise RuntimeError(f"No images under {args.images}")
        return paths, {}

    # TorchVision (with graceful fallback to picsum)
    cache = root / "torchvision_cache"; cache.mkdir(parents=True, exist_ok=True)
    try:
        name = args.tv_dataset
        out = root / f"tv_{name}_{args.tv_split}"
        if not args.force_download and dir_has_images(out, (".jpg",)):
            cached = list_image_files(out, (".jpg",))
            path2label_cached = read_label_index(out)
            print(f"[cache] Reusing {len(cached)} materialized TorchVision images from {out}")
            return cached, path2label_cached
        if name == "oxford_pets":
            ds = tv.datasets.OxfordIIITPet(root=str(cache), split=args.tv_split, download=True)
        elif name == "caltech101":
            ds = tv.datasets.Caltech101(root=str(cache), download=True)
        elif name == "flowers102":
            ds = tv.datasets.Flowers102(root=str(cache), split=args.tv_split, download=True)
        elif name == "food101":
            ds = tv.datasets.Food101(root=str(cache), split=args.tv_split, download=True)
        elif name == "cifar100":
            ds = tv.datasets.CIFAR100(root=str(cache), train=(args.tv_split!="test"), download=True)
        elif name == "stl10":
            ds = tv.datasets.STL10(root=str(cache), split=args.tv_split, download=True)
        else:
            raise ValueError(f"Unsupported tv-dataset: {name}")
        paths, p2l = materialize_dataset(ds, out, args.num_images)
        if not paths:
            raise RuntimeError("TorchVision dataset materialization yielded no images.")
        return paths, p2l
    except Exception as e:
        print(f"[WARN] TorchVision dataset '{args.tv_dataset}' failed ({e}). Falling back to picsum.")
        args.source = "picsum"
        return collect_source(args)

# ---------------------------
# Global dataset de-dup
# ---------------------------
def dedupe_paths(paths: List[str], mode: str) -> Tuple[List[str], List[str]]:
    if mode == "none":
        return paths, list(paths)
    if mode == "path":
        seen = set(); kept = []; keys = []
        for p in paths:
            if p in seen: continue
            seen.add(p); kept.append(p); keys.append(p)
        return kept, keys
    if mode == "hash":
        seen = set(); kept = []; keys = []
        for p in tqdm(paths, desc="dedupe(hash)", unit="img"):
            try:
                k = md5_file(p)
            except Exception:
                continue
            if k in seen: continue
            seen.add(k); kept.append(p); keys.append(k)
        return kept, keys
    raise ValueError(f"Unknown mode: {mode}")

# ---------------------------
# Activations & Top-K
# ---------------------------
def pool_activations(acts: torch.Tensor, pool: str) -> torch.Tensor:
    if acts.ndim == 4:
        if pool == "mean": return acts.mean(dim=(2,3))
        if pool == "max":  return acts.amax(dim=(2,3))
        raise ValueError(f"Unknown pool: {pool}")
    if acts.ndim == 2: return acts
    raise ValueError(f"Unexpected activation shape: {acts.shape}")

@torch.no_grad()
def collect_scores_and_logits(model: nn.Module, hook_layer: nn.Module,
                              loader: torch.utils.data.DataLoader, device: str, pool: str):
    buf: List[torch.Tensor] = []
    def _hook(_m, _in, out): buf.append(out.detach())
    h = hook_layer.register_forward_hook(_hook)

    all_s, all_p, all_l = [], [], []
    for batch, paths in loader:
        batch = batch.to(device); buf.clear()
        logits = model(batch)
        if isinstance(logits, (tuple, list)): logits = logits[0]
        a = buf.pop(); s = pool_activations(a, pool=pool)
        all_s.append(s.cpu().numpy()); all_p.extend(list(paths))
        all_l.append(logits.detach().cpu().numpy())
    h.remove()
    scores = np.concatenate(all_s, axis=0)
    logits = np.concatenate(all_l, axis=0)
    return scores, all_p, logits

def topk_unique_exact(scores_np: np.ndarray, k: int, keys_seq: Sequence[str]) -> np.ndarray:
    """
    Exact-K selection per channel with uniqueness by 'keys_seq' (e.g., content hash).
    Raises if any channel cannot supply K unique items.
    Returns: [k, C] int array of indices.
    """
    N_loc, C_loc = scores_np.shape
    if N_loc < k:
        raise RuntimeError(f"Dataset has only N={N_loc} images < topk={k}. Increase --num-images or lower --topk.")
    order = np.argsort(-scores_np, axis=0)  # [N, C] descending
    out = np.zeros((k, C_loc), dtype=int)
    for c in range(C_loc):
        seen = set(); picked = []
        for i in order[:, c]:
            key = keys_seq[int(i)]
            if key in seen: continue
            picked.append(int(i)); seen.add(key)
            if len(picked) == k: break
        if len(picked) < k:
            raise RuntimeError(f"Channel {c} yielded only {len(picked)} unique items < topk={k}. Try lowering --topk or increasing --num-images.")
        out[:, c] = picked
    return out

# ---------------------------
# Visualization
# ---------------------------
def label_for_tile(path: str, pred_name: str, path2dslabel: Dict[str, str], label_source: str) -> str:
    if label_source == "dataset":
        return path2dslabel.get(path, pred_name)
    if label_source == "pred":
        return pred_name
    if label_source == "both":
        ds = path2dslabel.get(path, "")
        if ds and ds != pred_name:
            return f"{ds} | {pred_name}"
        return ds or pred_name
    # auto
    return path2dslabel.get(path, pred_name)

def show_grid(indices: Sequence[int],
              paths: Sequence[str],
              scores: np.ndarray,
              pred_names: Sequence[str],
              path2dslabel: Dict[str, str],
              channel: int,
              title: str,
              savepath: Optional[Path],
              ncols: int = 4,
              figsize: Tuple[int, int] = (10, 8),
              label_source: str = "auto") -> None:
    n = len(indices); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1: axes = np.array([[axes]])
    elif nrows == 1: axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for j, ax in enumerate(axes.ravel()):
        if j < n:
            idx = int(indices[j]); p = paths[idx]
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                ax.axis("off"); ax.set_title("Error", fontsize=9); continue
            ax.imshow(im)
            label_txt = label_for_tile(p, pred_names[idx], path2dslabel, label_source)
            if len(label_txt) > 40: label_txt = label_txt[:37] + "…"
            ax.set_title(f"#{idx}  {scores[idx, channel]:.3f}\n{label_txt}", fontsize=9)
            ax.axis("off")
        else:
            ax.axis("off")

    fig.suptitle(title, fontsize=12); fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
