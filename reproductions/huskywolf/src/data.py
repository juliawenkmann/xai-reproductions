from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import torch
import torchvision
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

preprocess_unnormalized = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
])

def resolve_dataset_dirs(root: Path) -> Tuple[Path, Path]:
    """Return (train_dir, test_dir) given a project root.
    We try a few common layouts:
      - root/dataset/{train,test}/... (preferred)
      - root/dataset/... (single folder with class subdirs)
      - root/dataset1/... (some repos use this name)
      - root/... (if user passes the dataset folder directly)
    """
    root = Path(root)
    c = root / "dataset"
    train = c / "train"
    test = c / "test"
    if train.exists() and train.is_dir():
        return train, (test if test.exists() and test.is_dir() else c)
    # Binary classification folders directly under c, e.g. c/0, c/1 or c/wolf, c/husky
    subdirs = [d for d in c.iterdir() if d.is_dir()]
    if any((c / "0").exists() and (c / "1").exists()) or len(subdirs) == 2:
        return c, c
    raise FileNotFoundError("Could not resolve dataset folders under: {root}")

def make_loaders(root: str | Path, batch_size: int = 5, shuffle: bool = True):
    train_dir, _ = resolve_dataset_dirs(Path(root))
    ds = torchvision.datasets.ImageFolder(train_dir, transform=preprocess)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader, ds.classes
