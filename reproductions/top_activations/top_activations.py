"""
slide12_top_activations_cli.py
--------------------------------
Clean, reliable "top activations" script with these guarantees:
  • **Exact K per channel**: every shown grid has exactly K images (no missing).
  • **No duplicates in dataset**: content-hash de-dup on collection/materialization.
  • **No within-grid duplicates**: selection enforces unique-by-hash for each channel.
  • **Pop-up grids** (plt.show) so results appear like in a notebook.
  • **Class names on tiles** (dataset label if available, else predicted ImageNet class).

Sources
  - picsum (default; robust, unique-by-hash)
  - folder (easiest for local ImageNet: --source folder --images /path/to/imagenet/val)
  - torchvision datasets (oxford_pets, caltech101, flowers102, food101, cifar100, stl10) with fallback to picsum
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import torch
from typing import Callable, Sequence

import numpy as np
from PIL import Image
from src.utils import pick_device, get_module_by_path, collect_source, collect_scores_and_logits, dedupe_paths, load_torchvision_model, topk_unique_exact, md5_file, show_grid

# ---------------------------
# CLI
# ---------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Top activations (exact K per channel, no dataset duplicates, pop-up grids).")
    # Source
    p.add_argument("--source", type=str, default="picsum",
                   choices=["picsum", "folder", "torchvision"],
                   help="Where to get images from.")
    p.add_argument("--images", type=str, default=None, help="Local folder (recursive) when --source folder.")
    p.add_argument("--data-root", type=str, default="datasets/random_top_activations", help="Root for downloads/materialized images.")
    p.add_argument("--num-images", type=int, default=400, help="Target number of unique images after de-dup.")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed for picsum.")
    # picsum-only
    p.add_argument("--image-size", type=int, default=512, help="Picsum fetch size (square).")
    p.add_argument("--max-attempts", type=int, default=4000, help="Picsum max attempts to get unique images.")
    # TorchVision datasets
    p.add_argument("--tv-dataset", type=str, default="oxford_pets",
                   choices=["oxford_pets", "caltech101", "flowers102", "food101", "cifar100", "stl10"],
                   help="TorchVision dataset to auto-download when --source torchvision.")
    p.add_argument("--tv-split", type=str, default="test", help="Dataset split (varies by dataset).")
    # De-duplication / downloads
    p.add_argument("--dedupe-source", type=str, default="hash", choices=["hash", "path", "none"],
                   help="Global dataset de-duplication key (hash recommended).")
    p.add_argument("--force-download", action="store_true",
                   help="Redownload/rematerialize even if cached folders already contain images.")
    # Model & layer
    p.add_argument("--model", type=str, default="googlenet", choices=["googlenet", "resnet50", "efficientnet_b0"])
    p.add_argument("--layer", type=str, default=None, help="Layer path to hook; default depends on model.")
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "max"], help="Pooling over spatial dims.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    # Top-K & display
    p.add_argument("--topk", type=int, default=8, help="Top-K images per channel (EXACT for every channel).")
    p.add_argument("--show-channels", type=int, default=6, help="How many channels to display (most mono-class first).")
    p.add_argument("--viz-cols", type=int, default=4, help="Columns in grids.")
    p.add_argument("--save-grids", action="store_true", help="Also save grids under outputs/grids/.")
    p.add_argument("--save-csv", action="store_true", help="Save CSVs (topk + dominance).")
    p.add_argument("--output-dir", type=str, default="outputs")
    # Labels
    p.add_argument("--label-source", type=str, default="auto", choices=["auto", "dataset", "pred", "both"],
                   help="Which class name to show under each tile.")
    p.add_argument("--dominance-by", type=str, default="auto", choices=["auto", "dataset", "pred"],
                   help="Which labels to use for dominance calculation (mono-class ranking).")
    return p

# ---------------------------
# Main
# ---------------------------
def main():
    args = build_parser().parse_args()

    # 1) Collect and de-dup source
    paths_raw, path2dslabel = collect_source(args)
    if not paths_raw: raise RuntimeError("No images collected.")

    if args.dedupe_source != "none":
        paths, keys = dedupe_paths(paths_raw, mode=args.dedupe_source)
    else:
        paths = paths_raw; keys = list(paths)
    if not paths: raise RuntimeError("No images remain after de-duplication.")

    # Ensure at least topk items exist
    if len(paths) < args.topk:
        raise RuntimeError(f"Need at least {args.topk} unique images; got {len(paths)}. Increase --num-images or lower --topk.")

    # If too many, truncate to num-images (after dedupe)
    if args.num_images and len(paths) > args.num_images:
        paths = paths[:args.num_images]

    # 2) Model
    device = pick_device(args.device)
    model, preprocess, categories, default_layer = load_torchvision_model(args.model, device)
    layer_path = args.layer or default_layer
    try:
        hook_layer = get_module_by_path(model, layer_path)
    except Exception as e:
        raise RuntimeError(f"Could not resolve layer '{layer_path}' on model '{args.model}': {e}")

    # 3) DataLoader
    class ImageListDataset(torch.utils.data.Dataset):
        def __init__(self, paths: Sequence[str], transform: Callable):
            self.paths = list(paths); self.t = transform
        def __len__(self) -> int: return len(self.paths)
        def __getitem__(self, i: int):
            p = self.paths[i]
            with Image.open(p) as im: im = im.convert("RGB")
            return self.t(im), p

    loader = torch.utils.data.DataLoader(ImageListDataset(paths, preprocess),
                                         batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)

    # 4) Forward
    scores, sample_paths, logits = collect_scores_and_logits(model, hook_layer, loader, device=device, pool=args.pool)
    N, C = scores.shape
    print(f"Pooled activation scores: shape = {scores.shape} (N={N}, C={C})")
    if N < args.topk:
        raise RuntimeError(f"Post-load N={N} < topk={args.topk}; lower --topk or increase images.")

    # 5) Predicted class names
    pred_ids = np.argmax(logits, axis=1)
    pred_names = [categories[i] if (categories is not None and i < len(categories)) else str(i) for i in pred_ids]

    # 6) Keys aligned to sample order
    if args.dedupe_source == "hash":
        keys_in_order = [md5_file(p) for p in sample_paths]
    elif args.dedupe_source == "path":
        keys_in_order = list(sample_paths)
    else:
        keys_in_order = [str(i) for i in range(len(sample_paths))]

    # 7) Exact-K top per channel (unique within channel)
    top_idx = topk_unique_exact(scores, k=args.topk, keys_seq=keys_in_order)  # [k, C]

    # 8) Dominance ranking (choose labels: dataset if available, else predicted, or per --dominance-by)
    if args.dominance_by == "dataset" or (args.dominance_by == "auto" and any(p in path2dslabel for p in sample_paths)):
        names_for_dom = [path2dslabel.get(p, pred_names[i]) for i, p in enumerate(sample_paths)]
    else:
        names_for_dom = pred_names

    dominance_rows = []
    for c in range(C):
        idxs = [int(top_idx[r, c]) for r in range(args.topk)]
        names = [names_for_dom[i] for i in idxs]
        counts = Counter(names)
        dom_name, dom_count = counts.most_common(1)[0] if counts else ("", 0)
        frac = dom_count / float(args.topk)
        dominance_rows.append({"channel": c, "dominant_class": dom_name,
                               "dominant_count": dom_count, "k": args.topk, "fraction": frac})
    dominance_rows.sort(key=lambda d: (-d["fraction"], -d["dominant_count"], d["channel"]))

    print("\nMost mono-class channels (K={} exact):".format(args.topk))
    for row in dominance_rows[:args.show_channels]:
        print(f"  ch {row['channel']:>4}: {row['dominant_class']}  ({row['dominant_count']}/{row['k']}; {row['fraction']:.2f})")

    # 9) CSVs (optional)
    out_dir = Path(args.output_dir)
    if args.save_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        topk_csv = out_dir / "topk_summary.csv"
        with open(topk_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["channel", "rank", "index", "path", "score", "pred_name", "dataset_name"])
            for c in range(C):
                for r in range(args.topk):
                    i = int(top_idx[r, c])
                    w.writerow([c, r, i, sample_paths[i], float(scores[i, c]), pred_names[i], path2dslabel.get(sample_paths[i], "")])
        dom_csv = out_dir / "channel_class_dominance.csv"
        with open(dom_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["channel", "dominant_class", "dominant_count", "k", "fraction"])
            w.writeheader(); w.writerows(dominance_rows)
        print(f"Saved CSVs to {out_dir}")

    # 10) Display (and optionally save) the top 'show-channels' monoclass grids, each with EXACT K
    label_src = args.label_source
    if label_src == "auto":
        label_src = "dataset" if any(p in path2dslabel for p in sample_paths) else "pred"

    if args.save_grids:
        (out_dir / "grids").mkdir(parents=True, exist_ok=True)

    for row in dominance_rows[:args.show_channels]:
        ch = row["channel"]
        idxs = [int(top_idx[r, ch]) for r in range(args.topk)]
        title = f"Channel {ch} — top {args.topk}"
        savepath = (out_dir / "grids" / f"grid_channel_{ch}.png") if args.save_grids else None
        show_grid(idxs, sample_paths, scores, pred_names, path2dslabel,
                  channel=ch, title=title, savepath=savepath,
                  ncols=args.viz_cols, label_source=label_src)

    print("\nDone.")

if __name__ == "__main__":
    main()
