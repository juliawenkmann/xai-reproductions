## Top Activations

Exact-K neuron grids that recreate the “max-activation image” slide: we build a dataset (picsum, folder, or TorchVision), de-dup by content hash, run a backbone, and show/save the top-k inputs for the most mono-class channels.

### Contents
- `top_activations.py` – CLI script with strict guarantees (exact K, no duplicates, pop-up grids, optional CSV exports).
- `top_activations.ipynb` – notebook wrapper if you prefer interactive tweaking.

### Quickstart
```bash
# 1) Default: unique picsum images, GoogLeNet inception5b, K=8
python reproductions/top_activations/top_activations.py

# 2) Local ImageNet folder, ResNet-50 layer4, top-12 per channel
python reproductions/top_activations/top_activations.py \
  --source folder --images /path/to/imagenet/val \
  --model resnet50 --layer layer4 --topk 12 --show-channels 8

# 3) TorchVision dataset (auto-download, falls back to picsum if needed)
python reproductions/top_activations/top_activations.py \
  --source torchvision --tv-dataset oxford_pets --num-images 400
```
Useful flags:
- `--save-grids` and `--save-csv` write artifacts under `outputs/`.
- `--dedupe-source hash|path|none` controls global deduplication (hash recommended).
- `--label-source dataset|pred|both` toggles subtitle text for each tile.
- `--force-download` refreshes Picsum/TorchVision caches (otherwise cached folders are reused if they contain any images).

### Dependencies
Install the shared environment via `pip install -r requirements.txt` (needs `torch`, `torchvision`, `requests`, `tqdm`, `matplotlib`, `pillow`, etc.). Cached folders (e.g., `data/picsum/`, `data/tv_oxford_pets_test/`) are reused automatically; delete them or pass `--force-download` to refresh. For completely offline runs, point `--source folder` at a local image directory.
