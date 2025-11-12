# XAI Reproductions

Compact, runnable references for the demos we covered: concept algebra, neuron visualizations, and the Husky-vs-Wolf case study. Everything lives under `reproductions/`, while shared assets stay in `datasets/` and `models/`.

## Repo Layout
- `datasets/` – optional sample data (e.g., husky/wolf splits).
- `models/model_weights/` – cached checkpoints such as `binary_classifier.pth`.
- `reproductions/concept_algebra/` – CLIP & Word2Vec word-analogy tooling (script + notebook).
- `reproductions/feature_visualization/` – Lucent-based feature- and class-visualization scripts/notebooks.
- `reproductions/huskywolf/` – notebook + helpers for the bias/Grad-CAM walkthrough.

## Quick Recipes

### Concept Algebra (CLIP / Word2Vec)
```bash
python reproductions/concept_algebra/concept_algebra.py \
  --model "ViT-B/32" \
  --layer final \
  --prompt-template "{}"
```
Pass `--w2v-path` (or rely on `gensim.downloader`) for the classic word2vec variant. Use `concept_algebra_minimal.ipynb` for a guided version.

### Feature & Class Visualizations (Lucent)
```bash
# channel / unit visualization
python reproductions/feature_visualization/feature_visualization.py \
  --layer inception4a.branch1.conv --unit 11 --steps 512

# class logit maximization
python reproductions/feature_visualization/class_visualization.py \
  --class-idx 543 --steps 640
```
Both scripts install-free aside from `lucent`, `torch`, `torchvision`, `pillow`, `matplotlib`. Each saves PNGs and shows them inline by default.

### Husky vs. Wolf (Bias + Grad-CAM)
Open `reproductions/huskywolf/HuskyWolf_minimal.ipynb` to:
1. Auto-load (or train) the lightweight binary classifier using `models/model_weights/binary_classifier.pth`.
2. Predict on any image (auto-pick helper included).
3. Generate Grad-CAM overlays via `grad_cam_explain`.

The notebook dynamically locates its helper package (`reproductions/huskywolf/src`) so you can run it from any working directory. Make sure the dependencies listed at the top of the notebook are installed first (`torch`, `torchvision`, `pytorch_grad_cam`, `pillow`, `imageio`, etc.).
