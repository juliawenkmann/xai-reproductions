## Concept Algebra

Reruns the analogy/bias demos with CLIP (text encoder only) and, optionally, Word2Vec. Everything is pure Python so you can run it headless or in a notebook.

### Contents
- `concept_algebra.py` – CLI entry point (plots cosine-sim matrices + PCA arrows, can mix CLIP and Word2Vec).
- `concept_algebra_minimal.ipynb` – step-by-step walkthrough that mirrors the talk.
- `arithmetic_in_image_gen.py` – DCGAN latent arithmetic sampler for quick “A − B + C” image grids.

### Quickstart
```bash
pip install torch torchvision clip-anytorch gensim pandas matplotlib scikit-learn

# CLIP-only run (final-layer embeddings, default prompt "{}")
python reproductions/concept_algebra/concept_algebra.py \
  --model "ViT-B/32" \
  --layer final \
  --prompt-template "{}"
```
If `gensim` is installed the script will also fetch `word2vec-google-news-300` automatically. Use `--w2v-path /path/to/GoogleNews-vectors-negative300.bin` to point at an offline copy or pass `--skip-w2v` when you only care about CLIP.

For the GAN arithmetic script:
```bash
python reproductions/concept_algebra/arithmetic_in_image_gen.py --rows 8 --cols 4
```
Outputs land in `reproductions/concept_algebra/out/` by default.
