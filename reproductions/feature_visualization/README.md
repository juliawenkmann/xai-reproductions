## Feature & Class Visualization

Lucent- and PyTorch-based recreations of the neuron/class synthesis demos. The scripts are intentionally small so it is easy to tweak layers, priors, or optimizers.

### Contents
- `feature_visualization.py` – Lucent helper that renders GoogLeNet `mixed4a:11` (and snapshots) out of the box. Tweak `goal`, `snapshot_steps`, etc. to explore other channels or architectures.
- `class_visualization.py` – Pure PyTorch Fourier-param synthesizer for ImageNet classes (defaults to **dumbbell**). Adjust the call under `if __name__ == "__main__"` for different categories or hyperparameters.
- `feature_visualization.ipynb` / `class_visualization.ipynb` – notebook counterparts with markdown commentary.
- `lucent/` – vendored Lucent fork so the Lucent import path resolves without extra setup.

### Quickstart
```bash
pip install lucent torch torchvision pillow matplotlib

# Feature viz (Lucent). Edits go inside the script/notebook.
python reproductions/feature_visualization/feature_visualization.py

# Class viz (Fourier parametrization, default "dumbbell").
python reproductions/feature_visualization/class_visualization.py
```
Each script logs the exact output path at the end—edit the constants near the bottom if you want to change filenames or layer targets. Use the notebooks when you prefer interactive tweaks or incremental visual logging.
