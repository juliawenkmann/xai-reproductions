## Husky vs. Wolf (Ribeiro 2016)

Small repro of the biased husky/wolf classifier plus Grad-CAM explanations.

### Contents
- `HuskyWolf_minimal.ipynb` – main entry point (auto-loads/ trains the binary classifier, runs predictions, and renders Grad-CAM overlays).
- `src/` – lightweight helper package (`io.py`, `train.py`, `inference.py`, `explain.py`, etc.). The notebook dynamically adds this folder to `sys.path`, so you can run it from any working directory.
- `cam.jpg` – sample Grad-CAM output for quick visual checks.

### Requirements
Install the usual vision stack before launching the notebook:
```bash
pip install torch torchvision pytorch-grad-cam pillow imageio matplotlib
```
Optionally download the provided dataset (`datasets/huskywolf/train|test`) and copy the pretrained weights into `models/huskywolf/binary_classifier.pth` (already included in this repo snapshot).

### Quickstart
1. Launch Jupyter from the repo root: `jupyter lab reproductions/huskywolf/HuskyWolf_minimal.ipynb`.
2. Run the first setup cell; it will locate `src/`, load the pretrained weights if present, or fall back to training with `train_if_missing=True`.
3. Use the prediction cell to auto-pick an image (or set `IMG` manually) and inspect the Grad-CAM output.

### Programmatic Use
Reuse the helpers in scripts by adding the `src` folder to `PYTHONPATH` and importing `src` (rename the module as you wish):
```bash
PYTHONPATH=$(pwd)/reproductions/huskywolf/src python - <<'PY'
import src as huskywolf
model = huskywolf.get_model("models/huskywolf/binary_classifier.pth",
                            data_root=".", train_if_missing=False)
print(huskywolf.predict_image("datasets/huskywolf/test/snowdog.jpg",
                              weights_path="models/huskywolf/binary_classifier.pth"))
PY
```
`grad_cam_explain` exposes the Grad-CAM helper if you need it outside the notebook.
