from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
import random

FILE_DIR = Path.cwd()
PROJECT_DIR = FILE_DIR
print(PROJECT_DIR)
WEIGHTS = PROJECT_DIR / "models" / "huskywolf" / "binary_classifier.pth"

from src import get_model, predict_image, grad_cam_explain
# load if weights exist, otherwise train on your dataset and save weights
model = get_model(
    weights_path=WEIGHTS,
    data_root=PROJECT_DIR,      # used only if training is needed
    train_if_missing=True,
    epochs=20, lr=0.01
)
print("Model is ready:", type(model).__name__)


# Try to auto-pick a test image; otherwise set IMG manually.
def pick_any_image(project_dir: Path) -> Path | None:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    all_image_files = []
    for ext in exts:
        all_image_files.extend(list(project_dir.rglob(ext)))
    if all_image_files:
        # 3. Return a random choice from the list
        return random.choice(all_image_files)
    else:
        # 4. Return None if no images were found
        return None

IMG = pick_any_image(PROJECT_DIR)
print("Picked image:" if IMG else "No image auto-found. Please set IMG =", IMG)

if IMG:
    probs = predict_image(IMG, weights_path=WEIGHTS)
    plt.imshow(Image.open(IMG)); plt.axis("off")
    plt.show()
    print("Predicted probabilities:", probs)

if IMG:
    try:
        out = PROJECT_DIR / "reproductions" / "huskywolf" / "cam.jpg"
        grad_cam_explain(IMG, weights_path=WEIGHTS, out_path=out)
        print("Saved Grad‑CAM to:", out)
        plt.imshow(Image.open(out)); plt.axis("off"); plt.show()
    except ImportError as e:
        print("Grad‑CAM not installed. Install with: pip install grad-cam imageio")