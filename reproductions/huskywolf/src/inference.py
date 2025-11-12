from __future__ import annotations
from pathlib import Path
from typing import Dict
from PIL import Image
import torch
from .io import load_model
from .data import preprocess

def _to_image(x) -> Image.Image:
    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    raise TypeError("x must be an image path or PIL.Image")

def predict_image(x, 
                  weights_path: str | Path = "model_weights/binary_classifier.pth",
                  device: str | torch.device | None = None) -> Dict[str, float]:
    """Return probabilities as a dict.
    Note: class "1" is treated as "wolf" for display; adjust to match your folder order if needed.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = load_model(weights_path, device=device)
    img = _to_image(x)
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob1 = torch.sigmoid(model(tensor)).item()
    return {"prob_class1_wolf": float(prob1), "prob_class0_husky": float(1 - prob1)}
