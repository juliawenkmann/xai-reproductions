from __future__ import annotations
from pathlib import Path
import torch
from .model import create_model

def save_model(model, weights_path: str | Path = "model_weights/binary_classifier.pth") -> Path:
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    return weights_path

def load_model(weights_path: str | Path = "model_weights/binary_classifier.pth", 
               device: str | torch.device | None = None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = create_model(device)
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at: {weights_path}. Train first or set train_if_missing=True in get_model.")
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model

def get_model(weights_path: str | Path = "models/huskywolf/binary_classifier.pth",
              data_root: str | Path | None = None,
              train_if_missing: bool = True,
              epochs: int = 20,
              lr: float = 0.01,
              device: str | torch.device | None = None):
    """High-level helper:
    - If weights exist, load and return.
    - If not and train_if_missing=True, train and save then return.
    """
    weights_path = Path(weights_path)
    try:
        return load_model(weights_path, device=device)
    except FileNotFoundError:
        if not train_if_missing:
            raise
        if data_root is None:
            raise ValueError("data_root must be provided to train the model when weights are missing.")
        from .train import train  # local import to keep import time small
        model = train(data_root, epochs=epochs, lr=lr, device=device)
        save_model(model, weights_path)
        return model
