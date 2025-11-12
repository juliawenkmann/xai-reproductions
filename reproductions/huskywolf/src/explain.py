from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
try:
    from pytorch_grad_cam import AblationCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:  # pragma: no cover
    AblationCAM = None
    show_cam_on_image = None

from .io import load_model
from .data import preprocess, preprocess_unnormalized

def grad_cam_explain(x,
                     weights_path: str | Path = "model_weights/binary_classifier.pth",
                     target_layer: str = "feature_extractor3",
                     out_path: str | Path | None = None,
                     device: str | torch.device | None = None) -> Optional[Path]:
    """Create a Grad-CAM visualization over the input image.
    Returns the saved image path if out_path is provided, otherwise None.
    Requires pytorch-grad-cam to be installed.
    """
    if AblationCAM is None or show_cam_on_image is None:
        raise ImportError("pytorch-grad-cam is required: pip install grad-cam")

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = load_model(weights_path, device=device)

    # Resolve target layer by attribute name
    layer = getattr(model, target_layer, None)
    if layer is None:
        raise ValueError(f"Target layer '{target_layer}' not found on model.")

    img = Image.open(x).convert("RGB") if not isinstance(x, Image.Image) else x.convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    cam = AblationCAM(model=model, target_layers=[layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # (H, W)

    # Convert to 0..1 and overlay with a jet-style colormap for better visibility
    un = preprocess_unnormalized(img).detach().cpu().numpy().transpose(1, 2, 0)
    un = (un - un.min()) / (un.max() - un.min() + 1e-8)
    heat = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
    overlay = show_cam_on_image(un, heat, use_rgb=True)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(out_path)
        return out_path
    return None
