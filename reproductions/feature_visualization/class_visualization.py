"""
Class visualization with Fourier parameterization (clean version)

What it does:
- Optimizes a frequency-domain tensor ("spec") so that, after inverse FFT,
  the resulting image strongly activates a chosen ImageNet class in GoogLeNet.
- Uses simple natural-image priors (TV + L2) and random-view augmentation
  for robustness.

Requirements:
  pip install torch torchvision pillow
"""

import math, random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image
from src.utils import class_index, init_spectrum,  spectrum_to_image, box_blur_3x3, random_view, total_variation, to_pil

# -------------------------------
# 0) Device & deterministic bits
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# -------------------------------
# 1) Load model & metadata
# -------------------------------
# We use GoogLeNet (a.k.a. Inception v1) pre-trained on ImageNet-1K.
WEIGHTS = models.GoogLeNet_Weights.IMAGENET1K_V1
NET = models.googlenet(weights=WEIGHTS, aux_logits=True).to(DEVICE).eval()

# Freeze model params (we optimize only the image).
for p in NET.parameters():
    p.requires_grad_(False)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE)[:, None, None]  # [3,1,1]
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE)[:, None, None]  # [3,1,1]
CATEGORIES = WEIGHTS.meta["categories"]  # length 1000, index -> class name


def synthesize_class(
    class_name: str = "dumbbell",
    *,
    steps: int = 700,
    n_views: int = 8,
    size: int = 384,
    lr: float = 0.08,
    tv_weight: float = 1e-4,
    l2_weight: float = 1e-6,
    decay_power: float = 1.5,
    blur_every: int = 60,
    print_every: int = 50
) -> Image.Image:
    """
    Optimize a Fourier spectrum to produce an image that the model thinks
    is the given ImageNet class.

    Core objective (maximize):
        class_score  - tv_weight * TV(img) - l2_weight * mean((img-0.5)^2)

    Returns: PIL.Image
    """
    target_idx = class_index(class_name)
    spec, freqs = init_spectrum(size, size, device=DEVICE)
    opt = torch.optim.Adam([spec], lr=lr)

    for t in range(steps):
        # 1) Build the current image from the spectrum
        img = spectrum_to_image(spec, freqs, decay_power=decay_power)  # [1,3,H,W]

        # Optional periodic blur for stability
        if blur_every and t > 0 and (t % blur_every == 0):
            img = box_blur_3x3(img)

        # 2) Robustness: evaluate mean class score over n random views
        # Each view: [1,3,224,224], batch becomes [n,3,224,224]
        batch = torch.cat([random_view(img) for _ in range(n_views)], dim=0)
        out = NET(batch)
        logits = out.logits if hasattr(out, "logits") else out  # handle aux-outputs
        class_score = logits[:, target_idx].mean()

        # 3) Priors
        tv = total_variation(img)
        l2 = ((img - 0.5) ** 2).mean()

        # 4) Maximize score with priors  ->  minimize negative
        objective = class_score - tv_weight * tv - l2_weight * l2
        loss = -objective

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # 5) Gentle spectrum damping to prevent runaway amplitudes
        with torch.no_grad():
            spec.mul_(0.995)

        if print_every and (t % print_every == 0 or t == steps - 1):
            print(f"[{t:04d}/{steps}] score={class_score.item():.3f}  TV={tv.item():.3e}  L2={l2.item():.3e}")

    final_img = spectrum_to_image(spec, freqs, decay_power=decay_power)
    return to_pil(final_img)


# -------------------------------
# 6) Example usage
# -------------------------------
if __name__ == "__main__":
    # Same spirit as your original example; uses a slightly bigger canvas and a lower decay.
    out = synthesize_class(
        "dumbbell",
        steps=1000,
        n_views=8,
        size=448,
        lr=0.01,
        decay_power=0.5,  # lower = more high-frequency detail allowed
        tv_weight=1e-4,
        l2_weight=1e-6,
        blur_every=60,
        print_every=50
    )
    out.save("dumbbell_classviz.png")
    print("Saved to dumbbell_classviz.png")
