import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    googlenet, GoogLeNet_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)
from PIL import Image
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------
# Build torchvision models
# -------------------------
def build_tv_models(device):
    specs = {}

    # ResNet50
    w_r50 = ResNet50_Weights.IMAGENET1K_V2
    m_r50 = resnet50(weights=w_r50).to(device).eval()
    specs["resnet50"] = {
        "model": m_r50,
        "preprocess": w_r50.transforms(),   # includes resize, crop, norm
        "labels": w_r50.meta["categories"],
    }

    # GoogLeNet (InceptionV1)
    w_g = GoogLeNet_Weights.IMAGENET1K_V1
    m_g = googlenet(weights=w_g).to(device).eval()
    specs["googlenet_tv"] = {
        "model": m_g,
        "preprocess": w_g.transforms(),
        "labels": w_g.meta["categories"],
    }

    # EfficientNet-B0
    w_e = EfficientNet_B0_Weights.IMAGENET1K_V1
    m_e = efficientnet_b0(weights=w_e).to(device).eval()
    specs["efficientnet_b0"] = {
        "model": m_e,
        "preprocess": w_e.transforms(),
        "labels": w_e.meta["categories"],
    }

    return specs

tv_models = build_tv_models(device)


def forward_logits(m, x):
    with torch.no_grad():
        out = m(x)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "logits"):  # e.g. some HF models
        return out.logits
    raise TypeError(f"Unexpected output type: {type(out)}")


def find_dumbbell_idx(labels):
    for i, name in enumerate(labels):
        if "dumbbell" in name.lower():
            return i
    raise ValueError("No 'dumbbell' label in this model's categories")


dumbbell_idx_by_model = {
    name: find_dumbbell_idx(spec["labels"])
    for name, spec in tv_models.items()
}

# -------------------------
# Compare with_arms vs without_arms
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "datasets" / "dumbbell_and_arms"
with_dir = DATA_ROOT / "with_arms"
without_dir = DATA_ROOT / "without_arms"

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

with_files = {
    p.name: p for p in with_dir.iterdir()
    if p.suffix.lower() in valid_exts
}
without_files = {
    p.name: p for p in without_dir.iterdir()
    if p.suffix.lower() in valid_exts
}

common_names = sorted(set(with_files.keys()) & set(without_files.keys()))

for name in common_names:
    print(f"\n================ {name} ================")
    pil_with = Image.open(with_files[name]).convert("RGB")
    pil_without = Image.open(without_files[name]).convert("RGB")

    for model_name, spec in tv_models.items():
        labels = spec["labels"]
        preprocess = spec["preprocess"]
        dumbbell_idx = dumbbell_idx_by_model[model_name]

        x_with = preprocess(pil_with).unsqueeze(0).to(device)
        x_without = preprocess(pil_without).unsqueeze(0).to(device)

        logits_with = forward_logits(spec["model"], x_with)
        logits_without = forward_logits(spec["model"], x_without)

        probs_with = F.softmax(logits_with, dim=1)
        probs_without = F.softmax(logits_without, dim=1)

        d_with = probs_with[0, dumbbell_idx].item()
        d_without = probs_without[0, dumbbell_idx].item()
        delta = d_without - d_with

        # top‑1 for readability
        top1_with_prob, top1_with_idx = probs_with.max(dim=1)
        top1_without_prob, top1_without_idx = probs_without.max(dim=1)

        print(f"\n[{model_name}]")
        print(f"  top1 WITH arms    : {labels[top1_with_idx.item()]}  "
              f"(prob={top1_with_prob.item():.3f})")
        print(f"  top1 WITHOUT arms : {labels[top1_without_idx.item()]}  "
              f"(prob={top1_without_prob.item():.3f})")

        print(f"  dumbbell prob WITH arms    : {d_with:.4f}")
        print(f"  dumbbell prob WITHOUT arms : {d_without:.4f}")
        print(f"  Δ dumbbell (without - with): {delta:+.4f}")
