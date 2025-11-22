import os
import sys
from pathlib import Path
import torch
ROOT = "/Users/juliawenkmann/Documents/CodingProjects/damien/sophia_summit/sophia_summit_presentation/src/lucent"
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from lucent.optvis import render
from lucent.modelzoo import inceptionv1



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = inceptionv1.inceptionv1(pretrained=True)
#model = resnet50(pretrained=True)
#model = googlenet(pretrained=True)
_ = model.to(device).eval()

goal = "mixed4a:11"
_ = render.render_vis(model, goal, show_inline=True, save_image=f"Visualization_{goal}.pdf")



obj = "mixed4a:11"

pdf_out = "/Users/juliawenkmann/Documents/CodingProjects/damien/sophia_summit/sophia_summit_presentation/figures/feature_progression_mixed4a_11random_in_fourier.pdf"

imgs = render.render_feature_with_snapshots(
    model,
    obj,
    snapshot_steps=[0, 4, 48, 2048],
    img_size=224,
    device="auto",  # cuda > mps > cpu
    pdf_path=pdf_out,
    left_text="Starting from random noise in fourier space, we optimize an image to activate a particular neuron (layer mixed4a, unit 11).",
)
print("Saved PDF:", pdf_out)


pdf_out = "/Users/juliawenkmann/Documents/CodingProjects/damien/sophia_summit/sophia_summit_presentation/figures/feature_progression_mixed4a_11_random_in_pixel.pdf"

imgs = render.render_feature_with_snapshots(
    model,
    obj,
    snapshot_steps=[0, 4, 48, 2048],
    img_size=224,
    device="auto",  # cuda > mps > cpu
    pdf_path=pdf_out,
    left_text="Starting from random noise in pixel space, we optimize an image to activate a particular neuron (layer mixed4a, unit 11).",
    fft = False
)
print("Saved PDF:", pdf_out)