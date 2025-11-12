from pathlib import Path
import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    """A tiny binary classifier matching the notebook's architecture.
    Three MaxPool2d layers followed by a linear head.
    Input is expected to be (N, 3, 224, 224).
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.feature_extractor2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.feature_extractor3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(3 * 28 * 28, 1)

    def forward(self, x):
        x = self.feature_extractor1(x)
        x = self.feature_extractor2(x)
        x = self.feature_extractor3(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def create_model(device: str | torch.device | None = None) -> nn.Module:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = BinaryClassifier().to(device)
    return model
