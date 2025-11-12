from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from .model import create_model
from .data import make_loaders

def train(data_root: str | Path,
          epochs: int = 20,
          lr: float = 0.01,
          device: str | torch.device | None = None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = create_model(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_loader, _ = make_loaders(data_root)
    model.train()
    for _ in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            out = torch.sigmoid(model(inputs)).squeeze(-1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
    return model
