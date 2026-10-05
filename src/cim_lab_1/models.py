from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """A simple fully connected classifier exposing a latent representation."""

    def __init__(self, hidden_dim: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            # TODO: add a layer mapping hidden_dim to latent_dim.
            nn.Linear(hidden_dim, latent_dim)
        )
        self.head = nn.Linear(latent_dim, 10)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        return self.head(z)


class CNNClassifier(nn.Module):
    """A compact CNN exposing a global latent representation."""

    def __init__(self, channels: list[int], latent_dim: int = 64) -> None:
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # TODO: add a second convolution, non-linearity and pooling operation.
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder = nn.Sequential(
            nn.Flatten(),
            # TODO: determine the flattened feature size after the feature extractor.
            nn.Linear(1568, latent_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(latent_dim, 10)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        return self.head(z)


def build_model(cfg) -> nn.Module:
    if cfg.name == "mlp":
        return MLPClassifier(cfg.hidden_dim, cfg.latent_dim)
    if cfg.name == "cnn":
        return CNNClassifier(list(cfg.channels), cfg.latent_dim)
    raise ValueError(f"Unknown model: {cfg.name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
