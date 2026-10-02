from __future__ import annotations

import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == targets).float().mean()


def prediction_consistency(
    reference_logits: torch.Tensor, transformed_logits: torch.Tensor
) -> torch.Tensor:
    """Fraction retaining the reference model prediction."""
    # TODO: implement this metric.
    raise NotImplementedError


def representation_similarity(
    reference_features: torch.Tensor, transformed_features: torch.Tensor
) -> torch.Tensor:
    """Mean cosine similarity between paired latent vectors."""
    # TODO: implement this metric using F.cosine_similarity.
    raise NotImplementedError
