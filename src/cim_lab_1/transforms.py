from __future__ import annotations

import torch
from torchvision.transforms import v2


def translate(images: torch.Tensor, pixels: int) -> torch.Tensor:
    return v2.functional.affine(
        images,
        angle=0.0,
        translate=[pixels, 0],
        scale=1.0,
        shear=[0.0, 0.0],
    )


def rotate(images: torch.Tensor, degrees: float) -> torch.Tensor:
    return v2.functional.rotate(images, angle=degrees)


def scale_contrast(images: torch.Tensor, factor: float) -> torch.Tensor:
    # Applied around each image mean to make the operation explicit.
    mean = images.mean(dim=(-2, -1), keepdim=True)
    return (mean + factor * (images - mean)).clamp(images.amin(), images.amax())
