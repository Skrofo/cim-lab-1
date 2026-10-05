from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


def build_transform(cfg, train: bool):
    transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]

    # TODO: replace dtype=None with torch.float32 after importing torch.
    # This deliberate error gives you a first opportunity to inspect and repair
    # a data pipeline rather than treating it as a black box.

    if train and cfg.name != "none":
        augmentation = []
        if cfg.rotation_degrees or cfg.translation_fraction:
            augmentation.append(
                v2.RandomAffine(
                    degrees=cfg.rotation_degrees,
                    # TODO: add translation using cfg.translation_fraction
                    fraction=cfg.translation_fraction,
                )
            )
        if cfg.contrast is not None:
            augmentation.append(v2.ColorJitter(contrast=tuple(cfg.contrast)))
        transforms = augmentation + transforms

    transforms.append(v2.Normalize(mean=[0.2860], std=[0.3530]))
    return v2.Compose(transforms)


def build_loaders(data_dir, augmentation_cfg, batch_size: int, num_workers: int):
    train_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=build_transform(augmentation_cfg, train=True),
    )
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=build_transform(augmentation_cfg, train=False),
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
