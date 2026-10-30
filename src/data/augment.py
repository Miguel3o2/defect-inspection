"""Augmentation policies for grayscale defect images (uint8 tensors, shape [1, H, W]).

Deliberately no RandomErasing/CutOut: erasing the defect would create label noise.
"""
from __future__ import annotations

import torch
from torchvision.transforms import v2

POLICIES = ("none", "flip", "geometric", "photometric", "full")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def to_three_channels(x: torch.Tensor) -> torch.Tensor:
    """Module-level (picklable) so DataLoader workers work under spawn on Windows/macOS."""
    return x.expand(3, -1, -1)


def _aug_ops(policy: str) -> list:
    if policy not in POLICIES:
        raise ValueError(f"unknown aug policy {policy!r}; choose from {POLICIES}")
    flip = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]
    geo = [v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1))]
    photo = [v2.ColorJitter(brightness=0.3, contrast=0.3)]
    return {
        "none": [],
        "flip": flip,
        "geometric": flip + geo,
        "photometric": flip + photo,
        "full": flip + geo + photo,
    }[policy]


def build_transform(policy: str, train: bool) -> v2.Compose:
    ops = _aug_ops(policy) if train else []
    return v2.Compose([
        *ops,
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(to_three_channels),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
