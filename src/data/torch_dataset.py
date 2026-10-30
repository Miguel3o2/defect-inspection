from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from src.features.handcrafted import load_gray


def load_images(paths, size: int) -> torch.Tensor:
    """Load all images once as a uint8 tensor [N, 1, H, W] (NEU-DET is small enough for RAM)."""
    arr = np.stack([load_gray(p, size) for p in paths])
    return torch.from_numpy(arr).unsqueeze(1)


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels, transform):
        self.images, self.labels, self.transform = images, torch.as_tensor(labels), transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.transform(tv_tensors.Image(self.images[i]))
        return x.as_subclass(torch.Tensor), int(self.labels[i])
