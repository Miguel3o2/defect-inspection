"""Simulated production degradations on uint8 grayscale images."""
from __future__ import annotations

import cv2
import numpy as np


def blur(img: np.ndarray, sigma: float, rng=None) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def brightness(img: np.ndarray, shift: float, rng=None) -> np.ndarray:
    return np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)


def contrast(img: np.ndarray, factor: float, rng=None) -> np.ndarray:
    m = img.mean()
    return np.clip((img.astype(np.float32) - m) * factor + m, 0, 255).astype(np.uint8)


def noise(img: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


CORRUPTIONS = {"blur": blur, "brightness": brightness, "contrast": contrast, "noise": noise}


def apply(name: str, img: np.ndarray, value: float, rng: np.random.Generator | None = None) -> np.ndarray:
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; choose from {list(CORRUPTIONS)}")
    return CORRUPTIONS[name](img, value, rng)
