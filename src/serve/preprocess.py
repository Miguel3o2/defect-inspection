"""Inference preprocessing. MUST match training (load_gray -> /255 -> 3ch -> ImageNet normalise)."""
from __future__ import annotations

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def decode_gray(data: bytes, size: int) -> np.ndarray:
    """Decode image bytes to uint8 [size, size] grayscale (same resize as training). Raises ValueError."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not decode image")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(gray: np.ndarray) -> np.ndarray:
    """uint8 [H, W] -> float32 [3, H, W]."""
    x = np.broadcast_to(gray.astype(np.float32) / 255.0, (3, *gray.shape))
    return ((x - MEAN) / STD).astype(np.float32)


def image_stats(gray: np.ndarray) -> dict:
    """Cheap input statistics, logged per request for drift monitoring."""
    return {"brightness": float(gray.mean()), "contrast": float(gray.std()),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())}
