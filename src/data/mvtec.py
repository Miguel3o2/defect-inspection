"""MVTec AD loading (torch-free). Layout: <root>/<category>/{train/good, test/<defect|good>, ground_truth/<defect>}."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


@dataclass
class MVTecCategory:
    name: str
    train: list[Path]
    test: list[Path]
    test_defect: list[str]            # "good" or the defect-type folder name
    test_mask: list[Path | None]      # None for good images


def discover(root: str | Path, category: str) -> MVTecCategory:
    base = Path(root) / category
    if not base.exists():
        raise FileNotFoundError(f"{base} not found. Download MVTec AD from mvtec.com into {root}.")
    train = sorted(p for p in (base / "train" / "good").glob("*") if p.suffix.lower() in IMG_EXT)
    test, defect, masks = [], [], []
    for d in sorted((base / "test").iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            test.append(p)
            defect.append(d.name)
            if d.name == "good":
                masks.append(None)
                continue
            m = base / "ground_truth" / d.name / f"{p.stem}_mask.png"
            if not m.exists():
                raise FileNotFoundError(f"missing ground-truth mask {m}")
            masks.append(m)
    if not train or not test:
        raise RuntimeError(f"{category}: found {len(train)} train and {len(test)} test images")
    return MVTecCategory(category, train, test, defect, masks)


def _crop(a: np.ndarray, crop: int) -> np.ndarray:
    top, left = (a.shape[0] - crop) // 2, (a.shape[1] - crop) // 2
    return a[top:top + crop, left:left + crop]


def load_display(path, resize: int = 256, crop: int = 224) -> np.ndarray:
    """uint8 RGB [crop, crop, 3], same geometry as the model input."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return _crop(img, crop)


def load_image(path, resize: int = 256, crop: int = 224) -> np.ndarray:
    """float32 [3, crop, crop], ImageNet-normalised."""
    x = load_display(path, resize, crop).astype(np.float32).transpose(2, 0, 1) / 255.0
    return (x - _MEAN) / _STD


def load_mask(path, resize: int = 256, crop: int = 224) -> np.ndarray:
    """bool [crop, crop]; all-False for good images (path=None)."""
    if path is None:
        return np.zeros((crop, crop), bool)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return _crop(cv2.resize(m, (resize, resize), interpolation=cv2.INTER_NEAREST), crop) > 127
