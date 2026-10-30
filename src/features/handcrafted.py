"""Handcrafted texture/shape features for the classical baseline: HOG + LBP + GLCM + intensity stats."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def load_gray(path, size: int = 200) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def hog_features(img: np.ndarray, ppc: int = 32) -> np.ndarray:
    return hog(img, orientations=9, pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2), feature_vector=True)


def lbp_features(img: np.ndarray, configs=((8, 1), (16, 2))) -> np.ndarray:
    out = []
    for p, r in configs:
        lbp = local_binary_pattern(img, P=p, R=r, method="uniform")
        hist, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2), density=True)
        out.append(hist)
    return np.concatenate(out)


def glcm_features(img: np.ndarray, levels: int = 32, distances=(1, 3), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)) -> np.ndarray:
    q = (img.astype(np.float32) / 256 * levels).astype(np.uint8)
    glcm = graycomatrix(q, distances=list(distances), angles=list(angles), levels=levels, symmetric=True, normed=True)
    return np.concatenate([graycoprops(glcm, p).ravel() for p in GLCM_PROPS])


def intensity_stats(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    return np.array([x.mean(), x.std(), *np.percentile(x, [5, 25, 50, 75, 95])])


def extract(img: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or {}
    return np.concatenate([
        hog_features(img, cfg.get("hog_ppc", 32)),
        lbp_features(img, [tuple(c) for c in cfg.get("lbp", [(8, 1), (16, 2)])]),
        glcm_features(img, cfg.get("glcm_levels", 32)),
        intensity_stats(img),
    ]).astype(np.float32)


def extract_from_path(path, size: int = 200, cfg: dict | None = None) -> np.ndarray:
    return extract(load_gray(path, size), cfg)
