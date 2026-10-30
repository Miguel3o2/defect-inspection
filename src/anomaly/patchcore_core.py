"""PatchCore core: coreset subsampling, nearest-neighbour scoring, anomaly maps.

numpy by default; when torch reports a CUDA device, the distance-heavy parts run on the GPU with
the exact same algorithm (same RNG draws, same float32 ops), so results match the numpy path up
to float rounding. Tests and CI run the numpy path unchanged.

Simplifications vs the paper/official code (state them in your README):
  * image score = max patch distance, no neighbourhood re-weighting
  * exact brute-force kNN (no faiss)
"""
from __future__ import annotations

import cv2
import numpy as np


def _torch_cuda():
    """torch if a CUDA device is usable, else None (keeps this module importable without torch)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


def greedy_coreset(X: np.ndarray, n_select: int, proj_dim: int = 128, seed: int = 0) -> np.ndarray:
    """Approximate greedy k-center selection (indices into X). Distances use a random projection."""
    n = len(X)
    if n_select >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if X.shape[1] > proj_dim:
        P = (X @ rng.standard_normal((X.shape[1], proj_dim), dtype=np.float32) / np.sqrt(proj_dim)).astype(np.float32)
    else:
        P = X.astype(np.float32)

    torch = _torch_cuda()
    if torch is not None:
        Pt = torch.from_numpy(np.ascontiguousarray(P)).cuda()
        sqn = (Pt * Pt).sum(1)
        sel = np.empty(n_select, dtype=np.int64)
        sel[0] = int(rng.integers(n))
        min_d = sqn - 2.0 * (Pt @ Pt[sel[0]]) + sqn[sel[0]]
        for k in range(1, n_select):
            i = int(torch.argmax(min_d).item())
            sel[k] = i
            torch.minimum(min_d, sqn - 2.0 * (Pt @ Pt[i]) + sqn[i], out=min_d)
        return sel

    sqn = (P * P).sum(1)
    sel = np.empty(n_select, dtype=np.int64)
    sel[0] = int(rng.integers(n))
    min_d = sqn - 2.0 * (P @ P[sel[0]]) + sqn[sel[0]]
    for k in range(1, n_select):
        i = int(np.argmax(min_d))
        sel[k] = i
        np.minimum(min_d, sqn - 2.0 * (P @ P[i]) + sqn[i], out=min_d)
    return sel


def nn_distances(Q: np.ndarray, bank: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Euclidean distance from each row of Q to its nearest neighbour in bank."""
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    bank = np.ascontiguousarray(bank, dtype=np.float32)

    torch = _torch_cuda()
    if torch is not None:
        Qt, Bt = torch.from_numpy(Q).cuda(), torch.from_numpy(bank).cuda()
        bsq = (Bt * Bt).sum(1)
        out = np.empty(len(Q), np.float32)
        for s in range(0, len(Q), chunk):
            q = Qt[s:s + chunk]
            d2 = (q * q).sum(1, keepdims=True) - 2.0 * (q @ Bt.T) + bsq[None, :]
            out[s:s + chunk] = torch.sqrt(torch.clamp(d2.min(1).values, min=0.0)).cpu().numpy()
        return out

    bsq = (bank * bank).sum(1)
    out = np.empty(len(Q), np.float32)
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        d2 = (q * q).sum(1, keepdims=True) - 2.0 * (q @ bank.T) + bsq[None, :]
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def patch_scores_to_map(scores_hw: np.ndarray, out_size: int, sigma: float = 4.0) -> np.ndarray:
    m = cv2.resize(scores_hw.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(m, (0, 0), sigma) if sigma > 0 else m


def score_features(feats: np.ndarray, hw: tuple[int, int], bank: np.ndarray, out_size: int, sigma: float):
    """feats [B, h*w, D] -> (image_scores [B], maps [B, out, out])."""
    h, w = hw
    if _torch_cuda() is not None:
        # one kNN over all patches in the batch instead of a call per image
        d = nn_distances(feats.reshape(-1, feats.shape[-1]), bank).reshape(len(feats), h, w)
        img = d.max(axis=(1, 2)).astype(np.float32)
        maps = np.stack([patch_scores_to_map(dd, out_size, sigma) for dd in d])
        return img, maps
    img, maps = [], []
    for f in feats:
        d = nn_distances(f, bank).reshape(h, w)
        img.append(float(d.max()))
        maps.append(patch_scores_to_map(d, out_size, sigma))
    return np.array(img, np.float32), np.stack(maps)
