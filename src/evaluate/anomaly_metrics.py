from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(is_defect: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(is_defect, scores))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """Pools all pixels of all test images (the usual MVTec convention)."""
    return float(roc_auc_score(masks.ravel(), maps.ravel()))


def per_defect_auroc(defect_types: list[str], scores: np.ndarray) -> dict[str, float]:
    """AUROC of 'good' vs each defect type separately. Useful for the unseen-defect comparison."""
    d = np.array(defect_types)
    good = scores[d == "good"]
    out = {}
    for t in sorted(set(defect_types) - {"good"}):
        s = scores[d == t]
        out[t] = float(roc_auc_score(np.r_[np.zeros(len(good)), np.ones(len(s))], np.r_[good, s]))
    return out


def threshold_metrics(is_defect: np.ndarray, scores: np.ndarray, thr: float) -> dict[str, float]:
    pred = scores > thr
    tp = int((pred & is_defect).sum())
    fp = int((pred & ~is_defect).sum())
    fn = int((~pred & is_defect).sum())
    tn = int((~pred & ~is_defect).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0}


def tpr_at_fpr(is_defect: np.ndarray, scores: np.ndarray, fpr_target: float = 0.05) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(is_defect, scores)
    return float(tpr[np.searchsorted(fpr, fpr_target, side="right") - 1])


def paired_bootstrap_diff(is_defect, scores_a, scores_b, n: int = 1000, seed: int = 0):
    """AUROC(a) - AUROC(b) on the same images; stratified bootstrap. Returns (observed, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(is_defect)[0], np.where(~is_defect)[0]
    obs = roc_auc_score(is_defect, scores_a) - roc_auc_score(is_defect, scores_b)
    diffs = []
    for _ in range(n):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        y = is_defect[idx]
        diffs.append(roc_auc_score(y, scores_a[idx]) - roc_auc_score(y, scores_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(obs), float(lo), float(hi)
