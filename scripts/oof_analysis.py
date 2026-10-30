"""5-fold out-of-fold LightGBM predictions on train+val (the test split is never touched).

Gives (a) a mean +/- std for the baseline instead of one noisy split and (b) enough errors for a failure analysis.
Run from the project root:   python scripts/oof_analysis.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.data.neu_det import CLASSES, discover_images  # noqa: E402
from src.data.splits import stratified_split  # noqa: E402
from src.evaluate.metrics import save_predictions  # noqa: E402
from src.train.train_baseline import build_features  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="artifacts/predictions_lightgbm_oof.csv")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))

    paths, y_all = discover_images(cfg["data_dir"])
    tr, va, _test = stratified_split(y_all, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"])
    idx = np.concatenate([tr, va])
    X, y = build_features(paths, cfg)[idx], y_all[idx]
    print(f"{len(idx)} images (train+val; test excluded), {a.folds}-fold stratified CV")

    oof = np.zeros((len(idx), len(CLASSES)))
    fold_f1 = []
    for k, (fit, hold) in enumerate(StratifiedKFold(a.folds, shuffle=True, random_state=cfg["seed"]).split(X, y)):
        model = LGBMClassifier(class_weight="balanced", random_state=cfg["seed"], subsample_freq=1, verbose=-1,
                               **cfg["lightgbm"])
        model.fit(X[fit], y[fit])
        oof[hold] = model.predict_proba(X[hold])
        fold_f1.append(f1_score(y[hold], oof[hold].argmax(1), average="macro"))
        print(f"  fold {k + 1}: macro-F1 {fold_f1[-1]:.3f}", flush=True)

    pred = oof.argmax(1)
    errors = int((pred != y).sum())
    lo, hi = wilson(len(y) - errors, len(y))
    print(f"\nLightGBM 5-fold macro-F1: {np.mean(fold_f1):.3f} +/- {np.std(fold_f1):.3f} (std across folds)")
    print(f"pooled accuracy {1 - errors / len(y):.3f}  (95% CI {lo:.3f} to {hi:.3f}), {errors} errors in {len(y)} images")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_predictions([paths[i] for i in idx], y, oof, CLASSES, out)
    print(f"wrote {out}")
    return {"mean_f1": float(np.mean(fold_f1)), "std_f1": float(np.std(fold_f1)), "errors": errors, "n": len(y)}


if __name__ == "__main__":
    main()
