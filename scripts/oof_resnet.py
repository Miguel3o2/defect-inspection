"""5-fold out-of-fold ResNet predictions on the SAME folds as scripts/oof_analysis.py (train+val only, test untouched),
then a paired comparison with LightGBM and a list of errors both models make.

Early stopping uses an inner validation set carved out of each fold's TRAINING part, never the held-out fold.
Run from the project root, after scripts/oof_analysis.py:   python scripts/oof_resnet.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.neu_det import CLASSES, discover_images
from src.data.splits import stratified_split
from src.evaluate.metrics import save_predictions


def mcnemar_exact(a_wrong: np.ndarray, b_wrong: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test. Returns (only A wrong, only B wrong, p-value)."""
    only_a, only_b = int((a_wrong & ~b_wrong).sum()), int((~a_wrong & b_wrong).sum())
    n = only_a + only_b
    return only_a, only_b, (float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0)


def make_torch_train_fn(cfg: dict, paths: list):
    """Returns fn(fit_idx, inner_idx, hold_idx, y, seed) -> class probabilities for hold_idx (global indices)."""
    cache: dict = {}

    def fn(fit_idx, inner_idx, hold_idx, y, seed):
        from src.data.torch_dataset import load_images
        from src.train.engine import evaluate_split, fit

        if "images" not in cache:
            cache["images"] = load_images(paths, cfg["image_size"])
        res = fit(cfg, cache["images"], y, fit_idx, inner_idx, len(CLASSES), train_seed=seed)
        print(f"    best inner-val macro-F1 {res['best_f1']:.3f} at epoch {res['best_epoch'] + 1}", flush=True)
        return evaluate_split(res["model"], cache["images"], y, hold_idx)

    return fn


def compare(resnet: pd.DataFrame, lgbm_csv: str, out_dir: Path) -> dict:
    lg = pd.read_csv(lgbm_csv)
    m = resnet.merge(lg, on="path", suffixes=("_r", "_l"))
    if len(m) != len(resnet):
        print(f"WARNING: only {len(m)} of {len(resnet)} images matched the LightGBM file (different split/data?)")
    r_wrong, l_wrong = (m["true_r"] != m["pred_r"]).to_numpy(), (m["true_l"] != m["pred_l"]).to_numpy()
    only_l, only_r, p = mcnemar_exact(l_wrong, r_wrong)
    print(f"\nPaired comparison on {len(m)} images: LightGBM wrong {int(l_wrong.sum())}, ResNet wrong {int(r_wrong.sum())}")
    print(f"  only LightGBM wrong: {only_l} | only ResNet wrong: {only_r} | both wrong: {int((l_wrong & r_wrong).sum())}")
    print(f"  exact McNemar p = {p:.4f}")
    both = m[l_wrong & r_wrong & (m["confidence_r"] >= 0.9)].copy()
    both["conf_min"] = both[["confidence_r", "confidence_l"]].min(axis=1)
    both = both.sort_values("conf_min", ascending=False)
    cols = {"path": "path", "true_r": "label", "pred_l": "lightgbm_pred", "pred_r": "resnet_pred",
            "confidence_l": "lightgbm_conf", "confidence_r": "resnet_conf"}
    out = out_dir / "consensus_errors.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    both[list(cols)].rename(columns=cols).to_csv(out, index=False)
    print(f"  errors BOTH models make, with ResNet confidence >= 0.9: {len(both)} -> {out}")
    print("  (these are the strongest label-noise candidates: two different models reject the label; look at them)")
    return {"only_lgbm_wrong": only_l, "only_resnet_wrong": only_r, "p": p, "consensus": len(both)}


def run(cfg: dict, folds: int, out: str, lgbm_csv: str | None, train_fn=None, seed: int = 0) -> dict:
    paths, y_all = discover_images(cfg["data_dir"])
    tr, va, _test = stratified_split(y_all, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"])
    idx = np.concatenate([tr, va])
    y = y_all[idx]
    train_fn = train_fn or make_torch_train_fn(cfg, paths)
    print(f"{len(idx)} images (train+val; test excluded), {folds}-fold stratified CV, same folds as oof_analysis.py")

    oof = np.zeros((len(idx), len(CLASSES)))
    fold_f1 = []
    for k, (fit_local, hold_local) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=cfg["seed"]).split(idx, y)):
        fit_g, hold_g = idx[fit_local], idx[hold_local]
        fit_part, inner = train_test_split(fit_g, test_size=0.1, stratify=y_all[fit_g], random_state=seed + k)
        print(f"  fold {k + 1}/{folds}: train {len(fit_part)}, inner-val {len(inner)}, held-out {len(hold_g)}", flush=True)
        oof[hold_local] = train_fn(fit_part, inner, hold_g, y_all, seed + k)
        fold_f1.append(f1_score(y[hold_local], oof[hold_local].argmax(1), average="macro"))
        print(f"    held-out macro-F1 {fold_f1[-1]:.3f}", flush=True)

    errors = int((oof.argmax(1) != y).sum())
    print(f"\nResNet 5-fold macro-F1: {np.mean(fold_f1):.3f} +/- {np.std(fold_f1):.3f}; {errors} errors in {len(y)} images")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_predictions([paths[i] for i in idx], y, oof, CLASSES, out_path)
    print(f"wrote {out_path}")
    res = {"mean_f1": float(np.mean(fold_f1)), "std_f1": float(np.std(fold_f1)), "errors": errors}
    if lgbm_csv and Path(lgbm_csv).exists():
        res.update(compare(pd.read_csv(out_path), lgbm_csv, Path("reports") / "oof"))
    elif lgbm_csv:
        print(f"\n{lgbm_csv} not found: run scripts/oof_analysis.py first to get the paired comparison")
    return res


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deep.yaml")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="artifacts/predictions_resnet_oof.csv")
    ap.add_argument("--lgbm", default="artifacts/predictions_lightgbm_oof.csv")
    a = ap.parse_args(argv)
    return run(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")), a.folds, a.out, a.lgbm)


if __name__ == "__main__":
    main()
