"""Confident-learning label-issue detection (cleanlab) on out-of-fold predictions.

Uses train+val only; the test split is never touched. Flags are CANDIDATES: look at them,
delete rows you disagree with, save as reports/label_issues_confirmed.csv, then retrain with
`train_baseline --exclude-train reports/label_issues_confirmed.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from cleanlab.filter import find_label_issues
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data.neu_det import CLASSES, discover_images
from src.data.splits import stratified_split
from src.evaluate.viz import montage
from src.train.train_baseline import build_features


def main(config_path: str, out: str, folds: int, top: int) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    paths, y = discover_images(cfg["data_dir"])
    tr, va, _te = stratified_split(y, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"])
    idx = np.concatenate([tr, va])
    X = build_features(paths, cfg)[idx]
    labels = y[idx]

    model = LGBMClassifier(class_weight="balanced", random_state=cfg["seed"], subsample_freq=1, verbose=-1,
                           **cfg["lightgbm"])
    oof = cross_val_predict(model, X, labels, method="predict_proba",
                            cv=StratifiedKFold(folds, shuffle=True, random_state=cfg["seed"]))
    issues = find_label_issues(labels, oof, return_indices_ranked_by="self_confidence", n_jobs=1)

    in_train = set(tr.tolist())
    df = pd.DataFrame({
        "path": [str(paths[idx[i]]) for i in issues],
        "given_label": [CLASSES[labels[i]] for i in issues],
        "suggested_label": [CLASSES[int(oof[i].argmax())] for i in issues],
        "self_confidence": [float(oof[i, labels[i]]) for i in issues],
        "split": ["train" if int(idx[i]) in in_train else "val" for i in issues],
    })
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "label_issues.csv", index=False)
    print(f"{len(df)} candidate issues out of {len(idx)} ({len(df) / len(idx):.1%}); "
          f"{(df['split'] == 'train').sum()} in train, {(df['split'] == 'val').sum()} in val")
    print(df.groupby(["given_label", "suggested_label"]).size().sort_values(ascending=False).head(8))
    top_df = df.head(top)
    montage([(r.path, f"given: {r.given_label}\nmodel: {r.suggested_label} ({r.self_confidence:.2f})")
             for r in top_df.itertuples()], f"Top {len(top_df)} label-issue candidates",
            str(outdir / "figures" / "label_issues_top.png"))
    print(f"Review {outdir / 'label_issues.csv'} and the montage; keep only rows you agree are mislabeled "
          f"-> {outdir / 'label_issues_confirmed.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top", type=int, default=24)
    a = ap.parse_args()
    main(a.config, a.out, a.folds, a.top)
