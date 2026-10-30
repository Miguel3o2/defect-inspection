"""Classical baseline: handcrafted features + LogReg / LightGBM on NEU-DET.

Models are compared on the validation split. The test split is only touched with --eval-test,
so you don't tune against it by accident.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.data.leakage import cross_split_duplicates, phash_all
from src.data.neu_det import CLASSES, discover_images
from src.data.splits import stratified_split
from src.evaluate.metrics import save_confusion_matrix, save_predictions, summarize
from src.features.handcrafted import extract_from_path
from src.models.baseline import build_models


def build_features(paths, cfg) -> np.ndarray:
    sig = json.dumps(cfg["features"], sort_keys=True) + str(cfg["image_size"]) + str(len(paths))
    cache = Path(cfg["cache_dir"]) / f"features_{hashlib.md5(sig.encode()).hexdigest()[:10]}.npy"
    if cache.exists():
        return np.load(cache)
    X = np.stack(
        Parallel(n_jobs=-1)(delayed(extract_from_path)(p, cfg["image_size"], cfg["features"]) for p in paths)
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    return X


def main(config_path: str, eval_test: bool, exclude_train: str | None = None) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    paths, y = discover_images(cfg["data_dir"])
    print(f"{len(paths)} images, class counts: {np.bincount(y, minlength=len(CLASSES)).tolist()}")

    tr, va, te = stratified_split(y, cfg["split"]["val"], cfg["split"]["test"], seed)

    n_dropped = 0
    if exclude_train:
        drop = set(pd.read_csv(exclude_train)["path"].astype(str))
        keep = np.array([str(paths[i]) not in drop for i in tr])
        n_dropped = int((~keep).sum())
        tr = tr[keep]
        print(f"Excluding {n_dropped} flagged samples from TRAIN only (val/test untouched)")
    suffix = "_cleaned" if n_dropped else ""

    hashes = phash_all(paths)
    max_d = cfg["leakage"]["phash_max_distance"]
    dup = cross_split_duplicates([hashes[i] for i in te], [hashes[i] for i in tr], max_d)
    print(f"Leakage check: {len(dup)} near-duplicate (test, train) pairs at phash distance <= {max_d}")

    X = build_features(paths, cfg)
    print(f"Feature matrix: {X.shape}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    for name, model in build_models(cfg, seed).items():
        with mlflow.start_run(run_name=f"{name}{suffix}"):
            mlflow.log_params({
                "model": name, "seed": seed, "n_features": X.shape[1], "n_train": len(tr), "dropped_train_n": n_dropped,
                **{f"feat_{k}": str(v) for k, v in cfg["features"].items()},
                **{f"{name}_{k}": v for k, v in cfg[name].items()},
            })
            mlflow.log_metric("leakage_pairs_test_train", len(dup))
            model.fit(X[tr], y[tr])

            splits = {"val": va, **({"test": te} if eval_test else {})}
            for split_name, idx in splits.items():
                probs = model.predict_proba(X[idx])
                pred = probs.argmax(1)
                s = summarize(y[idx], pred, CLASSES)
                mlflow.log_metrics({f"{split_name}_accuracy": s["accuracy"], f"{split_name}_macro_f1": s["macro_f1"]})
                cm_path = out_dir / f"cm_{name}_{split_name}{suffix}.png"
                save_confusion_matrix(y[idx], pred, CLASSES, str(cm_path))
                mlflow.log_artifact(str(cm_path))
                (out_dir / f"report_{name}_{split_name}{suffix}.json").write_text(json.dumps(s["report"], indent=2), encoding="utf-8")
                save_predictions([paths[i] for i in idx], y[idx], probs, CLASSES,
                                 out_dir / f"predictions_{name}_{split_name}{suffix}.csv")
                print(f"[{name}] {split_name}: acc={s['accuracy']:.3f} macro_f1={s['macro_f1']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--eval-test", action="store_true",
                    help="also evaluate on the held-out test split (do this once, at the end)")
    ap.add_argument("--exclude-train", help="CSV with a `path` column; those samples are dropped from train only")
    a = ap.parse_args()
    main(a.config, a.eval_test, a.exclude_train)
