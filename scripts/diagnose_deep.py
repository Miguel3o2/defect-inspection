from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.leakage import cross_split_duplicates, phash_all
from src.data.neu_det import discover_images
from src.data.splits import stratified_split


def leakage_report(cfg: dict) -> dict:
    paths, y = discover_images(cfg["data_dir"])
    tr, va, te = stratified_split(y, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"])
    splits = {"val": va, "test": te}
    stats: dict = {"n_train": len(tr), "n_val": len(va), "n_test": len(te)}

    md5 = [hashlib.md5(p.read_bytes()).hexdigest() for p in paths]
    train_md5 = {md5[i] for i in tr}
    hashes = phash_all(paths)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} images")
    for name, idx in splits.items():
        exact = sum(md5[i] in train_md5 for i in idx)
        hits = cross_split_duplicates([hashes[i] for i in idx], [hashes[i] for i in tr], 8)
        stats[f"{name}_exact"] = exact
        print(f"\n{name} vs train:")
        print(f"  byte-identical to a train image: {exact} images")
        for d in (0, 2, 4, 8):
            near = [h for h in hits if h[2] <= d]
            stats[f"{name}_phash_{d}"] = len({a for a, _, _ in near})
            print(f"  perceptual-hash distance <= {d}: {len(near)} pairs, {stats[f'{name}_phash_{d}']} distinct {name} images")
    leak = stats["val_exact"] > 0 or stats["val_phash_4"] > 0
    print("\nVERDICT: " + ("val images have (near-)duplicates in train: LEAKAGE LIKELY, tell Claude these numbers"
                           if leak else "no exact or near-duplicate (distance <= 4) val-train pairs found"))
    print("(distance <= 4 means practically the same image; 8 is loosely similar, expect some on textured surfaces)")
    return stats


def training_curve(cfg: dict, experiment: str) -> None:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        runs = client.search_runs([exp.experiment_id], order_by=["attribute.start_time DESC"], max_results=1)
        run = runs[0]
    except Exception as e:
        print(f"\nTraining curve: no MLflow run found for experiment {experiment!r} ({type(e).__name__}: {e})")
        return
    print(f"\nTraining curve of the latest run '{run.info.run_name}':")
    print("  best_epoch:", run.data.metrics.get("best_epoch"))
    for m in ("train_loss", "val_macro_f1", "val_acc"):
        hist = [round(x.value, 3) for x in client.get_metric_history(run.info.run_id, m)]
        print(f"  {m}: {hist}")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deep.yaml")
    ap.add_argument("--experiment", default="neu-det-deep")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    stats = leakage_report(cfg)
    training_curve(cfg, a.experiment)
    return stats


if __name__ == "__main__":
    main()
