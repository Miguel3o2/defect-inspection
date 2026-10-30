"""Drift monitoring: reference vs a window of the prediction log, reported with Evidently.

  build-reference : reference.csv for a model bundle (input stats + prediction behaviour on held-out images)
  check           : compare the latest PREDICTION_LOG window to the reference; exit 1 on alert

Alert rule: any monitored column whose two-sample test (K-S for numeric, chi-square for categorical; both give
p-values) has p < alpha / n_columns. Input drift is NOT the same as the model getting worse: a drift alert says
"investigate", not "retrain". src.monitoring.simulate measures both side by side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.neu_det import discover_images
from src.data.splits import stratified_split
from src.features.handcrafted import load_gray
from src.serve.app import Model
from src.serve.preprocess import image_stats, to_tensor

os.environ.setdefault("EVIDENTLY_DISABLE_TELEMETRY", "1")
INPUT_COLS = ["brightness", "contrast", "sharpness"]


def records_from_grays(model: Model, grays: list[np.ndarray], batch_size: int = 64) -> pd.DataFrame:
    """Same columns the service logs per request: input stats + confidence + predicted label."""
    rows = []
    for s in range(0, len(grays), batch_size):
        chunk = grays[s:s + batch_size]
        probs = model.predict_proba(np.stack([to_tensor(g) for g in chunk]))
        for g, p in zip(chunk, probs, strict=True):
            rows.append({**image_stats(g), "confidence": float(p.max()), "label": model.classes[int(p.argmax())]})
    return pd.DataFrame(rows)


def load_prediction_log(path: str, model_version: str | None = None, last: int | None = None) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    if model_version is not None and "model_version" in df:
        df = df[df["model_version"] == model_version]
    return df.tail(last).reset_index(drop=True) if last else df.reset_index(drop=True)


def check_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], alpha: float = 0.01,
                min_samples: int = 100, html_path: str | None = None) -> dict:
    n_ref, n_cur = len(reference), len(current)
    base = {"n_reference": n_ref, "n_current": n_cur, "alpha": alpha}
    if n_cur < min_samples or n_ref < min_samples:
        return {**base, "alert": None, "reason": f"need >= {min_samples} samples (reference {n_ref}, current {n_cur})",
                "drifted_columns": [], "per_column": {}}

    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    thr = alpha / len(columns)
    num = [c for c in columns if c != "label"]
    cat = [c for c in columns if c == "label"]
    dd = DataDefinition(numerical_columns=num, categorical_columns=cat)
    ref_ds = Dataset.from_pandas(reference[columns].copy(), data_definition=dd)
    cur_ds = Dataset.from_pandas(current[columns].copy(), data_definition=dd)
    snap = Report([DataDriftPreset(columns=columns, num_method="ks", cat_method="chisquare", threshold=thr)]).run(
        cur_ds, ref_ds)
    if html_path:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        snap.save_html(html_path)

    per_col = {}
    for m in snap.dict()["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type", "").endswith("ValueDrift"):
            if cfg["method"] not in ("ks", "chisquare"):
                raise RuntimeError(f"unexpected drift method {cfg['method']!r}")
            per_col[cfg["column"]] = {"p_value": float(m["value"]), "method": cfg["method"],
                                      "drifted": bool(m["value"] < thr)}
    if set(per_col) != set(columns):
        raise RuntimeError(f"Evidently output did not cover all columns: {sorted(per_col)} vs {columns}")
    drifted = [c for c, v in per_col.items() if v["drifted"]]
    return {**base, "alert": bool(drifted), "per_column_threshold": thr, "drifted_columns": drifted,
            "per_column": per_col}


def val_halves(cfg: dict, seed: int):
    """Validation split cut in two stratified halves: reference (A) and simulated production (B).
    Held-out data, so prediction confidence is not inflated the way it would be on training images."""
    paths, y = discover_images(cfg["data_dir"])
    _, va, _ = stratified_split(y, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"])
    ref_idx, prod_idx = train_test_split(va, test_size=0.5, stratify=y[va], random_state=seed)
    return paths, y, ref_idx, prod_idx


def cmd_build_reference(a) -> int:
    mdir = Path(a.model_dir)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    mon = yaml.safe_load(Path(a.monitoring_config).read_text(encoding="utf-8"))
    model = Model(mdir, a.model_file, 1)
    paths, _, ref_idx, _ = val_halves(cfg, mon["seed"])
    df = records_from_grays(model, [load_gray(paths[i], model.size) for i in ref_idx])
    df.to_csv(mdir / "reference.csv", index=False)
    (mdir / "reference_meta.json").write_text(json.dumps({
        "n": len(df), "source": "validation split, half A", "seed": mon["seed"], "model_version": model.version,
        "model_file": a.model_file}, indent=2), encoding="utf-8")
    print(f"wrote {mdir / 'reference.csv'} ({len(df)} rows) for model {model.version}")
    return 0


def cmd_check(a) -> int:
    mdir = Path(a.model_dir)
    mon = yaml.safe_load(Path(a.monitoring_config).read_text(encoding="utf-8"))
    meta = json.loads((mdir / "meta.json").read_text(encoding="utf-8"))
    ref = pd.read_csv(mdir / "reference.csv")
    cur = load_prediction_log(a.log, meta.get("model_version"), a.last)
    res = check_drift(ref, cur, mon["columns"], mon["alpha"], a.min_samples or mon["min_samples"], a.report)
    print(json.dumps(res, indent=2))
    if res["alert"] is None:
        print("NO VERDICT: " + res["reason"])
        return 0
    if res["alert"]:
        print(f"DRIFT ALERT on: {', '.join(res['drifted_columns'])}")
        return 1
    print("no drift detected")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("build-reference", cmd_build_reference), ("check", cmd_check)):
        sp = sub.add_parser(name)
        sp.add_argument("--model-dir", default="models/current")
        sp.add_argument("--monitoring-config", default="configs/monitoring.yaml")
        sp.set_defaults(fn=fn)
        if name == "build-reference":
            sp.add_argument("--config", default="configs/deep.yaml", help="data/split config (same as training)")
            sp.add_argument("--model-file", default="model.onnx")
        else:
            sp.add_argument("--log", required=True, help="PREDICTION_LOG jsonl written by the service")
            sp.add_argument("--last", type=int, default=500, help="use the most recent N predictions")
            sp.add_argument("--min-samples", type=int)
            sp.add_argument("--report", default="reports/drift/latest.html")
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
