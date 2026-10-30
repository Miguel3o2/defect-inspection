"""Drift simulation: push clean and corrupted held-out images through the SERVICE, then ask the monitor.

For every scenario it records (a) whether the monitor alerts and (b) how much the model's macro-F1 actually drops,
because the interesting cases are the mismatches: alert without harm, harm without alert.

  python -m src.monitoring.simulate --model-dir models/current
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi.testclient import TestClient
from sklearn.metrics import f1_score

from src.data.neu_det import CLASSES
from src.features.handcrafted import load_gray
from src.monitoring.corruptions import apply
from src.monitoring.drift import check_drift, load_prediction_log, records_from_grays, val_halves
from src.serve.app import Model, create_app


def scenarios(mon: dict) -> list[tuple[str, int, float | None]]:
    out: list[tuple[str, int, float | None]] = [("control", 0, None)]
    for name, spec in mon["corruptions"].items():
        out += [(name, i + 1, v) for i, v in enumerate(spec["levels"])]
    return out


def run(model_dir: str, cfg: dict, mon: dict, out_dir: str, model_file: str = "model.onnx") -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = Model(Path(model_dir), model_file, 1)
    paths, y, ref_idx, prod_idx = val_halves(cfg, mon["seed"])
    ref = records_from_grays(model, [load_gray(paths[i], model.size) for i in ref_idx])
    prod_gray = [load_gray(paths[i], model.size) for i in prod_idx]
    y_prod = y[prod_idx]
    html_wanted = set(mon.get("html_reports", []))

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, (name, level, value) in enumerate(scenarios(mon)):
            rng = np.random.default_rng(mon["seed"] + k)
            grays = prod_gray if value is None else [apply(name, g, value, rng) for g in prod_gray]
            log = Path(tmp) / f"{name}_{level}.jsonl"
            preds = []
            with TestClient(create_app(model_dir, model_file, prediction_log=str(log))) as client:
                for g in grays:
                    ok, buf = cv2.imencode(".png", g)
                    r = client.post("/predict", files={"file": ("x.png", buf.tobytes(), "image/png")})
                    r.raise_for_status()
                    preds.append(CLASSES.index(r.json()["label"]) if r.json()["label"] in CLASSES else -1)
            tag = "control" if value is None else f"{name}:{level}"
            res = check_drift(ref, load_prediction_log(str(log)), mon["columns"], mon["alpha"], mon["min_samples"],
                              str(out / f"{tag.replace(':', '_')}.html") if tag in html_wanted else None)
            f1 = float(f1_score(y_prod, preds, average="macro", labels=list(range(len(CLASSES)))))
            rows.append({"scenario": name, "severity": level, "value": value, "alert": res["alert"],
                         "drifted_columns": res["drifted_columns"],
                         "min_p": min(v["p_value"] for v in res["per_column"].values()), "macro_f1": f1,
                         "n_current": res["n_current"]})
            print(f"{tag:14s} alert={res['alert']!s:5s} drifted={res['drifted_columns']} F1={f1:.3f}", flush=True)
    base = rows[0]["macro_f1"]
    for r in rows:
        r["f1_drop"] = base - r["macro_f1"]
    (out / "simulation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def to_markdown(rows: list[dict], harm_threshold: float = 0.05) -> str:
    lines = ["| Scenario | Severity | Monitor | Drifted columns | min p | Macro-F1 | ΔF1 vs control | Verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        harmed = r["f1_drop"] > harm_threshold
        if r["scenario"] == "control":
            verdict = "false alarm" if r["alert"] else "ok (no false alarm)"
        elif r["alert"] and harmed:
            verdict = "detected, harmful"
        elif r["alert"]:
            verdict = "detected, benign (model robust)"
        elif harmed:
            verdict = "MISSED, harmful"
        else:
            verdict = "undetected, benign"
        sev = "-" if r["scenario"] == "control" else f"{r['severity']} ({r['value']})"
        lines.append(f"| {r['scenario']} | {sev} | {'ALERT' if r['alert'] else 'ok'} | "
                     f"{', '.join(r['drifted_columns']) or '-'} | {r['min_p']:.1e} | {r['macro_f1']:.3f} | "
                     f"{-r['f1_drop']:+.3f} | {verdict} |")
    n = rows[0]["n_current"]
    lines.append(f"\n_{n} images per window, reference = other half of the validation split. 'harmful' = macro-F1 "
                 f"drop > {harm_threshold}; with this few images small F1 differences are noise._")
    return "\n".join(lines) + "\n"


def print_console_safe(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/current")
    ap.add_argument("--model-file", default="model.onnx")
    ap.add_argument("--config", default="configs/deep.yaml")
    ap.add_argument("--monitoring-config", default="configs/monitoring.yaml")
    ap.add_argument("--out", default="reports/drift")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    mon = yaml.safe_load(Path(a.monitoring_config).read_text(encoding="utf-8"))
    rows = run(a.model_dir, cfg, mon, a.out, a.model_file)
    md = to_markdown(rows)
    (Path(a.out) / "drift_simulation.md").write_text(md, encoding="utf-8")
    print_console_safe("\n" + md)
    return rows


if __name__ == "__main__":
    main()
