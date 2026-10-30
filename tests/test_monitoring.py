import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from helpers import CLASSES, make_texture_stub_onnx

from scripts.smoke_pipeline import make_neudet
from src.data.neu_det import discover_images
from src.features.handcrafted import load_gray
from src.monitoring import drift, simulate
from src.monitoring.corruptions import apply
from src.serve.app import Model, create_app
from src.serve.preprocess import to_tensor

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Synthetic NEU-DET + a brightness-texture stub classifier bundle (no torch needed)."""
    root = tmp_path_factory.mktemp("monitoring")
    data = root / "NEU-DET"
    make_neudet(data, per_class=240, noisy_per_class=0)
    paths, y = discover_images(data)
    centers = np.array([np.mean([to_tensor(load_gray(p, 200)).mean((1, 2)).sum()
                                 for p, lab in zip(paths, y, strict=True) if lab == c]) for c in range(6)])
    bundle = root / "bundle"
    bundle.mkdir()
    make_texture_stub_onnx(bundle / "model.onnx", centers)
    (bundle / "meta.json").write_text(json.dumps({"model_version": "stub0001", "arch": "stub", "classes": CLASSES,
                                                  "image_size": 200}))
    cfg = {"data_dir": str(data), "split": {"val": 0.15, "test": 0.15}, "seed": 42, "image_size": 200}
    mon = yaml.safe_load((REPO / "configs" / "monitoring.yaml").read_text())
    mon["html_reports"] = ["blur:3"]
    (root / "deep.yaml").write_text(yaml.safe_dump(cfg))
    return {"root": root, "bundle": bundle, "cfg": cfg, "mon": mon, "cfg_path": root / "deep.yaml"}


def _log_for(bundle, grays, log_path):
    with TestClient(create_app(str(bundle), prediction_log=str(log_path))) as c:
        for g in grays:
            c.post("/predict", files={"file": ("x.png", cv2.imencode(".png", g)[1].tobytes(), "image/png")})


def test_corruptions_have_the_intended_effect():
    def lap(a):
        return cv2.Laplacian(a, cv2.CV_64F).var()

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    assert lap(apply("blur", img, 2.0)) < lap(img) / 5
    assert apply("brightness", img, 40).mean() > img.mean() + 10
    assert apply("contrast", img, 0.5).std() < img.std() * 0.6
    assert apply("noise", np.full((64, 64), 128, np.uint8), 20, rng).std() > 10
    assert all(apply(n, img, v, rng).dtype == np.uint8 for n, v in
               (("blur", 1), ("brightness", 300), ("contrast", 3), ("noise", 99)))
    with pytest.raises(ValueError):
        apply("nope", img, 1)


def test_check_drift_decisions():
    rng = np.random.default_rng(0)

    def frame(n, shift=0.0):
        return pd.DataFrame({"brightness": rng.normal(120 + shift, 10, n), "sharpness": rng.normal(300, 50, n),
                             "label": rng.choice(list("abc"), n)})

    cols = ["brightness", "sharpness", "label"]
    same = drift.check_drift(frame(300), frame(300), cols, alpha=0.01)
    assert same["alert"] is False and same["per_column_threshold"] == pytest.approx(0.01 / 3)
    shifted = drift.check_drift(frame(300), frame(300, shift=10), cols, alpha=0.01)
    assert shifted["alert"] is True and shifted["drifted_columns"] == ["brightness"]
    tiny = drift.check_drift(frame(300), frame(20), cols, min_samples=100)
    assert tiny["alert"] is None and "need >=" in tiny["reason"]


def test_load_prediction_log_filters_version_and_tail(tmp_path):
    rows = [{"model_version": "a" if i < 6 else "b", "brightness": float(i)} for i in range(10)]
    f = tmp_path / "log.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(drift.load_prediction_log(str(f), "b")) == 4
    assert drift.load_prediction_log(str(f), "a", last=2)["brightness"].tolist() == [4.0, 5.0]


def test_reference_records_match_what_the_service_logs(world, tmp_path):
    paths, _ = discover_images(world["cfg"]["data_dir"])
    grays = [load_gray(p, 200) for p in paths[:15]]
    direct = drift.records_from_grays(Model(world["bundle"], "model.onnx", 1), grays)
    log = tmp_path / "p.jsonl"
    _log_for(world["bundle"], grays, log)
    logged = drift.load_prediction_log(str(log))
    for col in ("brightness", "contrast", "sharpness", "confidence"):
        assert np.allclose(direct[col], logged[col], rtol=1e-4, atol=1e-4), col
    assert direct["label"].tolist() == logged["label"].tolist()


def test_simulation_detects_drift_without_false_alarm(world, tmp_path):
    rows = simulate.main(["--model-dir", str(world["bundle"]), "--config", str(world["cfg_path"]),
                          "--out", str(tmp_path / "drift"), "--monitoring-config", str(_write(world, tmp_path))])
    by = {(r["scenario"], r["severity"]): r for r in rows}
    assert by[("control", 0)]["alert"] is False
    assert by[("blur", 3)]["alert"] and "sharpness" in by[("blur", 3)]["drifted_columns"]
    assert by[("brightness", 3)]["alert"] and "brightness" in by[("brightness", 3)]["drifted_columns"]
    assert by[("control", 0)]["macro_f1"] > 0.8
    assert by[("brightness", 3)]["f1_drop"] > 0.3
    assert by[("blur", 3)]["f1_drop"] < 0.05
    md = (tmp_path / "drift" / "drift_simulation.md").read_text()
    assert "detected, harmful" in md and "detected, benign" in md and "no false alarm" in md
    assert (tmp_path / "drift" / "blur_3.html").stat().st_size > 10_000


def _write(world, tmp_path):
    p = tmp_path / "mon.yaml"
    p.write_text(yaml.safe_dump(world["mon"]))
    return p


def test_check_cli_exit_codes(world, tmp_path):
    mon_path = _write(world, tmp_path)
    bundle = str(world["bundle"])
    assert drift.main(["build-reference", "--model-dir", bundle, "--config", str(world["cfg_path"]),
                       "--monitoring-config", str(mon_path)]) == 0
    assert (world["bundle"] / "reference.csv").exists()

    paths, y, _, prod_idx = drift.val_halves(world["cfg"], world["mon"]["seed"])
    grays = [load_gray(paths[i], 200) for i in prod_idx]
    rng = np.random.default_rng(0)
    clean, blurred = tmp_path / "clean.jsonl", tmp_path / "blur.jsonl"
    _log_for(world["bundle"], grays, clean)
    _log_for(world["bundle"], [apply("blur", g, 4.0, rng) for g in grays], blurred)
    small = tmp_path / "small.jsonl"
    small.write_text("\n".join(clean.read_text().splitlines()[:30]))

    def check(log):
        return drift.main(["check", "--model-dir", bundle, "--log", str(log), "--monitoring-config", str(mon_path),
                           "--report", str(tmp_path / "r.html")])

    assert check(clean) == 0
    assert check(blurred) == 1
    assert check(small) == 0
