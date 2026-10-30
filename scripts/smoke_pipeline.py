"""End-to-end smoke test on synthetic data. Runs in CI; locally: `make smoke` (add --skip-torch without torch).

It checks that every stage runs and produces its outputs (and that the baseline clears a sanity bar). It says
nothing about real-data quality: the synthetic data is deliberately easy. Stages needing torch: deep training,
ONNX export/quantisation, drift simulation on the exported model, benchmark, PatchCore, and the unseen-defect comparison that consumes PatchCore scores.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.neu_det import CLASSES  # noqa: E402


def make_neudet(root: Path, per_class: int = 60, noisy_per_class: int = 2, seed: int = 0) -> None:
    """Class-specific stripe texture + noise. A few images per class carry another class's texture (label noise)."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for ci, name in enumerate(CLASSES):
        for i in range(1, per_class + 1):
            tex = (ci + 1) % len(CLASSES) if i <= noisy_per_class else ci
            img = rng.normal(120, 40, (200, 200)).clip(0, 255)
            img[:, int(rng.integers(0, tex + 3))::tex + 3] = 255 - tex * 20
            cv2.imwrite(str(root / f"{name}_{i}.png"), img.astype(np.uint8))


def make_mvtec(root: Path, category: str = "widget", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    base = root / category
    types = {"scratch": 90, "dent": 50, "stain": -90}
    for d in ["train/good", "test/good", *[f"test/{t}" for t in types], *[f"ground_truth/{t}" for t in types]]:
        (base / d).mkdir(parents=True, exist_ok=True)

    def normal() -> np.ndarray:
        yy, xx = np.mgrid[0:256, 0:256]
        b = np.stack([100 + yy * 0.1, 130 + xx * 0.05, 90 + 0 * yy], -1)
        return (b + rng.normal(0, 6, b.shape)).clip(0, 255).astype(np.uint8)

    for i in range(20):
        cv2.imwrite(str(base / f"train/good/{i:03d}.png"), normal())
    for i in range(10):
        cv2.imwrite(str(base / f"test/good/{i:03d}.png"), normal())
    for t, boost in types.items():
        for i in range(8):
            img, mask = normal(), np.zeros((256, 256), np.uint8)
            y, x = (int(v) for v in rng.integers(50, 180, 2))
            img[y:y + 24, x:x + 24] = np.clip(img[y:y + 24, x:x + 24].astype(int) + boost, 0, 255)
            mask[y:y + 24, x:x + 24] = 255
            cv2.imwrite(str(base / f"test/{t}/{i:03d}.png"), img)
            cv2.imwrite(str(base / f"ground_truth/{t}/{i:03d}_mask.png"), mask)


def load_cfg(name: str, **overrides) -> dict:
    cfg = yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="smoke")
    ap.add_argument("--models-out", help="where to write the ONNX bundle (default <workdir>/models)")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--min-baseline-f1", type=float, default=0.85)
    a = ap.parse_args()

    work = Path(a.workdir).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    models_out = Path(a.models_out).resolve() if a.models_out else work / "models"
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    mlflow = {"tracking_uri": f"sqlite:///{work / 'mlflow.db'}", "experiment": "smoke"}

    def step(title: str, *args: str) -> None:
        print(f"\n=== {title}", flush=True)
        subprocess.run([sys.executable, "-m", *args], cwd=work, env=env, check=True)

    def need(path: Path) -> Path:
        assert path.exists(), f"expected output missing: {path}"
        return path

    def write_cfg(name: str, cfg: dict) -> str:
        p = work / name
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return str(p)

    make_neudet(work / "data" / "NEU-DET")
    make_mvtec(work / "data" / "mvtec")
    sample = work / "data" / "NEU-DET" / "scratches_30.png"
    shutil.copy(sample, work / "sample.png")

    base_cfg = write_cfg("baseline.yaml", load_cfg("baseline.yaml", data_dir=str(work / "data" / "NEU-DET"),
                                                   cache_dir=str(work / "cache"), mlflow=mlflow))
    deep_cfg = write_cfg("deep.yaml", load_cfg(
        "deep.yaml", data_dir=str(work / "data" / "NEU-DET"), mlflow=mlflow,
        model={"pretrained": False}, train={"epochs": 2, "batch_size": 32, "num_workers": 0, "aug": "flip", "patience": 5}))

    # ---- torch-free stages ----
    step("baseline: handcrafted features + LogReg/LightGBM", "src.train.train_baseline", "--config", base_cfg)
    report = json.loads(need(work / "artifacts" / "report_lightgbm_val.json").read_text(encoding="utf-8"))
    f1 = report["macro avg"]["f1-score"]
    print(f"baseline LightGBM val macro-F1 = {f1:.3f} (gate {a.min_baseline_f1})")
    assert f1 >= a.min_baseline_f1, f"baseline macro-F1 {f1:.3f} below sanity bar {a.min_baseline_f1}"

    step("failure analysis", "src.evaluate.failure_analysis", "analyze", "--pred",
         "artifacts/predictions_lightgbm_val.csv", "--out", "reports")
    need(work / "reports" / "error_review.csv")

    step("label-noise candidates (cleanlab)", "src.evaluate.label_noise", "--config", base_cfg, "--out", "reports")
    shutil.copy(need(work / "reports" / "label_issues.csv"), work / "reports" / "label_issues_confirmed.csv")
    step("baseline retrained without confirmed issues", "src.train.train_baseline", "--config", base_cfg,
         "--exclude-train", "reports/label_issues_confirmed.csv")
    need(work / "artifacts" / "predictions_lightgbm_val_cleaned.csv")

    if a.skip_torch:
        print("\n--skip-torch: skipping deep training, ONNX export, serving, benchmark, PatchCore, comparison")
        print("\nSMOKE OK (torch-free stages)")
        return

    # ---- torch stages ----
    step("deep model: ResNet fine-tuning (2 epochs, random init)", "src.train.train_deep", "train", "--config", deep_cfg)
    ckpt = need(work / "artifacts" / "resnet_best.pt")
    need(work / "artifacts" / "predictions_deep_val.csv")

    step("export to ONNX + int8 (with PyTorch parity check)", "src.deploy.export_onnx", "--ckpt", str(ckpt),
         "--out-root", str(models_out), "--config", deep_cfg, "--calib-n", "32", "--set-current")
    bundle = models_out / "current"
    for f in ("model.onnx", "model_int8.onnx", "meta.json"):
        need(bundle / f)

    print("\n=== serving: in-process API check on both graphs", flush=True)
    from fastapi.testclient import TestClient

    from src.serve.app import create_app

    for model_file in ("model.onnx", "model_int8.onnx"):
        with TestClient(create_app(str(bundle), model_file=model_file)) as c:
            r = c.post("/predict", files={"file": ("s.png", (work / "sample.png").read_bytes(), "image/png")})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["label"] in CLASSES and abs(sum(body["probabilities"].values()) - 1) < 1e-4
            print(f"{model_file}: {body['label']} ({body['confidence']:.2f})")

    step("benchmark: PyTorch vs ONNX fp32 vs int8", "src.deploy.benchmark", "--model-dir", str(bundle), "--ckpt",
         str(ckpt), "--config", deep_cfg, "--n", "20", "--out", "reports/benchmark.md")
    assert need(work / "reports" / "benchmark.md").read_text(encoding="utf-8").count("onnxruntime") >= 3

    # tiny windows in CI (about 27 images): lower min_samples; only the strong, unambiguous effects are asserted
    mon_cfg = write_cfg("monitoring.yaml", load_cfg("monitoring.yaml", min_samples=20, html_reports=[]))
    step("drift: reference for the exported model", "src.monitoring.drift", "build-reference", "--model-dir", str(bundle),
         "--config", deep_cfg, "--monitoring-config", mon_cfg)
    need(bundle / "reference.csv")
    step("drift: simulation through the service", "src.monitoring.simulate", "--model-dir", str(bundle), "--config",
         deep_cfg, "--monitoring-config", mon_cfg, "--out", "reports/drift")
    sim = {(r["scenario"], r["severity"]): r
           for r in json.loads(need(work / "reports" / "drift" / "simulation.json").read_text(encoding="utf-8"))}
    assert sim[("control", 0)]["alert"] is False, "monitor raised a false alarm on clean data"
    assert sim[("blur", 3)]["alert"] and sim[("brightness", 3)]["alert"], "monitor missed a severe corruption"

    pc_cfg = write_cfg("patchcore.yaml", load_cfg(
        "patchcore.yaml", data_root=str(work / "data" / "mvtec"), categories=["widget"],
        out_dir=str(work / "artifacts" / "patchcore"), backbone="resnet18", pretrained=False, batch_size=8,
        coreset_ratio=0.05, mlflow=mlflow))
    step("PatchCore on synthetic MVTec-style data (resnet18, random init)", "src.train.train_patchcore", "--config", pc_cfg)
    res = json.loads(need(work / "artifacts" / "patchcore" / "widget" / "results.json").read_text(encoding="utf-8"))
    assert all(np.isfinite(res[k]) for k in ("image_auroc", "pixel_auroc")), res
    print(f"PatchCore image AUROC {res['image_auroc']:.3f}, pixel AUROC {res['pixel_auroc']:.3f} (random weights: not gated)")

    cmp_cfg = write_cfg("comparison.yaml", load_cfg(
        "comparison.yaml", data_root=str(work / "data" / "mvtec"), categories=["widget"],
        patchcore_dir=str(work / "artifacts" / "patchcore"), out_dir=str(work / "artifacts" / "comparison"),
        supervised={"models": ["handcrafted_lgbm"], "seeds": [0]}, bootstrap={"n": 100}, mlflow=mlflow))
    step("unseen-defect comparison (handcrafted supervised baseline)", "src.evaluate.unseen_defect_comparison",
         "--config", cmp_cfg, "--models", "handcrafted_lgbm")
    need(work / "artifacts" / "comparison" / "results.md")

    print(f"\nSMOKE OK. ONNX bundle: {bundle}")


if __name__ == "__main__":
    main()
