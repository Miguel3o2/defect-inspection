"""Latency / throughput / size / accuracy for PyTorch vs ONNX Runtime fp32 vs ONNX Runtime int8.

  python -m src.deploy.benchmark --model-dir models/<version> [--ckpt artifacts/resnet_best.pt] --threads 1

Accuracy is measured on the VALIDATION split so that quantisation damage is visible next to the speed-up.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from sklearn.metrics import accuracy_score, f1_score

from src.data.neu_det import discover_images
from src.data.splits import stratified_split
from src.features.handcrafted import load_gray
from src.serve.preprocess import to_tensor


def ort_backend(path: str, threads: int):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return lambda x: sess.run(None, {name: x})[0]


def torch_backend(ckpt: str, threads: int):
    import torch

    from src.models.classifier import build_classifier

    torch.set_num_threads(threads)
    ck = torch.load(ckpt, map_location="cpu")
    net = build_classifier(ck["arch"], len(ck["classes"]), pretrained=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    def fn(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.from_numpy(x)).numpy()

    return fn


def latency(fn, x1: np.ndarray, warmup: int = 20, n: int = 200) -> dict:
    for _ in range(warmup):
        fn(x1)
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn(x1)
        t.append((time.perf_counter() - s) * 1000)
    p50, p95, p99 = np.percentile(t, [50, 95, 99])
    return {"p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99), "mean_ms": float(np.mean(t))}


def throughput(fn, xb: np.ndarray, batches: int = 10) -> float:
    fn(xb)
    s = time.perf_counter()
    for _ in range(batches):
        fn(xb)
    return float(len(xb) * batches / (time.perf_counter() - s))


def predict_all(fn, X: np.ndarray, bs: int = 64) -> np.ndarray:
    return np.concatenate([fn(X[i:i + bs]) for i in range(0, len(X), bs)])


def run(backends: dict, X: np.ndarray, y: np.ndarray | None, sizes_mb: dict, n: int, batch: int) -> list[dict]:
    rows, ref_pred = [], None
    for name, fn in backends.items():
        row = {"backend": name, "size_mb": sizes_mb.get(name), **latency(fn, X[:1], n=n),
               "throughput_img_s": throughput(fn, np.resize(X, (batch, *X.shape[1:])).astype(np.float32))}
        if y is not None:
            pred = predict_all(fn, X).argmax(1)
            ref_pred = pred if ref_pred is None else ref_pred
            row.update(accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                       agreement_with_first=float((pred == ref_pred).mean()))
        rows.append(row)
    return rows


def to_markdown(rows: list[dict], meta: dict) -> str:
    has_acc = "accuracy" in rows[0]
    head = "| Backend | Size (MB) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (img/s) |" + (
        " Val acc | Val macro-F1 | Agreement |" if has_acc else "")
    lines = [head, "|---|---|---|---|---|---|" + ("---|---|---|" if has_acc else "")]
    for r in rows:
        line = (f"| {r['backend']} | {r['size_mb']:.1f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | "
                f"{r['throughput_img_s']:.1f} |")
        if has_acc:
            line += f" {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['agreement_with_first']:.3f} |"
        lines.append(line)
    lines.append(f"\n_{meta['cpu']} · {meta['cpu_count']} cores available · {meta['threads']} thread(s) · "
                 f"onnxruntime {meta['ort']} · latency = batch 1, throughput = batch {meta['batch']}_")
    return "\n".join(lines) + "\n"


def load_val(config: str):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    paths, y = discover_images(cfg["data_dir"])
    _, va, _ = stratified_split(y, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"])
    return np.stack([to_tensor(load_gray(paths[i], cfg["image_size"])) for i in va]), y[va]


def main(a) -> None:
    mdir = Path(a.model_dir)
    meta = json.loads((mdir / "meta.json").read_text(encoding="utf-8"))
    if a.skip_accuracy:
        X, y = np.random.default_rng(0).normal(size=(64, 3, meta["image_size"], meta["image_size"])).astype(np.float32), None
    else:
        X, y = load_val(a.config)
    backends, sizes = {}, {}
    if a.ckpt:
        backends["pytorch fp32"], sizes["pytorch fp32"] = torch_backend(a.ckpt, a.threads), Path(a.ckpt).stat().st_size / 1e6
    backends["onnxruntime fp32"] = ort_backend(str(mdir / "model.onnx"), a.threads)
    sizes["onnxruntime fp32"] = (mdir / "model.onnx").stat().st_size / 1e6
    if (mdir / "model_int8.onnx").exists():
        backends["onnxruntime int8"] = ort_backend(str(mdir / "model_int8.onnx"), a.threads)
        sizes["onnxruntime int8"] = (mdir / "model_int8.onnx").stat().st_size / 1e6
    rows = run(backends, X, y, sizes, a.n, a.batch)
    info = {"cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(), "threads": a.threads,
            "ort": ort.__version__, "batch": a.batch}
    md = to_markdown(rows, info)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md, encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(json.dumps({"env": info, "rows": rows}, indent=2), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/current")
    ap.add_argument("--ckpt", help="PyTorch checkpoint; adds the PyTorch row (needs torch)")
    ap.add_argument("--config", default="configs/deep.yaml")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--n", type=int, default=200, help="latency samples")
    ap.add_argument("--batch", type=int, default=32, help="throughput batch size")
    ap.add_argument("--skip-accuracy", action="store_true", help="speed only (random inputs); no dataset needed")
    ap.add_argument("--out", default="reports/benchmark.md")
    main(ap.parse_args())
