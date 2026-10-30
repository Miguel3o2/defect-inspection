"""Export the trained ResNet checkpoint to ONNX (+ optional int8) and write the serving bundle.

  python -m src.deploy.export_onnx --ckpt artifacts/resnet_best.pt [--set-current]

Output: models/<version>/{model.onnx, model_int8.onnx, meta.json}. <version> = first 8 hex of the checkpoint sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml

from src.data.neu_det import discover_images
from src.data.splits import stratified_split
from src.deploy.quantize import quantize_onnx
from src.features.handcrafted import load_gray
from src.serve.preprocess import to_tensor


def calibration_batch(cfg: dict, n: int) -> np.ndarray:
    """Preprocessed images from the TRAIN split only."""
    paths, y = discover_images(cfg["data_dir"])
    tr, _, _ = stratified_split(y, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"])
    pick = np.random.default_rng(0).choice(tr, size=min(n, len(tr)), replace=False)
    return np.stack([to_tensor(load_gray(paths[i], cfg["image_size"])) for i in pick])


def export(ckpt_path: str, out_root: str = "models", config: str = "configs/deep.yaml", opset: int = 17,
           quantize: bool = True, calib_n: int = 128, set_current: bool = False, tol: float = 1e-3) -> Path:
    import torch

    from src.models.classifier import build_classifier

    ck = torch.load(ckpt_path, map_location="cpu")
    classes, size = ck["classes"], ck["image_size"]
    net = build_classifier(ck["arch"], len(classes), pretrained=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    sha = hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest()
    version = sha[:8]
    out = Path(out_root) / version
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model.onnx"

    dummy = torch.randn(1, 3, size, size)
    kwargs = {"input_names": ["input"], "output_names": ["logits"], "opset_version": opset,
              "do_constant_folding": True, "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}}}
    try:
        torch.onnx.export(net, dummy, str(onnx_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(net, dummy, str(onnx_path), **kwargs)
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    x = torch.randn(4, 3, size, size)
    with torch.no_grad():
        ref = net(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    if diff > tol:
        raise RuntimeError(f"ONNX output differs from PyTorch by {diff:.2e} (> {tol})")
    print(f"exported {onnx_path} | max |torch - onnx| = {diff:.2e}")

    meta = {"model_version": version, "arch": ck["arch"], "classes": classes, "image_size": size,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "input_name": "input",
            "output_name": "logits", "opset": opset, "source_checkpoint_sha256": sha, "onnx_parity_max_diff": diff,
            "created": datetime.now(UTC).isoformat()}
    if quantize:
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        quantize_onnx(str(onnx_path), str(out / "model_int8.onnx"), calibration_batch(cfg, calib_n))
        meta["int8"] = {"file": "model_int8.onnx", "calibration_images": calib_n, "scheme": "static QDQ, per-channel"}
        print("wrote model_int8.onnx (check its accuracy with src.deploy.benchmark before using it)")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if set_current:
        cur = Path(out_root) / "current"
        shutil.rmtree(cur, ignore_errors=True)
        shutil.copytree(out, cur)
        print(f"copied to {cur}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/resnet_best.pt")
    ap.add_argument("--out-root", default="models")
    ap.add_argument("--config", default="configs/deep.yaml")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--calib-n", type=int, default=128)
    ap.add_argument("--set-current", action="store_true", help="also copy the bundle to models/current")
    a = ap.parse_args()
    export(a.ckpt, a.out_root, a.config, a.opset, not a.no_quantize, a.calib_n, a.set_current)
