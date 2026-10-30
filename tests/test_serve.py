import json

import cv2
import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient
from helpers import CLASSES, SIZE, make_stub_onnx

from src.deploy.benchmark import latency, ort_backend, run, to_markdown
from src.deploy.quantize import quantize_onnx
from src.serve.app import create_app
from src.serve.preprocess import MEAN, STD, decode_gray, image_stats, to_tensor


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    make_stub_onnx(d / "model.onnx")
    (d / "meta.json").write_text(json.dumps({"model_version": "abc12345", "arch": "stub", "classes": CLASSES,
                                             "image_size": SIZE}))
    return d


def _png(value, size=64, noise=0.0, seed=0):
    img = np.full((size, size), value, np.float32) + np.random.default_rng(seed).normal(0, noise, (size, size))
    ok, buf = cv2.imencode(".png", img.clip(0, 255).astype(np.uint8))
    assert ok
    return buf.tobytes()


# ---------------- preprocessing ----------------
def test_preprocess_shape_dtype_and_formula():
    gray = np.full((SIZE, SIZE), 128, np.uint8)
    t = to_tensor(gray)
    assert t.shape == (3, SIZE, SIZE) and t.dtype == np.float32
    assert np.allclose(t[:, 0, 0], ((128 / 255.0 - MEAN) / STD).ravel(), atol=1e-6)
    assert decode_gray(_png(100), SIZE).shape == (SIZE, SIZE)
    with pytest.raises(ValueError):
        decode_gray(b"not an image", SIZE)
    assert set(image_stats(gray)) == {"brightness", "contrast", "sharpness"}


# ---------------- API ----------------
def test_health_model_and_predict(model_dir, tmp_path):
    log = tmp_path / "pred.jsonl"
    with TestClient(create_app(str(model_dir), prediction_log=str(log))) as c:
        assert c.get("/health").json() == {"status": "ok", "model_version": "abc12345"}
        assert c.get("/model").json()["classes"] == CLASSES
        dark = c.post("/predict", files={"file": ("a.png", _png(0), "image/png")}).json()
        bright = c.post("/predict", files={"file": ("b.png", _png(255), "image/png")}).json()
    assert dark["label"] == "crazing" and bright["label"] == "scratches"
    assert abs(sum(dark["probabilities"].values()) - 1) < 1e-5 and dark["model_version"] == "abc12345"
    assert 0 < dark["confidence"] <= 1 and dark["latency_ms"] >= 0
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 2 and {"ts", "label", "confidence", "brightness", "contrast", "sharpness"} <= set(rows[0])


def test_batch_endpoint_and_limits(model_dir):
    with TestClient(create_app(str(model_dir))) as c:
        files = [("files", (f"{i}.png", _png(v), "image/png")) for i, v in enumerate((0, 255))]
        out = c.post("/predict/batch", files=files).json()["predictions"]
        assert [p["label"] for p in out] == ["crazing", "scratches"]
        too_many = [("files", (f"{i}.png", _png(5), "image/png")) for i in range(17)]
        assert c.post("/predict/batch", files=too_many).status_code == 400


def test_bad_inputs_are_rejected(model_dir):
    with TestClient(create_app(str(model_dir), max_upload_bytes=2000)) as c:
        assert c.post("/predict", files={"file": ("x.png", b"garbage", "image/png")}).status_code == 400
        assert c.post("/predict", files={"file": ("x.png", b"", "image/png")}).status_code == 400
        big = _png(120, size=256, noise=60)
        assert len(big) > 2000
        assert c.post("/predict", files={"file": ("x.png", big, "image/png")}).status_code == 413
        assert c.post("/predict").status_code == 422


def test_service_matches_direct_onnx_inference(model_dir):
    data = _png(77, noise=20, seed=3)
    x = to_tensor(decode_gray(data, SIZE))[None]
    logits = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0][0]
    expected = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    with TestClient(create_app(str(model_dir))) as c:
        got = c.post("/predict", files={"file": ("a.png", data, "image/png")}).json()["probabilities"]
    assert np.allclose([got[k] for k in CLASSES], expected, atol=1e-5)


# ---------------- quantisation + benchmark ----------------
def test_quantize_and_benchmark(model_dir, tmp_path):
    rng = np.random.default_rng(0)
    calib = rng.normal(size=(32, 3, SIZE, SIZE)).astype(np.float32)
    int8 = tmp_path / "model_int8.onnx"
    quantize_onnx(str(model_dir / "model.onnx"), str(int8), calib)
    assert int8.exists()
    X = rng.normal(size=(40, 3, SIZE, SIZE)).astype(np.float32)
    y = ort_backend(str(model_dir / "model.onnx"), 1)(X).argmax(1)           # fp32 predictions as pseudo-labels
    backends = {"onnxruntime fp32": ort_backend(str(model_dir / "model.onnx"), 1),
                "onnxruntime int8": ort_backend(str(int8), 1)}
    rows = run(backends, X, y, {"onnxruntime fp32": 0.0, "onnxruntime int8": 0.0}, n=20, batch=8)
    assert rows[0]["accuracy"] == 1.0 and rows[1]["agreement_with_first"] > 0.7
    lat = latency(backends["onnxruntime fp32"], X[:1], warmup=2, n=30)
    assert lat["p50_ms"] <= lat["p95_ms"] <= lat["p99_ms"]
    md = to_markdown(rows, {"cpu": "test", "cpu_count": 1, "threads": 1, "ort": ort.__version__, "batch": 8})
    assert "onnxruntime int8" in md and "Val macro-F1" in md
