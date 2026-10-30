"""FastAPI inference service for the ONNX defect classifier.

Config (env): MODEL_DIR (default models/current: model.onnx + meta.json), MODEL_FILE (default model.onnx; use
model_int8.onnx for the quantised graph), MAX_UPLOAD_BYTES, ORT_THREADS, PREDICTION_LOG (JSONL path, optional).
Rollback = point MODEL_DIR at an older version directory and restart.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.serve.preprocess import decode_gray, image_stats, to_tensor

MAX_BATCH = 16


class Prediction(BaseModel):
    label: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    latency_ms: float


class BatchPrediction(BaseModel):
    predictions: list[Prediction]


class Model:
    def __init__(self, model_dir: Path, model_file: str, threads: int):
        self.meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / model_file), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.classes: list[str] = self.meta["classes"]
        self.size: int = self.meta["image_size"]
        self.version: str = self.meta.get("model_version", "unknown")
        self.model_file = model_file

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input_name: batch})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def create_app(model_dir: str | None = None, model_file: str | None = None, prediction_log: str | None = None,
               max_upload_bytes: int | None = None) -> FastAPI:
    model_dir_ = Path(model_dir or os.getenv("MODEL_DIR", "models/current"))
    model_file_ = model_file or os.getenv("MODEL_FILE", "model.onnx")
    log_path = prediction_log or os.getenv("PREDICTION_LOG")
    max_bytes = max_upload_bytes or int(os.getenv("MAX_UPLOAD_BYTES", "5000000"))
    threads = int(os.getenv("ORT_THREADS", "1"))
    log_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = Model(model_dir_, model_file_, threads)
        yield

    app = FastAPI(title="Surface defect classifier", lifespan=lifespan)

    def _log(record: dict) -> None:
        if not log_path:
            return
        with log_lock, open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _infer(model: Model, grays: list[np.ndarray]) -> list[Prediction]:
        t0 = time.perf_counter()
        probs = model.predict_proba(np.stack([to_tensor(g) for g in grays]))
        ms = (time.perf_counter() - t0) * 1000 / len(grays)
        out = []
        for g, p in zip(grays, probs, strict=True):
            i = int(p.argmax())
            pred = Prediction(label=model.classes[i], class_index=i, confidence=float(p[i]),
                              probabilities={c: float(v) for c, v in zip(model.classes, p, strict=True)},
                              model_version=model.version, latency_ms=round(ms, 3))
            _log({"ts": datetime.now(UTC).isoformat(), "model_version": model.version,
                  "label": pred.label, "confidence": pred.confidence, **image_stats(g)})
            out.append(pred)
        return out

    async def _read(file: UploadFile, size: int) -> np.ndarray:
        data = await file.read(max_bytes + 1)
        if not data:
            raise HTTPException(400, f"{file.filename}: empty file")
        if len(data) > max_bytes:
            raise HTTPException(413, f"{file.filename}: larger than {max_bytes} bytes")
        try:
            return await run_in_threadpool(decode_gray, data, size)
        except ValueError as e:
            raise HTTPException(400, f"{file.filename}: {e}") from e

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health(request: Request):
        m = getattr(request.app.state, "model", None)
        if m is None:
            raise HTTPException(503, "model not loaded")
        return {"status": "ok", "model_version": m.version}

    @app.get("/model")
    def model_info(request: Request):
        m: Model = request.app.state.model
        return {"model_version": m.version, "onnx_file": m.model_file, "classes": m.classes,
                "image_size": m.size, "arch": m.meta.get("arch")}

    @app.post("/predict", response_model=Prediction)
    async def predict(request: Request, file: Annotated[UploadFile, File()]):
        m: Model = request.app.state.model
        gray = await _read(file, m.size)
        return (await run_in_threadpool(_infer, m, [gray]))[0]

    @app.post("/predict/batch", response_model=BatchPrediction)
    async def predict_batch(request: Request, files: Annotated[list[UploadFile], File()]):
        m: Model = request.app.state.model
        if len(files) > MAX_BATCH:
            raise HTTPException(400, f"at most {MAX_BATCH} files per request")
        grays = [await _read(f, m.size) for f in files]
        return {"predictions": await run_in_threadpool(_infer, m, grays)}

    return app


app = create_app()
