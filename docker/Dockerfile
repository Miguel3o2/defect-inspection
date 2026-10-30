FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY docker/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# runtime code only: no torch, no training code
COPY src/__init__.py src/__init__.py
COPY src/serve src/serve
# model bundle: models/current/{model.onnx, meta.json}  (created by `python -m src.deploy.export_onnx --set-current`)
COPY models/current models/current

RUN useradd --create-home app && chown -R app /app
USER app
ENV MODEL_DIR=/app/models/current MODEL_FILE=model.onnx ORT_THREADS=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
