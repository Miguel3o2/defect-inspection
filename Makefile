.PHONY: install lint test baseline deep ablate tune failure-analysis label-noise patchcore compare export benchmark serve docker-build ci smoke drift-reference drift-simulate drift-check results
install:
	pip install -r requirements.txt
lint:
	ruff check src tests scripts
test:
	pytest -q
baseline:
	python -m src.train.train_baseline --config configs/baseline.yaml

deep:
	python -m src.train.train_deep train
ablate:
	python -m src.train.train_deep ablate
tune:
	python -m src.train.train_deep tune

failure-analysis:
	python -m src.evaluate.failure_analysis analyze --pred artifacts/predictions_lightgbm_val.csv
label-noise:
	python -m src.evaluate.label_noise
patchcore:
	python -m src.train.train_patchcore

compare:
	python -m src.evaluate.unseen_defect_comparison

export:
	python -m src.deploy.export_onnx --set-current
benchmark:
	python -m src.deploy.benchmark --ckpt artifacts/resnet_best.pt
serve:
	uvicorn src.serve.app:app --port 8000
docker-build:
	docker build -f docker/Dockerfile -t defect-inspection .
ci: lint test
smoke:
	python scripts/smoke_pipeline.py --workdir smoke --models-out models

drift-reference:
	python -m src.monitoring.drift build-reference --model-dir models/current
drift-simulate:
	python -m src.monitoring.simulate --model-dir models/current
drift-check:
	python -m src.monitoring.drift check --model-dir models/current --log $(LOG)
results:
	python scripts/collect_results.py --email
