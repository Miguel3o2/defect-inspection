# Runbook: from this repo to real results

**State of play.** The code was developed without access to the real datasets or a working PyTorch install, so:
the torch parts (ResNet training, ONNX export, PatchCore backbone, `resnet18` supervised scorer) have **never run**,
and no number in this repo comes from real data. Follow the stages in order; each ends with a gate. Stop at a
failed gate and fix it before continuing. If something fails, send me the full log of the failing command.

Rules that protect the credibility of the results:
1. The **test split is evaluated once**, at the very end (Stage 6), for each model. Everything before uses val.
2. Never type a number into the README or email by hand. `python scripts/collect_results.py` produces them.
3. Change one thing at a time; keep MLflow runs (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
4. A result that looks too good is a bug until proven otherwise (see "Sanity checks").

## Stage 0: it runs on your machine
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision        # GPU machine: use the CUDA index instead, e.g.
                                     #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
                                     # (a plain `pip install torch` on Windows gives the CPU-only wheel)
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # gate for GPU machines: must print True
make ci                              # lint + all tests (the 3 torch test files run for the first time here)
make smoke                           # full pipeline on synthetic data incl. ONNX export, PatchCore, drift
```
**Gate:** both green. Likeliest failures, in order: `torch.onnx.export` call (version differences), PatchCore
backbone wrapper (`src/anomaly/patchcore_extractor.py`), `resnet18_binary` scorer.

## Stage 1: CI on GitHub
Push, add the CI badge to the top of the README
(`[![CI](https://github.com/<user>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/<repo>/actions/workflows/ci.yml)`),
open the Actions tab.
**Gate:** `lint-test` and `smoke` green. Then download `frozen-requirements.txt` from the `smoke-outputs` artifact,
commit it as a lock file, and require the CI checks on `main` in the repo settings.

## Stage 2: data
- NEU-DET (1800 grayscale 200x200 images, 6 classes x 300): put the images under `data/raw/NEU-DET/` (any folder
  depth; class is read from file names like `crazing_12.jpg`). Check the source's terms of use.
- MVTec AD: download from mvtec.com (CC BY-NC-SA 4.0, non-commercial) into `data/raw/mvtec_ad/<category>/...`.
**Gate:** the baseline prints `1800 images, class counts: [300, 300, 300, 300, 300, 300]`.

## Stage 3: baseline
```bash
make baseline
```
**Gate:** read the leakage line. Any near-duplicate (test, train) pairs? If yes, deduplicate *before* splitting and
rerun everything; otherwise every later number is inflated. Note the baseline val macro-F1: it is the bar the deep
model has to beat to justify itself. If it is already very high, say so in the README: choosing the simple model
is the finding the job description asks for.

## Stage 4: deep model (validation only)
```bash
make deep        # sanity run; watch the loss go down
make ablate      # augmentation ablation, 3 seeds
make tune        # Optuna; long on CPU, reduce n_trials in configs/deep.yaml if needed
python -m src.train.train_deep train --params artifacts/best_params.yaml     # NO --eval-test yet
```
**Gate:** the deep model's val macro-F1 is plausible (well above chance, 0.167). If the ablation differences are
within the seed std, report them as "no significant effect".

## Stage 5: understand the errors (manual, the part reviewers read)
```bash
make failure-analysis                 # then label every error in reports/error_review.csv by hand
python -m src.evaluate.failure_analysis summarize --pred artifacts/predictions_lightgbm_val.csv
make label-noise                      # review the montage; keep only rows you agree with -> reports/label_issues_confirmed.csv
python -m src.train.train_baseline --exclude-train reports/label_issues_confirmed.csv
```
Write `reports/failure_analysis.md` from the template. Decide, using val only, whether cleaning helped.
**Gate:** the report states at least one change made *because of* the analysis and its measured effect.

## Stage 6: final test evaluation (once)
```bash
python -m src.train.train_baseline [--exclude-train reports/label_issues_confirmed.csv] --eval-test
python -m src.train.train_deep train --params artifacts/best_params.yaml --eval-test
```
Do not tune anything after seeing these numbers. If a model must change, say so in the README.

## Stage 7: anomaly detection and the unseen-defect comparison
```bash
make patchcore                         # start with 2-3 categories (configs/patchcore.yaml)
make compare                           # add `--models handcrafted_lgbm` for a quick first pass
```
**Gate:** PatchCore image AUROC on `bottle` is high (it is an easy category). If it is near 0.5, suspect the backbone
wrapper before anything else. Report the comparison whichever way it goes; check the CIs.

## Stage 8: deployment and monitoring
```bash
python -m src.deploy.export_onnx --ckpt artifacts/resnet_best.pt --set-current
make benchmark                         # run on the machine you will describe in the README
make drift-reference && make drift-simulate
docker build -f docker/Dockerfile -t defect-inspection . && docker run -p 8000:8000 defect-inspection
```
**Gate:** export reports parity < 1e-3; the int8 row's val macro-F1 is close to fp32 (if not, drop int8 and say so).
Take a screenshot of the Evidently report and of `drift_simulation.md`.

## Stage 9: results into the README
```bash
python scripts/collect_results.py --email
```
Paste `reports/RESULTS.md` content into the README results section (keep the generated tables as-is), fill the
remaining hand-written parts (limitations, what you learned), add 2-3 images (confusion matrix, error montage,
PatchCore overlays, drift report), record a short demo GIF of `curl` against the container.

## Stage 10: before you send anything
- [ ] CI green on `main`; a fresh `git clone` + `make ci` works.
- [ ] No data, `artifacts/`, `mlruns/`, or `models/*` committed (see `.gitignore`).
- [ ] README renders on GitHub; all links and the badge work.
- [ ] Limitations section is honest (see "Known gaps" below).
- [ ] `reports/email_draft.md` rendered from real numbers; the bracketed personal fields filled by you.
- [ ] You can explain every number and every design choice out loud; expect to be asked why not X.
- [ ] Send to a named person (engineer or hiring manager), not a generic inbox.

## Sanity checks when a result looks too good
Near-duplicates across splits (Stage 3 leakage line) · test split touched more than once · val used for model
selection and then quoted as the final number · labels derived from file names that leak the split · identical
seeds hiding variance.

## Known gaps (say so; do not claim these)
No object detection / segmentation model (classification and anomaly maps only, so no mAP) · no time-series work ·
no model registry (versions are hash-named directories; rollback = point `MODEL_DIR` at an older one) · no DVC
dataset versioning · PatchCore is not served by the API · retraining is a manual decision by design.
