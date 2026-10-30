"""Deep-model CLI: `train`, `ablate` (augmentation ablation), `tune` (Optuna).

Uses the same split seed as the baseline, so models are compared on identical val/test sets.
The test split is only evaluated with --eval-test on `train`.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
import yaml

from src.data.neu_det import CLASSES, discover_images
from src.data.splits import stratified_split
from src.data.torch_dataset import load_images
from src.evaluate.metrics import save_confusion_matrix, save_predictions, summarize
from src.train.engine import evaluate_split, fit

OUT = Path("artifacts")


def _setup(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    paths, y = discover_images(cfg["data_dir"])
    tr, va, te = stratified_split(y, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"])
    images = load_images(paths, cfg["image_size"])
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    OUT.mkdir(exist_ok=True)
    return cfg, paths, y, images, (tr, va, te)


def _flat_params(cfg: dict) -> dict:
    return {f"{k}.{kk}": vv for k, v in cfg.items() if isinstance(v, dict) and k in ("model", "train")
            for kk, vv in v.items()}


def cmd_train(a) -> None:
    cfg, paths, y, images, (tr, va, te) = _setup(a.config)
    if a.params:
        best = yaml.safe_load(Path(a.params).read_text(encoding="utf-8"))
        for k, v in best.items():
            section, key = k.split(".")
            cfg[section][key] = v
    with mlflow.start_run(run_name=f"{cfg['model']['arch']}-{cfg['train']['aug']}"):
        mlflow.log_params(_flat_params(cfg))
        res = fit(cfg, images, y, tr, va, len(CLASSES), train_seed=cfg["seed"],
                  on_epoch=lambda e, f1, m: mlflow.log_metrics(m, step=e))
        mlflow.log_metrics({"best_val_macro_f1": res["best_f1"], "best_epoch": res["best_epoch"]})

        splits = {"val": (va, res["val_probs"])}
        if a.eval_test:
            splits["test"] = (te, evaluate_split(res["model"], images, y, te))
        for name, (idx, probs) in splits.items():
            pred = probs.argmax(1)
            save_predictions([paths[i] for i in idx], y[idx], probs, CLASSES, OUT / f"predictions_deep_{name}.csv")
            s = summarize(y[idx], pred, CLASSES)
            mlflow.log_metrics({f"{name}_accuracy": s["accuracy"], f"{name}_macro_f1": s["macro_f1"]})
            cm = OUT / f"cm_deep_{name}.png"
            save_confusion_matrix(y[idx], pred, CLASSES, str(cm))
            mlflow.log_artifact(str(cm))
            (OUT / f"report_deep_{name}.json").write_text(json.dumps(s["report"], indent=2), encoding="utf-8")
            print(f"[deep] {name}: acc={s['accuracy']:.3f} macro_f1={s['macro_f1']:.3f}")

        # misclassified val samples -> input for reports/failure_analysis.md
        with (OUT / "misclassified_val.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "true", "pred", "confidence"])
            for i, p in zip(va, res["val_probs"], strict=True):
                if p.argmax() != y[i]:
                    w.writerow([paths[i], CLASSES[y[i]], CLASSES[p.argmax()], f"{p.max():.3f}"])

        ckpt = OUT / "resnet_best.pt"
        torch.save({"state_dict": {k: v.cpu() for k, v in res["model"].state_dict().items()}, "arch": cfg["model"]["arch"],
                    "classes": CLASSES, "image_size": cfg["image_size"]}, ckpt)
        mlflow.log_artifact(str(ckpt))


def cmd_ablate(a) -> None:
    cfg, _, y, images, (tr, va, _) = _setup(a.config)
    ab = cfg["ablation"]
    rows = {}
    for policy in ab["policies"]:
        scores = []
        for seed in ab["seeds"]:
            c = copy.deepcopy(cfg)
            c["train"].update(aug=policy, epochs=ab["epochs"])
            with mlflow.start_run(run_name=f"ablate-{policy}-s{seed}"):
                mlflow.log_params({**_flat_params(c), "train_seed": seed})
                res = fit(c, images, y, tr, va, len(CLASSES), train_seed=seed)
                mlflow.log_metric("best_val_macro_f1", res["best_f1"])
            scores.append(res["best_f1"])
            print(f"{policy} seed={seed}: {res['best_f1']:.4f}")
        rows[policy] = (float(np.mean(scores)), float(np.std(scores)), scores)

    lines = ["| Augmentation | Val macro-F1 (mean ± std) | Per-seed |", "|---|---|---|"]
    for p, (m, s, sc) in rows.items():
        lines.append(f"| {p} | {m:.4f} ± {s:.4f} | {', '.join(f'{x:.3f}' for x in sc)} |")
    (OUT / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_tune(a) -> None:
    cfg, _, y, images, (tr, va, _) = _setup(a.config)
    o = cfg["optuna"]

    def objective(trial: optuna.Trial) -> float:
        c = copy.deepcopy(cfg)
        c["model"].update(arch=trial.suggest_categorical("arch", ["resnet18", "resnet34"]),
                          dropout=trial.suggest_float("dropout", 0.0, 0.5),
                          freeze_backbone_epochs=trial.suggest_int("freeze_backbone_epochs", 0, 3))
        c["train"].update(lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                          weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                          batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
                          label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.2),
                          aug=trial.suggest_categorical("aug", ["flip", "geometric", "photometric", "full"]),
                          epochs=o["epochs"])

        def on_epoch(epoch, f1, _m):
            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned

        with mlflow.start_run(run_name=f"optuna-trial-{trial.number}"):
            mlflow.log_params(trial.params)
            res = fit(c, images, y, tr, va, len(CLASSES), train_seed=0, on_epoch=on_epoch)
            mlflow.log_metric("best_val_macro_f1", res["best_f1"])
        return res["best_f1"]

    study = optuna.create_study(
        study_name=o["study_name"], storage=o["storage"], load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]), pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=o["n_trials"])
    best = {("model." if k in ("arch", "dropout", "freeze_backbone_epochs") else "train.") + k: v
            for k, v in study.best_params.items()}
    (OUT / "best_params.yaml").write_text(yaml.safe_dump(best), encoding="utf-8")
    print(f"best val macro-F1 {study.best_value:.4f}\n{yaml.safe_dump(best)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("train", cmd_train), ("ablate", cmd_ablate), ("tune", cmd_tune)]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="configs/deep.yaml")
        sp.set_defaults(fn=fn)
        if name == "train":
            sp.add_argument("--params", help="YAML of tuned params, e.g. artifacts/best_params.yaml")
            sp.add_argument("--eval-test", action="store_true")
    args = ap.parse_args()
    args.fn(args)
