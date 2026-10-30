"""PatchCore (trained on good images only) vs supervised classifiers on UNSEEN defect types.

Leave-one-defect-type-out, per MVTec category. For each held-out type C:
  supervised train = all train/good (label 0) + the TRAIN half of every other defect type (label 1)
  unseen eval      = test/good + ALL images of type C
  seen eval        = test/good + the EVAL half of every other type (never trained on)
PatchCore never sees any defect, so its scores are the same in every fold.
Both models are scored on identical images; the AUROC gap gets a paired-bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.anomaly.supervised import SCORERS
from src.data.mvtec import MVTecCategory, discover
from src.evaluate.anomaly_metrics import paired_bootstrap_diff, tpr_at_fpr


def make_splits(cat: MVTecCategory, seed: int):
    """-> (good_idx, defect_types, half_train, half_eval); all indices refer to cat.test."""
    rng = np.random.default_rng(seed)
    d = np.array(cat.test_defect)
    types = sorted(set(d) - {"good"})
    half_train, half_eval = {}, {}
    for t in types:
        idx = rng.permutation(np.where(d == t)[0])
        k = len(idx) // 2
        half_train[t], half_eval[t] = np.sort(idx[:k]), np.sort(idx[k:])
    return np.where(d == "good")[0], types, half_train, half_eval


def load_patchcore_scores(pc_dir: Path, cat: MVTecCategory) -> np.ndarray:
    f = pc_dir / cat.name / "test_scores.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run `make patchcore` for {cat.name} first")
    m = dict(zip(pd.read_csv(f)["path"].astype(str), pd.read_csv(f)["score"], strict=True))
    missing = [str(p) for p in cat.test if str(p) not in m]
    if missing:
        raise ValueError(f"{len(missing)} test images have no PatchCore score (different data_root?): {missing[:2]}")
    return np.array([m[str(p)] for p in cat.test], np.float64)


def run_category(cfg: dict, cat: MVTecCategory, pc_scores: np.ndarray, scorers: dict) -> list[dict]:
    good, types, htr, hev = make_splits(cat, cfg["seed"])
    if len(types) < 2:
        print(f"[{cat.name}] only {len(types)} defect type(s): leave-one-type-out needs >= 2, skipping")
        return []
    defects = np.array(cat.test_defect)
    all_paths = [str(p) for p in cat.test]
    rows = []
    for held in types:
        seen = [t for t in types if t != held]
        tr_def = np.concatenate([htr[t] for t in seen])
        if len(tr_def) == 0:
            print(f"[{cat.name}/{held}] no defect images to train the supervised model, skipping")
            continue
        evals = {"unseen": np.where(defects == held)[0], "seen": np.concatenate([hev[t] for t in seen])}
        train_paths = [str(p) for p in cat.train] + [all_paths[i] for i in tr_def]
        y = np.r_[np.zeros(len(cat.train), int), np.ones(len(tr_def), int)]
        for name, scorer in scorers.items():
            sup = np.mean([scorer(train_paths, y, all_paths, s, cfg) for s in cfg["supervised"]["seeds"]], axis=0)
            for kind, def_idx in evals.items():
                if len(def_idx) == 0:
                    continue
                ids = np.concatenate([good, def_idx])
                if set(ids.tolist()) & set(tr_def.tolist()):
                    raise AssertionError("evaluation images overlap supervised training images")
                is_def = np.r_[np.zeros(len(good), bool), np.ones(len(def_idx), bool)]
                a, b = pc_scores[ids], sup[ids]
                diff, lo, hi = paired_bootstrap_diff(is_def, a, b, cfg["bootstrap"]["n"], cfg["seed"])
                fpr = cfg["bootstrap"]["fpr"]
                rows.append({"category": cat.name, "model": name, "held_out": held, "kind": kind,
                             "n_defect": len(def_idx), "n_good": len(good), "n_sup_train_defects": len(tr_def),
                             "patchcore_auroc": float(roc_auc_score(is_def, a)),
                             "supervised_auroc": float(roc_auc_score(is_def, b)),
                             "diff": diff, "ci_lo": lo, "ci_hi": hi,
                             "patchcore_tpr": tpr_at_fpr(is_def, a, fpr), "supervised_tpr": tpr_at_fpr(is_def, b, fpr)})
        print(f"[{cat.name}] held-out {held}: done")
    return rows


def _table(df: pd.DataFrame, fpr: float) -> list[str]:
    head = (f"| Category | Held-out type | n defect / good | PatchCore AUROC | Supervised AUROC | "
            f"Δ (PC − sup) [95% CI] | PC TPR@{fpr:.0%}FPR | Sup TPR@{fpr:.0%}FPR |")
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.category} | {r.held_out} | {r.n_defect} / {r.n_good} | {r.patchcore_auroc:.3f} | "
                     f"{r.supervised_auroc:.3f} | {r.diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | "
                     f"{r.patchcore_tpr:.2f} | {r.supervised_tpr:.2f} |")
    lines.append(f"| **mean over {len(df)} folds** | | | {df.patchcore_auroc.mean():.3f} | "
                 f"{df.supervised_auroc.mean():.3f} | {df['diff'].mean():+.3f} | {df.patchcore_tpr.mean():.2f} | "
                 f"{df.supervised_tpr.mean():.2f} |")
    pc_sig, sup_sig = int((df.ci_lo > 0).sum()), int((df.ci_hi < 0).sum())
    lines.append(f"\nPatchCore ahead in {int((df['diff'] > 0).sum())}/{len(df)} folds "
                 f"(CI excludes 0 in its favour: {pc_sig}; in the supervised model's favour: {sup_sig}).")
    return lines


def summarize(rows: list[dict], fpr: float = 0.05) -> str:
    df = pd.DataFrame(rows)
    out = []
    for model in df["model"].unique():
        for kind, title in (("unseen", "Unseen defect types (held out of supervised training)"),
                            ("seen", "Seen defect types (held-out images of types the supervised model trained on)")):
            sub = df[(df["model"] == model) & (df["kind"] == kind)]
            if len(sub):
                out += [f"### {title}: PatchCore vs `{model}`", "", *_table(sub, fpr), ""]
    out.append("_Small evaluation sets: wide CIs are expected. Supervised scores are averaged over training seeds._")
    return "\n".join(out) + "\n"


def main(config_path: str, categories: list[str] | None = None, models: list[str] | None = None) -> list[dict]:
    import mlflow

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    scorers = {m: SCORERS[m] for m in (models or cfg["supervised"]["models"])}
    rows: list[dict] = []
    for name in categories or cfg["categories"]:
        cat = discover(cfg["data_root"], name)
        rows += run_category(cfg, cat, load_patchcore_scores(Path(cfg["patchcore_dir"]), cat), scorers)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    md = summarize(rows, cfg["bootstrap"]["fpr"])
    (out / "results.md").write_text(md, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name="unseen-defect-comparison"):
        df = pd.DataFrame(rows)
        for (model, kind), g in df.groupby(["model", "kind"]):
            mlflow.log_metrics({f"{model}_{kind}_patchcore_auroc": g.patchcore_auroc.mean(),
                                f"{model}_{kind}_supervised_auroc": g.supervised_auroc.mean()})
        mlflow.log_artifact(str(out / "results.md"))
    print(md)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/comparison.yaml")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--models", nargs="*", choices=list(SCORERS), help="e.g. --models handcrafted_lgbm for a quick run")
    a = ap.parse_args()
    main(a.config, a.categories, a.models)
