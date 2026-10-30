"""PatchCore on MVTec AD. Trains on defect-free images only; evaluates image/pixel AUROC.

The deployment threshold is set from held-out GOOD train images (calib_fraction), never from the test set.
`extract_fn` is injectable so everything except the backbone can be tested without torch.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from src.anomaly.patchcore_core import greedy_coreset, score_features
from src.data.mvtec import MVTecCategory, discover, load_display, load_image, load_mask
from src.evaluate.anomaly_metrics import (
    image_auroc,
    per_defect_auroc,
    pixel_auroc,
    threshold_metrics,
)
from src.evaluate.viz import overlay_grid


def make_extract_fn(cfg: dict):
    import torch

    from src.anomaly.patchcore_extractor import PatchCoreExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = PatchCoreExtractor(cfg["backbone"], cfg["out_dim"], cfg["pretrained"]).to(device)

    def fn(batch: np.ndarray):
        feats, hw = ext(torch.from_numpy(batch).to(device))
        return feats.cpu().numpy(), hw

    return fn


def extract_paths(extract_fn, paths, cfg):
    """Yield (start, feats [B,P,D], hw) per batch."""
    bs = cfg["batch_size"]
    for s in range(0, len(paths), bs):
        batch = np.stack([load_image(p, cfg["resize"], cfg["crop"]) for p in paths[s:s + bs]])
        feats, hw = extract_fn(batch)
        yield s, feats.astype(np.float32), hw


def fit_category(cfg: dict, cat: MVTecCategory, extract_fn, out_dir: Path) -> dict:
    seed, crop = cfg["seed"], cfg["crop"]
    chunks, hw = [], None
    for _, f, hw_ in extract_paths(extract_fn, cat.train, cfg):
        chunks.append(f)
        hw = hw_
    feats = np.concatenate(chunks)
    order = np.random.default_rng(seed).permutation(len(feats))
    n_cal = max(1, round(cfg["calib_fraction"] * len(feats))) if len(feats) > 2 else 0
    cal_idx, fit_idx = order[:n_cal], order[n_cal:]

    cand = feats[fit_idx].reshape(-1, feats.shape[-1])
    sel = greedy_coreset(cand, max(1, int(cfg["coreset_ratio"] * len(cand))), cfg["coreset_proj_dim"], seed)
    bank = np.ascontiguousarray(cand[sel])
    print(f"[{cat.name}] {len(fit_idx)} train imgs -> {len(cand)} patches -> bank {bank.shape}")

    cal_scores = score_features(feats[cal_idx], hw, bank, crop, cfg["blur_sigma"])[0] if n_cal else np.array([])
    threshold = float(cal_scores.max()) if n_cal else None
    del feats, cand

    img_scores, maps = [], []
    for _, f, _hw in extract_paths(extract_fn, cat.test, cfg):
        s, m = score_features(f, hw, bank, crop, cfg["blur_sigma"])
        img_scores.append(s)
        maps.append(m)
    img_scores, maps = np.concatenate(img_scores), np.concatenate(maps)
    is_defect = np.array([d != "good" for d in cat.test_defect])
    masks = np.stack([load_mask(m, cfg["resize"], crop) for m in cat.test_mask])

    res = {"category": cat.name, "n_train": len(cat.train), "n_test": len(cat.test), "bank_size": len(bank),
           "image_auroc": image_auroc(is_defect, img_scores), "pixel_auroc": pixel_auroc(masks, maps),
           "per_defect_image_auroc": per_defect_auroc(cat.test_defect, img_scores), "threshold": threshold}
    if threshold is not None:
        res["at_threshold"] = threshold_metrics(is_defect, img_scores, threshold)

    cdir = out_dir / cat.name
    cdir.mkdir(parents=True, exist_ok=True)
    np.save(cdir / "bank.npy", bank)
    with (cdir / "test_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "defect", "score"])
        for pth, d, sc in zip(cat.test, cat.test_defect, img_scores, strict=True):
            w.writerow([str(pth), d, f"{sc:.6f}"])
    meta = {k: cfg[k] for k in ("backbone", "out_dim", "resize", "crop", "blur_sigma", "pretrained")}
    (cdir / "meta.json").write_text(json.dumps({**meta, "hw": list(hw), "threshold": threshold,
                                                "cal_scores": cal_scores.tolist(), "category": cat.name}, indent=2), encoding="utf-8")
    seen, recs = {}, []
    for i, d in enumerate(cat.test_defect):
        if d != "good" and seen.get(d, 0) < 2:
            seen[d] = seen.get(d, 0) + 1
            recs.append({"img": load_display(cat.test[i], cfg["resize"], crop), "map": maps[i], "mask": masks[i],
                         "title": f"{d} | score {img_scores[i]:.2f}"})
    overlay_grid(recs, str(cdir / "overlays.png"), vmax=float(np.percentile(maps, 99.9)))
    (cdir / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def results_table(results: list[dict]) -> str:
    lines = ["| Category | Image AUROC | Pixel AUROC | Precision@thr | Recall@thr | Bank size |", "|---|---|---|---|---|---|"]
    for r in results:
        t = r.get("at_threshold", {})
        lines.append(f"| {r['category']} | {r['image_auroc']:.3f} | {r['pixel_auroc']:.3f} | "
                     f"{t.get('precision', float('nan')):.2f} | {t.get('recall', float('nan')):.2f} | {r['bank_size']} |")
    n = len(results)
    lines.append(f"| **mean** | {sum(r['image_auroc'] for r in results) / n:.3f} | "
                 f"{sum(r['pixel_auroc'] for r in results) / n:.3f} | | | |")
    return "\n".join(lines) + "\n"


def main(config_path: str, categories: list[str] | None = None, extract_fn=None) -> list[dict]:
    import mlflow

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cats = categories or cfg["categories"]
    extract_fn = extract_fn or make_extract_fn(cfg)
    out_dir = Path(cfg["out_dir"])
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    results = []
    for name in cats:
        cat = discover(cfg["data_root"], name)
        with mlflow.start_run(run_name=f"patchcore-{name}"):
            mlflow.log_params({k: cfg[k] for k in ("backbone", "out_dim", "coreset_ratio", "calib_fraction", "blur_sigma",
                                                    "resize", "crop")} | {"category": name})
            res = fit_category(cfg, cat, extract_fn, out_dir)
            mlflow.log_metrics({"image_auroc": res["image_auroc"], "pixel_auroc": res["pixel_auroc"]})
            mlflow.log_artifact(str(out_dir / name / "overlays.png"))
        print(f"[{name}] image AUROC {res['image_auroc']:.3f} | pixel AUROC {res['pixel_auroc']:.3f} | "
              f"per-defect {({k: round(v, 3) for k, v in res['per_defect_image_auroc'].items()})}")
        results.append(res)
    (out_dir / "results.md").write_text(results_table(results), encoding="utf-8")
    print(results_table(results))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/patchcore.yaml")
    ap.add_argument("--categories", nargs="*", help="override categories in the config")
    a = ap.parse_args()
    main(a.config, a.categories)
