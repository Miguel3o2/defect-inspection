"""Failure analysis tooling.

  analyze   : errors -> confusion pairs, per-error auto hints, montages, error_review.csv to label by hand
  summarize : filled error_review.csv -> markdown tables for reports/failure_analysis.md

Run this on the VALIDATION predictions. Looking at test errors and then fixing them = tuning on test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.evaluate.viz import montage
from src.features.handcrafted import load_gray

CAUSES = ["label_noise", "ambiguous_class_boundary", "low_contrast", "small_or_faint_defect",
          "unusual_lighting", "background_texture", "model_error_other"]
STATS = ["brightness", "contrast", "sharpness"]


def image_stats(path: str, size: int = 200) -> dict:
    img = load_gray(path, size)
    return {"brightness": float(img.mean()), "contrast": float(img.std()),
            "sharpness": float(cv2.Laplacian(img, cv2.CV_64F).var())}


def add_hints(df: pd.DataFrame) -> pd.DataFrame:
    """Automatic hints only: z-score of image stats vs correctly classified samples of the same true class."""
    df = df.copy()
    df["correct"] = df["true"] == df["pred"]
    df = pd.concat([df, pd.DataFrame([image_stats(p) for p in df["path"]], index=df.index)], axis=1)
    ref = df[df["correct"]].groupby("true")[STATS].agg(["mean", "std"])
    names = {"brightness": ("dark", "bright"), "contrast": ("low_contrast", "high_contrast"),
             "sharpness": ("blurry", "sharp")}

    def hint(row) -> str:
        if row["correct"]:
            return ""
        h = []
        for col, (lo, hi) in names.items():
            if row["true"] not in ref.index:
                continue
            m, s = ref.loc[row["true"], (col, "mean")], ref.loc[row["true"], (col, "std")]
            if pd.notna(s) and s > 0:
                z = (row[col] - m) / s
                h += [lo] if z < -2 else [hi] if z > 2 else []
        h.append("high_conf_error(label_noise_suspect)" if row["confidence"] >= 0.9
                 else "low_conf_error(ambiguous?)" if row["confidence"] < 0.5 else "")
        return ";".join(x for x in h if x)

    df["auto_hint"] = df.apply(hint, axis=1)
    return df


def overview(df: pd.DataFrame) -> dict:
    labels = sorted(df["true"].unique())
    recall = {c: float((df[df["true"] == c]["pred"] == c).mean()) for c in labels}
    pairs = (df[df["true"] != df["pred"]].groupby(["true", "pred"]).size()
             .sort_values(ascending=False).rename("count").reset_index())
    return {"acc": accuracy_score(df["true"], df["pred"]),
            "macro_f1": f1_score(df["true"], df["pred"], average="macro"),
            "recall": recall, "pairs": pairs, "n": len(df), "n_err": int((df["true"] != df["pred"]).sum())}


def cmd_analyze(a) -> None:
    out = Path(a.out)
    df = add_hints(pd.read_csv(a.pred))
    o = overview(df)
    print(f"{o['n']} samples, {o['n_err']} errors | acc={o['acc']:.3f} macro_f1={o['macro_f1']:.3f}")
    print("per-class recall:", {k: round(v, 3) for k, v in o["recall"].items()})
    print("top confusions:\n", o["pairs"].head(8).to_string(index=False))

    errors = df[~df["correct"]]
    for _, r in o["pairs"].head(a.top_pairs).iterrows():
        sub = errors[(errors["true"] == r["true"]) & (errors["pred"] == r["pred"])].head(12)
        montage([(p, f"conf {c:.2f}") for p, c in zip(sub["path"], sub["confidence"], strict=True)],
                f"true={r['true']}  predicted={r['pred']}  (n={r['count']})",
                str(out / "figures" / f"errors_{r['true']}_as_{r['pred']}.png"))

    review = out / "error_review.csv"
    if review.exists():
        print(f"{review} already exists: NOT overwriting your manual labels.")
    else:
        errors = errors[["path", "true", "pred", "confidence", "auto_hint"]].copy()
        errors["cause"] = ""
        out.mkdir(parents=True, exist_ok=True)
        errors.to_csv(review, index=False)
        print(f"Wrote {review}. Look at every error image, fill `cause` with one of:\n  {CAUSES}")


def cmd_summarize(a) -> None:
    df = pd.read_csv(a.pred)
    o = overview(df)
    rev = pd.read_csv(a.review)
    cause = rev["cause"].fillna("").astype(str).str.strip()
    done = rev[cause != ""].assign(cause=cause[cause != ""])
    unknown = sorted(set(done["cause"]) - set(CAUSES))
    if unknown:
        print(f"WARNING: unknown causes {unknown}; allowed: {CAUSES}")
    if len(done) < len(rev):
        print(f"WARNING: {len(rev) - len(done)} of {len(rev)} errors still unlabeled")

    summary = f"**Out-of-fold predictions (5-fold CV, train+val):** {o['n']} samples, {o['n_err']} errors; accuracy {o['acc']:.3f}, macro-F1 {o['macro_f1']:.3f}."
    lines = [summary, "",
             "| Class | Recall |", "|---|---|", *[f"| {k} | {v:.3f} |" for k, v in o["recall"].items()], "",
             "| True | Predicted | Count |", "|---|---|---|",
             *[f"| {r.true} | {r.pred} | {r.count} |" for r in o["pairs"].head(8).itertuples()], "",
             f"**Error causes (manual review of {len(done)} errors):**", "",
             "| Cause | Count | Share |", "|---|---|---|"]
    for c, n in done["cause"].value_counts().items():
        lines.append(f"| {c} | {n} | {n / max(len(done), 1):.0%} |")
    if len(done) < 30:
        lines += ["", f"_Only {len(done)} errors reviewed: treat shares as indicative, not precise._"]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("analyze")
    p1.add_argument("--pred", required=True, help="predictions_<model>_val.csv")
    p1.add_argument("--out", default="reports")
    p1.add_argument("--top-pairs", type=int, default=3)
    p1.set_defaults(fn=cmd_analyze)
    p2 = sub.add_parser("summarize")
    p2.add_argument("--pred", required=True)
    p2.add_argument("--review", default="reports/error_review.csv")
    p2.add_argument("--out", default="reports/failure_analysis_generated.md")
    p2.set_defaults(fn=cmd_summarize)
    args = ap.parse_args()
    args.fn(args)
