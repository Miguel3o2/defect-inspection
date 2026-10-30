"""Label the cause of each prediction error, with typical correct examples shown next to it.

Run from the project root:   python scripts/label_errors.py
Progress is saved after every answer; rerun to continue. Close the CSV in Excel/your editor first.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CAUSES = ["label_noise", "ambiguous_class_boundary", "low_contrast", "small_or_faint_defect",
          "unusual_lighting", "background_texture", "model_error_other"]
HELP = ["the image clearly looks like the PREDICTED class (you'd bet the label is wrong)",
        "both classes are plausible", "low contrast", "small or faint defect", "unusual lighting",
        "background texture confuses it", "looks clearly like its label, no visible reason"]


def _img(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return im if im is not None else np.zeros((200, 200), np.uint8)


def pick_examples(pred: pd.DataFrame, cls: str, n: int, rng: np.random.Generator) -> list[str]:
    ok = pred[(pred["true"] == cls) & (pred["pred"] == cls)].nlargest(40, "confidence")
    return list(ok["path"].sample(min(n, len(ok)), random_state=int(rng.integers(1 << 30))))


def make_figure(row: pd.Series, pred: pd.DataFrame, rng: np.random.Generator):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[2, 1, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(_img(row["path"]), cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"LABELLED {row['true']}\nmodel says {row['pred']} (conf {row['confidence']})\n{row['auto_hint']}", fontsize=9)
    ax.axis("off")
    for r, cls in enumerate((row["true"], row["pred"])):
        for c, p in enumerate(pick_examples(pred, cls, 3, rng)):
            a = fig.add_subplot(gs[r, 1 + c])
            a.imshow(_img(p), cmap="gray", vmin=0, vmax=255)
            a.set_title(f"typical {cls}" if c == 0 else "", fontsize=8)
            a.axis("off")
    fig.tight_layout()
    return fig


def default_show(fig) -> None:
    import matplotlib.pyplot as plt

    if plt.get_backend().lower() == "agg":                 # no GUI backend: save and open with the OS viewer
        path = Path("reports") / "_current_error.png"
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=100)
        if hasattr(os, "startfile"):
            os.startfile(path)  # noqa: S606
        else:
            print(f"image saved to {path}")
    else:
        plt.show(block=False)
        plt.pause(0.5)


def run(review: str, pred_csv: str, ask=input, show=default_show, seed: int = 0) -> int:
    rev = pd.read_csv(review, dtype=str, keep_default_na=False)
    pred = pd.read_csv(pred_csv)
    rng = np.random.default_rng(seed)
    todo = [i for i in rev.index if not rev.at[i, "cause"].strip()]
    print("\n".join(f"  {k + 1} = {c}: {h}" for k, (c, h) in enumerate(zip(CAUSES, HELP, strict=True))))
    print("  s = skip, q = save and quit\n")
    for n, i in enumerate(todo, 1):
        fig = make_figure(rev.loc[i], pred, rng)
        show(fig)
        while True:
            ans = ask(f"[{n}/{len(todo)}] {rev.at[i, 'true']} -> {rev.at[i, 'pred']}: cause 1-7, s, q > ").strip().lower()
            if ans in ("s", "q") or (ans.isdigit() and 1 <= int(ans) <= len(CAUSES)):
                break
            print("  enter a number 1-7, s or q")
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pass
        if ans == "q":
            break
        if ans != "s":
            rev.at[i, "cause"] = CAUSES[int(ans) - 1]
            rev.to_csv(review, index=False)
    left = int((rev["cause"].str.strip() == "").sum())
    rev.to_csv(review, index=False)
    print(f"saved {review}: {len(rev) - left} labelled, {left} left")
    return left


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="reports/oof/error_review.csv")
    ap.add_argument("--pred", default="artifacts/predictions_lightgbm_oof.csv")
    a = ap.parse_args(argv)
    return run(a.review, a.pred)


if __name__ == "__main__":
    main()
