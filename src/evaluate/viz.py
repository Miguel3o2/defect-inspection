from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features.handcrafted import load_gray


def montage(items: list[tuple[str, str]], title: str, out: str, size: int = 200, cols: int = 4) -> None:
    """items: [(image_path, caption), ...] -> grid image saved at `out`."""
    if not items:
        return
    cols = min(cols, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (path, caption) in zip(axes.ravel(), items, strict=False):
        ax.imshow(load_gray(path, size), cmap="gray", vmin=0, vmax=255)
        ax.set_title(caption, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def overlay_grid(records: list[dict], out: str, vmax: float | None = None) -> None:
    """records: dicts with img (uint8 RGB), map (float), mask (bool), title. Columns: image | heatmap | ground truth."""
    if not records:
        return
    fig, axes = plt.subplots(len(records), 3, figsize=(9, 3 * len(records)), squeeze=False)
    for row, r in zip(axes, records, strict=True):
        row[0].imshow(r["img"])
        row[0].set_title(r["title"], fontsize=8)
        row[1].imshow(r["img"])
        row[1].imshow(r["map"], cmap="jet", alpha=0.5, vmin=0, vmax=vmax or float(r["map"].max()))
        row[1].set_title("anomaly map", fontsize=8)
        row[2].imshow(r["mask"], cmap="gray", vmin=0, vmax=1)
        row[2].set_title("ground truth", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
