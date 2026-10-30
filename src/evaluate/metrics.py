from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def summarize(y_true, y_pred, class_names) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": classification_report(
            y_true, y_pred, labels=range(len(class_names)), target_names=class_names,
            output_dict=True, zero_division=0,
        ),
    }


def save_confusion_matrix(y_true, y_pred, class_names, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_predictions(paths, y_true, probs, class_names, path) -> None:
    """One row per sample: path, true, pred, confidence, per-class probabilities."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence", *[f"p_{c}" for c in class_names]])
        for p, t, pr in zip(paths, y_true, probs, strict=True):
            w.writerow([str(p), class_names[t], class_names[int(pr.argmax())], f"{pr.max():.4f}",
                        *[f"{v:.4f}" for v in pr]])
