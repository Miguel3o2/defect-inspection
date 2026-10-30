"""Reusable training loop used by single runs, the augmentation ablation and Optuna."""
from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.augment import build_transform
from src.data.torch_dataset import TensorImageDataset
from src.models.classifier import build_classifier, set_backbone_trainable


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    return torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits = torch.cat([model(x.to(device, non_blocking=True)).cpu() for x, _ in loader])
    return logits.softmax(1).numpy()


def fit(cfg: dict, images: torch.Tensor, y: np.ndarray, tr: np.ndarray, va: np.ndarray, n_classes: int,
        train_seed: int = 0, on_epoch=None) -> dict:
    """Train on images[tr], select the best epoch on images[va] by macro-F1.

    `on_epoch(epoch, val_f1, metrics)` may raise to abort (used for Optuna pruning).
    Returns {"model": best model on CPU, "best_f1", "best_epoch", "history", "val_probs"}.
    """
    seed_everything(train_seed)
    device = get_device()
    t, m = cfg["train"], cfg["model"]
    use_amp = bool(t.get("amp", False)) and device.type == "cuda"

    train_ds = TensorImageDataset(images[tr], y[tr], build_transform(t["aug"], train=True))
    val_ds = TensorImageDataset(images[va], y[va], build_transform("none", train=False))
    pin = device.type == "cuda"
    workers = t["num_workers"]
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=workers,
                          pin_memory=pin, persistent_workers=workers > 0,
                          drop_last=len(train_ds) > t["batch_size"])
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=workers, pin_memory=pin,
                        persistent_workers=workers > 0)

    model = build_classifier(m["arch"], n_classes, m["pretrained"], m["dropout"]).to(device)
    weight = class_weights(y[tr], n_classes).to(device) if t["class_weights"] == "balanced" else None
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=t["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    freeze_epochs = m.get("freeze_backbone_epochs", 0)
    if freeze_epochs:
        set_backbone_trainable(model, False)

    def snapshot(state) -> dict:
        return {k: v.detach().cpu().clone() for k, v in state.items()}

    best = {"f1": -1.0, "epoch": -1, "state": None, "probs": None}
    history, bad_epochs = [], 0
    for epoch in range(t["epochs"]):
        if freeze_epochs and epoch == freeze_epochs:
            set_backbone_trainable(model, True)
        model.train()
        running = torch.zeros((), device=device)
        for x_cpu, y_cpu in train_dl:
            x, yb = x_cpu.to(device, non_blocking=True), y_cpu.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(x), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.detach() * len(yb)
        sched.step()

        probs = predict(model, val_dl, device)
        f1 = f1_score(y[va], probs.argmax(1), average="macro")
        metrics = {"train_loss": running.item() / len(train_ds), "val_macro_f1": f1,
                   "val_acc": float((probs.argmax(1) == y[va]).mean())}
        history.append(metrics)
        if f1 > best["f1"]:
            best.update(f1=f1, epoch=epoch, state=snapshot(model.state_dict()), probs=probs)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if on_epoch:
            on_epoch(epoch, f1, metrics)
        if bad_epochs >= t["patience"]:
            break

    model.load_state_dict(best["state"])
    return {"model": model.cpu(), "best_f1": best["f1"], "best_epoch": best["epoch"],
            "history": history, "val_probs": best["probs"]}


def evaluate_split(model, images, y, idx, device=None) -> np.ndarray:
    device = device or get_device()
    ds = TensorImageDataset(images[idx], y[idx], build_transform("none", train=False))
    return predict(model.to(device), DataLoader(ds, batch_size=64), device)
