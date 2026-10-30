"""Supervised good-vs-defective scorers for the unseen-defect comparison.

A scorer is `fn(train_paths, y, all_paths, seed, cfg) -> np.ndarray` of defect scores for `all_paths`
(higher = more defective). It must be trained ONLY on (train_paths, y). Fixed hyperparameters/epochs:
no model selection on the evaluation images.
"""
from __future__ import annotations

from functools import cache

import cv2
import numpy as np

from src.data.mvtec import load_display, load_image
from src.features.handcrafted import extract


@cache
def _handcrafted(path: str, resize: int, crop: int) -> np.ndarray:
    gray = cv2.cvtColor(load_display(path, resize, crop), cv2.COLOR_RGB2GRAY)
    return extract(gray)


def handcrafted_lgbm(train_paths, y, all_paths, seed, cfg) -> np.ndarray:
    from lightgbm import LGBMClassifier

    def f(paths):
        return np.stack([_handcrafted(p, cfg["resize"], cfg["crop"]) for p in paths])

    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", random_state=seed,
                         verbose=-1)
    clf.fit(f(train_paths), y)
    return clf.predict_proba(f(all_paths))[:, 1]


def resnet18_binary(train_paths, y, all_paths, seed, cfg) -> np.ndarray:
    import torch
    from torch import nn
    from torchvision import models

    r = cfg["resnet"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if r["pretrained"] else None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.to(device)

    X = torch.from_numpy(np.stack([load_image(p, cfg["resize"], cfg["crop"]) for p in train_paths]))
    Y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    counts = np.bincount(np.asarray(y), minlength=2)
    weight = torch.tensor(len(y) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(net.parameters(), lr=r["lr"], weight_decay=r["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=r["epochs"])
    bs = r["batch_size"]
    for _ in range(r["epochs"]):
        net.train()
        perm = torch.randperm(len(Y))
        for s in range(0, len(Y), bs):
            idx = perm[s:s + bs]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            opt.zero_grad(set_to_none=True)
            loss_fn(net(X[idx].to(device, non_blocking=True)), Y[idx].to(device, non_blocking=True)).backward()
            opt.step()
        sched.step()

    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(all_paths), 32):
            xb = np.stack([load_image(p, cfg["resize"], cfg["crop"]) for p in all_paths[s:s + 32]])
            out.append(net(torch.from_numpy(xb).to(device)).softmax(1)[:, 1].cpu())
    return torch.cat(out).numpy()


SCORERS = {"handcrafted_lgbm": handcrafted_lgbm, "resnet18": resnet18_binary}
