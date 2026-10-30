import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import numpy as np

from src.data.augment import POLICIES, build_transform
from src.models.classifier import build_classifier
from src.train.engine import class_weights, fit


@pytest.mark.parametrize("policy", POLICIES)
def test_transform_output_shape(policy):
    from torchvision import tv_tensors
    x = tv_tensors.Image(torch.randint(0, 256, (1, 200, 200), dtype=torch.uint8))
    out = build_transform(policy, train=True)(x)
    assert tuple(out.shape) == (3, 200, 200) and out.dtype == torch.float32


def test_classifier_forward():
    net = build_classifier("resnet18", 6, pretrained=False, dropout=0.1)
    assert net(torch.randn(2, 3, 200, 200)).shape == (2, 6)


def test_class_weights_upweight_rare_class():
    w = class_weights(np.array([0] * 90 + [1] * 10), 2)
    assert w[1] > w[0]


def test_fit_runs_one_epoch_on_tiny_data():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 12)
    images = torch.from_numpy(rng.integers(0, 256, (len(y), 1, 64, 64), dtype=np.uint8))
    cfg = {"model": {"arch": "resnet18", "pretrained": False, "dropout": 0.0, "freeze_backbone_epochs": 0},
           "train": {"epochs": 1, "batch_size": 12, "lr": 1e-3, "weight_decay": 0.0, "label_smoothing": 0.0,
                     "class_weights": "balanced", "patience": 3, "num_workers": 0, "aug": "flip"}}
    idx = np.arange(len(y))
    res = fit(cfg, images, y, idx[::2], idx[1::2], 3)
    assert 0.0 <= res["best_f1"] <= 1.0 and len(res["history"]) == 1
