from __future__ import annotations

from torch import nn
from torchvision import models

_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet34": models.ResNet34_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
}


def build_classifier(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    if arch not in _WEIGHTS:
        raise ValueError(f"unsupported arch {arch!r}; choose from {list(_WEIGHTS)}")
    net = getattr(models, arch)(weights=_WEIGHTS[arch] if pretrained else None)
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))
    return net


def set_backbone_trainable(net: nn.Module, trainable: bool) -> None:
    for name, p in net.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = trainable
