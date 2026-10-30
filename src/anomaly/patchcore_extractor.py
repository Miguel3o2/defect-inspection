"""Torch feature extractor for PatchCore: frozen ImageNet backbone, layer2+layer3 with local aggregation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

_WEIGHTS = {
    "wide_resnet50_2": models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
    "resnet18": models.ResNet18_Weights.DEFAULT,
}


class PatchCoreExtractor(nn.Module):
    def __init__(self, backbone: str = "wide_resnet50_2", out_dim: int = 1024, pretrained: bool = True):
        super().__init__()
        if backbone not in _WEIGHTS:
            raise ValueError(f"unsupported backbone {backbone!r}; choose from {list(_WEIGHTS)}")
        net = getattr(models, backbone)(weights=_WEIGHTS[backbone] if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.out_dim = out_dim
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """x [B,3,H,W] -> (features [B, h*w, D], (h, w)); h = H/8, w = W/8."""
        f2 = self.layer2(self.layer1(self.stem(x)))
        f3 = self.layer3(f2)
        f2, f3 = F.avg_pool2d(f2, 3, 1, 1), F.avg_pool2d(f3, 3, 1, 1)   # local neighbourhood aggregation
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        f = F.adaptive_avg_pool1d(f, min(self.out_dim, c)).squeeze(1)
        return f.reshape(b, h * w, -1), (h, w)
