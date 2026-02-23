# ESPDet — AGPL-3.0 License
# Ported from https://github.com/espressif/esp-detection/blob/main/nn/modules/esp_conv.py
"""Depthwise Separable Convolution used throughout ESPDet."""

import torch.nn as nn

__all__ = ("DSConv",)


class DSConv(nn.Module):
    """Depthwise Separable Convolution: DW Conv2d → BN → Act → PW Conv2d → BN → Act."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1,
                 p: int = 1, act=nn.ReLU(inplace=False)):
        super().__init__()
        self.depthwise = nn.Conv2d(c1, c1, k, s, p, groups=c1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pointwise = nn.Conv2d(c1, c2, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = act

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x
