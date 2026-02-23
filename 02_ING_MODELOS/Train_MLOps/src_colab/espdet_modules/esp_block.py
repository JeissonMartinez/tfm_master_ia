# ESPDet — AGPL-3.0 License
# Ported from https://github.com/espressif/esp-detection/blob/main/nn/modules/esp_block.py
"""ESP building blocks: DSBottleneck, DSC3k2, ESPBlock, ESPBlockLite."""

import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, C3, Bottleneck

from .esp_conv import DSConv

__all__ = (
    "C3k",
    "DSBottleneck",
    "DSC3k2",
    "ESPBlock",
    "ESPBlockLite",
    "ESPSerial",
    "ESPSerialLite",
)


# ─────────────────────────────────────────────────────────────
#  C3k — CSP bottleneck with custom kernel sizes
# ─────────────────────────────────────────────────────────────

class C3k(C3):
    """C3k: CSP bottleneck with customizable kernel sizes."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


# ─────────────────────────────────────────────────────────────
#  DSBottleneck — Bottleneck with DSConv instead of Conv
# ─────────────────────────────────────────────────────────────

class DSBottleneck(nn.Module):
    """Replace Conv in standard bottleneck with DSConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k[0], 1)
        self.cv2 = DSConv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ─────────────────────────────────────────────────────────────
#  DSC3k2 — C2f variant with DSBottleneck
# ─────────────────────────────────────────────────────────────

class DSC3k2(C2f):
    """Replace the standard bottleneck in C2f with DSBottleneck (or C3k)."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k
            else DSBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


# ─────────────────────────────────────────────────────────────
#  ESPSerial / ESPBlock / ESPBlockLite
# ─────────────────────────────────────────────────────────────

class ESPSerial(nn.Module):
    """Serial ESP block for the neck (top-down / bottom-up fusion)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(n * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(2 * self.c, self.c, shortcut, g,
                       k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        x = [self.cv1(x)]
        x.extend(m(x[-1]) for m in self.m)
        return self.cv2(x[-1])


class ESPSerialLite(nn.Module):
    """Lite variant of ESPSerial."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g,
                       k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        x = [self.cv1(x)]
        x.extend(m(x[-1]) for m in self.m)
        return self.cv2(x[-1])


class ESPBlock(ESPSerial):
    """ESP block with optional C3k inside."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(2 * self.c, self.c, 2, shortcut, g) if c3k
            else DSBottleneck(2 * self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class ESPBlockLite(ESPSerialLite):
    """Lite variant of ESPBlock with optional C3k inside."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=False):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k
            else DSBottleneck(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )
