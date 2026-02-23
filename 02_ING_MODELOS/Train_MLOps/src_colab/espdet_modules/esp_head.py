# ESPDet — AGPL-3.0 License
# Ported from https://github.com/espressif/esp-detection/blob/main/nn/modules/esp_head.py
"""ESPDetect-style detection head (standalone, no Ultralytics Detect inheritance).

Replicates the cv2 (box) / cv3 (cls) branches of the official ESPDetect head
but returns separate cls/reg tensors compatible with our custom training loop
(``build_espdet_loss`` expects ``{"cls": [...], "reg": [...]}``).
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv

from .esp_conv import DSConv

__all__ = ("ESPDetectHead", "ESPDetect", "ESPDLDetect")


class ESPDetectHead(nn.Module):
    """Detection head that mirrors the official ESPDetect from esp-detection.

    For each FPN level it produces:
        - box: (B, 4*reg_max, H, W)  — l,t,r,b regression
        - cls: (B, nc, H, W)         — class logits

    ``forward()`` returns ``{"cls": [P3, P4, P5], "reg": [P3, P4, P5]}``
    for training.

    ``export_onnx_forward(feats)`` returns a flat tuple
    ``(box0, score0, box1, score1, box2, score2)`` for esp-ppq export.
    """

    def __init__(self, nc: int = 5, ch: tuple = (32, 128, 128)):
        super().__init__()
        self.nc = nc
        self.reg_max = 1
        self.nl = len(ch)  # number of detection levels

        # Channel sizing identical to official ESPDetect
        c2 = max(16, ch[0] // 4, self.reg_max * 4)   # box branch hidden ch
        c3 = max(ch[0], min(self.nc, 100))             # cls branch hidden ch

        # Box regression branches (cv2) — per level
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DSConv(x, c2, 3),
                DSConv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )

        # Classification branches (cv3) — per level
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

    def forward(self, feats: list) -> dict:
        """Training forward: returns dict with cls & reg lists."""
        cls_list = []
        reg_list = []
        for i in range(self.nl):
            reg_list.append(self.cv2[i](feats[i]))
            cls_list.append(self.cv3[i](feats[i]))
        return {"cls": cls_list, "reg": reg_list}

    def export_onnx_forward(self, feats: list) -> tuple:
        """ONNX export forward: interleaved (box0, score0, box1, score1, box2, score2)."""
        outputs = []
        for i in range(self.nl):
            outputs.append(self.cv2[i](feats[i]))   # box_i
            outputs.append(self.cv3[i](feats[i]))   # score_i
        return tuple(outputs)


# Aliases for pickle compatibility with esp-detection repo
ESPDetect = ESPDetectHead
ESPDLDetect = ESPDetectHead
