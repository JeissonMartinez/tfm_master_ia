"""Unified model loading/building for Cycle 2 families (PyTorch only).

Supports: FCOS (MobileNetV3-Small + FPN + FCOS head),
          YOLO26_CUSTOM (Ultralytics backbone, manual training loop),
          ESPDet (ESPDet-Pico, custom Espressif-inspired architecture).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .utils_io import log


# =====================================================================
#  Model specs (reference)
# =====================================================================

FCOS_SPECS: Dict[str, Dict[str, Any]] = {
    "fcos_v3s": {
        "backbone": "MobileNetV3-Small",
        "params_est": "~1.5-2M",
        "fpn_channels": 64,
        "target_size_kb": "< 2 MB INT8",
    },
}

YOLO26_CUSTOM_SPECS: Dict[str, Dict[str, Any]] = {
    "yolo26n": {
        "params": "2.4M",
        "gflops": "5.4",
        "map50_95_coco": 40.1,
        "cpu_ms": 38.9,
        "gpu_ms": 1.7,
    },
}

ESPDET_SPECS: Dict[str, Dict[str, Any]] = {
    "espdet_pico": {
        "params_est": "~0.36M",
        "target_size_kb": "< 500 KB INT8",
        "reg_max": 1,
        "description": "Ultra-light anchor-free head, custom for ESP32-S3",
    },
}


def get_all_specs() -> Dict[str, Dict[str, Any]]:
    merged = {}
    merged.update(FCOS_SPECS)
    merged.update(YOLO26_CUSTOM_SPECS)
    merged.update(ESPDET_SPECS)
    return merged


# =====================================================================
#  FCOS — MobileNetV3-Small + FPN + FCOS Head
# =====================================================================

class SimpleFPN(nn.Module):
    """Lightweight Feature Pyramid Network for small backbones."""

    def __init__(self, in_channels_list: List[int], out_channels: int = 64):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        # Top-down path
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + nn.functional.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode="nearest"
            )
        return [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]


class FCOSHead(nn.Module):
    """Fully Convolutional One-Stage detection head."""

    def __init__(self, in_channels: int, num_classes: int, num_convs: int = 2):
        super().__init__()
        cls_layers = []
        reg_layers = []
        for _ in range(num_convs):
            cls_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(8, in_channels),
                nn.ReLU(inplace=True),
            ])
            reg_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(8, in_channels),
                nn.ReLU(inplace=True),
            ])
        self.cls_tower = nn.Sequential(*cls_layers)
        self.reg_tower = nn.Sequential(*reg_layers)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Init
        for modules in [self.cls_tower, self.reg_tower]:
            for layer in modules:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.constant_(layer.bias, 0)
        # Bias init for cls to avoid explosion
        nn.init.constant_(self.cls_logits.bias, -math.log(99))

    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        cls_outs, reg_outs, ctr_outs = [], [], []
        for feat in features:
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)
            cls_outs.append(self.cls_logits(cls_feat))
            reg_outs.append(nn.functional.relu(self.bbox_pred(reg_feat)))
            ctr_outs.append(self.centerness(cls_feat))
        return {"cls": cls_outs, "reg": reg_outs, "centerness": ctr_outs}


class FCOSModel(nn.Module):
    """MobileNetV3-Small + FPN + FCOS head."""

    def __init__(
        self,
        num_classes: int = 5,
        fpn_channels: int = 64,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        import torchvision.models as models
        from torchvision.models.feature_extraction import create_feature_extractor

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
        )
        # Extract features at 3 scales
        self.backbone = create_feature_extractor(backbone.features, {
            "3": "feat0",    # stride 8
            "8": "feat1",    # stride 16
            "12": "feat2",   # stride 32 (last inverted residual)
        })
        # Determine channel dims with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            in_channels = [feats[k].shape[1] for k in sorted(feats.keys())]

        self.fpn = SimpleFPN(in_channels, fpn_channels)
        self.head = FCOSHead(fpn_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        feats = self.backbone(x)
        fpn_feats = self.fpn([feats[k] for k in sorted(feats.keys())])
        return self.head(fpn_feats)


def build_fcos_model(
    num_classes: int = 5,
    fpn_channels: int = 64,
    pretrained_backbone: bool = True,
    device: str = "cpu",
) -> FCOSModel:
    """Build an FCOS model with MobileNetV3-Small backbone."""
    model = FCOSModel(num_classes, fpn_channels, pretrained_backbone)
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"✅ FCOS (MobileNetV3-S) construido: {total:,} params ({trainable:,} trainable)")
    log(f"   FPN channels: {fpn_channels} | Classes: {num_classes}")
    return model


# =====================================================================
#  YOLO26_CUSTOM — Ultralytics backbone, manual training loop
# =====================================================================

def build_yolo26_custom_model(
    variant: str = "yolo26n",
    num_classes: int = 5,
    pretrained: bool = True,
    device: str = "cpu",
) -> Any:
    """Load a YOLO26 model via Ultralytics for manual training.

    Returns the Ultralytics YOLO object.  The actual ``nn.Module`` is
    accessible at ``model.model``.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        log("❌ Ultralytics no instalada.  pip install ultralytics")
        return None

    pt_name = f"{variant}.pt" if pretrained else f"{variant}.yaml"
    log(f"🔄 Cargando YOLO26 Custom: {pt_name}")
    model = YOLO(pt_name)

    specs = YOLO26_CUSTOM_SPECS.get(variant)
    if specs:
        log(f"   📊 Referencia COCO: {specs.get('params', '?')} params | "
            f"mAP50-95={specs.get('map50_95_coco', '?')}%")

    return model


def get_yolo26_custom_torch_model(yolo_obj: Any) -> nn.Module:
    """Extract the underlying ``nn.Module`` from an Ultralytics YOLO object."""
    return yolo_obj.model


# =====================================================================
#  ESPDet-Pico — Ultra-light anchor-free detector
# =====================================================================

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, act: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride, kernel // 2,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class ESPDetPicoBackbone(nn.Module):
    """Ultra-light backbone for ESPDet-Pico (~0.36M params total)."""

    def __init__(self, width_mult: float = 0.5):
        super().__init__()
        # Channel list scaled by width multiplier
        def ch(c: int) -> int:
            return max(8, int(c * width_mult))

        self.stem = nn.Sequential(
            nn.Conv2d(3, ch(16), 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch(16)),
            nn.ReLU6(inplace=True),
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(ch(16), ch(32), stride=2),
            DepthwiseSeparableConv(ch(32), ch(32)),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(ch(32), ch(64), stride=2),
            DepthwiseSeparableConv(ch(64), ch(64)),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(ch(64), ch(128), stride=2),
            DepthwiseSeparableConv(ch(128), ch(128)),
        )
        self.out_channels = [ch(32), ch(64), ch(128)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c1 = self.stage1(x)   # stride 4
        c2 = self.stage2(c1)  # stride 8
        c3 = self.stage3(c2)  # stride 16
        return [c1, c2, c3]


class ESPDetPicoHead(nn.Module):
    """Anchor-free detection head with minimal reg_max."""

    def __init__(self, in_channels: int, num_classes: int, reg_max: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.cls_conv = DepthwiseSeparableConv(in_channels, in_channels)
        self.reg_conv = DepthwiseSeparableConv(in_channels, in_channels)
        self.cls_out = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_out = nn.Conv2d(in_channels, 4 * (reg_max + 1), 1)

        nn.init.constant_(self.cls_out.bias, -math.log(99))

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls = self.cls_out(self.cls_conv(feat))
        reg = self.reg_out(self.reg_conv(feat))
        return cls, reg


class ESPDetPico(nn.Module):
    """ESPDet-Pico: ultra-lightweight detector for ESP32-S3."""

    def __init__(
        self,
        num_classes: int = 5,
        width_mult: float = 0.5,
        reg_max: int = 1,
    ):
        super().__init__()
        self.backbone = ESPDetPicoBackbone(width_mult)
        self.num_classes = num_classes

        # Simple FPN for multi-scale fusion
        ch_list = self.backbone.out_channels
        fpn_ch = ch_list[0]  # smallest channel as FPN dim
        self.fpn = SimpleFPN(ch_list, fpn_ch)
        self.heads = nn.ModuleList([
            ESPDetPicoHead(fpn_ch, num_classes, reg_max) for _ in ch_list
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
        features = self.backbone(x)
        fpn_feats = self.fpn(features)
        outputs = [head(feat) for head, feat in zip(self.heads, fpn_feats)]
        return {
            "cls": [o[0] for o in outputs],
            "reg": [o[1] for o in outputs],
        }


def build_espdet_pico(
    num_classes: int = 5,
    width_mult: float = 0.5,
    reg_max: int = 1,
    pretrained_weights: Optional[str] = None,
    device: str = "cpu",
) -> ESPDetPico:
    """Build an ESPDet-Pico model."""
    model = ESPDetPico(num_classes, width_mult, reg_max)
    if pretrained_weights:
        state = torch.load(pretrained_weights, map_location="cpu")
        model.load_state_dict(state, strict=False)
        log(f"✅ Pesos pre-entrenados cargados: {pretrained_weights}")
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"✅ ESPDet-Pico construido: {total:,} params ({trainable:,} trainable)")
    log(f"   Width mult: {width_mult} | reg_max: {reg_max} | Classes: {num_classes}")
    return model


# =====================================================================
#  Unified helpers
# =====================================================================

def freeze_backbone(model: nn.Module, family: str) -> int:
    """Freeze backbone parameters (Phase 1). Returns count of frozen params."""
    frozen = 0
    if family == "FCOS":
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
                frozen += param.numel()
    elif family == "YOLO26_CUSTOM":
        # Freeze first N layers of the Ultralytics model
        torch_model = model.model if hasattr(model, "model") else model
        for i, (name, param) in enumerate(torch_model.named_parameters()):
            if i < 50:  # freeze ~ first 50 param groups (backbone)
                param.requires_grad = False
                frozen += param.numel()
    elif family == "ESPDet":
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
                frozen += param.numel()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"🔒 Backbone congelado: {frozen:,} params frozen")
    log(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return frozen


def unfreeze_all(model: nn.Module) -> int:
    """Unfreeze all parameters (Phase 2). Returns count of unfrozen params."""
    unfrozen = 0
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen += param.numel()
    total = sum(p.numel() for p in model.parameters())
    log(f"🔓 Todas las capas desbloqueadas: {unfrozen:,} params unfrozen")
    log(f"   Total trainable: {total:,}")
    return unfrozen


def print_model_summary(model: nn.Module, family: str) -> None:
    """Print a standardized parameter summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print(f"\n📦 Modelo: {family}")
    print(f"  Total params:     {total:>12,}")
    print(f"  Trainable:        {trainable:>12,}")
    print(f"  Non-trainable:    {non_trainable:>12,}")
    est_fp32 = total * 4 / 1024 / 1024
    est_int8 = total / 1024 / 1024
    print(f"  Est. float32:     {est_fp32:>10.2f} MB")
    print(f"  Est. INT8:        {est_int8:>10.2f} MB")


def estimate_model_size(model: nn.Module) -> Dict[str, float]:
    """Estimate model sizes (float32, int8) in MB."""
    total = sum(p.numel() for p in model.parameters())
    return {
        "params": total,
        "float32_mb": total * 4 / 1024 / 1024,
        "int8_mb": total / 1024 / 1024,
    }
