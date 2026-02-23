"""Unified model loading/building for Cycle 2 families (PyTorch only).

Supports: FCOS (MobileNetV3-Small + FPN + FCOS head),
          YOLO26_CUSTOM (Ultralytics backbone, manual training loop),
          ESPDet (ESPDet-Pico, official Espressif architecture v2.6).
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
        "target_size_kb": "< 1.5 MB ONNX",
        "reg_max": 1,
        "strides": [8, 16, 32],
        "description": "Official Espressif architecture (esp-detection repo)",
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
#  ESPDet-Pico — Official Espressif Architecture (0.36M params)
# =====================================================================
#
# Faithful reimplementation of the ESPDet-Pico topology from
# https://github.com/espressif/esp-detection (AGPL-3.0).
#
# Architecture YAML reference: cfg/models/espdet_pico.yaml
#   scale 'n': [depth=0.50, width=0.25, max_channels=512]
#
# Channel computation: each YAML arg is scaled by width (0.25) and
# made divisible by 8 via ``make_divisible``.  Depth (0.50) affects
# repeat counts (rounded up to 1).
#
# Strides: P3/8, P4/16, P5/32 (official, NOT the old [4,8,16]).
# =====================================================================

def _make_divisible(v: int, divisor: int = 8) -> int:
    """Equivalent to ultralytics ``make_divisible``."""
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


def _ch(yaml_ch: int, width: float = 0.25, max_ch: int = 512) -> int:
    """Scale a YAML channel arg by width multiplier (ESPDet-Pico 'n' scale)."""
    return _make_divisible(min(yaml_ch, max_ch) * width)


class ESPDetPico(nn.Module):
    """ESPDet-Pico: official Espressif ultra-light detector (0.36M params).

    Topology assembled manually from the YAML definition to avoid
    depending on Ultralytics ``DetectionModel`` / ``parse_model``.

    Strides: [8, 16, 32]  (P3, P4, P5)
    """

    def __init__(self, nc: int = 5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.block import SPPF
        from ultralytics.nn.modules.head import Detect  # only for SCDown
        # SCDown is in ultralytics.nn.modules.block for recent versions
        try:
            from ultralytics.nn.modules.block import SCDown
        except ImportError:
            from ultralytics.nn.modules import SCDown  # type: ignore

        from .espdet_modules import (
            DSConv, DSC3k2, ESPBlockLite, ESPBlock,
        )
        from .espdet_modules.esp_head import ESPDetectHead

        self.nc = nc
        w = 0.25   # width multiplier for 'n' scale
        mx = 512   # max_channels

        # ── Backbone (layers 0-10) ──────────────────────────────
        #  0: Conv(3 → 64*w=16, k=3, s=2)            → P1/2
        self.layer0 = Conv(3, _ch(64, w, mx), 3, 2)
        #  1: DSConv(16 → 128*w=32, k=3, s=2)         → P2/4
        self.layer1 = DSConv(_ch(64, w, mx), _ch(128, w, mx), 3, 2)
        #  2: ESPBlockLite(32 → 256*w=64, n=1*0.5→1, c3k=False)
        self.layer2 = ESPBlockLite(_ch(128, w, mx), _ch(256, w, mx),
                                   n=max(1, round(1 * 0.5)))
        #  3: DSConv(64 → 256*w=64, k=3, s=2)         → P3/8
        self.layer3 = DSConv(_ch(256, w, mx), _ch(256, w, mx), 3, 2)
        #  4: DSC3k2(64 → 256*w=64, n=2*0.5→1, c3k=False)
        self.layer4 = DSC3k2(_ch(256, w, mx), _ch(256, w, mx),
                             n=max(1, round(2 * 0.5)), c3k=False)
        #  5: SCDown(64 → 256*w=64, k=3, s=2)         → P4/16
        self.layer5 = SCDown(_ch(256, w, mx), _ch(256, w, mx), 3, 2)
        #  6: DSC3k2(64 → 256*w=64, n=2*0.5→1, c3k=True)
        self.layer6 = DSC3k2(_ch(256, w, mx), _ch(256, w, mx),
                             n=max(1, round(2 * 0.5)), c3k=True)
        #  7: SCDown(64 → 512*w=128, k=3, s=2)        → P5/32
        self.layer7 = SCDown(_ch(256, w, mx), _ch(512, w, mx), 3, 2)
        #  8: DSC3k2(128 → 512*w=128, n=2*0.5→1, c3k=True)
        self.layer8 = DSC3k2(_ch(512, w, mx), _ch(512, w, mx),
                             n=max(1, round(2 * 0.5)), c3k=True)
        #  9: SPPF(128 → 512*w=128, k=5)
        self.layer9 = SPPF(_ch(512, w, mx), _ch(512, w, mx), 5)
        # 10: DSConv(128 → 512*w=128, k=7, s=1, p=3)
        self.layer10 = DSConv(_ch(512, w, mx), _ch(512, w, mx), 7, 1, 3)

        # ── Neck — Top-down ─────────────────────────────────────
        # 11: Upsample (layer10 output → 2×)
        self.up11 = nn.Upsample(scale_factor=2, mode="nearest")
        # 12: Concat(up11, layer6) → 128+64=192 ch  (implicit in forward)
        # 13: ESPBlock(192 → 256*w=64, n=2*0.5→1)
        _cat12 = _ch(512, w, mx) + _ch(256, w, mx)  # concat channels
        self.layer13 = ESPBlock(_cat12, _ch(256, w, mx),
                                n=max(1, round(2 * 0.5)))

        # 14: Upsample (layer13 output → 2×)
        self.up14 = nn.Upsample(scale_factor=2, mode="nearest")
        # 15: Concat(up14, layer4) → 64+64=128 ch
        _cat15 = _ch(256, w, mx) + _ch(256, w, mx)
        # 16: ESPBlock(128 → 128*w=32, n=2*0.5→1)  — P3/8 output
        self.layer16 = ESPBlock(_cat15, _ch(128, w, mx),
                                n=max(1, round(2 * 0.5)))

        # ── Neck — Bottom-up ────────────────────────────────────
        # 17: DSConv(32 → 128*w=32, k=3, s=2)
        self.layer17 = DSConv(_ch(128, w, mx), _ch(128, w, mx), 3, 2)
        # 18: Concat(layer17, layer13) → 32+64=96 ch
        _cat18 = _ch(128, w, mx) + _ch(256, w, mx)
        # 19: ESPBlock(96 → 512*w=128, n=2*0.5→1) — P4/16 output
        self.layer19 = ESPBlock(_cat18, _ch(512, w, mx),
                                n=max(1, round(2 * 0.5)))

        # 20: DSConv(128 → 256*w=64, k=3, s=2)
        self.layer20 = DSConv(_ch(512, w, mx), _ch(256, w, mx), 3, 2)
        # 21: Concat(layer20, layer10) → 64+128=192 ch
        _cat21 = _ch(256, w, mx) + _ch(512, w, mx)
        # 22: ESPBlock(192 → 512*w=128, n=2*0.5→1) — P5/32 output
        self.layer22 = ESPBlock(_cat21, _ch(512, w, mx),
                                n=max(1, round(2 * 0.5)))

        # ── Detection Head ──────────────────────────────────────
        # Channel list for P3, P4, P5
        self.det_ch = (_ch(128, w, mx), _ch(512, w, mx), _ch(512, w, mx))
        self.head = ESPDetectHead(nc=nc, ch=self.det_ch)

        # flag for export mode
        self._export_mode = False

    def set_export_mode(self, mode: bool = True):
        """Toggle ONNX export mode (interleaved box/score output)."""
        self._export_mode = mode

    def forward(self, x: torch.Tensor):
        # ── Backbone ────────────────────────────────────────────
        x0 = self.layer0(x)        # P1/2   — 16ch
        x1 = self.layer1(x0)       # P2/4   — 32ch
        x2 = self.layer2(x1)       #         — 64ch
        x3 = self.layer3(x2)       # P3/8   — 64ch
        x4 = self.layer4(x3)       #         — 64ch  ← skip to neck top-down
        x5 = self.layer5(x4)       # P4/16  — 64ch
        x6 = self.layer6(x5)       #         — 64ch  ← skip to neck top-down
        x7 = self.layer7(x6)       # P5/32  — 128ch
        x8 = self.layer8(x7)       #         — 128ch
        x9 = self.layer9(x8)       #         — 128ch
        x10 = self.layer10(x9)     #         — 128ch ← skip to neck bottom-up

        # ── Neck — Top-down ─────────────────────────────────────
        up11 = self.up11(x10)               # 128ch, P4 resolution
        cat12 = torch.cat([up11, x6], 1)    # 128+64 = 192ch
        x13 = self.layer13(cat12)           # 64ch

        up14 = self.up14(x13)               # 64ch, P3 resolution
        cat15 = torch.cat([up14, x4], 1)    # 64+64 = 128ch
        x16 = self.layer16(cat15)           # 32ch   ← P3/8 detect

        # ── Neck — Bottom-up ────────────────────────────────────
        x17 = self.layer17(x16)             # 32ch, P4 resolution
        cat18 = torch.cat([x17, x13], 1)    # 32+64 = 96ch
        x19 = self.layer19(cat18)           # 128ch  ← P4/16 detect

        x20 = self.layer20(x19)             # 64ch, P5 resolution
        cat21 = torch.cat([x20, x10], 1)    # 64+128 = 192ch
        x22 = self.layer22(cat21)           # 128ch  ← P5/32 detect

        # ── Head ────────────────────────────────────────────────
        feats = [x16, x19, x22]
        if self._export_mode:
            return self.head.export_onnx_forward(feats)
        return self.head(feats)


def _convert_ultralytics_espdet_weights(
    ultralytics_state: dict,
    target_model: ESPDetPico,
) -> Tuple[dict, List[str], List[str]]:
    """Convert Ultralytics ESPDetPico state_dict keys to our naming.

    The Ultralytics model stores layers as ``model.{N}.{submodule}``
    where N is the layer index from the YAML (0-22 backbone+neck, 23 head).

    Our model uses ``layer{N}`` for backbone/neck and ``head.cv2/cv3``
    for the detection head.

    Returns:
        (converted_state_dict, matched_keys, unmatched_keys)
    """
    # Build mapping: ultralytics key prefix → our key prefix
    # Backbone layers 0-10
    prefix_map = {}
    for i in range(11):
        prefix_map[f"model.{i}."] = f"layer{i}."

    # Neck layers
    # 11 = Upsample (no params)
    # 12 = Concat (no params)
    prefix_map["model.13."] = "layer13."
    # 14 = Upsample (no params)
    # 15 = Concat (no params)
    prefix_map["model.16."] = "layer16."
    prefix_map["model.17."] = "layer17."
    # 18 = Concat (no params)
    prefix_map["model.19."] = "layer19."
    prefix_map["model.20."] = "layer20."
    # 21 = Concat (no params)
    prefix_map["model.22."] = "layer22."

    # Head (layer 23 in Ultralytics)
    prefix_map["model.23.cv2."] = "head.cv2."
    prefix_map["model.23.cv3."] = "head.cv3."

    converted = {}
    matched = []
    unmatched = []

    for key, value in ultralytics_state.items():
        found = False
        for old_prefix, new_prefix in prefix_map.items():
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix, 1)
                converted[new_key] = value
                matched.append(f"{key} → {new_key}")
                found = True
                break
        if not found:
            unmatched.append(key)

    return converted, matched, unmatched


def _load_ultralytics_espdet_pt(pt_path: str) -> dict:
    """Load a .pt from Ultralytics YOLO format and extract the raw state_dict.

    Ultralytics .pt files contain a pickled dict with key 'model' holding
    the full DetectionModel.  We extract its ``state_dict()``.

    The esp-detection repo pickles models with a local ``nn`` module namespace.
    We add shims so that ``torch.load`` can resolve the classes.
    """
    import sys as _sys

    # Shim: esp-detection pickle uses 'nn' as top-level module
    # which maps to ultralytics.nn (or the esp-detection nn extension)
    _shims_added = []
    try:
        import ultralytics.nn as _unn
        for mod_name in ["nn", "nn.modules", "nn.modules.conv",
                         "nn.modules.block", "nn.modules.head",
                         "nn.modules.transformer", "nn.modules.utils"]:
            if mod_name not in _sys.modules:
                # Map nn.X → ultralytics.nn.X
                ul_name = "ultralytics." + mod_name
                src = _sys.modules.get(ul_name)
                if src is not None:
                    _sys.modules[mod_name] = src
                    _shims_added.append(mod_name)

        # Shim: esp-detection custom modules (esp_conv, esp_block, esp_head)
        # Map nn.modules.esp_conv → our espdet_modules.esp_conv, etc.
        from . import espdet_modules
        from .espdet_modules import esp_conv, esp_block, esp_head
        for name, mod in [("nn.modules.esp_conv", esp_conv),
                          ("nn.modules.esp_block", esp_block),
                          ("nn.modules.esp_head", esp_head)]:
            if name not in _sys.modules:
                _sys.modules[name] = mod
                _shims_added.append(name)
    except ImportError:
        pass

    try:
        checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    finally:
        # Clean up shims to avoid polluting sys.modules
        for mod_name in _shims_added:
            _sys.modules.pop(mod_name, None)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_obj = checkpoint["model"]
        if hasattr(model_obj, "state_dict"):
            state = model_obj.float().state_dict()
        elif isinstance(model_obj, dict):
            state = model_obj
        else:
            state = model_obj
    elif hasattr(checkpoint, "state_dict"):
        state = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError(f"Cannot extract state_dict from {pt_path}")

    return state


def build_espdet_pico(
    num_classes: int = 5,
    pretrained_weights: Optional[str] = None,
    device: str = "cpu",
) -> ESPDetPico:
    """Build an ESPDet-Pico model with official Espressif architecture.

    Args:
        num_classes: Number of detection classes (default 5 for IODC).
        pretrained_weights: Path to Ultralytics .pt file (e.g. cat detection
            checkpoint from esp-detection repo).  Loaded with strict=False;
            only the final cls Conv2d layers won't match if nc differs.
        device: Target device.

    Returns:
        ESPDetPico model on the requested device.
    """
    model = ESPDetPico(nc=num_classes)

    if pretrained_weights and pretrained_weights.lower() not in ("none", "null", ""):
        log(f"🔄 Cargando pesos pretrained: {pretrained_weights}")
        raw_state = _load_ultralytics_espdet_pt(pretrained_weights)
        converted, matched, unmatched = _convert_ultralytics_espdet_weights(
            raw_state, model
        )

        # Filter out shape-mismatched params (e.g. nc=1 → nc=5 in cls head).
        # strict=False only skips missing/unexpected keys, NOT size mismatches.
        model_state = model.state_dict()
        skipped_shape = []
        filtered = {}
        for k, v in converted.items():
            if k in model_state and model_state[k].shape != v.shape:
                skipped_shape.append(k)
            else:
                filtered[k] = v

        load_info = model.load_state_dict(filtered, strict=False)
        n_loaded = len(filtered) - len(load_info.unexpected_keys)
        n_total = sum(1 for _ in model.named_parameters())
        log(f"  ✅ Transfer learning: {n_loaded} param groups cargados")
        if skipped_shape:
            log(f"  ℹ️  Shape mismatch (random init): {skipped_shape}")
        if load_info.missing_keys:
            log(f"  ℹ️  Missing keys (random init): {load_info.missing_keys[:10]}")
        if load_info.unexpected_keys:
            log(f"  ⚠️  Unexpected keys (ignored): {load_info.unexpected_keys[:10]}")
        if unmatched:
            log(f"  ℹ️  Ultralytics keys sin mapping ({len(unmatched)}): "
                f"{unmatched[:5]}")
    else:
        log("ℹ️  Sin pesos pretrained — inicialización aleatoria")

    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"✅ ESPDet-Pico (oficial) construido: {total:,} params "
        f"({trainable:,} trainable)")
    log(f"   Strides: [8, 16, 32] | Classes: {num_classes}")
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
        # Backbone = layers 0-10 in the official ESPDet-Pico topology
        backbone_prefixes = tuple(f"layer{i}." for i in range(11))
        for name, param in model.named_parameters():
            if name.startswith(backbone_prefixes):
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
