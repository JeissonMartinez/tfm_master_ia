"""Unified model loading/building for all supported families.

Supports: YOLO11, YOLO26 (Ultralytics), MobileNetV2, MobileNetV3 (TF/Keras).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .utils_io import log

# ── YOLO Model Specs ─────────────────────────────────────────────────
YOLO11_SPECS: Dict[str, Dict[str, Any]] = {
    "yolo11n": {"params": "2.6M", "gflops": "6.5", "map50_95_coco": 39.5, "cpu_ms": 56.1, "gpu_ms": 1.5},
    "yolo11s": {"params": "9.4M", "gflops": "21.5", "map50_95_coco": 47.0, "cpu_ms": 90.0, "gpu_ms": 2.5},
    "yolo11m": {"params": "20.1M", "gflops": "68.0", "map50_95_coco": 51.5, "cpu_ms": 183.2, "gpu_ms": 4.7},
    "yolo11l": {"params": "25.3M", "gflops": "86.9", "map50_95_coco": 53.4, "cpu_ms": 238.6, "gpu_ms": 6.2},
    "yolo11x": {"params": "56.9M", "gflops": "194.9", "map50_95_coco": 54.7, "cpu_ms": 462.8, "gpu_ms": 11.3},
}

YOLO26_SPECS: Dict[str, Dict[str, Any]] = {
    "yolo26n": {"params": "2.4M", "gflops": "5.4", "map50_95_coco": 40.1, "cpu_ms": 38.9, "gpu_ms": 1.7},
    "yolo26s": {"params": "9.5M", "gflops": "20.7", "map50_95_coco": 47.8, "cpu_ms": 87.2, "gpu_ms": 2.5},
    "yolo26m": {"params": "20.4M", "gflops": "68.2", "map50_95_coco": 52.5, "cpu_ms": 220.0, "gpu_ms": 4.7},
    "yolo26l": {"params": "24.8M", "gflops": "86.4", "map50_95_coco": 54.4, "cpu_ms": 286.2, "gpu_ms": 6.2},
    "yolo26x": {"params": "55.7M", "gflops": "193.9", "map50_95_coco": 56.9, "cpu_ms": 525.8, "gpu_ms": 11.8},
}

MOBILENET_SPECS: Dict[str, Dict[str, Any]] = {
    "MobileNetV2_SSDLite": {
        "backbone": "MobileNetV2",
        "default_alpha": 0.35,
        "params_est": "~0.5-1.5M",
        "int8_size_est_kb": "200-600",
    },
    "MobileNetV3S_SSDLite": {
        "backbone": "MobileNetV3 Small",
        "default_alpha": 1.0,
        "params_est": "~1.0-1.5M",
        "int8_size_est_kb": "600-800",
    },
    "MobileNetV3L_SSDLite": {
        "backbone": "MobileNetV3 Large",
        "default_alpha": 0.75,
        "params_est": "~2.0-3.0M",
        "int8_size_est_kb": "1000-1500",
    },
}


def get_all_specs() -> Dict[str, Dict[str, Any]]:
    """Return a merged dict of all model specs."""
    merged = {}
    merged.update(YOLO11_SPECS)
    merged.update(YOLO26_SPECS)
    merged.update(MOBILENET_SPECS)
    return merged


# ── YOLO model loading ───────────────────────────────────────────────

def _check_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
        return True
    except ImportError:
        log("❌ Ultralytics no disponible. Instala con: pip install ultralytics")
        return False


def load_yolo_model(
    family: str,
    variant: str,
    verbose: bool = True,
) -> Optional[Any]:
    """Load a YOLO model via Ultralytics.

    Works identically for YOLO11 and YOLO26 — the architecture is
    resolved from the ``.pt`` file metadata.

    Args:
        family: ``"YOLO11"`` or ``"YOLO26"``
        variant: e.g. ``"yolo11n"`` or ``"yolo26n"``
        verbose: Print loading info

    Returns:
        YOLO model or None.
    """
    if not _check_ultralytics():
        return None
    from ultralytics import YOLO  # type: ignore

    model_name = f"{variant}.pt"
    try:
        if verbose:
            log(f"🔄 Cargando modelo: {model_name}")
        model = YOLO(model_name)

        specs_dict = YOLO11_SPECS if family == "YOLO11" else YOLO26_SPECS
        specs = specs_dict.get(variant)
        if verbose and specs:
            log(f"✅ Modelo cargado: {model_name}")
            log(f"   📊 Params: {specs['params']} | FLOPs: {specs['gflops']}G")
            log(f"   🎯 mAP50-95 COCO: {specs['map50_95_coco']}%")
            log(f"   ⚡ CPU: {specs['cpu_ms']}ms | GPU: {specs['gpu_ms']}ms")
        elif verbose:
            log(f"✅ Modelo cargado: {model_name}")
        return model
    except Exception as exc:
        log(f"❌ Error cargando {model_name}: {exc}")
        return None


def get_yolo_model_info(model: Any) -> Dict[str, Any]:
    """Extract parameter counts from a loaded YOLO model."""
    if model is None:
        return {}
    try:
        info: Dict[str, Any] = {
            "task": getattr(model, "task", "detect"),
            "names": getattr(model, "names", {}),
            "num_classes": len(getattr(model, "names", {})),
        }
        if hasattr(model, "model"):
            pm = model.model
            total = sum(p.numel() for p in pm.parameters())
            trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
            info["total_params"] = total
            info["trainable_params"] = trainable
            info["total_params_m"] = total / 1e6
            info["trainable_params_m"] = trainable / 1e6
        return info
    except Exception as exc:
        log(f"⚠️ Error info modelo: {exc}")
        return {}


# ── MobileNet model building ────────────────────────────────────────

def _ssd_lite_conv_block(x, filters, kernel=3, stride=1, bn=True, name=""):
    import tensorflow as tf
    x = tf.keras.layers.DepthwiseConv2D(
        kernel, strides=stride, padding="same", use_bias=not bn,
        name=f"{name}_dw" if name else None)(x)
    if bn:
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1" if name else None)(x)
    x = tf.keras.layers.ReLU(6.0, name=f"{name}_r1" if name else None)(x)
    x = tf.keras.layers.Conv2D(
        filters, 1, padding="same", use_bias=not bn,
        name=f"{name}_pw" if name else None)(x)
    if bn:
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2" if name else None)(x)
    x = tf.keras.layers.ReLU(6.0, name=f"{name}_r2" if name else None)(x)
    return x


def _ssd_lite_head(features, num_anchors, num_classes, feat_ch=128, name="ssd"):
    import tensorflow as tf
    x = _ssd_lite_conv_block(features, feat_ch, name=f"{name}_sh1")
    x = _ssd_lite_conv_block(x, feat_ch, name=f"{name}_sh2")
    h, w = x.shape[1], x.shape[2]
    total = h * w * num_anchors

    obj = tf.keras.layers.Conv2D(num_anchors, 1, padding="same", name=f"{name}_obj_c")(x)
    obj = tf.keras.layers.Reshape((total, 1), name=f"{name}_obj_r")(obj)
    obj = tf.keras.layers.Activation("sigmoid", name="objectness")(obj)

    cls = tf.keras.layers.Conv2D(num_anchors * num_classes, 1, padding="same", name=f"{name}_cls_c")(x)
    cls = tf.keras.layers.Reshape((total, num_classes), name=f"{name}_cls_r")(cls)
    cls = tf.keras.layers.Activation("sigmoid", name="class_out")(cls)

    bbox = tf.keras.layers.Conv2D(num_anchors * 4, 1, padding="same", name=f"{name}_bb_c")(x)
    bbox = tf.keras.layers.Reshape((total, 4), name=f"{name}_bb_r")(bbox)
    bbox = tf.keras.layers.Activation("sigmoid", name="bbox_out")(bbox)

    return {"objectness": obj, "class_out": cls, "bbox_out": bbox}


def build_mobilenet_ssd(
    version: str = "V3",
    variant: str = "Small",
    num_classes: int = 4,
    num_anchors_per_cell: int = 9,
    img_size: int = 224,
    alpha: float = 1.0,
    minimalistic: bool = True,
    dropout_rate: float = 0.2,
    feature_channels: int = 128,
    l2_reg: float = 0.0,
) -> Any:
    """Build a MobileNet + SSD-Lite detection model.

    Args:
        version: ``"V2"`` or ``"V3"``
        variant: ``"Small"`` or ``"Large"`` (only for V3)
        Other args map directly to Keras builder params.

    Returns:
        Compiled Keras ``Model`` with outputs objectness, class_out, bbox_out.
    """
    import tensorflow as tf

    input_shape = (img_size, img_size, 3)

    if version == "V2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, alpha=alpha,
            include_top=False, weights="imagenet",
        )
        model_name = "MobileNetV2_SSDLite"
    elif version == "V3" and variant == "Small":
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape, alpha=alpha,
            minimalistic=minimalistic, include_top=False,
            weights="imagenet", include_preprocessing=False,
        )
        model_name = "MobileNetV3S_SSDLite"
    elif version == "V3" and variant == "Large":
        base = tf.keras.applications.MobileNetV3Large(
            input_shape=input_shape, alpha=alpha,
            minimalistic=minimalistic, include_top=False,
            weights="imagenet", include_preprocessing=False,
        )
        model_name = "MobileNetV3L_SSDLite"
    else:
        raise ValueError(f"Versión/variante no soportada: {version}/{variant}")

    features = base.output
    if dropout_rate > 0:
        features = tf.keras.layers.SpatialDropout2D(
            dropout_rate, name="feat_dropout")(features)

    outputs = _ssd_lite_head(
        features, num_anchors_per_cell, num_classes,
        feat_ch=feature_channels, name="ssd_lite",
    )

    model = tf.keras.Model(inputs=base.input, outputs=outputs, name=model_name)
    log(f"✅ Modelo construido: {model_name} (alpha={alpha})")
    return model


def get_backbone_layers(model, backbone_name: str = "mobilenetv3") -> List[str]:
    """Return layer names belonging to the backbone (for freezing)."""
    names = []
    for layer in model.layers:
        n = layer.name.lower()
        if any(p in n for p in [
            "expanded_conv", "block_", "conv", "bn", "re_lu",
            "multiply", "add", "rescaling", "normalization"
        ]):
            if "ssd" not in n:
                names.append(layer.name)
    return names


# ── Unified summary ──────────────────────────────────────────────────

def print_model_summary(model: Any, model_family: str) -> None:
    """Print a standardised summary for any supported model."""
    from .config import is_yolo_family

    if is_yolo_family(model_family):
        info = get_yolo_model_info(model)
        print(f"\n📦 Modelo YOLO: {model_family}")
        if "total_params_m" in info:
            print(f"  Total params:     {info['total_params']:>12,}")
            print(f"  Trainable params: {info['trainable_params']:>12,}")
            print(f"  Params (M):       {info['total_params_m']:>12.2f}")
        print(f"  Clases:           {info.get('num_classes', '?')}")
    else:
        import tensorflow as tf
        trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
        total = model.count_params()
        print(f"\n📦 Modelo: {model.name}")
        print(f"  Total params:     {total:>12,}")
        print(f"  Trainable:        {trainable:>12,}")
        print(f"  Non-trainable:    {non_trainable:>12,}")
        est_fp32 = total * 4 / 1024 / 1024
        est_int8 = total / 1024 / 1024
        print(f"  Est. float32:     {est_fp32:>10.2f} MB")
        print(f"  Est. INT8:        {est_int8:>10.2f} MB")


def estimate_model_size(model: Any, model_family: str) -> Dict[str, float]:
    """Estimate model sizes (float32, int8) in MB."""
    from .config import is_yolo_family

    if is_yolo_family(model_family):
        info = get_yolo_model_info(model)
        total = info.get("total_params", 0)
    else:
        total = model.count_params()

    return {
        "params": total,
        "float32_mb": total * 4 / 1024 / 1024,
        "int8_mb": total / 1024 / 1024,
    }


def estimate_esp32_inference(
    model_family: str,
    variant: str,
) -> Optional[Dict[str, Any]]:
    """Look up estimated ESP32-S3 inference time from benchmarks."""
    specs = get_all_specs().get(variant)
    if specs and "cpu_ms" in specs:
        cpu_ms = specs["cpu_ms"]
        # ESP32-S3 is ~10-15x slower than desktop CPU for INT8
        estimated_esp32_ms = cpu_ms * 12
        return {
            "desktop_cpu_ms": cpu_ms,
            "estimated_esp32_ms": estimated_esp32_ms,
            "estimated_esp32_fps": 1000 / estimated_esp32_ms if estimated_esp32_ms > 0 else 0,
        }
    return None
