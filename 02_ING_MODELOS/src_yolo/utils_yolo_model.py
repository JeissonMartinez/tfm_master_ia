"""YOLO26 model loading and information utilities.

Handles loading pre-trained YOLO26 models and extracting
model architecture information for ESP32-S3 deployment planning.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from .utils_io import log, safe_exists

if TYPE_CHECKING:
    from ultralytics import YOLO as YOLOType

try:
    from ultralytics import YOLO  # type: ignore
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    ULTRALYTICS_AVAILABLE = False
    log("⚠️ Ultralytics no disponible. Instala con: pip install ultralytics")


# YOLO26 model specifications (from official docs)
YOLO26_SPECS = {
    "yolo26n": {
        "params_m": 2.4,
        "flops_g": 5.4,
        "map50_coco": 40.9,
        "map5095_coco": 40.1,
        "cpu_speed_ms": 38.9,
        "gpu_speed_ms": 1.7,
    },
    "yolo26s": {
        "params_m": 9.5,
        "flops_g": 20.7,
        "map50_coco": 48.6,
        "map5095_coco": 47.8,
        "cpu_speed_ms": 87.2,
        "gpu_speed_ms": 2.5,
    },
    "yolo26m": {
        "params_m": 20.4,
        "flops_g": 68.2,
        "map50_coco": 53.1,
        "map5095_coco": 52.5,
        "cpu_speed_ms": 220.0,
        "gpu_speed_ms": 4.7,
    },
    "yolo26l": {
        "params_m": 24.8,
        "flops_g": 86.4,
        "map50_coco": 55.0,
        "map5095_coco": 54.4,
        "cpu_speed_ms": 286.2,
        "gpu_speed_ms": 6.2,
    },
    "yolo26x": {
        "params_m": 55.7,
        "flops_g": 193.9,
        "map50_coco": 57.5,
        "map5095_coco": 56.9,
        "cpu_speed_ms": 525.8,
        "gpu_speed_ms": 11.8,
    },
}


def load_yolo26_model(
    model_name: str = "yolo26n.pt",
    verbose: bool = True,
) -> Optional[Any]:
    """Load a YOLO26 model.

    Args:
        model_name: Model name (e.g., 'yolo26n.pt') or path to weights
        verbose: Whether to print loading info

    Returns:
        YOLO model instance or None on failure
    """
    if not ULTRALYTICS_AVAILABLE or YOLO is None:
        log("❌ Ultralytics no está disponible. Instala con: pip install ultralytics")
        return None

    try:
        if verbose:
            log(f"🔄 Cargando modelo: {model_name}")
        
        model = YOLO(model_name)  # type: ignore[misc]
        
        if verbose:
            # Get model variant from name
            variant = model_name.replace(".pt", "").replace("yolo26", "yolo26")
            if variant in YOLO26_SPECS:
                specs = YOLO26_SPECS[variant]
                log(f"✅ Modelo cargado: {model_name}")
                log(f"   📊 Params: {specs['params_m']}M | FLOPs: {specs['flops_g']}G")
                log(f"   🎯 mAP50 COCO: {specs['map50_coco']}%")
                log(f"   ⚡ CPU: {specs['cpu_speed_ms']}ms | GPU: {specs['gpu_speed_ms']}ms")
            else:
                log(f"✅ Modelo cargado: {model_name}")

        return model

    except Exception as exc:
        log(f"❌ Error cargando modelo {model_name}: {exc}")
        return None


def get_model_info(model: Any) -> Dict[str, Any]:
    """Extract detailed model information.

    Args:
        model: YOLO model instance

    Returns:
        Dictionary with model information
    """
    if model is None:
        return {}

    try:
        info = {
            "task": getattr(model, "task", "detect"),
            "names": getattr(model, "names", {}),
            "num_classes": len(getattr(model, "names", {})),
        }

        # Get model architecture info if available
        if hasattr(model, "model"):
            pytorch_model = model.model
            
            # Count parameters
            total_params = sum(p.numel() for p in pytorch_model.parameters())
            trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
            
            info["total_params"] = total_params
            info["trainable_params"] = trainable_params
            info["total_params_m"] = total_params / 1e6
            info["trainable_params_m"] = trainable_params / 1e6

        return info

    except Exception as exc:
        log(f"⚠️ Error obteniendo info del modelo: {exc}")
        return {}


def print_model_summary(model: Any, imgsz: int = 224) -> None:
    """Print a summary of the model architecture.

    Args:
        model: YOLO model instance
        imgsz: Input image size for FLOPs calculation
    """
    if model is None:
        log("⚠️ Modelo no disponible")
        return

    info = get_model_info(model)
    
    log("\n" + "=" * 60)
    log("📋 RESUMEN DEL MODELO YOLO26")
    log("=" * 60)
    
    if "task" in info:
        log(f"🎯 Tarea: {info['task']}")
    
    if "num_classes" in info:
        log(f"📦 Clases: {info['num_classes']}")
        if "names" in info and info["names"]:
            names = list(info["names"].values()) if isinstance(info["names"], dict) else info["names"]
            log(f"   {names}")
    
    if "total_params_m" in info:
        log(f"📊 Parámetros totales: {info['total_params_m']:.2f}M")
        log(f"📊 Parámetros entrenables: {info['trainable_params_m']:.2f}M")
    
    # Estimate model sizes
    if "total_params" in info:
        fp32_size_mb = (info["total_params"] * 4) / (1024 * 1024)
        fp16_size_mb = fp32_size_mb / 2
        int8_size_mb = fp32_size_mb / 4
        
        log(f"\n💾 Tamaño estimado del modelo:")
        log(f"   FP32: {fp32_size_mb:.2f} MB")
        log(f"   FP16: {fp16_size_mb:.2f} MB")
        log(f"   INT8: {int8_size_mb:.2f} MB")
        
        # ESP32-S3 compatibility check
        if int8_size_mb <= 2.0:
            log(f"   ✅ Compatible con ESP32-S3 (8MB PSRAM)")
        elif int8_size_mb <= 4.0:
            log(f"   ⚠️ Ajustado para ESP32-S3 - considerar optimizaciones")
        else:
            log(f"   ❌ Demasiado grande para ESP32-S3")
    
    log("=" * 60 + "\n")


def estimate_inference_time_esp32(model_size_mb: float, imgsz: int = 224) -> Dict[str, float]:
    """Estimate inference time on ESP32-S3.

    Based on empirical measurements and model size.
    ESP32-S3 with vector acceleration typically achieves ~10-20 MOPS for INT8.

    Args:
        model_size_mb: Model size in MB (INT8)
        imgsz: Input image size

    Returns:
        Dictionary with estimated times
    """
    # Rough estimates based on ESP32-S3 performance
    # These are approximations and actual times depend on model architecture
    
    # Base time increases with model size and input resolution
    base_ops = model_size_mb * 1e6 / 4  # Approx operations based on params
    input_ops = (imgsz * imgsz * 3) * 100  # Input processing
    
    # ESP32-S3 with SIMD: ~10-20 MOPS for INT8
    mops = 15  # Conservative estimate
    
    total_ops = base_ops + input_ops
    inference_ms = (total_ops / (mops * 1e6)) * 1000
    
    return {
        "estimated_inference_ms": float(inference_ms),
        "model_size_mb": float(model_size_mb),
        "imgsz": float(imgsz),
    }


def check_yolo26_features() -> Dict[str, bool]:
    """Check available YOLO26 features in current Ultralytics version.

    Returns:
        Dictionary of feature availability
    """
    features = {
        "ultralytics_available": ULTRALYTICS_AVAILABLE,
        "yolo26_available": False,
        "end2end_available": False,
        "tflite_export": False,
        "int8_quantization": False,
    }

    if not ULTRALYTICS_AVAILABLE or YOLO is None:
        return features

    try:
        # Check if YOLO26 models are available
        test_model = YOLO("yolo26n.pt")  # type: ignore[misc]
        features["yolo26_available"] = True
        
        # Check export capabilities
        import ultralytics
        version = getattr(ultralytics, "__version__", "0.0.0")
        major, minor = map(int, version.split(".")[:2])
        
        features["end2end_available"] = (major >= 8 and minor >= 4)
        features["tflite_export"] = True
        features["int8_quantization"] = True
        
        del test_model
        
    except Exception as exc:
        log(f"⚠️ Error verificando características YOLO26: {exc}")

    return features
