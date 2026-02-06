"""YOLO26 training configuration and training utilities.

Optimized for YOLO26n deployment on ESP32-S3 with:
- MuSGD optimizer (auto-selected for YOLO26)
- End-to-end NMS-free inference
- ProgLoss + STAL for small object detection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils_io import log, safe_exists

try:
    from ultralytics import YOLO  # type: ignore
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    ULTRALYTICS_AVAILABLE = False


@dataclass
class Yolo26TrainConfig:
    """Configuration for YOLO26 training optimized for ESP32-S3.

    Key YOLO26 features:
    - optimizer='auto' selects MuSGD automatically for YOLO26
    - end2end=True for NMS-free inference (default in YOLO26)
    - DFL-free architecture simplifies INT8 quantization
    - ProgLoss + STAL improve small object detection
    """

    # Model selection
    model: str = "yolo26n.pt"

    # Input configuration
    imgsz: int = 224  # Reduced for ESP32-S3 (default 640)

    # Training schedule
    epochs: int = 100
    patience: int = 50  # Longer patience for YOLO26 warmup
    batch: int = 16
    
    # Optimizer - 'auto' selects MuSGD for YOLO26
    optimizer: str = "auto"
    lr0: float = 0.01  # Initial learning rate
    lrf: float = 0.01  # Final LR fraction (lr0 * lrf)
    cos_lr: bool = True  # Cosine annealing
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # Warmup
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Augmentation (aggressive for class imbalance)
    mosaic: float = 1.0  # High for class imbalance
    mixup: float = 0.1  # Light mixup
    close_mosaic: int = 10  # Disable mosaic last N epochs
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5

    # Loss weights
    box: float = 7.5  # Box loss weight
    cls: float = 0.5  # Classification loss weight (increase for minority classes)
    
    # Validation and saving
    val: bool = True
    save: bool = True
    save_period: int = -1  # Save every epoch if > 0
    plots: bool = True
    
    # Device and performance
    device: Optional[str] = None  # Auto-select GPU/CPU
    amp: bool = True  # Automatic mixed precision
    workers: int = 8
    
    # Project organization
    project: str = "logs"
    name: str = "yolo26_run"
    exist_ok: bool = True
    
    # Advanced settings
    freeze: Optional[List[int]] = None  # Layers to freeze
    multi_scale: float = 0.0  # Multi-scale training
    rect: bool = False  # Rectangular training
    single_cls: bool = False  # Treat as single class
    
    # YOLO26 specific
    max_det: int = 300  # Max detections per image (end2end head limit)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for model.train()."""
        return {
            "epochs": self.epochs,
            "patience": self.patience,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "cos_lr": self.cos_lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "close_mosaic": self.close_mosaic,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "box": self.box,
            "cls": self.cls,
            "val": self.val,
            "save": self.save,
            "save_period": self.save_period,
            "plots": self.plots,
            "amp": self.amp,
            "workers": self.workers,
            "project": self.project,
            "name": self.name,
            "exist_ok": self.exist_ok,
            "rect": self.rect,
            "single_cls": self.single_cls,
            "max_det": self.max_det,
            "verbose": True,
        }

    def summary(self) -> str:
        """Return a formatted summary of the configuration."""
        lines = [
            "=" * 60,
            "🔧 CONFIGURACIÓN DE ENTRENAMIENTO YOLO26",
            "=" * 60,
            f"📦 Modelo: {self.model}",
            f"📐 Imagen: {self.imgsz}x{self.imgsz}",
            "",
            "⏱️ Entrenamiento:",
            f"   Épocas: {self.epochs} | Paciencia: {self.patience}",
            f"   Batch: {self.batch} | Workers: {self.workers}",
            "",
            "🎯 Optimizador:",
            f"   Tipo: {self.optimizer} (MuSGD para YOLO26)",
            f"   LR inicial: {self.lr0} → final: {self.lr0 * self.lrf}",
            f"   Cosine LR: {self.cos_lr}",
            "",
            "🔄 Augmentación:",
            f"   Mosaic: {self.mosaic} | Mixup: {self.mixup}",
            f"   Close mosaic: {self.close_mosaic} épocas",
            f"   Scale: {self.scale} | Flip LR: {self.fliplr}",
            "",
            "⚖️ Loss Weights:",
            f"   Box: {self.box} | Cls: {self.cls}",
            "",
            f"📁 Proyecto: {self.project}/{self.name}",
            "=" * 60,
        ]
        return "\n".join(lines)


def train_yolo26(
    data_yaml: str,
    cfg: Yolo26TrainConfig,
    resume: bool = False,
    resume_path: Optional[str] = None,
) -> Optional[Any]:
    """Train YOLO26 model using Ultralytics API.

    YOLO26 uses end-to-end training (no separate warm-up/fine-tuning phases)
    with automatic MuSGD optimizer selection for stable convergence.

    Args:
        data_yaml: Path to data.yaml configuration
        cfg: Training configuration
        resume: Whether to resume from checkpoint
        resume_path: Path to checkpoint for resuming

    Returns:
        Training results object or None on failure
    """
    if not ULTRALYTICS_AVAILABLE:
        log("❌ Ultralytics no disponible. Instala con: pip install ultralytics")
        return None

    if not safe_exists(data_yaml):
        log(f"❌ data.yaml no encontrado: {data_yaml}")
        return None

    log("\n" + cfg.summary())

    try:
        # Load model
        if resume and resume_path and safe_exists(resume_path):
            log(f"🔄 Reanudando desde: {resume_path}")
            model = YOLO(resume_path)
        else:
            log(f"🔄 Cargando modelo base: {cfg.model}")
            model = YOLO(cfg.model)

        # Get training parameters
        train_params = cfg.to_dict()
        train_params["data"] = data_yaml
        
        if resume:
            train_params["resume"] = True

        # Add device if specified
        if cfg.device is not None:
            train_params["device"] = cfg.device

        # Add freeze if specified
        if cfg.freeze is not None:
            train_params["freeze"] = cfg.freeze

        log("\n🚀 Iniciando entrenamiento YOLO26...")
        log("   💡 YOLO26 usa MuSGD optimizer y entrenamiento end-to-end")
        log("   💡 ProgLoss + STAL activos para mejor detección de objetos pequeños")
        log("   💡 Modelo NMS-free para deployment simplificado\n")

        # Train
        results = model.train(**train_params)

        log("\n✅ Entrenamiento completado")
        
        # Print final metrics if available
        if results is not None:
            try:
                metrics = results.results_dict
                if metrics:
                    log("\n📊 Métricas finales:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            log(f"   {key}: {value:.4f}")
            except Exception:
                pass

        return results

    except KeyboardInterrupt:
        log("\n⚠️ Entrenamiento interrumpido por el usuario")
        return None
    except Exception as exc:
        log(f"❌ Error durante entrenamiento: {exc}")
        import traceback
        traceback.print_exc()
        return None


def validate_yolo26(
    model_path: str,
    data_yaml: str,
    split: str = "val",
    imgsz: int = 224,
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 300,
    end2end: bool = True,
    project: Optional[str] = None,
    name: str = "val",
) -> Optional[Any]:
    """Validate YOLO26 model on a dataset split.

    Args:
        model_path: Path to model weights
        data_yaml: Path to data.yaml
        split: Dataset split ('val' or 'test')
        imgsz: Image size
        conf: Confidence threshold
        iou: IoU threshold for NMS (ignored if end2end=True)
        max_det: Maximum detections
        end2end: Use end-to-end inference (NMS-free)
        project: Directory to save validation results. If None, derived
            automatically from model_path so that results are stored
            inside the experiment folder (e.g. logs/yolo26n_v1/val).
        name: Subdirectory name inside *project* (default ``"val"``).

    Returns:
        Validation results or None on failure
    """
    if not ULTRALYTICS_AVAILABLE:
        log("❌ Ultralytics no disponible")
        return None

    # Derive project from model_path if not provided
    if project is None:
        try:
            # e.g. "logs/yolo26n_v1/weights/best.pt" → "logs/yolo26n_v1"
            project = str(Path(model_path).parent.parent)
        except Exception:
            pass  # Fall back to Ultralytics default (runs/detect/val)

    try:
        log(f"\n🔍 Validando modelo en split: {split}")
        if project:
            log(f"   📂 Resultados en: {project}/{name}")
        model = YOLO(model_path)
        
        val_kwargs = dict(
            data=data_yaml,
            split=split,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            end2end=end2end,
            verbose=True,
        )
        if project is not None:
            val_kwargs["project"] = project
            val_kwargs["name"] = name

        results = model.val(**val_kwargs)
        
        return results

    except Exception as exc:
        log(f"❌ Error durante validación: {exc}")
        return None


def get_training_results_path(cfg: Yolo26TrainConfig) -> str:
    """Get the expected path to training results directory."""
    return f"{cfg.project}/{cfg.name}"
