"""YOLOv11 training helpers (Ultralytics)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .utils_io import log, safe_exists

try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover - defensive
    YOLO = None
    log(f"⚠️ Ultralytics no disponible: {exc}")


@dataclass
class YoloTrainConfig:
    model: str = "yolo11n.pt"
    imgsz: int = 224
    epochs: int = 100
    patience: int = 15
    batch: int = 32
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    cos_lr: bool = True
    warmup_epochs: int = 3
    weight_decay: float = 0.0005
    mosaic: float = 0.3
    mixup: float = 0.0
    hsv_h: float = 0.0
    hsv_s: float = 0.1
    hsv_v: float = 0.1
    degrees: float = 5.0
    translate: float = 0.05
    scale: float = 0.1
    fliplr: float = 0.3
    project: str = "logs"
    name: str = "yolo_run"
    exist_ok: bool = True
    amp: bool = True
    plots: bool = True
    save: bool = True
    save_period: int = -1
    val: bool = True


def train_yolo(data_yaml: str, cfg: YoloTrainConfig) -> Optional[Any]:
    """Train YOLO model using Ultralytics.

    Returns results object or None on failure.
    """
    if YOLO is None:
        log("⚠️ Ultralytics no disponible. Entrenamiento omitido.")
        return None
    if not safe_exists(data_yaml):
        log(f"⚠️ data.yaml no encontrado: {data_yaml}")
        return None

    try:
        model = YOLO(cfg.model)
        results = model.train(
            data=data_yaml,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            patience=cfg.patience,
            optimizer=cfg.optimizer,
            lr0=cfg.lr0,
            lrf=cfg.lrf,
            cos_lr=cfg.cos_lr,
            warmup_epochs=cfg.warmup_epochs,
            weight_decay=cfg.weight_decay,
            mosaic=cfg.mosaic,
            mixup=cfg.mixup,
            hsv_h=cfg.hsv_h,
            hsv_s=cfg.hsv_s,
            hsv_v=cfg.hsv_v,
            degrees=cfg.degrees,
            translate=cfg.translate,
            scale=cfg.scale,
            fliplr=cfg.fliplr,
            project=cfg.project,
            name=cfg.name,
            exist_ok=cfg.exist_ok,
            amp=cfg.amp,
            plots=cfg.plots,
            save=cfg.save,
            save_period=cfg.save_period,
            val=cfg.val,
            verbose=True,
        )
        return results
    except Exception as exc:  # pragma: no cover - defensive
        log(f"⚠️ Error entrenando YOLO: {exc}")
        return None
