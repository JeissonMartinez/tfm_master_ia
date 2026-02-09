"""YOLO inference helpers for test evaluation."""
from __future__ import annotations

from typing import List, Tuple
import time

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover - defensive
    YOLO = None

try:
    from .utils_eval import BoundingBox, DetectionResult
except ImportError:
    from utils_eval import BoundingBox, DetectionResult


def load_yolo_model(weights_path: str):
    if YOLO is None:
        raise RuntimeError("Ultralytics no está disponible. Instala ultralytics para usar YOLO.")
    return YOLO(weights_path)


def run_yolo_inference(
    model,
    image_batch: np.ndarray,
    class_names: List[str],
    image_ids: List[int],
    ground_truths: List[List[BoundingBox]],
    model_name: str = "YOLO",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    image_size: Tuple[int, int] = (224, 224),
) -> List[DetectionResult]:
    if model is None:
        raise RuntimeError("YOLO model no está cargado.")

    start = time.time()
    batch_list = [img for img in image_batch]
    results = model.predict(
        source=batch_list,
        imgsz=image_size[0],
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    elapsed = (time.time() - start) * 1000.0 / len(image_batch)

    out: List[DetectionResult] = []
    for idx, res in enumerate(results):
        preds: List[BoundingBox] = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
                x = float(x1)
                y = float(y1)
                w = float(x2 - x1)
                h = float(y2 - y1)
                if 0 <= cls_id < len(class_names):
                    class_name = class_names[cls_id]
                else:
                    class_name = "unknown"
                preds.append(
                    BoundingBox(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        class_id=int(cls_id),
                        class_name=class_name,
                        confidence=float(conf),
                    )
                )

        out.append(
            DetectionResult(
                image_id=image_ids[idx],
                model_name=model_name,
                predictions=preds,
                ground_truth=ground_truths[idx],
                inference_time_ms=elapsed,
            )
        )

    return out
