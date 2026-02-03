"""SSD inference helpers."""
from __future__ import annotations

from typing import List, Tuple
import time

import numpy as np
import tensorflow as tf

try:
    from .utils_eval import BoundingBox, DetectionResult
except ImportError:
    from utils_eval import BoundingBox, DetectionResult


def _nms(boxes: List[BoundingBox], iou_threshold: float = 0.5) -> List[BoundingBox]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    keep: List[BoundingBox] = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        filtered = []
        for box in boxes:
            if best.class_id != box.class_id:
                filtered.append(box)
                continue
            iou = _iou_xyxy(best, box)
            if iou < iou_threshold:
                filtered.append(box)
        boxes = filtered
    return keep


def _iou_xyxy(b1: BoundingBox, b2: BoundingBox) -> float:
    x1_1, y1_1, x2_1, y2_1 = b1.to_xyxy()
    x1_2, y1_2, x2_2, y2_2 = b2.to_xyxy()
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def decode_ssd_predictions(
    class_logits: np.ndarray,
    bbox_preds: np.ndarray,
    class_names: List[str],
    conf_threshold: float = 0.3,
    nms_iou: float = 0.5,
    image_size: Tuple[int, int] = (224, 224),
    top_k: int | None = None,
) -> List[BoundingBox]:
    if class_logits.ndim != 2 or bbox_preds.ndim != 2:
        raise ValueError("Expected class_logits and bbox_preds with shape (N, C) and (N, 4).")

    num_classes = len(class_names)
    class_probs = tf.nn.softmax(class_logits, axis=-1).numpy()
    boxes: List[BoundingBox] = []
    dst_h, dst_w = image_size

    candidates: List[BoundingBox] = []
    if top_k is not None and num_classes > 0:
        non_bg_probs = class_probs[:, 1:]
        best_cls = np.argmax(non_bg_probs, axis=-1)
        best_conf = np.max(non_bg_probs, axis=-1)
        for i in range(non_bg_probs.shape[0]):
            x, y, w, h = bbox_preds[i]
            x = float(np.clip(x, 0.0, 1.0)) * dst_w
            y = float(np.clip(y, 0.0, 1.0)) * dst_h
            w = float(np.clip(w, 0.0, 1.0)) * dst_w
            h = float(np.clip(h, 0.0, 1.0)) * dst_h
            class_id = int(best_cls[i])
            class_name = class_names[class_id]
            conf = float(best_conf[i])
            candidates.append(
                BoundingBox(x=x, y=y, w=w, h=h, class_id=class_id, class_name=class_name, confidence=conf)
            )
        filtered = sorted(candidates, key=lambda b: b.confidence, reverse=True)
        boxes = filtered[:top_k]
    else:
        for i in range(class_probs.shape[0]):
            class_idx = int(np.argmax(class_probs[i]))
            conf = float(np.max(class_probs[i]))
            if class_idx == 0 or conf < conf_threshold:
                continue
            x, y, w, h = bbox_preds[i]
            x = float(np.clip(x, 0.0, 1.0)) * dst_w
            y = float(np.clip(y, 0.0, 1.0)) * dst_h
            w = float(np.clip(w, 0.0, 1.0)) * dst_w
            h = float(np.clip(h, 0.0, 1.0)) * dst_h
            class_name = class_names[class_idx - 1]
            class_id = class_idx - 1
            boxes.append(
                BoundingBox(x=x, y=y, w=w, h=h, class_id=class_id, class_name=class_name, confidence=conf)
            )

    return _nms(boxes, iou_threshold=nms_iou)


def run_ssd_inference(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    class_names: List[str],
    image_ids: List[int],
    ground_truths: List[List[BoundingBox]],
    model_name: str = "SSD",
    conf_threshold: float = 0.3,
    nms_iou: float = 0.5,
    image_size: Tuple[int, int] = (224, 224),
    top_k: int | None = None,
) -> List[DetectionResult]:
    start = time.time()
    outputs = model.predict(image_batch, verbose=0)
    class_logits = outputs["class_out"]
    bbox_preds = outputs["bbox_out_sigmoid"]
    elapsed = (time.time() - start) * 1000.0 / len(image_batch)

    results: List[DetectionResult] = []
    for idx in range(len(image_batch)):
        boxes = decode_ssd_predictions(
            class_logits[idx],
            bbox_preds[idx],
            class_names=class_names,
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
            image_size=image_size,
            top_k=top_k,
        )
        results.append(
            DetectionResult(
                image_id=image_ids[idx],
                model_name=model_name,
                predictions=boxes,
                ground_truth=ground_truths[idx],
                inference_time_ms=elapsed,
            )
        )
    return results
