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
    bbox_format: str = "xywh_center",  # "xywh_center" (model output) or "xywh_corner"
) -> List[BoundingBox]:
    """Decode SSD predictions to bounding boxes.
    
    Args:
        class_logits: (N, num_classes+1) with background at index 0
        bbox_preds: (N, 4) normalized coords. Format depends on bbox_format.
        class_names: List of class names (without background)
        conf_threshold: Confidence threshold for filtering
        nms_iou: IoU threshold for NMS
        image_size: (height, width) of the image
        top_k: If set, return top_k predictions by confidence (ignores conf_threshold)
        bbox_format: "xywh_center" if bbox_preds are (xc, yc, w, h), 
                     "xywh_corner" if (x, y, w, h) with x,y as top-left corner
    """
    if class_logits.ndim != 2 or bbox_preds.ndim != 2:
        raise ValueError("Expected class_logits and bbox_preds with shape (N, C) and (N, 4).")

    num_classes = len(class_names)
    class_probs = tf.nn.softmax(class_logits, axis=-1).numpy()
    boxes: List[BoundingBox] = []
    dst_h, dst_w = image_size

    def decode_bbox(xc_or_x: float, yc_or_y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """Convert from model format to (x, y, w, h) corner format for BoundingBox."""
        xc_or_x = float(np.clip(xc_or_x, 0.0, 1.0))
        yc_or_y = float(np.clip(yc_or_y, 0.0, 1.0))
        w = float(np.clip(w, 0.0, 1.0))
        h = float(np.clip(h, 0.0, 1.0))
        
        if bbox_format == "xywh_center":
            # Convert center to corner: x = xc - w/2, y = yc - h/2
            x = (xc_or_x - w / 2) * dst_w
            y = (yc_or_y - h / 2) * dst_h
            w = w * dst_w
            h = h * dst_h
        else:
            # Already corner format
            x = xc_or_x * dst_w
            y = yc_or_y * dst_h
            w = w * dst_w
            h = h * dst_h
        
        # Clamp to image bounds
        x = max(0.0, x)
        y = max(0.0, y)
        w = min(w, dst_w - x)
        h = min(h, dst_h - y)
        return x, y, w, h

    candidates: List[BoundingBox] = []
    if top_k is not None and num_classes > 0:
        non_bg_probs = class_probs[:, 1:]
        best_cls = np.argmax(non_bg_probs, axis=-1)
        best_conf = np.max(non_bg_probs, axis=-1)
        for i in range(non_bg_probs.shape[0]):
            x, y, w, h = decode_bbox(*bbox_preds[i])
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
            x, y, w, h = decode_bbox(*bbox_preds[i])
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


def decode_ssd_v4_predictions(
    objectness: np.ndarray,
    class_probs: np.ndarray,
    bbox_preds: np.ndarray,
    class_names: List[str],
    obj_threshold: float = 0.5,
    cls_threshold: float = 0.3,
    nms_iou: float = 0.5,
    image_size: Tuple[int, int] = (224, 224),
    top_k: int | None = None,
) -> List[BoundingBox]:
    """Decode SSD V4 predictions (objectness + sigmoid classes).
    
    Args:
        objectness: (N, 1) objectness scores from sigmoid
        class_probs: (N, num_classes) class probabilities from sigmoid (NO background)
        bbox_preds: (N, 4) normalized coords [xc, yc, w, h]
        class_names: List of class names
        obj_threshold: Objectness threshold to filter anchors
        cls_threshold: Class probability threshold
        nms_iou: IoU threshold for NMS
        image_size: (height, width) of the image
        top_k: If set, return top_k predictions
    
    Returns:
        List of BoundingBox predictions
    """
    num_classes = len(class_names)
    dst_h, dst_w = image_size
    
    def decode_bbox(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
        xc = float(np.clip(xc, 0.0, 1.0))
        yc = float(np.clip(yc, 0.0, 1.0))
        w = float(np.clip(w, 0.0, 1.0))
        h = float(np.clip(h, 0.0, 1.0))
        x = (xc - w / 2) * dst_w
        y = (yc - h / 2) * dst_h
        w_px = w * dst_w
        h_px = h * dst_h
        x = max(0.0, x)
        y = max(0.0, y)
        w_px = min(w_px, dst_w - x)
        h_px = min(h_px, dst_h - y)
        return x, y, w_px, h_px

    boxes: List[BoundingBox] = []
    objectness = objectness.squeeze(-1)  # (N,)
    
    # Filtrar por objectness
    obj_mask = objectness >= obj_threshold
    
    for i in np.where(obj_mask)[0]:
        obj_score = float(objectness[i])
        
        # Obtener clase con mayor probabilidad
        cls_scores = class_probs[i]
        best_cls = int(np.argmax(cls_scores))
        best_cls_score = float(cls_scores[best_cls])
        
        # Confianza final = objectness * class_prob
        final_conf = obj_score * best_cls_score
        
        if final_conf < cls_threshold:
            continue
        
        x, y, w, h = decode_bbox(*bbox_preds[i])
        
        if w <= 0 or h <= 0:
            continue
        
        boxes.append(
            BoundingBox(
                x=x, y=y, w=w, h=h,
                class_id=best_cls,
                class_name=class_names[best_cls],
                confidence=final_conf,
            )
        )
    
    # Aplicar NMS
    boxes = _nms(boxes, iou_threshold=nms_iou)
    
    # Aplicar top_k si se especifica
    if top_k is not None and len(boxes) > top_k:
        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)[:top_k]
    
    return boxes


def run_ssd_v4_inference_coco(
    model: tf.keras.Model,
    coco_dict: dict,
    images_dir: str,
    class_names: List[str],
    target_size: Tuple[int, int] = (224, 224),
    obj_threshold: float = 0.5,
    cls_threshold: float = 0.3,
    nms_iou: float = 0.5,
    batch_size: int = 16,
    verbose: bool = True,
) -> dict:
    """Run SSD V4 inference on COCO format dataset.
    
    Args:
        model: Trained SSD V4 model
        coco_dict: COCO format dictionary with 'images' and 'annotations'
        images_dir: Directory containing the images
        class_names: List of class names
        target_size: (height, width) for resizing
        obj_threshold: Objectness threshold
        cls_threshold: Class probability threshold
        nms_iou: NMS IoU threshold
        batch_size: Batch size for inference
        verbose: Print progress
    
    Returns:
        dict mapping image_id -> list of predictions (dict with bbox_xyxy, class_name, confidence)
    """
    import os
    import cv2
    
    images_info = coco_dict["images"]
    all_preds = {}
    
    # Process in batches
    for batch_start in range(0, len(images_info), batch_size):
        batch_imgs = images_info[batch_start:batch_start + batch_size]
        images = []
        img_ids = []
        
        for img_info in batch_imgs:
            img_path = os.path.join(images_dir, img_info["file_name"])
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            img_ids.append(img_info["id"])
        
        if not images:
            continue
        
        images_batch = np.array(images)
        outputs = model.predict(images_batch, verbose=0)
        
        objectness = outputs["objectness"]
        class_probs = outputs["class_out"]
        bbox_preds = outputs["bbox_out_sigmoid"]
        
        for idx, img_id in enumerate(img_ids):
            boxes = decode_ssd_v4_predictions(
                objectness[idx],
                class_probs[idx],
                bbox_preds[idx],
                class_names=class_names,
                obj_threshold=obj_threshold,
                cls_threshold=cls_threshold,
                nms_iou=nms_iou,
                image_size=target_size,
            )
            
            # Convert to dict format for compatibility
            preds_list = []
            for box in boxes:
                x1, y1, x2, y2 = box.to_xyxy()
                preds_list.append({
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "class_name": box.class_name,
                    "class_id": box.class_id,
                    "confidence": box.confidence,
                })
            all_preds[img_id] = preds_list
        
        if verbose and batch_start % (batch_size * 5) == 0:
            print(f"  Processed {min(batch_start + batch_size, len(images_info))}/{len(images_info)} images")
    
    return all_preds


def run_ssd_v4_inference(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    class_names: List[str],
    image_ids: List[int],
    ground_truths: List[List[BoundingBox]],
    model_name: str = "SSD_V4",
    obj_threshold: float = 0.5,
    cls_threshold: float = 0.3,
    nms_iou: float = 0.5,
    image_size: Tuple[int, int] = (224, 224),
    top_k: int | None = None,
) -> List[DetectionResult]:
    """Run inference with SSD V4 model (objectness + sigmoid classes).
    
    Args:
        model: Trained SSD V4 model
        image_batch: Batch of images normalized to [0, 1]
        class_names: List of class names
        image_ids: List of image IDs
        ground_truths: List of ground truth boxes per image
        model_name: Name for logging
        obj_threshold: Objectness threshold
        cls_threshold: Class probability threshold
        nms_iou: NMS IoU threshold
        image_size: Image size (H, W)
        top_k: Max predictions per image
    
    Returns:
        List of DetectionResult objects
    """
    start = time.time()
    outputs = model.predict(image_batch, verbose=0)
    
    objectness = outputs["objectness"]
    class_probs = outputs["class_out"]
    bbox_preds = outputs["bbox_out_sigmoid"]
    
    elapsed = (time.time() - start) * 1000.0 / len(image_batch)

    results: List[DetectionResult] = []
    for idx in range(len(image_batch)):
        boxes = decode_ssd_v4_predictions(
            objectness[idx],
            class_probs[idx],
            bbox_preds[idx],
            class_names=class_names,
            obj_threshold=obj_threshold,
            cls_threshold=cls_threshold,
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
