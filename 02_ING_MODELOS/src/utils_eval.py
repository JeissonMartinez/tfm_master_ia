"""Evaluation utilities for object detection models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    from .utils_io import log
except ImportError:  # fallback when running as a script/notebook
    from utils_io import log


@dataclass
class BoundingBox:
    """Bounding box in absolute coords (x, y, w, h)."""

    x: float
    y: float
    w: float
    h: float
    class_id: int
    class_name: str
    confidence: float = 1.0

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def scale_to(self, src_w: float, src_h: float, dst_w: float, dst_h: float) -> "BoundingBox":
        scale_x = dst_w / src_w
        scale_y = dst_h / src_h
        return BoundingBox(
            x=self.x * scale_x,
            y=self.y * scale_y,
            w=self.w * scale_x,
            h=self.h * scale_y,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
        )


@dataclass
class DetectionResult:
    image_id: int
    model_name: str
    predictions: List[BoundingBox] = field(default_factory=list)
    ground_truth: List[BoundingBox] = field(default_factory=list)
    inference_time_ms: float = 0.0


@dataclass
class MetricsResult:
    model_name: str
    map_50: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    detection_accuracy: float = 0.0
    avg_iou: float = 0.0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    inference_time_ms: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((5, 5)))


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1.to_xyxy()
    x1_2, y1_2, x2_2, y2_2 = box2.to_xyxy()
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


def match_predictions(
    predictions: List[BoundingBox],
    ground_truth: List[BoundingBox],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if not predictions or not ground_truth:
        return [], list(range(len(predictions))), list(range(len(ground_truth)))

    iou_matrix = []
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truth):
            if pred.class_id != gt.class_id:
                continue
            iou = compute_iou(pred, gt)
            if iou >= iou_threshold:
                iou_matrix.append((pred_idx, gt_idx, iou))

    iou_matrix.sort(key=lambda x: x[2], reverse=True)

    matches = []
    matched_preds = set()
    matched_gts = set()
    for pred_idx, gt_idx, iou in iou_matrix:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matches.append((pred_idx, gt_idx, iou))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)

    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_preds]
    unmatched_gt = [i for i in range(len(ground_truth)) if i not in matched_gts]
    return matches, unmatched_preds, unmatched_gt


def calculate_ap(predictions: List[Tuple[float, bool]], total_gt: int) -> float:
    if total_gt == 0 or not predictions:
        return 0.0
    sorted_preds = sorted(predictions, key=lambda x: x[0], reverse=True)
    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for _, is_tp in sorted_preds:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt)

    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p = [precisions[i] for i in range(len(recalls)) if recalls[i] >= r]
        if p:
            ap += max(p)
    return ap / 11.0


def calculate_map_50(
    results: List[DetectionResult],
    class_names: List[str],
    iou_threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    class_predictions: Dict[str, List[Tuple[float, bool]]] = {c: [] for c in class_names}
    class_gt_count: Dict[str, int] = {c: 0 for c in class_names}

    for det in results:
        gt_boxes = det.ground_truth
        pred_boxes = det.predictions
        for gt in gt_boxes:
            class_gt_count[gt.class_name] += 1

        gt_matched = [False] * len(gt_boxes)
        sorted_preds = sorted(pred_boxes, key=lambda x: x.confidence, reverse=True)
        for pred in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_matched[gt_idx] or gt.class_name != pred.class_name:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            is_tp = best_iou >= iou_threshold and best_gt_idx >= 0
            if is_tp:
                gt_matched[best_gt_idx] = True
            class_predictions[pred.class_name].append((pred.confidence, is_tp))

    per_class_ap = {}
    for c in class_names:
        per_class_ap[c] = calculate_ap(class_predictions[c], class_gt_count[c])

    valid_aps = [ap for c, ap in per_class_ap.items() if class_gt_count[c] > 0]
    map_50 = float(np.mean(valid_aps)) if valid_aps else 0.0
    return map_50, per_class_ap


def calculate_precision_recall_f1(
    results: List[DetectionResult],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float, int, int, int, float]:
    total_tp = total_fp = total_fn = 0
    iou_sum = 0.0
    for det in results:
        gt_boxes = det.ground_truth
        pred_boxes = det.predictions
        gt_matched = [False] * len(gt_boxes)
        sorted_preds = sorted(pred_boxes, key=lambda x: x.confidence, reverse=True)
        for pred in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_matched[gt_idx] or gt.class_name != pred.class_name:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                total_tp += 1
                gt_matched[best_gt_idx] = True
                iou_sum += best_iou
            else:
                total_fp += 1
        total_fn += sum(1 for m in gt_matched if not m)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = iou_sum / total_tp if total_tp > 0 else 0.0
    return precision, recall, f1, total_tp, total_fp, total_fn, avg_iou


def calculate_detection_accuracy(
    results: List[DetectionResult],
    iou_threshold: float = 0.5,
) -> float:
    correct = 0
    for det in results:
        gt_boxes = det.ground_truth
        pred_boxes = det.predictions
        if not gt_boxes:
            if not pred_boxes:
                correct += 1
            continue
        found = False
        for pred in pred_boxes:
            for gt in gt_boxes:
                if pred.class_name != gt.class_name:
                    continue
                if compute_iou(pred, gt) >= iou_threshold:
                    found = True
                    break
            if found:
                break
        if found:
            correct += 1
    return correct / len(results) if results else 0.0


def calculate_confusion_matrix(
    results: List[DetectionResult],
    class_names: List[str],
    iou_threshold: float = 0.5,
) -> np.ndarray:
    n_classes = len(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    confusion = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

    for det in results:
        gt_boxes = det.ground_truth
        pred_boxes = det.predictions
        gt_matched = [False] * len(gt_boxes)
        sorted_preds = sorted(pred_boxes, key=lambda x: x.confidence, reverse=True)
        for pred in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            pred_idx = class_to_idx.get(pred.class_name, n_classes)
            if best_gt_idx >= 0:
                gt = gt_boxes[best_gt_idx]
                gt_idx = class_to_idx.get(gt.class_name, n_classes)
                confusion[pred_idx, gt_idx] += 1
                gt_matched[best_gt_idx] = True
            else:
                confusion[pred_idx, n_classes] += 1
        for gt_idx, gt in enumerate(gt_boxes):
            if not gt_matched[gt_idx]:
                gt_i = class_to_idx.get(gt.class_name, n_classes)
                confusion[n_classes, gt_i] += 1

    return confusion


def evaluate_model(
    model_name: str,
    results: List[DetectionResult],
    class_names: List[str],
) -> MetricsResult:
    map_50, per_class_ap = calculate_map_50(results, class_names)
    precision, recall, f1, tp, fp, fn, avg_iou = calculate_precision_recall_f1(results)
    detection_acc = calculate_detection_accuracy(results)
    confusion = calculate_confusion_matrix(results, class_names)
    avg_time = float(np.mean([r.inference_time_ms for r in results])) if results else 0.0
    return MetricsResult(
        model_name=model_name,
        map_50=map_50,
        precision=precision,
        recall=recall,
        f1_score=f1,
        detection_accuracy=detection_acc,
        avg_iou=avg_iou,
        total_tp=tp,
        total_fp=fp,
        total_fn=fn,
        inference_time_ms=avg_time,
        per_class_ap=per_class_ap,
        confusion_matrix=confusion,
    )
