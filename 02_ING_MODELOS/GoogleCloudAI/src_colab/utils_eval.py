"""Unified evaluation utilities for all model families.

Provides standardized detection-level evaluation (mAP, P, R, F1,
per-class AP, confusion matrix) across YOLO and MobileNet models.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils_io import log, safe_mkdir, write_json
from .utils_infer import _nms_multiclass, predict_tflite


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class Detection:
    """Single detection / ground-truth box."""
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    matched: bool = False


@dataclass
class EvaluationResults:
    """Unified evaluation results — same schema for every family."""
    model_name: str = ""
    family: str = ""
    split: str = "val"
    # -- global metrics --
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # -- per-class --
    per_class_ap50: Dict[str, float] = field(default_factory=dict)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    # -- confusion matrix --
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)
    # -- counts --
    n_images: int = 0
    n_detections: int = 0
    n_ground_truths: int = 0
    # -- timings --
    avg_inference_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"\n📊 Evaluación: {self.model_name} ({self.family}) – split={self.split}",
            f"  mAP@50:    {self.mAP50:.4f}",
            f"  mAP@50-95: {self.mAP50_95:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall:    {self.recall:.4f}",
            f"  F1-Score:  {self.f1:.4f}",
            f"  Imágenes: {self.n_images}  |  Detecciones: {self.n_detections}  |"
            f"  GT: {self.n_ground_truths}",
        ]
        if self.avg_inference_ms > 0:
            lines.append(f"  Avg inference: {self.avg_inference_ms:.1f} ms")
        if self.per_class_ap50:
            lines.append(f"\n  Per-class AP@50:")
            for cls, ap in sorted(self.per_class_ap50.items()):
                lines.append(f"    {cls:20s}  {ap:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "model_name": self.model_name,
            "family": self.family,
            "split": self.split,
            "mAP50": self.mAP50,
            "mAP50_95": self.mAP50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "per_class_ap50": self.per_class_ap50,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "n_images": self.n_images,
            "n_detections": self.n_detections,
            "n_ground_truths": self.n_ground_truths,
            "avg_inference_ms": self.avg_inference_ms,
            "class_names": self.class_names,
        }
        if self.confusion_matrix is not None:
            d["confusion_matrix"] = self.confusion_matrix.tolist()
        return d


# =====================================================================
#  YOLO evaluation
# =====================================================================

def evaluate_yolo_model(
    model_path: str,
    data_yaml: str,
    split: str = "val",
    imgsz: int = 224,
    conf: float = 0.25,
    iou: float = 0.6,
    max_det: int = 300,
    class_names: Optional[List[str]] = None,
) -> EvaluationResults:
    """Evaluate a YOLO model (best.pt) on a dataset split.

    Wraps ``model.val()`` and normalises into ``EvaluationResults``.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        log("❌ Ultralytics no disponible")
        return EvaluationResults()

    log(f"\n🔍 Evaluando YOLO: {model_path}  split={split}")
    model = YOLO(model_path)
    val_results = model.val(
        data=data_yaml, split=split, imgsz=imgsz,
        conf=conf, iou=iou, max_det=max_det, verbose=True,
    )

    ev = EvaluationResults()
    ev.model_name = Path(model_path).stem
    ev.family = "yolo"
    ev.split = split

    try:
        rd = val_results.results_dict
        ev.mAP50 = rd.get("metrics/mAP50(B)", 0.0)
        ev.mAP50_95 = rd.get("metrics/mAP50-95(B)", 0.0)
        ev.precision = rd.get("metrics/precision(B)", 0.0)
        ev.recall = rd.get("metrics/recall(B)", 0.0)
        ev.f1 = 2 * ev.precision * ev.recall / (ev.precision + ev.recall + 1e-8)
    except Exception:
        pass

    # per-class
    if class_names is None:
        try:
            class_names = list(model.names.values())
        except Exception:
            class_names = []
    ev.class_names = class_names

    try:
        ap50 = val_results.box.ap50  # shape (num_classes,)
        if ap50 is not None and class_names:
            for i, name in enumerate(class_names):
                if i < len(ap50):
                    ev.per_class_ap50[name] = float(ap50[i])
    except Exception:
        pass

    # ── Per-class Precision / Recall / F1 ──
    try:
        p_arr = val_results.box.p    # per-class precision (array)
        r_arr = val_results.box.r    # per-class recall    (array)
        if p_arr is not None and r_arr is not None and class_names:
            for i, name in enumerate(class_names):
                pi = float(p_arr[i]) if i < len(p_arr) else 0.0
                ri = float(r_arr[i]) if i < len(r_arr) else 0.0
                ev.per_class_precision[name] = pi
                ev.per_class_recall[name] = ri
                ev.per_class_f1[name] = 2 * pi * ri / (pi + ri + 1e-8)
    except Exception:
        pass

    # ── Image / detection / GT counts ──
    try:
        ev.n_images = int(getattr(val_results, "seen", 0))
    except Exception:
        pass
    try:
        cm_raw = val_results.confusion_matrix.matrix
        # GT count = sum of all columns except last (background) row
        nc = len(class_names)
        gt_total = int(np.sum(cm_raw[:nc, :]))   # rows 0..nc-1
        det_total = int(np.sum(cm_raw[:, :nc]))   # cols 0..nc-1
        ev.n_ground_truths = gt_total
        ev.n_detections = det_total
    except Exception:
        pass

    try:
        cm = val_results.confusion_matrix.matrix
        ev.confusion_matrix = np.array(cm)
    except Exception:
        pass

    try:
        ev.avg_inference_ms = val_results.speed.get("inference", 0.0)
    except Exception:
        pass

    log(ev.summary())
    return ev


# =====================================================================
#  MobileNet evaluation
# =====================================================================

def evaluate_mobilenet_model(
    model,
    val_ds,
    class_names: List[str],
    imgsz: int = 224,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    anchors: Optional[np.ndarray] = None,
    model_name: str = "mobilenet_ssd",
) -> EvaluationResults:
    """Evaluate a MobileNet-SSD model on a tf.data.Dataset.

    Runs inference on all batches, decodes detections, matches to
    ground truth, and computes mAP.
    """
    import tensorflow as tf

    ev = EvaluationResults()
    ev.model_name = model_name
    ev.family = "mobilenet"
    ev.split = "val"
    ev.class_names = class_names

    all_detections = []  # (img_idx, cls, conf, bbox)
    all_ground_truths = []  # (img_idx, cls, bbox)
    times = []

    img_idx = 0
    for batch in val_ds:
        images = batch[0]
        targets = batch[1]
        batch_size = images.shape[0]

        t0 = tf.timestamp()
        preds = model(images, training=False)
        t1 = tf.timestamp()
        times.append(float(t1 - t0) * 1000 / batch_size)

        objectness = preds["objectness"].numpy()
        class_out = preds["class_out"].numpy()
        bbox_out = preds["bbox_out"].numpy()
        gt_obj = targets["objectness"].numpy()
        gt_cls = targets["class_out"].numpy()
        gt_bbox = targets["bbox_out"].numpy()

        for b in range(batch_size):
            # decode predictions
            obj_scores = objectness[b, :, 0]
            valid_mask = obj_scores > conf_threshold
            if valid_mask.any():
                cls_probs = class_out[b][valid_mask]
                cls_ids = np.argmax(cls_probs, axis=-1)
                cls_confs = np.max(cls_probs, axis=-1)
                combined_conf = obj_scores[valid_mask] * cls_confs
                bboxes = bbox_out[b][valid_mask]

                # Modelo predice coordenadas absolutas [xc,yc,w,h] (sigmoid)
                # Convertir a [x1,y1,x2,y2] para cálculo de IoU
                bboxes = _xywh_to_xyxy(bboxes)

                # NMS per-class para eliminar duplicados
                keep = _nms_multiclass(cls_ids, combined_conf, bboxes,
                                       iou_thr=iou_threshold)
                cls_ids = cls_ids[keep]
                combined_conf = combined_conf[keep]
                bboxes = bboxes[keep]

                for j in range(len(cls_ids)):
                    all_detections.append((
                        img_idx, int(cls_ids[j]), float(combined_conf[j]),
                        tuple(bboxes[j].tolist()),
                    ))
                ev.n_detections += len(cls_ids)

            # ground truths — también en [xc,yc,w,h], convertir a [x1,y1,x2,y2]
            gt_pos = gt_obj[b, :, 0] > 0.5
            if gt_pos.any():
                gt_c = np.argmax(gt_cls[b][gt_pos], axis=-1)
                gt_b = gt_bbox[b][gt_pos]
                gt_b = _xywh_to_xyxy(gt_b)  # convertir formato
                for j in range(len(gt_c)):
                    all_ground_truths.append((
                        img_idx, int(gt_c[j]), tuple(gt_b[j].tolist()),
                    ))
                ev.n_ground_truths += len(gt_c)

            img_idx += 1

    ev.n_images = img_idx
    ev.avg_inference_ms = float(np.mean(times)) if times else 0.0

    # Compute mAP
    _compute_map(ev, all_detections, all_ground_truths, class_names, iou_threshold)

    log(ev.summary())
    return ev


# =====================================================================
#  TFLite evaluation  (mAP sobre dataset completo)
# =====================================================================

def evaluate_tflite_model(
    tflite_path: str,
    class_names: List[str],
    imgsz: int = 224,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    model_name: str = "tflite_model",
    # --- MobileNet path ---
    test_ds=None,
    # --- YOLO path ---
    dataset_dir: Optional[str] = None,
    split: str = "test",
) -> EvaluationResults:
    """Evaluate a TFLite model on a complete dataset split.

    Supports two modes:
    - **MobileNet**: pass ``test_ds`` (tf.data.Dataset with (images, targets)).
    - **YOLO**: pass ``dataset_dir`` + ``split`` to load images/labels.

    Returns an ``EvaluationResults`` with mAP@50, P, R, F1, per-class AP,
    confusion matrix, etc.  — same schema used by
    ``evaluate_mobilenet_model`` and ``evaluate_yolo_model``.
    """
    ev = EvaluationResults()
    ev.model_name = model_name
    ev.family = "tflite"
    ev.split = split
    ev.class_names = class_names

    all_detections: list = []   # (img_idx, cls, conf, bbox)
    all_ground_truths: list = []  # (img_idx, cls, bbox)
    all_times: List[float] = []

    img_idx = 0

    # ── MobileNet path: iterate tf.data batches ──
    if test_ds is not None:
        import tensorflow as tf
        for batch in test_ds:
            images = batch[0].numpy()
            targets = batch[1]
            batch_size = images.shape[0]

            # TFLite inference for the batch
            tfl_dets, avg_ms = predict_tflite(
                tflite_path=tflite_path,
                images=images,
                class_names=class_names,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            all_times.append(avg_ms)

            # Gather ground truths from batch targets
            gt_obj = targets["objectness"].numpy()
            gt_cls = targets["class_out"].numpy()
            gt_bbox = targets["bbox_out"].numpy()

            for b in range(batch_size):
                # Detections from TFLite
                for det in tfl_dets[b] if b < len(tfl_dets) else []:
                    all_detections.append((
                        img_idx, det.class_id, det.confidence, det.bbox,
                    ))
                    ev.n_detections += 1

                # Ground truths
                gt_pos = gt_obj[b, :, 0] > 0.5
                if gt_pos.any():
                    gt_c = np.argmax(gt_cls[b][gt_pos], axis=-1)
                    gt_b = gt_bbox[b][gt_pos]
                    gt_b = _xywh_to_xyxy(gt_b)
                    for j in range(len(gt_c)):
                        all_ground_truths.append((
                            img_idx, int(gt_c[j]),
                            tuple(gt_b[j].tolist()),
                        ))
                        ev.n_ground_truths += 1

                img_idx += 1

    # ── YOLO path: load images + .txt labels ──
    elif dataset_dir is not None:
        import cv2
        from pathlib import Path as _P

        ds_root = _P(dataset_dir)
        # Resolve images/labels dirs (Roboflow vs Ultralytics layout)
        imgs_dir = lbls_dir = None
        for variant in [split, "images"]:
            if variant == split:
                _imgs = ds_root / split / "images"
                _lbls = ds_root / split / "labels"
            else:
                _imgs = ds_root / "images" / split
                _lbls = ds_root / "labels" / split
            if _imgs.is_dir() and _lbls.is_dir():
                imgs_dir, lbls_dir = _imgs, _lbls
                break

        if imgs_dir is None:
            log(f"❌ No se encontró split '{split}' en {dataset_dir}")
            return ev

        img_files = sorted(
            list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png"))
        )
        log(f"🔍 Evaluando TFLite sobre {len(img_files)} imágenes ({split})")

        # Process in batches
        BATCH = 16
        for start in range(0, len(img_files), BATCH):
            batch_files = img_files[start:start + BATCH]
            batch_imgs = []
            for img_p in batch_files:
                img = cv2.imread(str(img_p))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (imgsz, imgsz))
                batch_imgs.append(img / 255.0)
            batch_arr = np.array(batch_imgs, dtype=np.float32)

            tfl_dets, avg_ms = predict_tflite(
                tflite_path=tflite_path,
                images=batch_arr,
                class_names=class_names,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            all_times.append(avg_ms)

            for b_i, img_p in enumerate(batch_files):
                # TFLite detections
                for det in tfl_dets[b_i] if b_i < len(tfl_dets) else []:
                    all_detections.append((
                        img_idx, det.class_id, det.confidence, det.bbox,
                    ))
                    ev.n_detections += 1

                # Ground truths from .txt label file
                lbl_path = lbls_dir / f"{img_p.stem}.txt"
                if lbl_path.exists():
                    for line in lbl_path.read_text().strip().splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, bw, bh = (
                                float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]),
                            )
                            x1 = np.clip(cx - bw / 2, 0, 1)
                            y1 = np.clip(cy - bh / 2, 0, 1)
                            x2 = np.clip(cx + bw / 2, 0, 1)
                            y2 = np.clip(cy + bh / 2, 0, 1)
                            all_ground_truths.append((
                                img_idx, cls_id, (x1, y1, x2, y2),
                            ))
                            ev.n_ground_truths += 1

                img_idx += 1
    else:
        log("❌ evaluate_tflite_model: proporciona test_ds o dataset_dir")
        return ev

    ev.n_images = img_idx
    ev.avg_inference_ms = float(np.mean(all_times)) if all_times else 0.0

    # Compute mAP con la misma lógica usada en evaluate_mobilenet_model
    _compute_map(ev, all_detections, all_ground_truths, class_names, iou_threshold)

    log(ev.summary())
    return ev


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [xc, yc, w, h] → [x1, y1, x2, y2], clipped to [0, 1]."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = np.clip(cx - w / 2, 0, 1)
    y1 = np.clip(cy - h / 2, 0, 1)
    x2 = np.clip(cx + w / 2, 0, 1)
    y2 = np.clip(cy + h / 2, 0, 1)
    return np.stack([x1, y1, x2, y2], axis=-1)


def _compute_map(
    ev: EvaluationResults,
    detections: list,
    ground_truths: list,
    class_names: List[str],
    iou_threshold: float = 0.5,
) -> None:
    """Compute mAP@50 and per-class AP via 101-point interpolation."""
    num_classes = len(class_names)
    aps = []

    # group GT by (img_idx, class_id)
    gt_by_img_cls: Dict[tuple, list] = defaultdict(list)
    for img_idx, cls_id, bbox in ground_truths:
        gt_by_img_cls[(img_idx, cls_id)].append({"bbox": bbox, "matched": False})

    for c in range(num_classes):
        # filter detections for this class
        dets_c = [(d[0], d[2], d[3]) for d in detections if d[1] == c]
        dets_c.sort(key=lambda x: x[1], reverse=True)  # sort by conf

        tp = np.zeros(len(dets_c))
        fp = np.zeros(len(dets_c))
        n_gt_c = sum(1 for gt in ground_truths if gt[1] == c)

        # reset matched flags
        for key in gt_by_img_cls:
            for gt in gt_by_img_cls[key]:
                gt["matched"] = False

        for i, (img_idx, conf, bbox) in enumerate(dets_c):
            gts = gt_by_img_cls.get((img_idx, c), [])
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gts):
                iou = _compute_iou(bbox, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            if best_iou >= iou_threshold and best_gt_idx >= 0 and not gts[best_gt_idx]["matched"]:
                tp[i] = 1
                gts[best_gt_idx]["matched"] = True
            else:
                fp[i] = 1

        # precision-recall curve
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / (n_gt_c + 1e-8)
        prec = tp_cum / (tp_cum + fp_cum + 1e-8)

        # 101-point interpolation
        ap = _ap_interp(rec, prec)
        aps.append(ap)

        if c < len(class_names):
            ev.per_class_ap50[class_names[c]] = float(ap)
            if n_gt_c > 0:
                final_p = float(prec[-1]) if len(prec) > 0 else 0.0
                final_r = float(rec[-1]) if len(rec) > 0 else 0.0
                ev.per_class_precision[class_names[c]] = final_p
                ev.per_class_recall[class_names[c]] = final_r
                f1 = 2 * final_p * final_r / (final_p + final_r + 1e-8)
                ev.per_class_f1[class_names[c]] = f1

    ev.mAP50 = float(np.mean(aps)) if aps else 0.0

    # global P/R/F1 (from per-class averages already computed above)
    ev.precision = float(np.mean(list(ev.per_class_precision.values()))) if ev.per_class_precision else 0.0
    ev.recall = float(np.mean(list(ev.per_class_recall.values()))) if ev.per_class_recall else 0.0
    ev.f1 = 2 * ev.precision * ev.recall / (ev.precision + ev.recall + 1e-8)

    # ── Confusion matrix (num_classes+1 × num_classes+1) ──
    # Última fila/columna = background (FP / FN)
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.float64)

    # Resetear matched flags para construir la CM
    for key in gt_by_img_cls:
        for gt in gt_by_img_cls[key]:
            gt["matched"] = False
            gt["matched_cls"] = -1

    # Recorrer detecciones (ordenadas por confianza desc) y asignar a GT
    all_dets_sorted = sorted(detections, key=lambda d: d[2], reverse=True)
    for img_idx, pred_cls, conf, bbox in all_dets_sorted:
        gts = gt_by_img_cls.get((img_idx, pred_cls), [])
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, gt in enumerate(gts):
            iou_val = _compute_iou(bbox, gt["bbox"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = g_idx
        if best_iou >= iou_threshold and best_gt_idx >= 0 and not gts[best_gt_idx]["matched"]:
            # TP: GT class == pred class (same key)
            gts[best_gt_idx]["matched"] = True
            gts[best_gt_idx]["matched_cls"] = pred_cls
            cm[pred_cls, pred_cls] += 1
        else:
            # Intentar match cross-class para la CM
            matched_cross = False
            for gt_cls in range(num_classes):
                if gt_cls == pred_cls:
                    continue
                cross_gts = gt_by_img_cls.get((img_idx, gt_cls), [])
                for g_idx, gt in enumerate(cross_gts):
                    if gt["matched"]:
                        continue
                    iou_val = _compute_iou(bbox, gt["bbox"])
                    if iou_val >= iou_threshold:
                        gt["matched"] = True
                        gt["matched_cls"] = pred_cls
                        cm[gt_cls, pred_cls] += 1  # GT=gt_cls, Pred=pred_cls
                        matched_cross = True
                        break
                if matched_cross:
                    break
            if not matched_cross:
                # FP: predicción sin GT → background row
                cm[num_classes, pred_cls] += 1

    # FN: GT no matcheados → background column
    for (img_idx, gt_cls), gts in gt_by_img_cls.items():
        for gt in gts:
            if not gt["matched"]:
                cm[gt_cls, num_classes] += 1

    ev.confusion_matrix = cm


def _ap_interp(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point AP interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    points = np.linspace(0, 1, 101)
    ap = 0.0
    for t in points:
        idx = np.where(mrec >= t)[0]
        if len(idx) > 0:
            ap += mpre[idx[0]]
    return ap / 101.0


def _compute_iou(box1, box2) -> float:
    """IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-8)


# =====================================================================
#  Visualization
# =====================================================================

def plot_confusion_matrix(
    ev: EvaluationResults,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True,
) -> None:
    """Plot confusion matrix from evaluation results."""
    import matplotlib.pyplot as plt

    if ev.confusion_matrix is None:
        log("⚠️  No confusion matrix available")
        return

    cm = ev.confusion_matrix.copy()
    labels = ev.class_names + ["background"]
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / (row_sums + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Confusion Matrix — {ev.model_name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"📊 Confusion matrix guardada: {save_path}")
    plt.show()


def plot_per_class_metrics(
    ev: EvaluationResults,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Bar plot of per-class AP@50, P, R, F1."""
    import matplotlib.pyplot as plt

    if not ev.per_class_ap50:
        log("⚠️  No per-class metrics available")
        return

    classes = sorted(ev.per_class_ap50.keys())
    x = np.arange(len(classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=figsize)
    for offset, (metric_dict, label, color) in enumerate([
        (ev.per_class_ap50, "AP@50", "tab:blue"),
        (ev.per_class_precision, "Precision", "tab:green"),
        (ev.per_class_recall, "Recall", "tab:orange"),
        (ev.per_class_f1, "F1", "tab:red"),
    ]):
        vals = [metric_dict.get(c, 0.0) for c in classes]
        ax.bar(x + offset * width - 1.5 * width, vals, width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Métricas por clase — {ev.model_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"📊 Per-class metrics guardados: {save_path}")
    plt.show()


def save_evaluation(ev: EvaluationResults, output_path: str) -> None:
    """Save evaluation results as JSON."""
    write_json(output_path, ev.to_dict())
    log(f"💾 Evaluación guardada: {output_path}")
