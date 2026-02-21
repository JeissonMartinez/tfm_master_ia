"""Unified inference and visualization — Cycle 2 (PyTorch only).

Provides family-specific ``predict_*`` functions and common
visualization helpers for FCOS, YOLO26_CUSTOM, and ESPDet.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils_io import log, safe_mkdir


# =====================================================================
#  Standard detection structure
# =====================================================================

class DetectedObject:
    """One detected object, family-agnostic."""
    __slots__ = ("class_id", "class_name", "confidence", "bbox")

    def __init__(self, class_id: int, class_name: str, confidence: float,
                 bbox: Tuple[float, float, float, float]):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2) normalized [0-1]

    def __repr__(self):
        return (f"Det({self.class_name}, {self.confidence:.2f}, "
                f"[{self.bbox[0]:.3f},{self.bbox[1]:.3f},"
                f"{self.bbox[2]:.3f},{self.bbox[3]:.3f}])")


# =====================================================================
#  FCOS inference
# =====================================================================

def predict_fcos(
    model: torch.nn.Module,
    images: torch.Tensor,
    conf_threshold: float = 0.15,
    nms_threshold: float = 0.45,
    class_names: Optional[List[str]] = None,
    strides: List[int] = [8, 16, 32],
    ctr_power: float = 1.0,
    iou_aware: bool = False,
) -> List[List[DetectedObject]]:
    """Run FCOS inference on a batch of images.

    Args:
        model: Trained FCOSModel.
        images: (B, C, H, W) tensor, already normalized.
        conf_threshold: Minimum confidence for detection.
        nms_threshold: IoU threshold for NMS.
        class_names: List of class names.
        strides: Feature map strides for each FPN level.
        ctr_power: Exponent for centerness in scoring. Values < 1
            (e.g. 0.5) soften the suppressive effect of low centerness,
            recovering detections near object edges.
        iou_aware: If True, compute a geometric quality factor from
            predicted (l, t, r, b) and multiply it into the score.
            This acts as an IoU-aware localization quality signal.

    Returns:
        List of DetectedObject lists (one per image).
    """
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        preds = model(images)

    batch_size = images.shape[0]
    img_h, img_w = images.shape[2], images.shape[3]
    all_dets: List[List[DetectedObject]] = []

    for b in range(batch_size):
        boxes_all, scores_all, labels_all = [], [], []

        for lvl, (cls_pred, reg_pred, ctr_pred) in enumerate(
            zip(preds["cls"], preds["reg"], preds["centerness"])
        ):
            stride = strides[lvl]
            h_feat, w_feat = cls_pred.shape[2], cls_pred.shape[3]

            cls_scores = cls_pred[b].sigmoid()       # (C, H, W)
            centerness = ctr_pred[b].sigmoid()        # (1, H, W)
            reg_vals = reg_pred[b]                    # (4, H, W)

            # Flatten
            num_classes = cls_scores.shape[0]
            cls_flat = cls_scores.permute(1, 2, 0).reshape(-1, num_classes)
            ctr_flat = centerness.reshape(-1)  # (1,H,W) → (H*W,)
            reg_flat = reg_vals.permute(1, 2, 0).reshape(-1, 4)

            # Grid coordinates
            y_grid, x_grid = torch.meshgrid(
                torch.arange(h_feat, device=device),
                torch.arange(w_feat, device=device),
                indexing="ij",
            )
            cx = (x_grid.flatten().float() + 0.5) * stride
            cy = (y_grid.flatten().float() + 0.5) * stride

            # Filter by cls score only (not multiplied by centerness)
            max_cls, max_labels = cls_flat.max(dim=-1)
            mask = max_cls > conf_threshold

            if mask.sum() == 0:
                continue

            cls_sel = max_cls[mask]
            labels_sel = max_labels[mask]
            ctr_sel = ctr_flat[mask]
            reg_sel = reg_flat[mask]
            cx_sel = cx[mask]
            cy_sel = cy[mask]

            # --- Scoring: softened centerness + IoU-aware quality ---
            quality = ctr_sel ** ctr_power  # ctr^0.5 is less aggressive

            if iou_aware:
                # Geometric quality from predicted l,t,r,b
                l, t, r, b_val = reg_sel[:, 0], reg_sel[:, 1], reg_sel[:, 2], reg_sel[:, 3]
                lr = torch.min(l, r) / (torch.max(l, r) + 1e-6)
                tb = torch.min(t, b_val) / (torch.max(t, b_val) + 1e-6)
                geo_quality = torch.sqrt(torch.clamp(lr * tb, min=0, max=1))
                quality = quality * geo_quality

            scores_sel = cls_sel * quality

            # Decode boxes: FCOS predicts (l, t, r, b) in stride-normalized units
            x1 = (cx_sel - reg_sel[:, 0] * stride) / img_w
            y1 = (cy_sel - reg_sel[:, 1] * stride) / img_h
            x2 = (cx_sel + reg_sel[:, 2] * stride) / img_w
            y2 = (cy_sel + reg_sel[:, 3] * stride) / img_h

            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            boxes = boxes.clamp(0, 1)

            boxes_all.append(boxes)
            scores_all.append(scores_sel)
            labels_all.append(labels_sel)

        if boxes_all:
            boxes_cat = torch.cat(boxes_all)
            scores_cat = torch.cat(scores_all)
            labels_cat = torch.cat(labels_all)
            keep = _nms_multiclass_torch(
                labels_cat, scores_cat, boxes_cat, nms_threshold
            )
            dets = _build_dets(
                labels_cat[keep], scores_cat[keep], boxes_cat[keep], class_names
            )
        else:
            dets = []

        all_dets.append(dets)

    return all_dets


# =====================================================================
#  YOLO26 Custom inference (Ultralytics)
# =====================================================================

def predict_yolo26_custom(
    model_path: str,
    image_paths: List[str],
    imgsz: int = 224,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    class_names: Optional[List[str]] = None,
) -> List[List[DetectedObject]]:
    """Run YOLO26 inference via Ultralytics on a list of images."""
    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_path)
    results = model(
        image_paths, imgsz=imgsz, conf=conf, iou=iou,
        max_det=max_det, verbose=False,
    )

    if class_names is None:
        class_names = list(model.names.values())

    all_dets: List[List[DetectedObject]] = []
    for r in results:
        dets = []
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                c = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                dets.append(DetectedObject(cls_id, name, c, (x1, y1, x2, y2)))
        all_dets.append(dets)
    return all_dets


# =====================================================================
#  ESPDet inference
# =====================================================================

def predict_espdet(
    model: torch.nn.Module,
    images: torch.Tensor,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    class_names: Optional[List[str]] = None,
    strides: List[int] = [4, 8, 16],
) -> List[List[DetectedObject]]:
    """Run ESPDet-Pico inference on a batch of images.

    Similar to FCOS but without centerness output.
    """
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        preds = model(images)

    batch_size = images.shape[0]
    img_h, img_w = images.shape[2], images.shape[3]
    all_dets: List[List[DetectedObject]] = []

    for b in range(batch_size):
        boxes_all, scores_all, labels_all = [], [], []

        for lvl, (cls_pred, reg_pred) in enumerate(
            zip(preds["cls"], preds["reg"])
        ):
            stride = strides[lvl]
            h_feat, w_feat = cls_pred.shape[2], cls_pred.shape[3]

            cls_scores = cls_pred[b].sigmoid()   # (C, H, W)
            reg_vals = reg_pred[b]               # (4*(reg_max+1), H, W)

            num_classes = cls_scores.shape[0]
            cls_flat = cls_scores.permute(1, 2, 0).reshape(-1, num_classes)
            # Take first 4 channels of reg (simplified for reg_max=1)
            reg_flat = reg_vals[:4].permute(1, 2, 0).reshape(-1, 4)

            y_grid, x_grid = torch.meshgrid(
                torch.arange(h_feat, device=device),
                torch.arange(w_feat, device=device),
                indexing="ij",
            )
            cx = (x_grid.flatten().float() + 0.5) * stride
            cy = (y_grid.flatten().float() + 0.5) * stride

            max_scores, max_labels = cls_flat.max(dim=-1)
            mask = max_scores > conf_threshold

            if mask.sum() == 0:
                continue

            scores_sel = max_scores[mask]
            labels_sel = max_labels[mask]
            reg_sel = torch.relu(reg_flat[mask])  # distances must be positive
            cx_sel, cy_sel = cx[mask], cy[mask]

            # Decode boxes: ESPDet predicts (l, t, r, b) in stride-normalized units
            x1 = (cx_sel - reg_sel[:, 0] * stride) / img_w
            y1 = (cy_sel - reg_sel[:, 1] * stride) / img_h
            x2 = (cx_sel + reg_sel[:, 2] * stride) / img_w
            y2 = (cy_sel + reg_sel[:, 3] * stride) / img_h

            boxes = torch.stack([x1, y1, x2, y2], dim=1).clamp(0, 1)
            boxes_all.append(boxes)
            scores_all.append(scores_sel)
            labels_all.append(labels_sel)

        if boxes_all:
            boxes_cat = torch.cat(boxes_all)
            scores_cat = torch.cat(scores_all)
            labels_cat = torch.cat(labels_all)
            keep = _nms_multiclass_torch(
                labels_cat, scores_cat, boxes_cat, nms_threshold
            )
            dets = _build_dets(
                labels_cat[keep], scores_cat[keep], boxes_cat[keep], class_names
            )
        else:
            dets = []

        all_dets.append(dets)

    return all_dets


# =====================================================================
#  NMS helpers
# =====================================================================

def _nms_multiclass_torch(
    cls_ids: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    iou_thr: float,
) -> torch.Tensor:
    """Per-class NMS using torchvision if available, else greedy."""
    try:
        from torchvision.ops import batched_nms
        keep = batched_nms(boxes, scores, cls_ids, iou_thr)
        return keep
    except ImportError:
        # Fallback: simple greedy NMS
        keep: List[int] = []
        order = torch.argsort(scores, descending=True)
        boxes_np = boxes.cpu().numpy()
        cls_np = cls_ids.cpu().numpy()
        scores_np = scores.cpu().numpy()
        remaining = set(range(len(order)))

        for idx in order.tolist():
            if idx not in remaining:
                continue
            keep.append(idx)
            remaining.discard(idx)
            to_remove = []
            for other in remaining:
                if cls_np[idx] == cls_np[other]:
                    iou = _iou_single(boxes_np[idx], boxes_np[other])
                    if iou >= iou_thr:
                        to_remove.append(other)
            for r in to_remove:
                remaining.discard(r)
        return torch.tensor(keep, dtype=torch.long)


def _nms_multiclass(cls_ids, scores, boxes, iou_thr=0.45) -> List[int]:
    """Numpy per-class greedy NMS (for compatibility)."""
    keep: List[int] = []
    for c in np.unique(cls_ids):
        idxs = np.where(cls_ids == c)[0]
        order = idxs[np.argsort(-scores[idxs])]
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))
            if len(order) == 1:
                break
            rest = order[1:]
            ious = _iou_vectorized(boxes[i], boxes[rest])
            order = rest[ious < iou_thr]
    return keep


def _iou_single(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def _iou_vectorized(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    a2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-8)


def _build_dets(
    labels: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    class_names: Optional[List[str]],
) -> List[DetectedObject]:
    dets = []
    for i in range(len(labels)):
        cid = int(labels[i])
        name = class_names[cid] if class_names and cid < len(class_names) else str(cid)
        dets.append(DetectedObject(
            cid, name, float(scores[i]),
            tuple(boxes[i].cpu().tolist()),
        ))
    return dets


# =====================================================================
#  Visualization
# =====================================================================

_COLORS = [
    (0, 114, 189),
    (217, 83, 25),
    (237, 177, 32),
    (126, 47, 142),
    (119, 172, 48),
    (77, 190, 238),
    (162, 20, 47),
    (128, 128, 128),
]


def visualize_predictions(
    images: list,
    predictions: List[List[DetectedObject]],
    ground_truths: Optional[List[List[DetectedObject]]] = None,
    max_images: int = 8,
    cols: int = 4,
    figsize_per_img: float = 4.0,
    save_path: Optional[str] = None,
    title: str = "",
) -> None:
    """Grid visualization of predicted (and optionally GT) bboxes."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n = min(len(images), max_images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * figsize_per_img,
                                      rows * figsize_per_img))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        img = images[i]
        if isinstance(img, str):
            img = plt.imread(img)
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        h, w = img.shape[:2]
        ax.imshow(img)
        ax.axis("off")

        for det in predictions[i]:
            x1, y1, x2, y2 = det.bbox
            rx, ry, rw, rh = x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h
            color = np.array(_COLORS[det.class_id % len(_COLORS)]) / 255.0
            rect = patches.Rectangle((rx, ry), rw, rh, linewidth=2,
                                     edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(rx, ry - 3, f"{det.class_name} {det.confidence:.2f}",
                    fontsize=7, color="white",
                    bbox=dict(facecolor=color, alpha=0.7, pad=1))

        if ground_truths and i < len(ground_truths):
            for gt in ground_truths[i]:
                x1, y1, x2, y2 = gt.bbox
                rx, ry, rw, rh = x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h
                color = np.array(_COLORS[gt.class_id % len(_COLORS)]) / 255.0
                rect = patches.Rectangle((rx, ry), rw, rh, linewidth=1.5,
                                         edgecolor=color, facecolor="none",
                                         linestyle="--")
                ax.add_patch(rect)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"📊 Visualización guardada: {save_path}")
    plt.close(fig)


def compare_predictions_side_by_side(
    image,
    dets_a: List[DetectedObject],
    dets_b: List[DetectedObject],
    label_a: str = "PyTorch",
    label_b: str = "ONNX",
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side detection comparison (2 panels)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if isinstance(image, str):
        image = plt.imread(image)
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    h, w = image.shape[:2]

    for ax, dets, label in [(ax1, dets_a, label_a), (ax2, dets_b, label_b)]:
        ax.imshow(image)
        ax.set_title(f"{label} ({len(dets)} dets)")
        ax.axis("off")
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            rx, ry, rw, rh = x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h
            color = np.array(_COLORS[det.class_id % len(_COLORS)]) / 255.0
            rect = patches.Rectangle((rx, ry), rw, rh, linewidth=2,
                                     edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(rx, ry - 3, f"{det.class_name} {det.confidence:.2f}",
                    fontsize=7, color="white",
                    bbox=dict(facecolor=color, alpha=0.7, pad=1))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
