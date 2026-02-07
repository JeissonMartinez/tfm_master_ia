"""Unified inference and visual prediction utilities.

Provides a single ``predict_and_visualize`` entry-point that works
for YOLO (Ultralytics) and MobileNet-SSD (Keras) models, as well
as TFLite runtime inference.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
        self.bbox = bbox  # (x1, y1, x2, y2) normalised [0-1]

    def __repr__(self):
        return (f"Det({self.class_name}, {self.confidence:.2f}, "
                f"[{self.bbox[0]:.3f},{self.bbox[1]:.3f},"
                f"{self.bbox[2]:.3f},{self.bbox[3]:.3f}])")


# =====================================================================
#  YOLO inference
# =====================================================================

def predict_yolo(
    model_path: str,
    image_paths: List[str],
    imgsz: int = 224,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    class_names: Optional[List[str]] = None,
) -> List[List[DetectedObject]]:
    """Run YOLO inference on a list of images.

    Returns one list of :class:`DetectedObject` per image.
    """
    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_path)
    results = model(image_paths, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)

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
#  MobileNet inference
# =====================================================================

def predict_mobilenet(
    model,
    images: np.ndarray,
    class_names: List[str],
    anchors: Optional[np.ndarray] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 224,
) -> List[List[DetectedObject]]:
    """Run MobileNet-SSD inference on a batch of images [0-1]."""
    import tensorflow as tf

    if images.ndim == 3:
        images = np.expand_dims(images, 0)

    preds = model(images, training=False)
    objectness = preds["objectness"].numpy()
    class_out = preds["class_out"].numpy()
    bbox_out = preds["bbox_out"].numpy()

    all_dets: List[List[DetectedObject]] = []
    for b in range(images.shape[0]):
        obj = objectness[b, :, 0]
        mask = obj > conf_threshold
        if not mask.any():
            all_dets.append([])
            continue

        cls_probs = class_out[b][mask]
        cls_ids = np.argmax(cls_probs, axis=-1)
        cls_confs = np.max(cls_probs, axis=-1)
        combined = obj[mask] * cls_confs
        bboxes = bbox_out[b][mask]

        if anchors is not None:
            bboxes = _decode_anchors(bboxes, anchors[mask])

        # NMS per class
        keep = _nms_multiclass(cls_ids, combined, bboxes, iou_threshold)
        dets = []
        for idx in keep:
            cid = int(cls_ids[idx])
            name = class_names[cid] if cid < len(class_names) else str(cid)
            dets.append(DetectedObject(cid, name, float(combined[idx]),
                                       tuple(bboxes[idx].tolist())))
        all_dets.append(dets)
    return all_dets


def _decode_anchors(offsets: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    cx_a = (anchors[:, 0] + anchors[:, 2]) / 2
    cy_a = (anchors[:, 1] + anchors[:, 3]) / 2
    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    cx = offsets[:, 0] * w_a + cx_a
    cy = offsets[:, 1] * h_a + cy_a
    w = np.exp(offsets[:, 2]) * w_a
    h = np.exp(offsets[:, 3]) * h_a
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def _nms_multiclass(cls_ids, scores, boxes, iou_thr=0.45) -> List[int]:
    """Simple per-class greedy NMS."""
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
            ious = np.array([_iou(boxes[i], boxes[j]) for j in rest])
            order = rest[ious < iou_thr]
    return keep


def _iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


# =====================================================================
#  TFLite inference
# =====================================================================

def predict_tflite(
    tflite_path: str,
    images: np.ndarray,
    class_names: List[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[List[List[DetectedObject]], float]:
    """Run TFLite inference. Returns (detections, avg_ms)."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()
    input_shape = inp["shape"]  # e.g. [1, 224, 224, 3]
    is_quantized = inp["dtype"] == np.uint8 or inp["dtype"] == np.int8

    all_dets: List[List[DetectedObject]] = []
    times: List[float] = []

    if images.ndim == 3:
        images = np.expand_dims(images, 0)

    for i in range(images.shape[0]):
        img = images[i]
        if is_quantized:
            scale, zero = inp.get("quantization", (1.0, 0))
            if isinstance(scale, tuple):
                scale, zero = scale
            img = (img / scale + zero).astype(inp["dtype"])
        else:
            img = img.astype(np.float32)

        img = np.expand_dims(img, 0)
        interpreter.set_tensor(inp["index"], img)

        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

        # read outputs
        out_data = {}
        for o in outs:
            name = o["name"]
            data = interpreter.get_tensor(o["index"]).astype(np.float32)
            # dequantize if needed
            if "quantization_parameters" in o:
                qp = o["quantization_parameters"]
                if "scales" in qp and len(qp["scales"]) > 0:
                    data = (data - qp["zero_points"][0]) * qp["scales"][0]
            out_data[name] = data

        dets = _parse_tflite_outputs(out_data, class_names, conf_threshold, iou_threshold)
        all_dets.append(dets)

    avg_ms = float(np.mean(times)) if times else 0.0
    return all_dets, avg_ms


def _parse_tflite_outputs(
    outputs: Dict[str, np.ndarray],
    class_names: List[str],
    conf_thr: float,
    iou_thr: float,
) -> List[DetectedObject]:
    """Parse TFLite output tensors into DetectedObject list."""
    # Try to find standard output names
    obj_key = _find_key(outputs, ["objectness", "output_0", "Identity"])
    cls_key = _find_key(outputs, ["class_out", "output_1", "Identity_1"])
    box_key = _find_key(outputs, ["bbox_out", "output_2", "Identity_2"])

    if obj_key is None or cls_key is None or box_key is None:
        # If we can't identify outputs, try positional
        keys = sorted(outputs.keys())
        if len(keys) >= 3:
            obj_key, cls_key, box_key = keys[0], keys[1], keys[2]
        else:
            return []

    obj = outputs[obj_key].squeeze()
    cls = outputs[cls_key].squeeze()
    box = outputs[box_key].squeeze()

    if obj.ndim == 1:
        mask = obj > conf_thr
    else:
        mask = obj[:, 0] > conf_thr if obj.ndim == 2 else obj > conf_thr

    if not mask.any():
        return []

    obj_s = obj[mask] if obj.ndim == 1 else obj[mask, 0]
    cls_s = cls[mask]
    box_s = box[mask]

    cls_ids = np.argmax(cls_s, axis=-1)
    cls_confs = np.max(cls_s, axis=-1)
    combined = obj_s * cls_confs

    keep = _nms_multiclass(cls_ids, combined, box_s, iou_thr)
    dets = []
    for idx in keep:
        cid = int(cls_ids[idx])
        name = class_names[cid] if cid < len(class_names) else str(cid)
        dets.append(DetectedObject(cid, name, float(combined[idx]),
                                   tuple(box_s[idx].tolist())))
    return dets


def _find_key(d: dict, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        for k in d:
            if c.lower() in k.lower():
                return k
    return None


# =====================================================================
#  Visualization
# =====================================================================

# color palette – consistent per class
_COLORS = [
    (0, 114, 189),  # azul
    (217, 83, 25),  # naranja
    (237, 177, 32),  # amarillo
    (126, 47, 142),  # morado
    (119, 172, 48),  # verde
    (77, 190, 238),  # cian
    (162, 20, 47),  # rojo
    (128, 128, 128),  # gris
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
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_img, rows * figsize_per_img))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        img = images[i]
        if isinstance(img, str):
            img = plt.imread(img)
        h, w = img.shape[:2]
        ax.imshow(img)
        ax.axis("off")

        # predictions (solid bbox)
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

        # ground truths (dashed bbox)
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
    plt.show()


# =====================================================================
#  Visualize GT samples per class (no inference)
# =====================================================================

def visualize_gt_samples_per_class(
    dataset_dir: str,
    class_names: List[str],
    split: str = "train",
    samples_per_class: int = 3,
    cols: int = 0,
    figsize_per_img: float = 4.0,
    save_path: Optional[str] = None,
    title: str = "",
    seed: int = 42,
) -> None:
    """Show random ground-truth images per class with bboxes and labels.

    For each class, selects *samples_per_class* random images from *split*
    that contain at least one annotation of that class, and draws all
    ground-truth bounding boxes present in those images.

    Supports both Roboflow (``{split}/images``, ``{split}/labels``) and
    Ultralytics (``images/{split}``, ``labels/{split}``) directory layouts.

    Parameters
    ----------
    dataset_dir : str
        Root of the YOLO dataset.
    class_names : list[str]
        Ordered class names matching label IDs (index = class_id).
    split : str
        Dataset split to sample from (``"train"``, ``"valid"``, ``"test"``).
    samples_per_class : int
        Number of random images per class.
    cols : int
        Grid columns.  ``0`` = auto (``samples_per_class``).
    figsize_per_img : float
        Size (inches) of each subplot.
    save_path : str, optional
        If given, save figure to this path.
    title : str
        Figure super-title.
    seed : int
        Random seed for reproducibility.
    """
    import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    dataset_dir = Path(dataset_dir)

    # ── Resolve images/labels dirs (Roboflow vs Ultralytics) ──
    imgs_dir = lbls_dir = None
    for variant in [split, "images"]:
        if variant == split:
            _imgs = dataset_dir / split / "images"
            _lbls = dataset_dir / split / "labels"
        else:
            _imgs = dataset_dir / "images" / split
            _lbls = dataset_dir / "labels" / split
        if _imgs.is_dir() and _lbls.is_dir():
            imgs_dir, lbls_dir = _imgs, _lbls
            break

    if imgs_dir is None or lbls_dir is None:
        raise FileNotFoundError(
            f"No se encontró split '{split}' en {dataset_dir}. "
            f"Buscando: {split}/images o images/{split}"
        )

    # ── Index: class_id → list of (image_path, label_path) ──
    class_index: Dict[int, List[Tuple[Path, Path]]] = {
        i: [] for i in range(len(class_names))
    }

    for lbl_file in sorted(lbls_dir.glob("*.txt")):
        stem = lbl_file.stem
        # Find matching image
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = imgs_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        # Which classes appear in this label file?
        classes_in_file = set()
        for line in lbl_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                cls_id = int(parts[0])
                if 0 <= cls_id < len(class_names):
                    classes_in_file.add(cls_id)

        for cid in classes_in_file:
            class_index[cid].append((img_path, lbl_file))

    # ── Sample random images per class ──
    rng = random.Random(seed)
    classes_with_data = [cid for cid in range(len(class_names)) if class_index[cid]]

    if not classes_with_data:
        log("⚠️  No se encontraron imágenes anotadas para visualizar.")
        return

    # ── Build grid: one row per class ──
    actual_cols = cols if cols > 0 else samples_per_class
    actual_rows = len(classes_with_data)

    fig, axes = plt.subplots(
        actual_rows, actual_cols,
        figsize=(actual_cols * figsize_per_img, actual_rows * figsize_per_img),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for row, cid in enumerate(classes_with_data):
        cname = class_names[cid]
        pool = class_index[cid]
        chosen = rng.sample(pool, min(samples_per_class, len(pool)))

        for col_i, (img_p, lbl_p) in enumerate(chosen):
            if col_i >= actual_cols:
                break
            ax = axes[row][col_i]
            img = plt.imread(str(img_p))
            h, w = img.shape[:2]
            ax.imshow(img)
            ax.axis("off")

            # Draw ALL annotations in this image
            for line in lbl_p.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                lbl_cls = int(parts[0])
                if lbl_cls < 0 or lbl_cls >= len(class_names):
                    continue
                cx, cy, bw, bh = (
                    float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]),
                )
                # YOLO (cx, cy, w, h) normalised → pixel rect
                rx = (cx - bw / 2) * w
                ry = (cy - bh / 2) * h
                rw = bw * w
                rh_px = bh * h
                color = np.array(_COLORS[lbl_cls % len(_COLORS)]) / 255.0
                rect = patches.Rectangle(
                    (rx, ry), rw, rh_px, linewidth=2,
                    edgecolor=color, facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    rx, ry - 3, class_names[lbl_cls],
                    fontsize=7, color="white",
                    bbox=dict(facecolor=color, alpha=0.7, pad=1),
                )

            if col_i == 0:
                ax.set_ylabel(
                    cname, fontsize=11, fontweight="bold",
                    rotation=0, labelpad=60, va="center",
                )

        # Hide unused columns
        for col_i in range(len(chosen), actual_cols):
            axes[row][col_i].axis("off")

    plt.tight_layout()
    if save_path:
        safe_mkdir(str(Path(save_path).parent))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"📊 GT samples guardado: {save_path}")
    plt.show()


def compare_predictions_side_by_side(
    image,
    dets_a: List[DetectedObject],
    dets_b: List[DetectedObject],
    label_a: str = "Framework",
    label_b: str = "TFLite",
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side detection comparison (2 panels)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if isinstance(image, str):
        image = plt.imread(image)
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
    plt.show()
