"""Framework vs TFLite comparison utilities.

Runs the same images through both the original framework model
(YOLO/Keras) and the exported TFLite INT8 model, then computes
agreement rate, IoU distribution and side-by-side visualisations.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils_io import log, write_json, safe_mkdir
from .utils_infer import (
    DetectedObject, predict_yolo, predict_mobilenet,
    predict_tflite, compare_predictions_side_by_side,
)
from .utils_export import TFLiteVerificationResult


# =====================================================================
#  High-level comparison
# =====================================================================

def compare_framework_vs_tflite(
    framework_model,
    tflite_path: str,
    images: np.ndarray,
    class_names: List[str],
    family: str,
    anchors: Optional[np.ndarray] = None,
    imgsz: int = 224,
    conf: float = 0.25,
    iou_thr: float = 0.45,
    model_path: Optional[str] = None,
) -> TFLiteVerificationResult:
    """Compare framework model to its TFLite counterpart.

    Parameters
    ----------
    framework_model
        - YOLO: path string (best.pt)
        - MobileNet: Keras model instance
    tflite_path
        Path to the .tflite file.
    images
        Batch of images [B, H, W, 3] in [0, 1].
    """
    from .config import is_yolo_family

    log(f"\n🔄 Comparando framework vs TFLite ({family})")
    log(f"   TFLite: {tflite_path}")
    log(f"   Muestras: {images.shape[0]}")

    # framework predictions
    if is_yolo_family(family):
        fw_dets = predict_yolo(
            model_path=model_path or str(framework_model),
            image_paths=[images[i] for i in range(images.shape[0])],
            imgsz=imgsz, conf=conf, iou=iou_thr,
            class_names=class_names,
        )
    else:
        fw_dets = predict_mobilenet(
            model=framework_model, images=images,
            class_names=class_names, anchors=anchors,
            conf_threshold=conf, iou_threshold=iou_thr,
        )

    # TFLite predictions
    tfl_dets, avg_ms = predict_tflite(
        tflite_path=tflite_path, images=images,
        class_names=class_names, conf_threshold=conf,
        iou_threshold=iou_thr,
    )

    # compute agreement
    result = _compute_agreement(fw_dets, tfl_dets)
    result.tflite_path = tflite_path
    result.n_samples = images.shape[0]
    result.avg_inference_ms = avg_ms
    result.passed = result.agreement_rate >= 0.7 and result.avg_iou >= 0.4

    log(result.summary())
    return result


def _compute_agreement(
    fw: List[List[DetectedObject]],
    tfl: List[List[DetectedObject]],
) -> TFLiteVerificationResult:
    """Compute per-image detection agreement metrics."""
    total_agree = 0
    total_fw = 0
    ious: List[float] = []
    conf_diffs: List[float] = []

    for fw_dets, tfl_dets in zip(fw, tfl):
        total_fw += len(fw_dets)
        matched_tfl = set()

        for fd in fw_dets:
            best_iou = 0.0
            best_idx = -1
            for j, td in enumerate(tfl_dets):
                if j in matched_tfl:
                    continue
                if fd.class_id != td.class_id:
                    continue
                iou_val = _iou(fd.bbox, td.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = j
            if best_iou >= 0.3 and best_idx >= 0:
                total_agree += 1
                matched_tfl.add(best_idx)
                ious.append(best_iou)
                conf_diffs.append(abs(fd.confidence - tfl_dets[best_idx].confidence))

    res = TFLiteVerificationResult()
    res.agreement_rate = total_agree / max(total_fw, 1)
    res.avg_iou = float(np.mean(ious)) if ious else 0.0
    res.avg_conf_diff = float(np.mean(conf_diffs)) if conf_diffs else 0.0
    return res


def _iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


# =====================================================================
#  Visualization helpers
# =====================================================================

def plot_iou_distribution(
    fw_dets: List[List[DetectedObject]],
    tfl_dets: List[List[DetectedObject]],
    save_path: Optional[str] = None,
) -> None:
    """Histogram of pair-wise IoU between matched detections."""
    import matplotlib.pyplot as plt

    ious = []
    for fds, tds in zip(fw_dets, tfl_dets):
        for fd in fds:
            for td in tds:
                if fd.class_id == td.class_id:
                    ious.append(_iou(fd.bbox, td.bbox))

    if not ious:
        log("⚠️  Sin pares para distribución IoU")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ious, bins=30, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(ious), color="red", linestyle="--", label=f"Mean={np.mean(ious):.3f}")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title("Framework vs TFLite — IoU Distribution")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confidence_scatter(
    fw_dets: List[List[DetectedObject]],
    tfl_dets: List[List[DetectedObject]],
    save_path: Optional[str] = None,
) -> None:
    """Scatter plot of framework vs TFLite confidences for matched dets."""
    import matplotlib.pyplot as plt

    fw_confs, tfl_confs = [], []
    for fds, tds in zip(fw_dets, tfl_dets):
        matched = set()
        for fd in fds:
            best_iou, best_j = 0, -1
            for j, td in enumerate(tds):
                if j in matched or fd.class_id != td.class_id:
                    continue
                v = _iou(fd.bbox, td.bbox)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= 0.3 and best_j >= 0:
                fw_confs.append(fd.confidence)
                tfl_confs.append(tds[best_j].confidence)
                matched.add(best_j)

    if not fw_confs:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(fw_confs, tfl_confs, alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], "r--", label="y=x")
    ax.set_xlabel("Framework Confidence")
    ax.set_ylabel("TFLite Confidence")
    ax.set_title("Conf. Correlation")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def visualize_comparison_grid(
    images: list,
    fw_dets: List[List[DetectedObject]],
    tfl_dets: List[List[DetectedObject]],
    max_images: int = 4,
    save_path: Optional[str] = None,
) -> None:
    """Grid: left = framework, right = TFLite, for N images."""
    n = min(len(images), max_images)
    for i in range(n):
        sp = None
        if save_path:
            base, ext = os.path.splitext(save_path)
            sp = f"{base}_{i}{ext}"
        compare_predictions_side_by_side(
            images[i], fw_dets[i], tfl_dets[i],
            label_a="Framework", label_b="TFLite",
            save_path=sp,
        )


def save_comparison_result(
    result: TFLiteVerificationResult,
    output_path: str,
) -> None:
    """Save comparison result as JSON."""
    d = {
        "tflite_path": result.tflite_path,
        "n_samples": result.n_samples,
        "agreement_rate": result.agreement_rate,
        "avg_iou": result.avg_iou,
        "avg_conf_diff": result.avg_conf_diff,
        "avg_inference_ms": result.avg_inference_ms,
        "passed": result.passed,
    }
    write_json(output_path, d)
    log(f"💾 Comparison result: {output_path}")


# =====================================================================
#  Full-dataset metric comparison:  Framework vs TFLite
# =====================================================================

def plot_fw_vs_tflite_metrics(
    fw_ev,
    tfl_ev,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 6),
) -> None:
    """3-panel comparison of Framework vs TFLite EvaluationResults.

    Panel 1: Global metrics bar chart (mAP@50, mAP@50-95, P, R, F1).
    Panel 2: Per-class AP@50 grouped bar chart.
    Panel 3: Delta table (difference FW − TFLite).

    Parameters
    ----------
    fw_ev : EvaluationResults
        Framework model evaluation (val or test split).
    tfl_ev : EvaluationResults
        TFLite model evaluation on the same split.
    save_path : str, optional
        Path to save the figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Framework vs TFLite — Métricas Comparadas", fontsize=14, fontweight="bold")

    # ── Panel 1: Global metrics ──
    ax = axes[0]
    metric_names = ["mAP@50", "mAP@50-95", "Precision", "Recall", "F1"]
    fw_vals = [fw_ev.mAP50, fw_ev.mAP50_95, fw_ev.precision, fw_ev.recall, fw_ev.f1]
    tfl_vals = [tfl_ev.mAP50, tfl_ev.mAP50_95, tfl_ev.precision, tfl_ev.recall, tfl_ev.f1]

    x = np.arange(len(metric_names))
    w = 0.35
    bars1 = ax.bar(x - w / 2, fw_vals, w, label="Framework", color="tab:blue", alpha=0.85)
    bars2 = ax.bar(x + w / 2, tfl_vals, w, label="TFLite INT8", color="tab:orange", alpha=0.85)

    # Annotate values
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Métricas Globales")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: Per-class AP@50 ──
    ax2 = axes[1]
    classes = sorted(set(list(fw_ev.per_class_ap50.keys()) +
                         list(tfl_ev.per_class_ap50.keys())))
    if classes:
        x2 = np.arange(len(classes))
        fw_ap = [fw_ev.per_class_ap50.get(c, 0.0) for c in classes]
        tfl_ap = [tfl_ev.per_class_ap50.get(c, 0.0) for c in classes]
        ax2.bar(x2 - w / 2, fw_ap, w, label="Framework", color="tab:blue", alpha=0.85)
        ax2.bar(x2 + w / 2, tfl_ap, w, label="TFLite INT8", color="tab:orange", alpha=0.85)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
        ax2.set_ylabel("AP@50")
        ax2.set_title("AP@50 por Clase")
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 1.15)
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Sin datos per-class", ha="center", va="center")
        ax2.set_title("AP@50 por Clase")

    # ── Panel 3: Delta table ──
    ax3 = axes[2]
    ax3.axis("off")
    row_labels = metric_names + [f"AP50 {c}" for c in classes]
    fw_all = fw_vals + [fw_ev.per_class_ap50.get(c, 0.0) for c in classes]
    tfl_all = tfl_vals + [tfl_ev.per_class_ap50.get(c, 0.0) for c in classes]
    deltas = [f - t for f, t in zip(fw_all, tfl_all)]

    table_data = []
    for label, fv, tv, d in zip(row_labels, fw_all, tfl_all, deltas):
        sign = "+" if d >= 0 else ""
        table_data.append([label, f"{fv:.4f}", f"{tv:.4f}", f"{sign}{d:.4f}"])

    table = ax3.table(
        cellText=table_data,
        colLabels=["Métrica", "Framework", "TFLite", "Δ (FW−TFL)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Color deltas: green if positive (FW better), red if negative
    for i, d in enumerate(deltas):
        cell = table[i + 1, 3]  # +1 because header row
        if d > 0.01:
            cell.set_facecolor("#d4edda")
        elif d < -0.01:
            cell.set_facecolor("#f8d7da")

    ax3.set_title("Tabla de Diferencias", pad=20)

    plt.tight_layout()
    if save_path:
        safe_mkdir(str(Path(save_path).parent))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"📊 FW vs TFLite metrics guardados: {save_path}")
    plt.show()


def visualize_fw_vs_tflite_samples(
    images: list,
    fw_dets: List[List[DetectedObject]],
    tfl_dets: List[List[DetectedObject]],
    class_names: List[str],
    samples_per_class: int = 1,
    save_path: Optional[str] = None,
    figsize_per_img: float = 5.0,
) -> None:
    """Side-by-side FW vs TFLite visualization, 1 image per class.

    For each class, selects an image where the **Framework** model
    detected at least one instance of that class, then renders:
        - Left panel: Framework predictions.
        - Right panel: TFLite predictions.

    Parameters
    ----------
    images : list
        List of images (np.ndarray [H,W,3] in [0,1] or file paths).
    fw_dets / tfl_dets : list of list of DetectedObject
        Parallel lists of detections for each image.
    class_names : list of str
        Ordered class names (index = class_id).
    samples_per_class : int
        How many images to show per class (default 1).
    save_path : str, optional
        If given, saves the combined figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Map class_id → list of image indices where FW detected that class
    from collections import defaultdict
    class_img_map: Dict[int, List[int]] = defaultdict(list)
    for idx, dets in enumerate(fw_dets):
        seen_classes = set()
        for det in dets:
            if det.class_id not in seen_classes:
                class_img_map[det.class_id].append(idx)
                seen_classes.add(det.class_id)

    # Build rows: one per class (or per sample_per_class)
    rows_data = []  # (class_name, img_idx)
    for cid in range(len(class_names)):
        candidates = class_img_map.get(cid, [])
        n = min(samples_per_class, len(candidates))
        for k in range(n):
            rows_data.append((class_names[cid], candidates[k]))

    if not rows_data:
        log("⚠️  No hay detecciones FW para seleccionar muestras por clase")
        return

    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(2 * figsize_per_img, n_rows * figsize_per_img),
                             squeeze=False)
    fig.suptitle("Framework vs TFLite — Predicciones por Clase",
                 fontsize=14, fontweight="bold")

    from .utils_infer import _COLORS

    for row, (cname, img_i) in enumerate(rows_data):
        img = images[img_i]
        if isinstance(img, str):
            img = plt.imread(img)
        h, w_img = img.shape[:2]

        for col, (dets, label) in enumerate([
            (fw_dets[img_i], "Framework"),
            (tfl_dets[img_i], "TFLite INT8"),
        ]):
            ax = axes[row][col]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{label} ({len(dets)} dets)", fontsize=9)

            for det in dets:
                x1, y1, x2, y2 = det.bbox
                rx = x1 * w_img
                ry = y1 * h
                rw = (x2 - x1) * w_img
                rh = (y2 - y1) * h
                color = np.array(_COLORS[det.class_id % len(_COLORS)]) / 255.0
                rect = patches.Rectangle(
                    (rx, ry), rw, rh, linewidth=2,
                    edgecolor=color, facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(rx, ry - 3,
                        f"{det.class_name} {det.confidence:.2f}",
                        fontsize=7, color="white",
                        bbox=dict(facecolor=color, alpha=0.7, pad=1))

            # Row label on left panel
            if col == 0:
                ax.set_ylabel(cname, fontsize=11, fontweight="bold",
                              rotation=0, labelpad=60, va="center")

    plt.tight_layout()
    if save_path:
        safe_mkdir(str(Path(save_path).parent))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"📊 FW vs TFLite samples guardados: {save_path}")
    plt.show()
