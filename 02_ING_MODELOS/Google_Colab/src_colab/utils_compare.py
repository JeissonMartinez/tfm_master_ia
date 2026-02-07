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
