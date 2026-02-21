"""Model comparison utilities — Cycle 2.

Compares PyTorch vs ONNX detections and provides plots / metrics
for degradation analysis.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils_io import log, safe_mkdir


# =====================================================================
#  Comparison container
# =====================================================================

@dataclass
class ComparisonResult:
    """Holds per-image comparison data between two models."""
    label_a: str = "PyTorch"
    label_b: str = "ONNX"
    per_image: List[Dict[str, Any]] = field(default_factory=list)
    iou_values: List[float] = field(default_factory=list)
    matched_conf_a: List[float] = field(default_factory=list)
    matched_conf_b: List[float] = field(default_factory=list)
    n_only_a: int = 0
    n_only_b: int = 0
    n_matched: int = 0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.iou_values)) if self.iou_values else 0.0

    @property
    def conf_delta_mean(self) -> float:
        if not self.matched_conf_a:
            return 0.0
        return float(np.mean(
            np.array(self.matched_conf_b) - np.array(self.matched_conf_a)
        ))


# =====================================================================
#  Core comparison
# =====================================================================

def compare_detections(
    dets_a_batch,   # List[List[DetectedObject]]
    dets_b_batch,   # List[List[DetectedObject]]
    iou_threshold: float = 0.5,
    label_a: str = "PyTorch",
    label_b: str = "ONNX",
) -> ComparisonResult:
    """Match detections from two sources via greedy IoU matching.

    Args:
        dets_a_batch: Detections from model A (per image).
        dets_b_batch: Detections from model B (per image).
        iou_threshold: Minimum IoU to consider a match.
        label_a, label_b: Human-readable labels.

    Returns:
        ComparisonResult with aggregate statistics.
    """
    result = ComparisonResult(label_a=label_a, label_b=label_b)

    for dets_a, dets_b in zip(dets_a_batch, dets_b_batch):
        matched_a = set()
        matched_b = set()

        for i, da in enumerate(dets_a):
            best_iou = 0.0
            best_j = -1
            for j, db in enumerate(dets_b):
                if j in matched_b:
                    continue
                if da.class_id != db.class_id:
                    continue
                iou = _iou(da.bbox, db.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0:
                matched_a.add(i)
                matched_b.add(best_j)
                result.iou_values.append(best_iou)
                result.matched_conf_a.append(da.confidence)
                result.matched_conf_b.append(dets_b[best_j].confidence)
                result.n_matched += 1

        result.n_only_a += len(dets_a) - len(matched_a)
        result.n_only_b += len(dets_b) - len(matched_b)

        result.per_image.append({
            "n_a": len(dets_a),
            "n_b": len(dets_b),
            "matched": len(matched_a),
            "only_a": len(dets_a) - len(matched_a),
            "only_b": len(dets_b) - len(matched_b),
        })

    return result


def _iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-8)


# =====================================================================
#  Plot helpers
# =====================================================================

def plot_iou_distribution(
    comparison: ComparisonResult,
    save_path: Optional[str] = None,
    title: str = "",
) -> None:
    """Histogram of IoU values for matched detections."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if comparison.iou_values:
        ax.hist(comparison.iou_values, bins=50, range=(0, 1),
                color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(comparison.mean_iou, color="red", ls="--",
                   label=f"Mean IoU={comparison.mean_iou:.3f}")
        ax.legend()
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title(title or f"IoU distribution ({comparison.label_a} vs {comparison.label_b})")
    plt.tight_layout()
    if save_path:
        safe_mkdir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_scatter(
    comparison: ComparisonResult,
    save_path: Optional[str] = None,
    title: str = "",
) -> None:
    """Scatter plot of confidence: model A vs model B for matched detections."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    if comparison.matched_conf_a:
        ax.scatter(comparison.matched_conf_a, comparison.matched_conf_b,
                   alpha=0.4, s=10, color="steelblue")
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="x=y")
        ax.legend()
    ax.set_xlabel(f"Confidence ({comparison.label_a})")
    ax.set_ylabel(f"Confidence ({comparison.label_b})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title or "Confidence comparison")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_detection_count_comparison(
    comparison: ComparisonResult,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart: detections per image for both models."""
    import matplotlib.pyplot as plt

    n_imgs = len(comparison.per_image)
    if n_imgs == 0:
        return

    x = np.arange(n_imgs)
    counts_a = [d["n_a"] for d in comparison.per_image]
    counts_b = [d["n_b"] for d in comparison.per_image]

    fig, ax = plt.subplots(figsize=(max(8, n_imgs * 0.3), 4))
    w = 0.35
    ax.bar(x - w / 2, counts_a, w, label=comparison.label_a, color="steelblue")
    ax.bar(x + w / 2, counts_b, w, label=comparison.label_b, color="coral")
    ax.set_xlabel("Image index")
    ax.set_ylabel("# detections")
    ax.legend()
    ax.set_title("Detection count per image")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_comparison_summary(comp: ComparisonResult) -> None:
    """Print human-readable comparison summary."""
    log(f"\n{'='*60}")
    log(f"  Comparación: {comp.label_a} vs {comp.label_b}")
    log(f"{'='*60}")
    log(f"  Imágenes evaluadas : {len(comp.per_image)}")
    log(f"  Matched detections : {comp.n_matched}")
    log(f"  Only {comp.label_a:12s} : {comp.n_only_a}")
    log(f"  Only {comp.label_b:12s} : {comp.n_only_b}")
    log(f"  Mean IoU (matched) : {comp.mean_iou:.4f}")
    log(f"  Conf delta (B - A) : {comp.conf_delta_mean:+.4f}")
    log(f"{'='*60}\n")


def visualize_comparison_grid(
    images: list,
    dets_a_batch,
    dets_b_batch,
    label_a: str = "PyTorch",
    label_b: str = "ONNX",
    max_images: int = 6,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side grid comparing two sets of detections."""
    from .utils_infer import compare_predictions_side_by_side

    # Use the per-image comparison from utils_infer
    n = min(len(images), max_images)
    for i in range(n):
        sp = None
        if save_path:
            base, ext = os.path.splitext(save_path)
            sp = f"{base}_{i}{ext}"
        compare_predictions_side_by_side(
            images[i], dets_a_batch[i], dets_b_batch[i],
            label_a=label_a, label_b=label_b,
            save_path=sp,
        )
