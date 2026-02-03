"""Advanced comparison utilities for object detection results."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .utils_eval import BoundingBox, DetectionResult, MetricsResult, match_predictions
    from .utils_viz import plot_detections
except ImportError:
    from utils_eval import BoundingBox, DetectionResult, MetricsResult, match_predictions
    from utils_viz import plot_detections


def _area(box: BoundingBox) -> float:
    return float(max(0.0, box.w) * max(0.0, box.h))


def _area_norm(box: BoundingBox, image_size: Tuple[int, int]) -> float:
    h, w = image_size
    denom = float(h * w) if h and w else 1.0
    return _area(box) / denom


def matched_iou_by_size(
    results: List[DetectionResult],
    image_size: Tuple[int, int] = (224, 224),
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    areas = []
    ious = []
    for det in results:
        matches, _, _ = match_predictions(det.predictions, det.ground_truth, iou_threshold)
        for _, gt_idx, iou in matches:
            gt = det.ground_truth[gt_idx]
            areas.append(_area_norm(gt, image_size))
            ious.append(iou)
    return np.array(areas), np.array(ious)


def pred_conf_by_size(
    results: List[DetectionResult],
    image_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    areas = []
    confs = []
    for det in results:
        for pred in det.predictions:
            areas.append(_area_norm(pred, image_size))
            confs.append(float(pred.confidence))
    return np.array(areas), np.array(confs)


def error_samples(
    results: List[DetectionResult],
    image_size: Tuple[int, int] = (224, 224),
    iou_threshold: float = 0.5,
    top_k: int = 5,
) -> Dict[str, List[Tuple[int, BoundingBox]]]:
    fp_samples: List[Tuple[int, BoundingBox]] = []
    fn_samples: List[Tuple[int, BoundingBox]] = []

    for det in results:
        matches, unmatched_preds, unmatched_gt = match_predictions(
            det.predictions, det.ground_truth, iou_threshold
        )
        for p_idx in unmatched_preds:
            fp_samples.append((det.image_id, det.predictions[p_idx]))
        for g_idx in unmatched_gt:
            fn_samples.append((det.image_id, det.ground_truth[g_idx]))

    fp_samples = sorted(fp_samples, key=lambda x: x[1].confidence, reverse=True)[:top_k]
    fn_samples = sorted(fn_samples, key=lambda x: _area_norm(x[1], image_size), reverse=True)[:top_k]

    return {"fp": fp_samples, "fn": fn_samples}


def build_comparison_table(
    ssd_metrics: MetricsResult,
    yolo_metrics: MetricsResult,
) -> pd.DataFrame:
    rows = [
        {
            "model": "SSD",
            "mAP@50": ssd_metrics.map_50,
            "precision": ssd_metrics.precision,
            "recall": ssd_metrics.recall,
            "f1": ssd_metrics.f1_score,
            "det_acc": ssd_metrics.detection_accuracy,
            "avg_iou": ssd_metrics.avg_iou,
            "time_ms": ssd_metrics.inference_time_ms,
        },
        {
            "model": "YOLO",
            "mAP@50": yolo_metrics.map_50,
            "precision": yolo_metrics.precision,
            "recall": yolo_metrics.recall,
            "f1": yolo_metrics.f1_score,
            "det_acc": yolo_metrics.detection_accuracy,
            "avg_iou": yolo_metrics.avg_iou,
            "time_ms": yolo_metrics.inference_time_ms,
        },
    ]
    return pd.DataFrame(rows)


def plot_metric_comparison(compare_df: pd.DataFrame) -> None:
    metrics_cols = ["mAP@50", "precision", "recall", "f1", "det_acc", "avg_iou"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(metrics_cols))
    vals = compare_df.set_index("model")[metrics_cols].reindex(["SSD", "YOLO"]).to_numpy(dtype=float)
    ax.bar([i - 0.2 for i in x], vals[0], width=0.4, label="SSD")
    ax.bar([i + 0.2 for i in x], vals[1], width=0.4, label="YOLO")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_cols, rotation=30)
    ax.set_ylim(0, 1)
    ax.set_title("Comparativa de métricas (SSD vs YOLO)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    ssd_metrics: MetricsResult,
    yolo_metrics: MetricsResult,
    class_names: List[str],
) -> None:
    def _plot(cm: np.ndarray, title: str, class_names: List[str]) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("GT")
        ax.set_ylabel("Pred")
        ax.set_xticks(range(len(class_names) + 1))
        ax.set_yticks(range(len(class_names) + 1))
        ax.set_xticklabels(class_names + ["bg"], rotation=45, ha="right")
        ax.set_yticklabels(class_names + ["bg"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.show()

    _plot(ssd_metrics.confusion_matrix, "SSD - Confusion Matrix", class_names)
    _plot(yolo_metrics.confusion_matrix, "YOLO - Confusion Matrix", class_names)


def plot_size_scatter(
    ssd_results: List[DetectionResult],
    yolo_results: List[DetectionResult],
    image_size: Tuple[int, int] = (224, 224),
) -> None:
    ssd_areas_iou, ssd_ious = matched_iou_by_size(ssd_results, image_size)
    yolo_areas_iou, yolo_ious = matched_iou_by_size(yolo_results, image_size)
    ssd_areas_conf, ssd_conf = pred_conf_by_size(ssd_results, image_size)
    yolo_areas_conf, yolo_conf = pred_conf_by_size(yolo_results, image_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(ssd_areas_iou, ssd_ious, alpha=0.6, label="SSD")
    axes[0].scatter(yolo_areas_iou, yolo_ious, alpha=0.6, label="YOLO")
    axes[0].set_title("Tamaño vs IoU (matches)")
    axes[0].set_xlabel("Área normalizada")
    axes[0].set_ylabel("IoU")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(ssd_areas_conf, ssd_conf, alpha=0.6, label="SSD")
    axes[1].scatter(yolo_areas_conf, yolo_conf, alpha=0.6, label="YOLO")
    axes[1].set_title("Tamaño vs Confianza (preds)")
    axes[1].set_xlabel("Área normalizada")
    axes[1].set_ylabel("Confianza")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_error_samples(
    title: str,
    samples: List[Tuple[int, BoundingBox]],
    test_loader,
    mode: str = "fp",
    max_samples: int = 3,
) -> None:
    print(f"\n{title} (mostrando {min(max_samples, len(samples))})")
    for image_id, box in samples[:max_samples]:
        img = test_loader.load_resized_image(image_id)
        gt = test_loader.get_scaled_ground_truth(image_id)
        preds = [box] if mode == "fp" else []
        plot_detections(img, gt, preds, title=f"{title} - img {image_id}")


def run_error_analysis(
    ssd_results: List[DetectionResult],
    yolo_results: List[DetectionResult],
    test_loader,
    image_size: Tuple[int, int] = (224, 224),
    iou_threshold: float = 0.5,
    top_k: int = 5,
) -> None:
    ssd_err = error_samples(ssd_results, image_size, iou_threshold=iou_threshold, top_k=top_k)
    yolo_err = error_samples(yolo_results, image_size, iou_threshold=iou_threshold, top_k=top_k)

    print("SSD FP/FN:", len(ssd_err["fp"]), len(ssd_err["fn"]))
    print("YOLO FP/FN:", len(yolo_err["fp"]), len(yolo_err["fn"]))

    visualize_error_samples("SSD - False Positives", ssd_err["fp"], test_loader, mode="fp")
    visualize_error_samples("SSD - False Negatives", ssd_err["fn"], test_loader, mode="fn")
    visualize_error_samples("YOLO - False Positives", yolo_err["fp"], test_loader, mode="fp")
    visualize_error_samples("YOLO - False Negatives", yolo_err["fn"], test_loader, mode="fn")


def _normalize_inverse(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin <= 1e-9:
        return np.ones_like(values, dtype=float)
    return (vmax - values) / (vmax - vmin)


def build_final_evaluation_matrix(
    ssd_metrics: MetricsResult,
    yolo_metrics: MetricsResult,
    weights: Dict[str, float] | None = None,
    model_info: Dict[str, Dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Build final evaluation matrix with normalized scores.

    Dimensions:
        - precision_score: mean of mAP@50, F1, avg IoU
        - speed_score: normalized inverse inference time
        - robustness_score: detection accuracy
        - complexity_score: normalized inverse model size (if provided), else proxy via speed
    """
    if weights is None:
        weights = {
            "precision_score": 0.4,
            "speed_score": 0.2,
            "robustness_score": 0.2,
            "complexity_score": 0.2,
        }

    precision_vals = np.array([
        np.mean([ssd_metrics.map_50, ssd_metrics.f1_score, ssd_metrics.avg_iou]),
        np.mean([yolo_metrics.map_50, yolo_metrics.f1_score, yolo_metrics.avg_iou]),
    ])
    speed_vals = _normalize_inverse(np.array([ssd_metrics.inference_time_ms, yolo_metrics.inference_time_ms]))
    robustness_vals = np.array([ssd_metrics.detection_accuracy, yolo_metrics.detection_accuracy])

    if model_info is not None and all(k in model_info for k in ("SSD", "YOLO")):
        ssd_size = model_info["SSD"].get("size_mb", ssd_metrics.inference_time_ms)
        yolo_size = model_info["YOLO"].get("size_mb", yolo_metrics.inference_time_ms)
        complexity_vals = _normalize_inverse(np.array([ssd_size, yolo_size]))
        complexity_basis = "size_mb"
    else:
        complexity_vals = speed_vals.copy()
        complexity_basis = "time_ms"

    df = pd.DataFrame(
        {
            "model": ["SSD", "YOLO"],
            "precision_score": precision_vals,
            "speed_score": speed_vals,
            "robustness_score": robustness_vals,
            "complexity_score": complexity_vals,
        }
    )

    df["overall_score"] = (
        df["precision_score"] * weights.get("precision_score", 0.0)
        + df["speed_score"] * weights.get("speed_score", 0.0)
        + df["robustness_score"] * weights.get("robustness_score", 0.0)
        + df["complexity_score"] * weights.get("complexity_score", 0.0)
    )

    df.attrs["complexity_basis"] = complexity_basis
    return df


def plot_final_matrix(df: pd.DataFrame) -> None:
    cols = ["precision_score", "speed_score", "robustness_score", "complexity_score", "overall_score"]
    data = df.set_index("model")[cols]
    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(data.values, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()
