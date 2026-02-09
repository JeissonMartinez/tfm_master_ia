"""YOLO26 evaluation utilities with mAP calculation.

Handles comprehensive model evaluation including:
- mAP@50 and mAP@50-95 calculation
- Per-class metrics
- Confusion matrix generation
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .utils_io import log, safe_exists, safe_mkdir
from .utils_yolo_infer import BoundingBox, DetectionResult


@dataclass
class EvaluationResults:
    """Container for model evaluation results."""
    model_name: str
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    per_class_ap50: Dict[str, float] = field(default_factory=dict)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)
    total_images: int = 0
    total_predictions: int = 0
    total_ground_truth: int = 0
    avg_inference_time_ms: float = 0.0

    def summary(self) -> str:
        """Return formatted summary of results."""
        lines = [
            "=" * 60,
            f"📊 RESULTADOS DE EVALUACIÓN: {self.model_name}",
            "=" * 60,
            f"🎯 mAP@50: {self.map50:.4f} ({self.map50*100:.1f}%)",
            f"🎯 mAP@50-95: {self.map50_95:.4f} ({self.map50_95*100:.1f}%)",
            f"📈 Precision: {self.precision:.4f}",
            f"📈 Recall: {self.recall:.4f}",
            f"📈 F1-Score: {self.f1_score:.4f}",
            "",
            "📋 mAP@50 por clase:",
        ]
        for cls_name, ap in self.per_class_ap50.items():
            lines.append(f"   {cls_name}: {ap:.4f}")
        
        lines.extend([
            "",
            f"📸 Imágenes: {self.total_images}",
            f"🔍 Predicciones: {self.total_predictions}",
            f"✅ Ground Truth: {self.total_ground_truth}",
            f"⏱️ Tiempo promedio: {self.avg_inference_time_ms:.2f} ms",
            "=" * 60,
        ])
        return "\n".join(lines)


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute IoU between two bounding boxes.

    Args:
        box1: First bounding box
        box2: Second bounding box

    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1.to_xyxy()
    x1_2, y1_2, x2_2, y2_2 = box2.to_xyxy()

    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    # Union
    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)
    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def match_predictions_to_gt(
    predictions: List[BoundingBox],
    ground_truth: List[BoundingBox],
    iou_threshold: float = 0.5,
) -> Tuple[List[bool], List[bool]]:
    """Match predictions to ground truth boxes.

    Args:
        predictions: List of predicted boxes
        ground_truth: List of ground truth boxes
        iou_threshold: IoU threshold for matching

    Returns:
        Tuple of (prediction_matched, gt_matched) boolean lists
    """
    pred_matched = [False] * len(predictions)
    gt_matched = [False] * len(ground_truth)

    if not predictions or not ground_truth:
        return pred_matched, gt_matched

    # Sort predictions by confidence (descending)
    sorted_indices = sorted(
        range(len(predictions)),
        key=lambda i: predictions[i].confidence,
        reverse=True
    )

    for pred_idx in sorted_indices:
        pred = predictions[pred_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            if pred.class_id != gt.class_id:
                continue
            
            iou = compute_iou(pred, gt)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            pred_matched[pred_idx] = True
            gt_matched[best_gt_idx] = True

    return pred_matched, gt_matched


def calculate_ap(
    precisions: List[float],
    recalls: List[float],
) -> float:
    """Calculate Average Precision using 11-point interpolation.

    Args:
        precisions: List of precision values
        recalls: List of recall values

    Returns:
        Average Precision value
    """
    if not precisions or not recalls:
        return 0.0

    # Sort by recall
    sorted_indices = sorted(range(len(recalls)), key=lambda i: recalls[i])
    sorted_precisions = [precisions[i] for i in sorted_indices]
    sorted_recalls = [recalls[i] for i in sorted_indices]

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec_at_recall = 0.0
        for p, r in zip(sorted_precisions, sorted_recalls):
            if r >= t:
                prec_at_recall = max(prec_at_recall, p)
        ap += prec_at_recall / 11

    return ap


def calculate_map50(
    detection_results: List[DetectionResult],
    class_names: List[str],
    iou_threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """Calculate mAP@50 from detection results.

    Args:
        detection_results: List of DetectionResult objects
        class_names: List of class names
        iou_threshold: IoU threshold (default 0.5)

    Returns:
        Tuple of (mean_ap, per_class_ap)
    """
    # Organize predictions and GT by class
    class_predictions: Dict[str, List[Tuple[float, bool]]] = {name: [] for name in class_names}
    class_num_gt: Dict[str, int] = {name: 0 for name in class_names}

    for result in detection_results:
        # Count ground truth per class
        for gt in result.ground_truth:
            if gt.class_name in class_num_gt:
                class_num_gt[gt.class_name] += 1

        # Match predictions
        pred_matched, _ = match_predictions_to_gt(
            result.predictions,
            result.ground_truth,
            iou_threshold
        )

        for pred, matched in zip(result.predictions, pred_matched):
            if pred.class_name in class_predictions:
                class_predictions[pred.class_name].append((pred.confidence, matched))

    # Calculate AP per class
    per_class_ap: Dict[str, float] = {}
    
    for class_name in class_names:
        preds = class_predictions[class_name]
        num_gt = class_num_gt[class_name]

        if num_gt == 0:
            per_class_ap[class_name] = 0.0
            continue

        # Sort by confidence
        preds.sort(key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for conf, is_tp in preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
            recall = tp_cumsum / num_gt if num_gt > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        per_class_ap[class_name] = calculate_ap(precisions, recalls)

    # Mean AP
    valid_aps = [ap for ap in per_class_ap.values() if ap > 0 or class_num_gt.get(list(per_class_ap.keys())[list(per_class_ap.values()).index(ap)], 0) > 0]
    mean_ap = np.mean(valid_aps) if valid_aps else 0.0

    return float(mean_ap), per_class_ap


def evaluate_model(
    model: Any,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 224,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    end2end: bool = True,
    verbose: bool = True,
    project: Optional[str] = None,
    name: str = "val",
) -> Optional[EvaluationResults]:
    """Evaluate YOLO26 model on a dataset split.

    Args:
        model: YOLO model instance
        data_yaml: Path to data.yaml
        split: Dataset split
        imgsz: Image size
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for mAP
        end2end: Use end-to-end inference
        verbose: Print progress
        project: Directory to save validation results. If None, derived
            automatically from the model's checkpoint path so that results
            are stored inside the experiment folder (e.g. logs/yolo26n_v1/val).
        name: Subdirectory name inside *project* (default ``"val"``).

    Returns:
        EvaluationResults object or None on failure
    """
    if model is None:
        log("❌ Modelo no disponible")
        return None

    # Derive project from model checkpoint path if not provided
    if project is None:
        try:
            ckpt_path = getattr(model, 'ckpt_path', None)
            if ckpt_path:
                # e.g. "logs/yolo26n_v1/weights/best.pt" → "logs/yolo26n_v1"
                project = str(Path(ckpt_path).parent.parent)
        except Exception:
            pass  # Fall back to Ultralytics default (runs/detect/val)

    if verbose:
        log(f"\n🔍 Evaluando modelo en split: {split}")
        if project:
            log(f"   📂 Resultados en: {project}/{name}")

    try:
        # Run validation with Ultralytics
        val_kwargs: Dict[str, Any] = dict(
            data=data_yaml,
            split=split,
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            end2end=end2end,
            verbose=verbose,
        )
        if project is not None:
            val_kwargs["project"] = project
            val_kwargs["name"] = name

        results = model.val(**val_kwargs)

        # Extract metrics
        eval_results = EvaluationResults(
            model_name=str(model.model_name) if hasattr(model, 'model_name') else "YOLO26",
        )

        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            eval_results.map50 = metrics.get("metrics/mAP50(B)", 0.0)
            eval_results.map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
            eval_results.precision = metrics.get("metrics/precision(B)", 0.0)
            eval_results.recall = metrics.get("metrics/recall(B)", 0.0)
            
            if eval_results.precision > 0 and eval_results.recall > 0:
                eval_results.f1_score = 2 * (eval_results.precision * eval_results.recall) / (eval_results.precision + eval_results.recall)

        # Get per-class metrics if available
        if hasattr(results, 'names') and hasattr(results, 'ap50'):
            names = results.names
            if isinstance(names, dict):
                names = list(names.values())
            eval_results.class_names = names
            
            if hasattr(results, 'ap50'):
                for i, name in enumerate(names):
                    if i < len(results.ap50):
                        eval_results.per_class_ap50[name] = float(results.ap50[i])

        # Get confusion matrix if available
        if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            eval_results.confusion_matrix = results.confusion_matrix.matrix

        if verbose:
            log(eval_results.summary())

        return eval_results

    except Exception as exc:
        log(f"❌ Error durante evaluación: {exc}")
        import traceback
        traceback.print_exc()
        return None


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    show: bool = True,
) -> None:
    """Plot confusion matrix.

    Args:
        confusion_matrix: NxN confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
        normalize: Whether to normalize the matrix
        show: Whether to display the plot
    """
    if confusion_matrix is None or len(confusion_matrix) == 0:
        log("⚠️ Confusion matrix no disponible")
        return

    # Normalize if requested
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        cm = np.divide(confusion_matrix, row_sums, where=row_sums != 0)
    else:
        cm = confusion_matrix

    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if value > thresh else "black",
                    fontsize=10)

    plt.tight_layout()
    safe_mkdir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    log(f"✅ Confusion matrix guardada en: {save_path}")


def compare_models(
    results_list: List[EvaluationResults],
    save_path: Optional[str] = None,
) -> None:
    """Compare evaluation results from multiple models.

    Args:
        results_list: List of EvaluationResults objects
        save_path: Optional path to save comparison plot
    """
    if not results_list:
        log("⚠️ No hay resultados para comparar")
        return

    # Print comparison table
    log("\n" + "=" * 80)
    log("📊 COMPARACIÓN DE MODELOS")
    log("=" * 80)
    log(f"{'Modelo':<20} {'mAP@50':>10} {'mAP@50-95':>12} {'Precision':>12} {'Recall':>10} {'F1':>8}")
    log("-" * 80)
    
    for result in results_list:
        log(f"{result.model_name:<20} {result.map50:>10.4f} {result.map50_95:>12.4f} "
            f"{result.precision:>12.4f} {result.recall:>10.4f} {result.f1_score:>8.4f}")
    
    log("=" * 80)

    if save_path:
        # Create comparison bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        model_names = [r.model_name for r in results_list]
        
        # mAP comparison
        ax = axes[0]
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, [r.map50 for r in results_list], width, label='mAP@50')
        ax.bar(x + width/2, [r.map50_95 for r in results_list], width, label='mAP@50-95')
        ax.set_ylabel('Score')
        ax.set_title('mAP Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Precision/Recall comparison
        ax = axes[1]
        ax.bar(x - width/2, [r.precision for r in results_list], width, label='Precision')
        ax.bar(x + width/2, [r.recall for r in results_list], width, label='Recall')
        ax.set_ylabel('Score')
        ax.set_title('Precision/Recall Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        safe_mkdir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        
        log(f"✅ Comparación guardada en: {save_path}")
