"""Evaluation metrics for object detection models.

This module provides:
- mAP@50 computation
- Average Precision per class
- Precision, Recall, F1-Score
- Confusion matrix with "No Detection" column
- Full evaluation pipeline for Keras and TFLite models
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from .utils_mobilenet_infer import (
    Detection,
    run_inference_keras,
    run_inference_tflite,
    batch_inference_keras,
)


@dataclass
class GroundTruth:
    """Ground truth bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    matched: bool = False  # Flag for matching with predictions
    
    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    # Overall metrics
    map50: float
    precision: float
    recall: float
    f1_score: float
    
    # Per-class metrics
    ap_per_class: Dict[int, float]
    precision_per_class: Dict[int, float]
    recall_per_class: Dict[int, float]
    f1_per_class: Dict[int, float]
    
    # Counts
    total_gt: int
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    
    # Confusion matrix (N+1 x N+2): rows=pred, cols=GT + "Background" + "No Detection"
    confusion_matrix: np.ndarray
    class_names: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "map50": self.map50,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "ap_per_class": {str(k): v for k, v in self.ap_per_class.items()},
            "precision_per_class": {str(k): v for k, v in self.precision_per_class.items()},
            "recall_per_class": {str(k): v for k, v in self.recall_per_class.items()},
            "f1_per_class": {str(k): v for k, v in self.f1_per_class.items()},
            "total_gt": self.total_gt,
            "total_predictions": self.total_predictions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "class_names": self.class_names,
        }
    
    def print_summary(self):
        """Print formatted summary of results."""
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📈 Overall Metrics:")
        print(f"   mAP@50:     {self.map50:.4f}")
        print(f"   Precision:  {self.precision:.4f}")
        print(f"   Recall:     {self.recall:.4f}")
        print(f"   F1-Score:   {self.f1_score:.4f}")
        
        print(f"\n📦 Detection Counts:")
        print(f"   Ground Truth:     {self.total_gt}")
        print(f"   Predictions:      {self.total_predictions}")
        print(f"   True Positives:   {self.true_positives}")
        print(f"   False Positives:  {self.false_positives}")
        print(f"   False Negatives:  {self.false_negatives}")
        
        print(f"\n📊 Per-Class Metrics:")
        print(f"   {'Class':<15} {'AP@50':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print(f"   {'-'*47}")
        
        for class_id in sorted(self.ap_per_class.keys()):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            ap = self.ap_per_class.get(class_id, 0)
            prec = self.precision_per_class.get(class_id, 0)
            rec = self.recall_per_class.get(class_id, 0)
            f1 = self.f1_per_class.get(class_id, 0)
            print(f"   {class_name:<15} {ap:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")
        
        print("="*60)


def compute_iou_boxes(box1: Tuple[float, float, float, float], 
                      box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h
    
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_detections_to_gt(
    detections: List[Detection],
    ground_truths: List[GroundTruth],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[Detection, GroundTruth]], List[Detection], List[GroundTruth]]:
    """Match detections to ground truth boxes.
    
    Args:
        detections: List of predictions
        ground_truths: List of GT boxes
        iou_threshold: Minimum IoU for a match
        
    Returns:
        Tuple of:
        - matched: List of (detection, gt) pairs
        - unmatched_dets: Detections without matching GT (false positives)
        - unmatched_gts: GT without matching detection (false negatives)
    """
    # Reset matched flags
    for gt in ground_truths:
        gt.matched = False
    
    # Sort detections by confidence (descending)
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    
    matched = []
    unmatched_dets = []
    
    for det in sorted_dets:
        best_iou = 0
        best_gt = None
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truths):
            # Only match same class
            if gt.class_id != det.class_id:
                continue
            
            # Skip already matched
            if gt.matched:
                continue
            
            iou = compute_iou_boxes(
                (det.x1, det.y1, det.x2, det.y2),
                (gt.x1, gt.y1, gt.x2, gt.y2)
            )
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt = gt
                best_gt_idx = i
        
        if best_gt is not None:
            best_gt.matched = True
            matched.append((det, best_gt))
        else:
            unmatched_dets.append(det)
    
    # Collect unmatched ground truths
    unmatched_gts = [gt for gt in ground_truths if not gt.matched]
    
    return matched, unmatched_dets, unmatched_gts


def compute_ap(
    detections: List[Detection],
    ground_truths: List[GroundTruth],
    class_id: int,
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision for a single class.
    
    Uses the 11-point interpolation method.
    """
    # Filter by class
    class_dets = [d for d in detections if d.class_id == class_id]
    class_gts = [g for g in ground_truths if g.class_id == class_id]
    
    if not class_gts:
        return 0.0 if class_dets else 1.0
    
    if not class_dets:
        return 0.0
    
    # Sort by confidence
    class_dets = sorted(class_dets, key=lambda d: d.confidence, reverse=True)
    
    # Reset matched flags
    for gt in class_gts:
        gt.matched = False
    
    # Compute TP/FP for each detection
    tp = np.zeros(len(class_dets))
    fp = np.zeros(len(class_dets))
    
    for i, det in enumerate(class_dets):
        best_iou = 0
        best_gt = None
        
        for gt in class_gts:
            if gt.matched:
                continue
            
            iou = compute_iou_boxes(
                (det.x1, det.y1, det.x2, det.y2),
                (gt.x1, gt.y1, gt.x2, gt.y2)
            )
            
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        
        if best_iou >= iou_threshold and best_gt is not None:
            best_gt.matched = True
            tp[i] = 1
        else:
            fp[i] = 1
    
    # Compute cumulative precision/recall
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    recalls = cum_tp / len(class_gts)
    precisions = cum_tp / (cum_tp + cum_fp)
    
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec_at_recall = precisions[recalls >= t]
        if len(prec_at_recall) > 0:
            ap += np.max(prec_at_recall)
    ap /= 11
    
    return ap


def compute_map50(
    all_detections: List[List[Detection]],
    all_ground_truths: List[List[GroundTruth]],
    num_classes: int,
) -> Tuple[float, Dict[int, float]]:
    """Compute mAP@50 across all images and classes.
    
    Args:
        all_detections: List of detection lists (one per image)
        all_ground_truths: List of GT lists (one per image)
        num_classes: Number of classes
        
    Returns:
        Tuple of (mAP@50, dict of AP per class)
    """
    # Flatten all detections and GTs
    flat_dets = []
    flat_gts = []
    
    for dets in all_detections:
        flat_dets.extend(dets)
    
    for gts in all_ground_truths:
        # Create new GT objects to avoid modifying originals
        for gt in gts:
            flat_gts.append(GroundTruth(
                x1=gt.x1, y1=gt.y1, x2=gt.x2, y2=gt.y2,
                class_id=gt.class_id, matched=False
            ))
    
    # Compute AP per class
    ap_per_class = {}
    for class_id in range(num_classes):
        ap = compute_ap(flat_dets, flat_gts, class_id, iou_threshold=0.5)
        ap_per_class[class_id] = ap
    
    # mAP is mean of AP values (only for classes with GT)
    classes_with_gt = [c for c in range(num_classes) 
                       if any(gt.class_id == c for gt in flat_gts)]
    
    if classes_with_gt:
        map50 = np.mean([ap_per_class[c] for c in classes_with_gt])
    else:
        map50 = 0.0
    
    return map50, ap_per_class


def build_confusion_matrix(
    all_detections: List[List[Detection]],
    all_ground_truths: List[List[GroundTruth]],
    num_classes: int,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """Build symmetric confusion matrix for object detection.
    
    Matrix layout (symmetric):
    - Rows: Predicted class (0..N-1) + "Background" (N)
    - Columns: GT class (0..N-1) + "Background" (N)
    
    Interpretation:
    - Diagonal: Correct detections (TP for each class)
    - Off-diagonal (rows 0..N-1, cols 0..N-1): Class confusion
    - Row i, col N (Background): False Positives for class i (detected but no GT)
    - Row N, col j: False Negatives for class j (GT not detected)
    - Cell [N, N]: True Negatives (not used in object detection, set to 0)
    
    Args:
        all_detections: List of detection lists per image
        all_ground_truths: List of GT lists per image
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
        score_threshold: Confidence threshold for predictions
        normalize: Whether to normalize by column (GT class) for recall view
        
    Returns:
        Confusion matrix (N+1) x (N+1)
    """
    # Matrix: both rows and cols = classes (0..N-1) + Background (N)
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.float32)
    
    for detections, ground_truths in zip(all_detections, all_ground_truths):
        # Filter by score threshold
        filtered_dets = [d for d in detections if d.confidence >= score_threshold]
        
        # Create fresh GT copies
        gts = [GroundTruth(x1=g.x1, y1=g.y1, x2=g.x2, y2=g.y2, 
                          class_id=g.class_id, matched=False) 
               for g in ground_truths]
        
        # Match detections to GT
        matched, unmatched_dets, unmatched_gts = match_detections_to_gt(
            filtered_dets, gts, iou_threshold
        )
        
        # True positives and class confusion: pred class vs GT class
        for det, gt in matched:
            cm[det.class_id, gt.class_id] += 1
        
        # False positives (detected but no GT): pred class row, Background column
        for det in unmatched_dets:
            cm[det.class_id, num_classes] += 1
        
        # False negatives (GT not detected): Background row, GT class column
        for gt in unmatched_gts:
            cm[num_classes, gt.class_id] += 1
    
    # Note: cm[num_classes, num_classes] stays 0 (TN not meaningful in detection)
    
    if normalize:
        # Normalize by column (GT class) to show recall per class
        # This answers: "Of all GT of class X, what fraction was detected as each class?"
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        cm = cm / col_sums
    
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
    show_tn_cell: bool = False,
) -> plt.Figure:
    """Plot symmetric confusion matrix for object detection.
    
    Matrix interpretation:
    - Diagonal (class, class): True Positives (correct detections)
    - Off-diagonal (pred_i, gt_j): Class confusion (detected as i, was j)
    - (pred_class, Background): False Positives (detected but no GT)
    - (Background, gt_class): False Negatives (GT not detected)
    - (Background, Background): Not applicable in detection (shown in gray)
    
    Args:
        cm: Confusion matrix (N+1 x N+1)
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
        show_tn_cell: Whether to show value in TN cell (usually N/A)
        
    Returns:
        Matplotlib figure
    """
    # Symmetric labels: classes + Background for both axes
    labels = class_names + ["Background"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create masked array to gray out the TN cell if needed
    masked_cm = np.ma.array(cm, mask=False)
    if not show_tn_cell:
        # Mask the bottom-right cell (Background, Background)
        mask = np.zeros_like(cm, dtype=bool)
        mask[-1, -1] = True
        masked_cm = np.ma.array(cm, mask=mask)
    
    im = ax.imshow(masked_cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Gray out the TN cell
    if not show_tn_cell:
        ax.add_patch(plt.Rectangle(
            (cm.shape[1] - 1.5, cm.shape[0] - 1.5), 1, 1,
            fill=True, facecolor='lightgray', edgecolor='gray', linewidth=1
        ))
    
    # Ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title=title,
           ylabel='Predicted',
           xlabel='Ground Truth')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Skip TN cell annotation if not showing
            if not show_tn_cell and i == cm.shape[0] - 1 and j == cm.shape[1] - 1:
                ax.text(j, i, "N/A",
                        ha="center", va="center",
                        color="gray", fontsize=9, fontstyle='italic')
                continue
                
            value = cm[i, j]
            text = f"{value:.2f}" if value < 1 else f"{value:.0f}"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if value > thresh else "black",
                    fontsize=9)
    
    # Add legend explaining the matrix
    legend_text = (
        "Diagonal: TP | Off-diag: Confusion\n"
        "Row→Bkg: FP | Col→Bkg: FN"
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Confusion matrix saved to: {save_path}")
    
    return fig


def evaluate_model_full(
    model,
    test_data: List[Dict],
    anchors: np.ndarray,
    num_classes: int,
    class_names: List[str],
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    eval_iou_threshold: float = 0.5,
    cm_score_threshold: float = 0.5,
    img_size: int = 224,
    batch_size: int = 32,
    is_tflite: bool = False,
    verbose: bool = True,
) -> EvaluationResults:
    """Complete evaluation pipeline.
    
    Args:
        model: Keras model or TFLite interpreter
        test_data: List of image data dicts with "path", "boxes", "classes"
        anchors: Anchor boxes for SSD
        num_classes: Number of classes
        class_names: List of class names
        score_threshold: Confidence threshold for NMS
        nms_iou_threshold: IoU threshold for NMS
        eval_iou_threshold: IoU threshold for evaluation matching
        cm_score_threshold: Score threshold for confusion matrix
        img_size: Input image size
        batch_size: Batch size for inference
        is_tflite: Whether model is TFLite interpreter
        verbose: Show progress
        
    Returns:
        EvaluationResults dataclass
    """
    import cv2
    
    if verbose:
        print(f"🔍 Evaluating model on {len(test_data)} images...")
    
    all_detections = []
    all_ground_truths = []
    
    # Load and process images
    images = []
    for item in test_data:
        img = cv2.imread(item["path"])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        images.append(img)
        
        # Convert GT to GroundTruth objects
        gts = []
        for box, cls in zip(item["boxes"], item["classes"]):
            # box is [x_center, y_center, width, height] normalized
            cx, cy, w, h = box
            gts.append(GroundTruth(
                x1=cx - w/2,
                y1=cy - h/2,
                x2=cx + w/2,
                y2=cy + h/2,
                class_id=cls,
            ))
        all_ground_truths.append(gts)
    
    if verbose:
        print(f"   Loaded {len(images)} images")
    
    # Run inference
    if is_tflite:
        # TFLite: process one at a time
        for i, img in enumerate(images):
            dets = run_inference_tflite(
                interpreter=model,
                image=img,
                anchors=anchors,
                score_threshold=score_threshold,
                nms_iou_threshold=nms_iou_threshold,
            )
            all_detections.append(dets)
            if verbose and (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(images)} images")
    else:
        # Keras: batch inference
        all_detections = batch_inference_keras(
            model=model,
            images=images,
            anchors=anchors,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            batch_size=batch_size,
            verbose=verbose,
        )
    
    if verbose:
        print(f"   Computing metrics...")
    
    # Compute mAP@50
    map50, ap_per_class = compute_map50(all_detections, all_ground_truths, num_classes)
    
    # Compute overall precision/recall
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    tp_per_class = {c: 0 for c in range(num_classes)}
    fp_per_class = {c: 0 for c in range(num_classes)}
    fn_per_class = {c: 0 for c in range(num_classes)}
    
    for detections, ground_truths in zip(all_detections, all_ground_truths):
        gts = [GroundTruth(x1=g.x1, y1=g.y1, x2=g.x2, y2=g.y2,
                          class_id=g.class_id, matched=False)
               for g in ground_truths]
        
        matched, unmatched_dets, unmatched_gts = match_detections_to_gt(
            detections, gts, eval_iou_threshold
        )
        
        total_tp += len(matched)
        total_fp += len(unmatched_dets)
        total_fn += len(unmatched_gts)
        
        # Per-class
        for det, gt in matched:
            tp_per_class[gt.class_id] += 1
        for det in unmatched_dets:
            fp_per_class[det.class_id] += 1
        for gt in unmatched_gts:
            fn_per_class[gt.class_id] += 1
    
    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Per-class precision/recall/F1
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    
    for c in range(num_classes):
        tp = tp_per_class[c]
        fp = fp_per_class[c]
        fn = fn_per_class[c]
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precision_per_class[c] = prec
        recall_per_class[c] = rec
        f1_per_class[c] = f1_c
    
    # Build confusion matrix
    cm = build_confusion_matrix(
        all_detections, all_ground_truths, num_classes,
        iou_threshold=eval_iou_threshold,
        score_threshold=cm_score_threshold,
        normalize=True,
    )
    
    # Count totals
    total_gt = sum(len(gts) for gts in all_ground_truths)
    total_preds = sum(len(dets) for dets in all_detections)
    
    results = EvaluationResults(
        map50=map50,
        precision=precision,
        recall=recall,
        f1_score=f1,
        ap_per_class=ap_per_class,
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        f1_per_class=f1_per_class,
        total_gt=total_gt,
        total_predictions=total_preds,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        confusion_matrix=cm,
        class_names=class_names,
    )
    
    if verbose:
        results.print_summary()
    
    return results


def compare_keras_vs_tflite(
    keras_results: EvaluationResults,
    tflite_results: EvaluationResults,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compare Keras and TFLite model performance.
    
    Returns dict with comparison metrics.
    """
    comparison = {
        "map50_diff": keras_results.map50 - tflite_results.map50,
        "precision_diff": keras_results.precision - tflite_results.precision,
        "recall_diff": keras_results.recall - tflite_results.recall,
        "f1_diff": keras_results.f1_score - tflite_results.f1_score,
        "ap_diff_per_class": {},
    }
    
    for c in keras_results.ap_per_class:
        keras_ap = keras_results.ap_per_class.get(c, 0)
        tflite_ap = tflite_results.ap_per_class.get(c, 0)
        comparison["ap_diff_per_class"][c] = keras_ap - tflite_ap
    
    print("\n" + "="*60)
    print("📊 KERAS vs TFLITE COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<20} {'Keras':>10} {'TFLite':>10} {'Diff':>10}")
    print("-"*50)
    print(f"{'mAP@50':<20} {keras_results.map50:>10.4f} {tflite_results.map50:>10.4f} {comparison['map50_diff']:>+10.4f}")
    print(f"{'Precision':<20} {keras_results.precision:>10.4f} {tflite_results.precision:>10.4f} {comparison['precision_diff']:>+10.4f}")
    print(f"{'Recall':<20} {keras_results.recall:>10.4f} {tflite_results.recall:>10.4f} {comparison['recall_diff']:>+10.4f}")
    print(f"{'F1-Score':<20} {keras_results.f1_score:>10.4f} {tflite_results.f1_score:>10.4f} {comparison['f1_diff']:>+10.4f}")
    
    print(f"\n{'AP per Class:'}")
    for c in sorted(keras_results.ap_per_class.keys()):
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        keras_ap = keras_results.ap_per_class.get(c, 0)
        tflite_ap = tflite_results.ap_per_class.get(c, 0)
        diff = keras_ap - tflite_ap
        print(f"   {name:<15} {keras_ap:>10.4f} {tflite_ap:>10.4f} {diff:>+10.4f}")
    
    # Assessment
    if abs(comparison['map50_diff']) < 0.02:
        print(f"\n✅ Cuantización exitosa: degradación mAP < 2%")
    elif abs(comparison['map50_diff']) < 0.05:
        print(f"\n⚠️ Degradación moderada: {abs(comparison['map50_diff'])*100:.1f}% pérdida en mAP")
    else:
        print(f"\n❌ Degradación significativa: {abs(comparison['map50_diff'])*100:.1f}% pérdida en mAP")
    
    print("="*60)
    
    return comparison
