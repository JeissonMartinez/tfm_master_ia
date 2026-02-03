"""Inference and post-processing utilities for SSD-Lite models.

This module provides:
- Decoding of SSD predictions (anchors → bounding boxes)
- Non-Maximum Suppression (NMS)
- Complete inference pipeline for Keras and TFLite models
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import tensorflow as tf


@dataclass
class Detection:
    """Single detection result."""
    x1: float  # Top-left x (normalized 0-1)
    y1: float  # Top-left y (normalized 0-1)
    x2: float  # Bottom-right x (normalized 0-1)
    y2: float  # Bottom-right y (normalized 0-1)
    confidence: float  # Objectness score
    class_id: int  # Class index
    class_score: float  # Class probability
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_score": self.class_score,
        }


def decode_ssd_predictions(
    objectness: np.ndarray,
    class_probs: np.ndarray,
    bbox_pred: np.ndarray,
    anchors: np.ndarray,
    score_threshold: float = 0.3,
) -> List[Detection]:
    """Decode raw SSD predictions to Detection objects.
    
    Args:
        objectness: (num_anchors, 1) - Objectness scores (sigmoid)
        class_probs: (num_anchors, num_classes) - Class probabilities (softmax)
        bbox_pred: (num_anchors, 4) - Predicted boxes [cx, cy, w, h] normalized
        anchors: (num_anchors, 4) - Anchor boxes [cx, cy, w, h] normalized
        score_threshold: Minimum objectness score to keep
        
    Returns:
        List of Detection objects
    """
    detections = []
    num_anchors = len(anchors)
    
    for i in range(num_anchors):
        obj_score = float(objectness[i, 0])
        
        if obj_score < score_threshold:
            continue
        
        # Get class prediction
        class_id = int(np.argmax(class_probs[i]))
        class_score = float(class_probs[i, class_id])
        
        # Decode bbox: predictions are [cx, cy, w, h] relative to anchors
        # For SSD-Lite, bbox_pred is already in [cx, cy, w, h] format
        cx, cy, w, h = bbox_pred[i]
        
        # Clamp values
        cx = np.clip(cx, 0, 1)
        cy = np.clip(cy, 0, 1)
        w = np.clip(w, 0.001, 1)
        h = np.clip(h, 0.001, 1)
        
        # Convert to x1, y1, x2, y2
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Clamp to valid range
        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        detections.append(Detection(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            confidence=obj_score,
            class_id=class_id,
            class_score=class_score,
        ))
    
    return detections


def compute_iou(box1: Detection, box2: Detection) -> float:
    """Compute IoU between two detections."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h
    
    union = box1.area + box2.area - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def apply_nms(
    detections: List[Detection],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> List[Detection]:
    """Apply Non-Maximum Suppression to detections.
    
    Args:
        detections: List of Detection objects
        iou_threshold: IoU threshold for suppression
        class_agnostic: If True, apply NMS across all classes;
                        if False, apply per-class NMS
    
    Returns:
        Filtered list of Detection objects
    """
    if not detections:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    
    keep = []
    
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        filtered = []
        for det in detections:
            # Skip NMS between different classes (unless class_agnostic)
            if not class_agnostic and det.class_id != best.class_id:
                filtered.append(det)
                continue
            
            iou = compute_iou(best, det)
            if iou < iou_threshold:
                filtered.append(det)
        
        detections = filtered
    
    return keep


def postprocess_detections(
    objectness: np.ndarray,
    class_probs: np.ndarray,
    bbox_pred: np.ndarray,
    anchors: np.ndarray,
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> List[Detection]:
    """Complete post-processing pipeline: decode + NMS.
    
    Args:
        objectness: (num_anchors, 1) - Objectness scores
        class_probs: (num_anchors, num_classes) - Class probabilities
        bbox_pred: (num_anchors, 4) - Predicted boxes
        anchors: (num_anchors, 4) - Anchor boxes
        score_threshold: Minimum confidence to keep
        nms_iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections to return
        
    Returns:
        List of filtered Detection objects
    """
    # Decode predictions
    detections = decode_ssd_predictions(
        objectness=objectness,
        class_probs=class_probs,
        bbox_pred=bbox_pred,
        anchors=anchors,
        score_threshold=score_threshold,
    )
    
    # Apply NMS
    detections = apply_nms(
        detections=detections,
        iou_threshold=nms_iou_threshold,
        class_agnostic=False,
    )
    
    # Limit number of detections
    if len(detections) > max_detections:
        detections = detections[:max_detections]
    
    return detections


def run_inference_keras(
    model: tf.keras.Model,
    image: np.ndarray,
    anchors: np.ndarray,
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> List[Detection]:
    """Run inference on a single image using Keras model.
    
    Args:
        model: Keras model with outputs [objectness, class_out, bbox_out]
        image: Input image (H, W, 3) normalized to [0, 1]
        anchors: Anchor boxes
        score_threshold: Minimum confidence
        nms_iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections
        
    Returns:
        List of Detection objects
    """
    # Ensure correct shape
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    
    # Run inference
    outputs = model.predict(image, verbose=0)
    
    # Handle dict output
    if isinstance(outputs, dict):
        objectness = outputs["objectness"][0]
        class_probs = outputs["class_out"][0]
        bbox_pred = outputs["bbox_out"][0]
    else:
        # Assume list/tuple: [objectness, class_out, bbox_out]
        objectness = outputs[0][0]
        class_probs = outputs[1][0]
        bbox_pred = outputs[2][0]
    
    return postprocess_detections(
        objectness=objectness,
        class_probs=class_probs,
        bbox_pred=bbox_pred,
        anchors=anchors,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )


def run_inference_tflite(
    interpreter: tf.lite.Interpreter,
    image: np.ndarray,
    anchors: np.ndarray,
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> List[Detection]:
    """Run inference on a single image using TFLite interpreter.
    
    Args:
        interpreter: TFLite interpreter (already allocated)
        image: Input image (H, W, 3) - will be normalized and quantized if needed
        anchors: Anchor boxes
        score_threshold: Minimum confidence
        nms_iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections
        
    Returns:
        List of Detection objects
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.uint8:
        # Quantized input
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.float32)
    
    # Set input and invoke
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    # Get outputs - need to identify which is which by shape or name
    # Model outputs: objectness (441,1), class_out (441,num_classes), bbox_out (441,4)
    outputs = {}
    shape_4_outputs = []  # Collect outputs with last dim = 4 (could be bbox or class)
    
    for detail in output_details:
        tensor = interpreter.get_tensor(detail['index'])
        shape = tensor.shape
        name = detail['name'].lower()
        
        # First try to identify by name
        if 'objectness' in name or 'obj' in name:
            outputs['objectness'] = tensor[0] if tensor.ndim == 3 else tensor
        elif 'class' in name or 'cls' in name:
            outputs['class_out'] = tensor[0] if tensor.ndim == 3 else tensor
        elif 'bbox' in name or 'box' in name or 'loc' in name:
            outputs['bbox_out'] = tensor[0] if tensor.ndim == 3 else tensor
        else:
            # Fallback: identify output by shape
            last_dim = shape[-1] if len(shape) >= 2 else shape[0]
            processed_tensor = tensor[0] if tensor.ndim == 3 else tensor
            
            if last_dim == 1:
                outputs['objectness'] = processed_tensor
            elif last_dim == 4:
                # Could be bbox OR class (if num_classes == 4)
                # Store for later disambiguation
                shape_4_outputs.append((detail['name'], processed_tensor))
            else:
                # Assume it's class output (num_classes != 4)
                outputs['class_out'] = processed_tensor
    
    # Disambiguate shape-4 outputs: 
    # bbox values are typically in range [0,1] (deltas), class probs sum to ~1
    if 'bbox_out' not in outputs or 'class_out' not in outputs:
        for name, tensor in shape_4_outputs:
            if 'bbox_out' not in outputs and 'class_out' not in outputs:
                # Need to distinguish: check if values look like probabilities or bbox deltas
                # Bbox deltas can be negative, class probs are 0-1 after sigmoid
                min_val = np.min(tensor)
                max_val = np.max(tensor)
                
                # If we have negative values or values outside [0,1], likely bbox
                if min_val < -0.1 or max_val > 1.1:
                    outputs['bbox_out'] = tensor
                else:
                    outputs['class_out'] = tensor
            elif 'bbox_out' not in outputs:
                outputs['bbox_out'] = tensor
            elif 'class_out' not in outputs:
                outputs['class_out'] = tensor
    
    # Validate we have all required outputs
    missing = [k for k in ['objectness', 'class_out', 'bbox_out'] if k not in outputs]
    if missing:
        # Debug: print output details
        print(f"⚠️ Missing outputs: {missing}")
        print(f"   Available outputs:")
        for detail in output_details:
            tensor = interpreter.get_tensor(detail['index'])
            print(f"      - {detail['name']}: shape={tensor.shape}")
        raise KeyError(f"Could not identify TFLite output tensors: {missing}")
    
    return postprocess_detections(
        objectness=outputs['objectness'],
        class_probs=outputs['class_out'],
        bbox_pred=outputs['bbox_out'],
        anchors=anchors,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )


def batch_inference_keras(
    model: tf.keras.Model,
    images: List[np.ndarray],
    anchors: np.ndarray,
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 100,
    batch_size: int = 32,
    verbose: bool = True,
) -> List[List[Detection]]:
    """Run inference on multiple images.
    
    Args:
        model: Keras model
        images: List of images (each H, W, 3 normalized)
        anchors: Anchor boxes
        score_threshold: Minimum confidence
        nms_iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
        batch_size: Batch size for inference
        verbose: Show progress
        
    Returns:
        List of detection lists (one per image)
    """
    all_detections = []
    num_images = len(images)
    
    for i in range(0, num_images, batch_size):
        batch_images = images[i:i + batch_size]
        batch_array = np.array(batch_images)
        
        # Run batch inference
        outputs = model.predict(batch_array, verbose=0)
        
        if isinstance(outputs, dict):
            obj_batch = outputs["objectness"]
            cls_batch = outputs["class_out"]
            box_batch = outputs["bbox_out"]
        else:
            obj_batch = outputs[0]
            cls_batch = outputs[1]
            box_batch = outputs[2]
        
        # Process each image in batch
        for j in range(len(batch_images)):
            detections = postprocess_detections(
                objectness=obj_batch[j],
                class_probs=cls_batch[j],
                bbox_pred=box_batch[j],
                anchors=anchors,
                score_threshold=score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
            )
            all_detections.append(detections)
        
        if verbose and (i + batch_size) % 100 == 0:
            print(f"   Processed {min(i + batch_size, num_images)}/{num_images} images")
    
    return all_detections


def visualize_detections(
    image: np.ndarray,
    detections: List[Detection],
    class_names: List[str],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    show_confidence: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw detections on an image.
    
    Args:
        image: Image (H, W, 3) in [0, 255] uint8 or [0, 1] float
        detections: List of Detection objects
        class_names: List of class names
        colors: List of RGB colors per class (default: auto-generate)
        show_confidence: Whether to show confidence scores
        line_thickness: Box line thickness
        
    Returns:
        Image with detections drawn
    """
    import cv2
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Make copy
    vis_image = image.copy()
    h, w = vis_image.shape[:2]
    
    # Default colors
    if colors is None:
        colors = [
            (46, 204, 113),   # Green
            (52, 152, 219),   # Blue
            (231, 76, 60),    # Red
            (155, 89, 182),   # Purple
            (241, 196, 15),   # Yellow
            (26, 188, 156),   # Teal
            (230, 126, 34),   # Orange
            (149, 165, 166),  # Gray
        ]
    
    for det in detections:
        # Convert normalized to pixel coordinates
        x1 = int(det.x1 * w)
        y1 = int(det.y1 * h)
        x2 = int(det.x2 * w)
        y2 = int(det.y2 * h)
        
        color = colors[det.class_id % len(colors)]
        
        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label
        if det.class_id < len(class_names):
            label = class_names[det.class_id]
        else:
            label = f"class_{det.class_id}"
        
        if show_confidence:
            label = f"{label}: {det.confidence:.2f}"
        
        # Label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(vis_image, label, (x1, y1 - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_image


def visualize_predictions_nms(
    model,
    images_data: List[Dict[str, Any]],
    anchors: np.ndarray,
    class_names: List[str],
    img_size: int = 224,
    num_samples: int = 4,
    score_threshold: float = 0.4,
    nms_iou_threshold: float = 0.5,
    random_seed: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize_per_image: Tuple[int, int] = (4, 4),
    title: Optional[str] = None,
    max_cols: int = 4,
) -> "plt.Figure":
    """Visualize model predictions with NMS on randomly selected images.
    
    Args:
        model: Keras model for inference
        images_data: List of dicts with image info (must have 'path' key)
        anchors: Anchor boxes array (num_anchors, 4)
        class_names: List of class names
        img_size: Input image size for the model
        num_samples: Number of images to visualize
        score_threshold: Minimum confidence score for detections
        nms_iou_threshold: IoU threshold for NMS
        random_seed: Seed for reproducible random selection (None = random each time)
        save_path: Optional path to save the figure
        figsize_per_image: Figure size per image (width, height)
        title: Optional title for the entire figure
        max_cols: Maximum number of columns per row (default: 4)
        
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import cv2
    import math
    import random
    
    # Set random seed if provided (for reproducibility)
    if random_seed is not None:
        random.seed(random_seed)
    
    # Randomly select images
    num_available = len(images_data)
    num_to_show = min(num_samples, num_available)
    selected_indices = random.sample(range(num_available), num_to_show)
    selected_images = [images_data[i] for i in selected_indices]
    
    # Calculate grid dimensions (max 4 columns per row)
    n_cols = min(num_to_show, max_cols)
    n_rows = math.ceil(num_to_show / max_cols)
    
    # Create figure with grid layout
    fig_width = figsize_per_image[0] * n_cols
    fig_height = figsize_per_image[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Flatten axes for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten().tolist()
    
    colors = ['green', 'blue', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    for idx, (ax, img_data) in enumerate(zip(axes, selected_images)):
        # Load and preprocess image
        img = cv2.imread(img_data["path"])
        if img is None:
            ax.set_title(f"Error loading image")
            ax.axis('off')
            continue
            
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = img.astype(np.float32) / 255.0
        
        # Run inference with NMS
        detections = run_inference_keras(
            model=model,
            image=img_input,
            anchors=anchors,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        
        ax.imshow(img)
        
        # Draw detections
        for det in detections:
            x1 = int(det.x1 * img_size)
            y1 = int(det.y1 * img_size)
            x2 = int(det.x2 * img_size)
            y2 = int(det.y2 * img_size)
            
            color = colors[det.class_id % len(colors)]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, color=color, linewidth=2
            )
            ax.add_patch(rect)
            
            label = f"{class_names[det.class_id]}: {det.confidence:.2f}"
            ax.text(
                x1, y1 - 5, label,
                color=color, fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
            )
        
        # Get original image index for title
        original_idx = selected_indices[idx]
        ax.set_title(f"Img {original_idx} ({len(detections)} dets)")
        ax.axis('off')
    
    # Hide unused subplots (when num_samples doesn't fill the grid)
    total_cells = n_rows * n_cols
    for idx in range(num_to_show, total_cells):
        axes[idx].axis('off')
    
    # Add figure title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
