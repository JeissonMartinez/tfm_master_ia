"""Loss functions for SSD-Lite object detection.

This module provides:
- Focal Loss for classification (handles class imbalance)
- Smooth L1 Loss for bounding box regression
- Combined SSD loss with configurable weights

Reference:
    - Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    - Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016
"""
from __future__ import annotations

from typing import Optional, Callable
import tensorflow as tf


def focal_loss(
    alpha: float = 0.25,
    gamma: float = 2.0,
    class_weights: Optional[tf.Tensor] = None,
) -> Callable:
    """Focal Loss for multi-class classification with class imbalance.
    
    Focal Loss reduces the weight of easy examples and focuses
    training on hard examples, which is crucial for object detection
    where background dominates.
    
    Loss = -alpha * (1 - pt)^gamma * log(pt)
    
    Args:
        alpha: Balance factor for positive/negative classes (0-1)
        gamma: Focusing parameter. Higher values reduce weight of easy examples.
               gamma=0 is equivalent to standard cross-entropy
        class_weights: Optional tensor of shape (num_classes,) for per-class weighting
    
    Returns:
        Loss function compatible with Keras
    
    Example:
        >>> loss_fn = focal_loss(alpha=0.25, gamma=2.0)
        >>> model.compile(loss={'class_out': loss_fn})
    """
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: (batch, num_anchors, num_classes) one-hot or multi-label
            y_pred: (batch, num_anchors, num_classes) predictions (sigmoid output)
        """
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Cross entropy
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Probability of correct class
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = tf.pow(1.0 - pt, gamma)
        
        # Alpha weighting
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Final focal loss
        focal_ce = alpha_weight * focal_weight * ce
        
        # Apply class weights if provided
        if class_weights is not None:
            # Expand weights to match shape
            weights = tf.reshape(class_weights, (1, 1, -1))
            focal_ce = focal_ce * weights
        
        # Average over classes, then over anchors that have objects
        # Use mean of non-zero targets as normalization
        has_object = tf.reduce_max(y_true, axis=-1, keepdims=True)
        focal_ce = focal_ce * has_object  # Only count anchors with objects
        
        # Normalize by number of positive anchors
        num_positives = tf.reduce_sum(has_object) + 1e-4
        
        return tf.reduce_sum(focal_ce) / num_positives
    
    return loss_fn


def binary_focal_loss(
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Callable:
    """Binary Focal Loss for objectness prediction.
    
    Used for the objectness output (object vs background).
    
    Args:
        alpha: Balance factor for positive class
        gamma: Focusing parameter
    
    Returns:
        Loss function compatible with Keras
    """
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: (batch, num_anchors, 1) binary targets
            y_pred: (batch, num_anchors, 1) predictions (sigmoid output)
        """
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal weight
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1.0 - pt, gamma)
        
        # Alpha weighting
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        focal_bce = alpha_weight * focal_weight * bce
        
        return tf.reduce_mean(focal_bce)
    
    return loss_fn


def smooth_l1_loss(delta: float = 1.0) -> Callable:
    """Smooth L1 Loss for bounding box regression.
    
    Smooth L1 is less sensitive to outliers than L2 loss
    and more stable than L1 for small errors.
    
    Loss = 0.5 * x^2           if |x| < delta
           delta * |x| - 0.5 * delta^2  otherwise
    
    Args:
        delta: Threshold for switching from L2 to L1
    
    Returns:
        Loss function compatible with Keras
    """
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: (batch, num_anchors, 4) target boxes [xc, yc, w, h]
            y_pred: (batch, num_anchors, 4) predicted boxes
        """
        diff = tf.abs(y_true - y_pred)
        
        # Smooth L1 formula
        loss = tf.where(
            diff < delta,
            0.5 * tf.square(diff),
            delta * diff - 0.5 * tf.square(delta)
        )
        
        # Only compute loss for positive anchors (where target is not zero)
        valid_mask = tf.reduce_max(tf.abs(y_true), axis=-1, keepdims=True)
        valid_mask = tf.cast(valid_mask > 0.001, tf.float32)
        
        loss = loss * valid_mask
        
        # Normalize by number of positive anchors
        num_positives = tf.reduce_sum(valid_mask) + 1e-4
        
        return tf.reduce_sum(loss) / num_positives
    
    return loss_fn


def giou_loss() -> Callable:
    """Generalized IoU Loss for bounding box regression.
    
    GIoU is scale-invariant and handles non-overlapping boxes better
    than standard IoU loss.
    
    Returns:
        Loss function compatible with Keras
    
    Reference:
        Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019
    """
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: (batch, num_anchors, 4) target boxes [xc, yc, w, h]
            y_pred: (batch, num_anchors, 4) predicted boxes [xc, yc, w, h]
        """
        # Convert center format to corner format
        t_x1 = y_true[..., 0] - y_true[..., 2] / 2
        t_y1 = y_true[..., 1] - y_true[..., 3] / 2
        t_x2 = y_true[..., 0] + y_true[..., 2] / 2
        t_y2 = y_true[..., 1] + y_true[..., 3] / 2
        
        p_x1 = y_pred[..., 0] - y_pred[..., 2] / 2
        p_y1 = y_pred[..., 1] - y_pred[..., 3] / 2
        p_x2 = y_pred[..., 0] + y_pred[..., 2] / 2
        p_y2 = y_pred[..., 1] + y_pred[..., 3] / 2
        
        # Intersection
        inter_x1 = tf.maximum(t_x1, p_x1)
        inter_y1 = tf.maximum(t_y1, p_y1)
        inter_x2 = tf.minimum(t_x2, p_x2)
        inter_y2 = tf.minimum(t_y2, p_y2)
        
        inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)
        
        # Areas
        t_area = y_true[..., 2] * y_true[..., 3]
        p_area = y_pred[..., 2] * y_pred[..., 3]
        
        # Union
        union = t_area + p_area - inter_area + 1e-7
        
        # IoU
        iou = inter_area / union
        
        # Enclosing box
        enclose_x1 = tf.minimum(t_x1, p_x1)
        enclose_y1 = tf.minimum(t_y1, p_y1)
        enclose_x2 = tf.maximum(t_x2, p_x2)
        enclose_y2 = tf.maximum(t_y2, p_y2)
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-7
        
        # GIoU
        giou = iou - (enclose_area - union) / enclose_area
        
        # GIoU loss
        loss = 1.0 - giou
        
        # Only compute for positive anchors
        valid_mask = tf.reduce_max(tf.abs(y_true), axis=-1)
        valid_mask = tf.cast(valid_mask > 0.001, tf.float32)
        
        loss = loss * valid_mask
        num_positives = tf.reduce_sum(valid_mask) + 1e-4
        
        return tf.reduce_sum(loss) / num_positives
    
    return loss_fn


def ssd_combined_loss(
    cls_weight: float = 1.0,
    obj_weight: float = 1.0,
    bbox_weight: float = 2.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    smooth_l1_delta: float = 1.0,
    class_weights: Optional[tf.Tensor] = None,
    use_giou: bool = False,
) -> dict:
    """Create combined loss dictionary for SSD model.
    
    Returns separate loss functions for each output head:
    - objectness: Binary focal loss
    - class_out: Focal loss
    - bbox_out: Smooth L1 or GIoU loss
    
    Args:
        cls_weight: Weight for classification loss
        obj_weight: Weight for objectness loss
        bbox_weight: Weight for bbox regression loss
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        smooth_l1_delta: Delta for smooth L1 loss
        class_weights: Optional per-class weights
        use_giou: Use GIoU loss instead of Smooth L1
    
    Returns:
        Dictionary with loss functions and weights for model.compile()
    
    Example:
        >>> losses = ssd_combined_loss(cls_weight=1.0, bbox_weight=2.0)
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss=losses['losses'],
        ...     loss_weights=losses['weights']
        ... )
    """
    # Build losses
    obj_loss = binary_focal_loss(alpha=focal_alpha, gamma=focal_gamma)
    cls_loss = focal_loss(alpha=focal_alpha, gamma=focal_gamma, class_weights=class_weights)
    
    if use_giou:
        box_loss = giou_loss()
    else:
        box_loss = smooth_l1_loss(delta=smooth_l1_delta)
    
    return {
        "losses": {
            "objectness": obj_loss,
            "class_out": cls_loss,
            "bbox_out": box_loss,
        },
        "weights": {
            "objectness": obj_weight,
            "class_out": cls_weight,
            "bbox_out": bbox_weight,
        }
    }


class SSDLoss(tf.keras.losses.Loss):
    """Combined SSD Loss as a single Keras Loss class.
    
    Alternative to ssd_combined_loss() for use with custom training loops.
    
    Args:
        num_classes: Number of object classes
        cls_weight: Weight for classification loss
        obj_weight: Weight for objectness loss
        bbox_weight: Weight for bbox loss
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        cls_weight: float = 1.0,
        obj_weight: float = 1.0,
        bbox_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        name: str = "ssd_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.bbox_weight = bbox_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.obj_loss_fn = binary_focal_loss(focal_alpha, focal_gamma)
        self.cls_loss_fn = focal_loss(focal_alpha, focal_gamma)
        self.bbox_loss_fn = smooth_l1_loss()
    
    def call(self, y_true: dict, y_pred: dict) -> tf.Tensor:
        """Compute combined loss.
        
        Args:
            y_true: Dict with 'objectness', 'class_out', 'bbox_out'
            y_pred: Dict with 'objectness', 'class_out', 'bbox_out'
        
        Returns:
            Combined weighted loss
        """
        obj_loss = self.obj_loss_fn(y_true["objectness"], y_pred["objectness"])
        cls_loss = self.cls_loss_fn(y_true["class_out"], y_pred["class_out"])
        bbox_loss = self.bbox_loss_fn(y_true["bbox_out"], y_pred["bbox_out"])
        
        total_loss = (
            self.obj_weight * obj_loss +
            self.cls_weight * cls_loss +
            self.bbox_weight * bbox_loss
        )
        
        return total_loss
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "cls_weight": self.cls_weight,
            "obj_weight": self.obj_weight,
            "bbox_weight": self.bbox_weight,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
        })
        return config
