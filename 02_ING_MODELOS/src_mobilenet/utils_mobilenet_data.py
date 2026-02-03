"""Data loading utilities for COCO format with SSD anchor encoding.

This module provides:
- COCO JSON annotation loading
- Class weight computation for imbalanced datasets
- Data generator with anchor-based target encoding
- Data augmentation for minority classes
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

import numpy as np
import cv2
import tensorflow as tf


def load_coco_annotations(
    json_path: str,
    image_dir: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, str], Dict[int, int]]:
    """Load COCO format annotations.
    
    Args:
        json_path: Path to COCO JSON annotation file
        image_dir: Directory containing images
    
    Returns:
        Tuple of:
        - List of image data dicts with annotations
        - Category id to name mapping
        - Category id to index mapping (0-indexed)
    
    Example:
        >>> images, cat_names, cat_to_idx = load_coco_annotations(
        ...     "train.json", "train_images/"
        ... )
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    
    # Build category mappings
    categories = coco_data.get("categories", [])
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    
    # Create 0-indexed mapping for model (sorted by category id)
    sorted_cat_ids = sorted(cat_id_to_name.keys())
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    
    # Build image id to annotations mapping
    annotations = coco_data.get("annotations", [])
    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # Build image data list
    images_data = []
    for img_info in coco_data.get("images", []):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = os.path.join(image_dir, file_name)
        
        if not os.path.exists(img_path):
            continue
        
        width = img_info.get("width", 224)
        height = img_info.get("height", 224)
        
        img_anns = img_id_to_anns.get(img_id, [])
        
        # Convert annotations to normalized format
        boxes = []
        classes = []
        for ann in img_anns:
            bbox = ann["bbox"]  # [x, y, w, h] in pixels
            cat_id = ann["category_id"]
            
            if cat_id not in cat_id_to_idx:
                continue
            
            # Convert to normalized center format [xc, yc, w, h]
            x, y, w, h = bbox
            xc = (x + w / 2) / width
            yc = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            
            # Clip to valid range
            xc = np.clip(xc, 0, 1)
            yc = np.clip(yc, 0, 1)
            w_norm = np.clip(w_norm, 0.001, 1)
            h_norm = np.clip(h_norm, 0.001, 1)
            
            boxes.append([xc, yc, w_norm, h_norm])
            classes.append(cat_id_to_idx[cat_id])
        
        images_data.append({
            "id": img_id,
            "path": img_path,
            "width": width,
            "height": height,
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            "classes": np.array(classes, dtype=np.int32) if classes else np.zeros((0,), dtype=np.int32),
        })
    
    print(f"✅ Loaded {len(images_data)} images with annotations")
    print(f"   Categories: {cat_id_to_name}")
    print(f"   Class mapping: {cat_id_to_idx}")
    
    return images_data, cat_id_to_name, cat_id_to_idx


def compute_class_weights(
    images_data: List[Dict[str, Any]],
    num_classes: int,
    method: str = "inverse_freq",
) -> np.ndarray:
    """Compute class weights for handling class imbalance.
    
    Args:
        images_data: List of image data dicts from load_coco_annotations
        num_classes: Total number of classes
        method: Weighting method:
            - 'inverse_freq': weight = total / (num_classes * count)
            - 'sqrt_inverse': weight = sqrt(max_count / count)
            - 'effective_samples': weight based on effective number of samples
    
    Returns:
        Array of shape (num_classes,) with class weights
    
    Example:
        >>> weights = compute_class_weights(images_data, num_classes=4)
        >>> # For imbalanced data: [0.5, 1.0, 2.0, 2.1]
    """
    # Count instances per class
    class_counts = Counter()
    for img_data in images_data:
        for cls in img_data["classes"]:
            class_counts[cls] += 1
    
    total_samples = sum(class_counts.values())
    
    weights = np.ones(num_classes, dtype=np.float32)
    
    if method == "inverse_freq":
        # Weight = total / (num_classes * count)
        for cls, count in class_counts.items():
            if count > 0:
                weights[cls] = total_samples / (num_classes * count)
    
    elif method == "sqrt_inverse":
        # Weight = sqrt(max_count / count)
        max_count = max(class_counts.values()) if class_counts else 1
        for cls, count in class_counts.items():
            if count > 0:
                weights[cls] = np.sqrt(max_count / count)
    
    elif method == "effective_samples":
        # Based on "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        for cls, count in class_counts.items():
            if count > 0:
                effective_num = (1 - beta ** count) / (1 - beta)
                weights[cls] = 1.0 / effective_num
        # Normalize
        weights = weights / weights.sum() * num_classes
    
    # Print summary
    print(f"\n📊 Class distribution and weights ({method}):")
    for cls in range(num_classes):
        count = class_counts.get(cls, 0)
        print(f"   Class {cls}: {count:5d} samples, weight={weights[cls]:.3f}")
    print(f"   Total: {total_samples} samples")
    
    return weights


def generate_anchors(
    feature_map_size: int = 7,
    scales: List[float] = None,
    aspect_ratios: List[float] = None,
) -> np.ndarray:
    """Generate anchor boxes for SSD.
    
    Args:
        feature_map_size: Size of feature map (e.g., 7 for 224/32)
        scales: Anchor scales (relative to image)
        aspect_ratios: Anchor aspect ratios (w/h)
    
    Returns:
        Anchors array of shape (num_anchors, 4) in [xc, yc, w, h] normalized
    """
    if scales is None:
        scales = [0.1, 0.2, 0.4]
    if aspect_ratios is None:
        aspect_ratios = [0.5, 1.0, 2.0]
    
    anchors = []
    step = 1.0 / feature_map_size
    
    for i in range(feature_map_size):
        for j in range(feature_map_size):
            cx = (j + 0.5) * step
            cy = (i + 0.5) * step
            
            for scale in scales:
                for ar in aspect_ratios:
                    w = scale * np.sqrt(ar)
                    h = scale / np.sqrt(ar)
                    anchors.append([cx, cy, w, h])
    
    return np.array(anchors, dtype=np.float32)


def compute_iou_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    """Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) boxes in [xc, yc, w, h]
        boxes2: (M, 4) boxes in [xc, yc, w, h]
    
    Returns:
        IoU matrix of shape (N, M)
    """
    # Convert to corners
    b1_x1 = boxes1[:, 0:1] - boxes1[:, 2:3] / 2
    b1_y1 = boxes1[:, 1:2] - boxes1[:, 3:4] / 2
    b1_x2 = boxes1[:, 0:1] + boxes1[:, 2:3] / 2
    b1_y2 = boxes1[:, 1:2] + boxes1[:, 3:4] / 2
    
    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Intersection
    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Union
    b1_area = boxes1[:, 2:3] * boxes1[:, 3:4]
    b2_area = boxes2[:, 2] * boxes2[:, 3]
    union = b1_area + b2_area - inter_area
    
    return np.where(union > 0, inter_area / union, 0).astype(np.float32)


def encode_targets(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    anchors: np.ndarray,
    num_classes: int,
    iou_threshold: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode ground truth boxes to anchor targets.
    
    Args:
        gt_boxes: (N, 4) ground truth boxes in [xc, yc, w, h]
        gt_classes: (N,) class indices
        anchors: (A, 4) anchor boxes
        num_classes: Number of classes
        iou_threshold: IoU threshold for positive anchors
    
    Returns:
        objectness_targets: (A, 1) binary targets
        class_targets: (A, num_classes) class targets (one-hot)
        bbox_targets: (A, 4) bbox targets
    """
    num_anchors = anchors.shape[0]
    
    objectness_targets = np.zeros((num_anchors, 1), dtype=np.float32)
    class_targets = np.zeros((num_anchors, num_classes), dtype=np.float32)
    bbox_targets = np.zeros((num_anchors, 4), dtype=np.float32)
    
    if gt_boxes.size == 0:
        return objectness_targets, class_targets, bbox_targets
    
    # Compute IoU matrix: (N_gt, A)
    iou_matrix = compute_iou_matrix(gt_boxes, anchors)
    
    # Strategy 1: Best anchor per GT
    best_anchor_per_gt = np.argmax(iou_matrix, axis=1)
    for gt_idx, anchor_idx in enumerate(best_anchor_per_gt):
        cls = int(gt_classes[gt_idx])
        objectness_targets[anchor_idx, 0] = 1.0
        class_targets[anchor_idx] = 0.0
        class_targets[anchor_idx, cls] = 1.0
        bbox_targets[anchor_idx] = gt_boxes[gt_idx]
    
    # Strategy 2: All anchors above threshold
    best_gt_per_anchor = np.argmax(iou_matrix, axis=0)
    max_iou_per_anchor = np.max(iou_matrix, axis=0)
    
    positive_mask = max_iou_per_anchor >= iou_threshold
    for anchor_idx in np.where(positive_mask)[0]:
        gt_idx = best_gt_per_anchor[anchor_idx]
        cls = int(gt_classes[gt_idx])
        objectness_targets[anchor_idx, 0] = 1.0
        class_targets[anchor_idx] = 0.0
        class_targets[anchor_idx, cls] = 1.0
        bbox_targets[anchor_idx] = gt_boxes[gt_idx]
    
    return objectness_targets, class_targets, bbox_targets


class COCODataGenerator(tf.keras.utils.Sequence):
    """Data generator for COCO format with anchor encoding.
    
    Features:
    - Loads images and annotations in batches
    - Encodes targets for SSD anchor-based detection
    - Supports data augmentation
    - Handles class imbalance through weighted sampling
    
    Args:
        images_data: List of image data dicts from load_coco_annotations
        anchors: Anchor boxes array
        batch_size: Batch size
        img_size: Target image size
        num_classes: Number of classes
        iou_threshold: IoU threshold for positive anchors
        augment: Whether to apply data augmentation
        shuffle: Whether to shuffle data each epoch
    
    Example:
        >>> images, _, _ = load_coco_annotations("train.json", "images/")
        >>> anchors = generate_anchors(7)
        >>> gen = COCODataGenerator(images, anchors, batch_size=32)
        >>> for images, targets in gen:
        ...     model.train_on_batch(images, targets)
    """
    
    def __init__(
        self,
        images_data: List[Dict[str, Any]],
        anchors: np.ndarray,
        batch_size: int = 32,
        img_size: int = 224,
        num_classes: int = 4,
        iou_threshold: float = 0.35,
        augment: bool = False,
        shuffle: bool = True,
    ):
        super().__init__()
        self.images_data = images_data
        self.anchors = anchors
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.augment = augment
        self.shuffle = shuffle
        
        self.indices = np.arange(len(images_data))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        return int(np.ceil(len(self.images_data) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def _augment_image(
        self, 
        img: np.ndarray, 
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation.
        
        Augmentations:
        - Horizontal flip (50% chance)
        - Brightness adjustment
        - Contrast adjustment
        """
        aug_img = img.copy()
        aug_boxes = boxes.copy()
        
        # Horizontal flip
        if np.random.random() > 0.5:
            aug_img = np.fliplr(aug_img).copy()
            if len(aug_boxes) > 0:
                aug_boxes[:, 0] = 1.0 - aug_boxes[:, 0]  # Flip x center
        
        # Brightness adjustment
        if np.random.random() > 0.5:
            delta = np.random.uniform(-0.2, 0.2)
            aug_img = np.clip(aug_img + delta, 0, 1)
        
        # Contrast adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            aug_img = np.clip((aug_img - 0.5) * factor + 0.5, 0, 1)
        
        return aug_img, aug_boxes
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_images = []
        batch_objectness = []
        batch_classes = []
        batch_bboxes = []
        
        for i in batch_indices:
            img_data = self.images_data[i]
            
            # Load image
            try:
                img = self._load_and_preprocess_image(img_data["path"])
            except Exception as e:
                print(f"⚠️ Error loading {img_data['path']}: {e}")
                continue
            
            boxes = img_data["boxes"].copy()
            classes = img_data["classes"].copy()
            
            # Apply augmentation
            if self.augment:
                img, boxes = self._augment_image(img, boxes)
            
            # Encode targets
            obj_targets, cls_targets, bbox_targets = encode_targets(
                boxes, classes, self.anchors, self.num_classes, self.iou_threshold
            )
            
            batch_images.append(img)
            batch_objectness.append(obj_targets)
            batch_classes.append(cls_targets)
            batch_bboxes.append(bbox_targets)
        
        return (
            np.array(batch_images, dtype=np.float32),
            {
                "objectness": np.array(batch_objectness, dtype=np.float32),
                "class_out": np.array(batch_classes, dtype=np.float32),
                "bbox_out": np.array(batch_bboxes, dtype=np.float32),
            }
        )


def create_tf_dataset(
    images_data: List[Dict[str, Any]],
    anchors: np.ndarray,
    batch_size: int = 32,
    img_size: int = 224,
    num_classes: int = 4,
    iou_threshold: float = 0.35,
    augment: bool = False,
    shuffle: bool = True,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """Create tf.data.Dataset for training.
    
    Alternative to COCODataGenerator using tf.data pipeline
    for better performance with GPU training.
    
    Args:
        images_data: List of image data dicts
        anchors: Anchor boxes
        batch_size: Batch size
        img_size: Target image size
        num_classes: Number of classes
        iou_threshold: IoU threshold for positive anchors
        augment: Whether to augment data
        shuffle: Whether to shuffle
        prefetch: Whether to prefetch batches
    
    Returns:
        tf.data.Dataset
    """
    generator = COCODataGenerator(
        images_data=images_data,
        anchors=anchors,
        batch_size=batch_size,
        img_size=img_size,
        num_classes=num_classes,
        iou_threshold=iou_threshold,
        augment=augment,
        shuffle=shuffle,
    )
    
    output_signature = (
        tf.TensorSpec(shape=(None, img_size, img_size, 3), dtype=tf.float32),
        {
            "objectness": tf.TensorSpec(shape=(None, anchors.shape[0], 1), dtype=tf.float32),
            "class_out": tf.TensorSpec(shape=(None, anchors.shape[0], num_classes), dtype=tf.float32),
            "bbox_out": tf.TensorSpec(shape=(None, anchors.shape[0], 4), dtype=tf.float32),
        }
    )
    
    dataset = tf.data.Dataset.from_generator(
        lambda: iter(generator),
        output_signature=output_signature,
    )
    
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def compute_anchor_statistics(
    images_data: List[Dict[str, Any]],
    anchors: np.ndarray,
    iou_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Compute statistics about anchor matching.
    
    Useful for debugging anchor configuration.
    
    Args:
        images_data: List of image data dicts
        anchors: Anchor boxes
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with matching statistics
    """
    total_gt = 0
    matched_gt = 0
    total_positive_anchors = 0
    max_ious = []
    
    for img_data in images_data:
        boxes = img_data["boxes"]
        if boxes.size == 0:
            continue
        
        total_gt += len(boxes)
        
        iou_matrix = compute_iou_matrix(boxes, anchors)
        max_iou_per_gt = np.max(iou_matrix, axis=1)
        max_ious.extend(max_iou_per_gt.tolist())
        
        matched_gt += np.sum(max_iou_per_gt >= iou_threshold)
        
        max_iou_per_anchor = np.max(iou_matrix, axis=0)
        total_positive_anchors += np.sum(max_iou_per_anchor >= iou_threshold)
    
    avg_positives_per_image = total_positive_anchors / len(images_data) if images_data else 0
    
    stats = {
        "total_gt_boxes": total_gt,
        "matched_gt_boxes": matched_gt,
        "match_rate": matched_gt / total_gt if total_gt > 0 else 0,
        "avg_max_iou": np.mean(max_ious) if max_ious else 0,
        "min_max_iou": np.min(max_ious) if max_ious else 0,
        "avg_positive_anchors_per_image": avg_positives_per_image,
        "total_anchors": len(anchors),
    }
    
    print(f"\n📊 Anchor Matching Statistics:")
    print(f"   Total GT boxes: {stats['total_gt_boxes']}")
    print(f"   Matched GT boxes (IoU≥{iou_threshold}): {stats['matched_gt_boxes']} ({stats['match_rate']*100:.1f}%)")
    print(f"   Avg max IoU per GT: {stats['avg_max_iou']:.3f}")
    print(f"   Min max IoU: {stats['min_max_iou']:.3f}")
    print(f"   Avg positive anchors/image: {stats['avg_positive_anchors_per_image']:.1f}")
    
    return stats
