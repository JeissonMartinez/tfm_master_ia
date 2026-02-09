"""YOLO26 inference utilities for evaluation and visualization.

Handles running inference with YOLO26 models on test images
and extracting detection results.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from .utils_io import log, safe_exists

try:
    from ultralytics import YOLO  # type: ignore
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    ULTRALYTICS_AVAILABLE = False


@dataclass
class BoundingBox:
    """Bounding box with class and confidence information."""
    x: float  # x_min
    y: float  # y_min
    w: float  # width
    h: float  # height
    class_id: int
    class_name: str
    confidence: float = 1.0

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Return (x, y, w, h) format."""
        return (self.x, self.y, self.w, self.h)


@dataclass
class DetectionResult:
    """Result of running detection on a single image."""
    image_id: int
    image_path: str
    model_name: str
    predictions: List[BoundingBox] = field(default_factory=list)
    ground_truth: List[BoundingBox] = field(default_factory=list)
    inference_time_ms: float = 0.0


def load_yolo_model(weights_path: str, verbose: bool = True) -> Any:
    """Load YOLO model from weights file.

    Args:
        weights_path: Path to model weights (.pt file)
        verbose: Whether to print loading info

    Returns:
        YOLO model instance
    """
    if not ULTRALYTICS_AVAILABLE:
        raise RuntimeError("Ultralytics no disponible. Instala con: pip install ultralytics")
    
    if not safe_exists(weights_path):
        raise FileNotFoundError(f"Modelo no encontrado: {weights_path}")

    if verbose:
        log(f"🔄 Cargando modelo: {weights_path}")
    
    model = YOLO(weights_path)
    
    if verbose:
        log(f"✅ Modelo cargado correctamente")
    
    return model


def run_yolo_inference(
    model: Any,
    images: List[np.ndarray],
    class_names: List[str],
    image_ids: Optional[List[int]] = None,
    image_paths: Optional[List[str]] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    imgsz: int = 224,
    end2end: bool = True,
) -> List[DetectionResult]:
    """Run inference on a batch of images.

    Args:
        model: YOLO model instance
        images: List of images as numpy arrays (H, W, C) in BGR format
        class_names: List of class names
        image_ids: Optional list of image IDs
        image_paths: Optional list of image paths
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS (ignored if end2end=True)
        imgsz: Image size for inference
        end2end: Use end-to-end inference (NMS-free for YOLO26)

    Returns:
        List of DetectionResult objects
    """
    if model is None:
        raise RuntimeError("Modelo no cargado")

    if image_ids is None:
        image_ids = list(range(len(images)))
    if image_paths is None:
        image_paths = [f"image_{i}" for i in image_ids]

    results_list = []
    
    start_time = time.time()
    
    # Run batch inference
    results = model.predict(
        source=images,
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
        end2end=end2end,
    )
    
    total_time = (time.time() - start_time) * 1000
    avg_time = total_time / len(images)

    # Process results
    for idx, res in enumerate(results):
        predictions: List[BoundingBox] = []
        
        if res.boxes is not None and len(res.boxes) > 0:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            
            for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
                if 0 <= cls_id < len(class_names):
                    class_name = class_names[cls_id]
                else:
                    class_name = "unknown"
                
                predictions.append(BoundingBox(
                    x=float(x1),
                    y=float(y1),
                    w=float(x2 - x1),
                    h=float(y2 - y1),
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                ))

        results_list.append(DetectionResult(
            image_id=image_ids[idx],
            image_path=image_paths[idx],
            model_name="YOLO26",
            predictions=predictions,
            inference_time_ms=avg_time,
        ))

    return results_list


def run_inference_on_dataset(
    model: Any,
    data_yaml: str,
    split: str = "test",
    conf_threshold: float = 0.25,
    imgsz: int = 224,
    max_images: Optional[int] = None,
) -> Tuple[List[DetectionResult], List[str]]:
    """Run inference on a dataset split.

    Args:
        model: YOLO model instance
        data_yaml: Path to data.yaml
        split: Dataset split ('train', 'val', 'test')
        conf_threshold: Confidence threshold
        imgsz: Image size
        max_images: Maximum number of images to process

    Returns:
        Tuple of (results_list, class_names)
    """
    import yaml
    import cv2

    with open(data_yaml, "r") as f:
        config = yaml.safe_load(f)

    class_names = config.get("names", [])
    if isinstance(class_names, dict):
        class_names = list(class_names.values())

    dataset_root = config.get("path", os.path.dirname(data_yaml))
    images_dir = os.path.join(dataset_root, "images", split)
    labels_dir = os.path.join(dataset_root, "labels", split)

    if not safe_exists(images_dir):
        log(f"⚠️ Directorio no encontrado: {images_dir}")
        return [], class_names

    # Get image files
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if max_images is not None:
        image_files = image_files[:max_images]

    log(f"🔍 Procesando {len(image_files)} imágenes del split '{split}'...")

    results_list = []
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Run inference
        start_time = time.time()
        results = model.predict(
            source=img,
            imgsz=imgsz,
            conf=conf_threshold,
            verbose=False,
        )
        inference_time = (time.time() - start_time) * 1000

        # Extract predictions
        predictions: List[BoundingBox] = []
        res = results[0]
        
        if res.boxes is not None and len(res.boxes) > 0:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            
            for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
                if 0 <= cls_id < len(class_names):
                    predictions.append(BoundingBox(
                        x=float(x1),
                        y=float(y1),
                        w=float(x2 - x1),
                        h=float(y2 - y1),
                        class_id=int(cls_id),
                        class_name=class_names[cls_id],
                        confidence=float(conf),
                    ))

        # Load ground truth
        ground_truth: List[BoundingBox] = []
        if safe_exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_c, y_c, w_n, h_n = map(float, parts[1:5])
                        
                        # Convert from normalized to absolute
                        box_w = w_n * w
                        box_h = h_n * h
                        x1 = (x_c - w_n / 2) * w
                        y1 = (y_c - h_n / 2) * h
                        
                        if 0 <= cls_id < len(class_names):
                            ground_truth.append(BoundingBox(
                                x=float(x1),
                                y=float(y1),
                                w=float(box_w),
                                h=float(box_h),
                                class_id=cls_id,
                                class_name=class_names[cls_id],
                                confidence=1.0,
                            ))

        results_list.append(DetectionResult(
            image_id=idx,
            image_path=img_path,
            model_name="YOLO26",
            predictions=predictions,
            ground_truth=ground_truth,
            inference_time_ms=inference_time,
        ))

    avg_time = np.mean([r.inference_time_ms for r in results_list]) if results_list else 0
    log(f"✅ Inferencia completada: {len(results_list)} imágenes, {avg_time:.1f}ms/img promedio")

    return results_list, class_names


def visualize_yolo_predictions_grid(
    model: Any,
    images_dir: str,
    class_names: List[str],
    imgsz: int = 224,
    num_samples: int = 12,
    conf_threshold: float = 0.25,
    random_seed: Optional[int] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    max_cols: int = 4,
    figsize_per_image: Tuple[float, float] = (4.0, 4.0),
) -> Any:
    """Visualize YOLO predictions on a grid of randomly selected images.

    Similar to MobileNet's visualize_predictions_nms but adapted for YOLO26.
    YOLO26 is NMS-free (end-to-end), so no NMS post-processing is needed.

    Args:
        model: YOLO model instance
        images_dir: Directory containing test images
        class_names: List of class names
        imgsz: Input image size for the model
        num_samples: Number of images to visualize
        conf_threshold: Minimum confidence score for detections
        random_seed: Seed for reproducible random selection (None = random each time)
        save_path: Optional path to save the figure
        title: Optional title for the entire figure
        max_cols: Maximum number of columns per row
        figsize_per_image: Figure size per image (width, height)

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    import math
    import random
    import os

    # Get list of image files
    if not os.path.exists(images_dir):
        log(f"⚠️ Directorio no encontrado: {images_dir}")
        return None

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not image_files:
        log(f"⚠️ No se encontraron imágenes en: {images_dir}")
        return None

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Randomly select images
    num_available = len(image_files)
    num_to_show = min(num_samples, num_available)
    selected_files = random.sample(image_files, num_to_show)

    # Calculate grid dimensions
    n_cols = min(num_to_show, max_cols)
    n_rows = math.ceil(num_to_show / max_cols)

    # Create figure
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

    # Color palette for classes
    colors = ['green', 'blue', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    for idx, (ax, img_path) in enumerate(zip(axes, selected_files)):
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            ax.set_title("Error loading")
            ax.axis('off')
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Run YOLO inference
        try:
            results = model.predict(
                source=img,
                imgsz=imgsz,
                conf=conf_threshold,
                verbose=False,
            )
            res = results[0]
        except Exception as e:
            ax.set_title(f"Error: {e}")
            ax.axis('off')
            continue

        # Display image
        ax.imshow(img_rgb)

        # Draw detections
        num_dets = 0
        if res.boxes is not None and len(res.boxes) > 0:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
                num_dets += 1
                color = colors[cls_id % len(colors)]

                # Draw rectangle
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, color=color, linewidth=2
                )
                ax.add_patch(rect)

                # Draw label
                if 0 <= cls_id < len(class_names):
                    label = f"{class_names[cls_id]}: {conf:.2f}"
                else:
                    label = f"cls{cls_id}: {conf:.2f}"

                ax.text(
                    x1, y1 - 5, label,
                    color=color, fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
                )

        # Set title with image name and detection count
        img_name = os.path.basename(img_path)
        ax.set_title(f"{img_name[:15]}... ({num_dets} dets)", fontsize=9)
        ax.axis('off')

    # Hide unused subplots
    total_cells = n_rows * n_cols
    for idx in range(num_to_show, total_cells):
        axes[idx].axis('off')

    # Add figure title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        from .utils_io import safe_mkdir
        save_dir = os.path.dirname(save_path)
        if save_dir:
            safe_mkdir(save_dir)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        log(f"✅ Visualización guardada en: {save_path}")

    return fig


def visualize_predictions(
    image: np.ndarray,
    predictions: List[BoundingBox],
    ground_truth: Optional[List[BoundingBox]] = None,
    class_names: Optional[List[str]] = None,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """Draw predictions and ground truth on an image.

    Args:
        image: Image as numpy array (H, W, C) in BGR format
        predictions: List of predicted bounding boxes
        ground_truth: Optional list of ground truth boxes
        class_names: List of class names for coloring
        conf_threshold: Minimum confidence to display

    Returns:
        Image with drawn boxes
    """
    import cv2

    img = image.copy()
    
    # Color palette for classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    # Draw ground truth (dashed, thinner)
    if ground_truth:
        for box in ground_truth:
            color = colors[box.class_id % len(colors)]
            x1, y1, x2, y2 = map(int, box.to_xyxy())
            # Dashed line effect
            for i in range(x1, x2, 8):
                cv2.line(img, (i, y1), (min(i + 4, x2), y1), color, 1)
                cv2.line(img, (i, y2), (min(i + 4, x2), y2), color, 1)
            for i in range(y1, y2, 8):
                cv2.line(img, (x1, i), (x1, min(i + 4, y2)), color, 1)
                cv2.line(img, (x2, i), (x2, min(i + 4, y2)), color, 1)

    # Draw predictions (solid)
    for box in predictions:
        if box.confidence < conf_threshold:
            continue
            
        color = colors[box.class_id % len(colors)]
        x1, y1, x2, y2 = map(int, box.to_xyxy())
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{box.class_name}: {box.confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img
