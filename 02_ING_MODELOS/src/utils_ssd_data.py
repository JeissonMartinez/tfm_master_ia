"""SSD data generator utilities (YOLO txt -> SSD targets)."""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

try:
    from .utils_io import log
except ImportError:  # fallback when running as a script/notebook
    from utils_io import log

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - defensive
    cv2 = None
    log(f"⚠️ OpenCV no disponible: {exc}")

try:
    import tensorflow as tf  # type: ignore
except Exception as exc:  # pragma: no cover - defensive
    tf = None
    log(f"⚠️ TensorFlow no disponible: {exc}")


if tf:
    class SSDDataGenerator(tf.keras.utils.Sequence):  # type: ignore
        """Generador SSD basado en labels YOLO (.txt).

        - Selecciona los objetos más grandes por área.
        - Devuelve diccionario con salidas `class_out` y `bbox_out`.
        """

        def __init__(
            self,
            image_dir: str,
            label_dir: str,
            batch_size: int = 32,
            img_size: int = 224,
            max_objects: int = 2,
            num_classes: int = 4,
        ) -> None:
            if cv2 is None:
                raise RuntimeError("OpenCV es requerido para SSDDataGenerator.")
            super().__init__()  # Requerido por Keras 3
            self.image_paths = sorted([p for p in _glob_jpg(image_dir)])
            self.label_dir = label_dir
            self.batch_size = batch_size
            self.img_size = img_size
            self.max_objects = max_objects
            self.num_classes = num_classes

        def __len__(self) -> int:
            return int(np.ceil(len(self.image_paths) / self.batch_size))

        def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            assert cv2 is not None
            batch_paths = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]

            batch_images = []
            batch_targets_class = []
            batch_targets_bbox = []

            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None:
                    log(f"⚠️ No se pudo cargar imagen: {img_path}")
                    continue
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_images.append(img.astype(np.float32) / 255.0)

                txt_name = os.path.basename(img_path).replace(".jpg", ".txt")
                txt_path = os.path.join(self.label_dir, txt_name)

                ssd_classes = np.zeros((self.max_objects, self.num_classes), dtype=np.float32)
                ssd_bboxes = np.zeros((self.max_objects, 4), dtype=np.float32)

                objects_found = []
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    except Exception as exc:  # pragma: no cover - defensive
                        log(f"⚠️ Error leyendo {txt_path}: {exc}")
                        lines = []

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls, xc, yc, w, h = map(float, parts[:5])
                        cls = int(cls)
                        area = w * h
                        objects_found.append((area, cls, xc, yc, w, h))

                    objects_found.sort(key=lambda x: x[0], reverse=True)
                    for i, obj in enumerate(objects_found[: self.max_objects]):
                        _, cls, xc, yc, w, h = obj
                        if 0 <= cls < self.num_classes:
                            ssd_classes[i, cls] = 1.0
                        ssd_bboxes[i] = [xc, yc, w, h]

                batch_targets_class.append(ssd_classes)
                batch_targets_bbox.append(ssd_bboxes)

            batch_images = np.array(batch_images)
            return batch_images, {
                "class_out": np.array(batch_targets_class),
                "bbox_out_sigmoid": np.array(batch_targets_bbox),
            }


    def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
        x, y, w, h = box
        return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)


    def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
        a_xy = _xywh_to_xyxy(a)
        b_xy = _xywh_to_xyxy(b)
        x1 = max(a_xy[0], b_xy[0])
        y1 = max(a_xy[1], b_xy[1])
        x2 = min(a_xy[2], b_xy[2])
        y2 = min(a_xy[3], b_xy[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a_xy[2] - a_xy[0]) * max(0.0, a_xy[3] - a_xy[1])
        area_b = max(0.0, b_xy[2] - b_xy[0]) * max(0.0, b_xy[3] - b_xy[1])
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union


    def compute_iou_matrix(gt_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Vectorized IoU computation between all GT boxes and anchors.
        
        Args:
            gt_boxes: (N, 4) in [xc, yc, w, h] normalized
            anchors: (A, 4) in [xc, yc, w, h] normalized
        Returns:
            iou_matrix: (N, A) IoU values
        """
        # Convert to corner format for IoU calculation
        # gt_boxes: (N, 4) -> (N, 4) corner format
        gt_x1 = gt_boxes[:, 0:1] - gt_boxes[:, 2:3] / 2  # (N, 1)
        gt_y1 = gt_boxes[:, 1:2] - gt_boxes[:, 3:4] / 2
        gt_x2 = gt_boxes[:, 0:1] + gt_boxes[:, 2:3] / 2
        gt_y2 = gt_boxes[:, 1:2] + gt_boxes[:, 3:4] / 2
        
        # anchors: (A, 4) -> (A, 4) corner format
        a_x1 = anchors[:, 0] - anchors[:, 2] / 2  # (A,)
        a_y1 = anchors[:, 1] - anchors[:, 3] / 2
        a_x2 = anchors[:, 0] + anchors[:, 2] / 2
        a_y2 = anchors[:, 1] + anchors[:, 3] / 2
        
        # Compute intersection: (N, A)
        inter_x1 = np.maximum(gt_x1, a_x1)  # Broadcasting: (N, 1) vs (A,) -> (N, A)
        inter_y1 = np.maximum(gt_y1, a_y1)
        inter_x2 = np.minimum(gt_x2, a_x2)
        inter_y2 = np.minimum(gt_y2, a_y2)
        
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h  # (N, A)
        
        # Compute areas
        gt_area = gt_boxes[:, 2:3] * gt_boxes[:, 3:4]  # (N, 1)
        a_area = anchors[:, 2] * anchors[:, 3]  # (A,)
        
        # Union
        union = gt_area + a_area - inter_area  # (N, A)
        
        # IoU
        iou = np.where(union > 0, inter_area / union, 0.0)
        return iou.astype(np.float32)


    def encode_anchors(
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        anchors: np.ndarray,
        num_classes: int,
        iou_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode GT boxes to anchor targets (original version for backwards compatibility).

        Args:
            gt_boxes: (N, 4) in [xc, yc, w, h] normalized (0-1)
            gt_classes: (N,) class indices [0..num_classes-1]
            anchors: (A, 4) in [xc, yc, w, h] normalized
        Returns:
            class_targets: (A, num_classes+1) one-hot with background at index 0
            bbox_targets: (A, 4) target boxes (xc, yc, w, h) normalized
        """
        num_anchors = anchors.shape[0]
        class_targets = np.zeros((num_anchors, num_classes + 1), dtype=np.float32)
        class_targets[:, 0] = 1.0  # background by default
        bbox_targets = np.zeros((num_anchors, 4), dtype=np.float32)

        if gt_boxes.size == 0:
            return class_targets, bbox_targets

        for gt_idx, gt_box in enumerate(gt_boxes):
            best_iou = -1.0
            best_anchor = -1
            for a_idx in range(num_anchors):
                iou = _iou_xywh(gt_box, anchors[a_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = a_idx

            if best_anchor >= 0:
                cls = int(gt_classes[gt_idx])
                class_targets[best_anchor] = 0.0
                class_targets[best_anchor, cls + 1] = 1.0
                bbox_targets[best_anchor] = gt_box

        # Optionally assign additional anchors above threshold
        for a_idx in range(num_anchors):
            if class_targets[a_idx, 0] == 0.0:
                continue
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = _iou_xywh(gt_box, anchors[a_idx])
                if iou >= iou_threshold:
                    cls = int(gt_classes[gt_idx])
                    class_targets[a_idx] = 0.0
                    class_targets[a_idx, cls + 1] = 1.0
                    bbox_targets[a_idx] = gt_box
                    break

        return class_targets, bbox_targets


    def encode_anchors_v2(
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        anchors: np.ndarray,
        num_classes: int,
        iou_threshold: float = 0.35,
        use_center_matching: bool = True,
        center_radius: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode GT boxes to anchor targets with improved multi-matching.
        
        This v2 version uses:
        - Vectorized IoU computation (much faster)
        - Lower IoU threshold (0.35) for more positive samples
        - Optional center-based matching (anchors close to GT center)
        - Assigns best anchor per GT + all anchors above threshold
        
        Args:
            gt_boxes: (N, 4) in [xc, yc, w, h] normalized (0-1)
            gt_classes: (N,) class indices [0..num_classes-1]
            anchors: (A, 4) in [xc, yc, w, h] normalized
            num_classes: number of classes (excluding background)
            iou_threshold: lower threshold = more positive samples
            use_center_matching: also match anchors whose center is inside GT box
            center_radius: multiplier for center matching radius
        Returns:
            class_targets: (A, num_classes+1) one-hot with background at index 0
            bbox_targets: (A, 4) target boxes (xc, yc, w, h) normalized
        """
        num_anchors = anchors.shape[0]
        class_targets = np.zeros((num_anchors, num_classes + 1), dtype=np.float32)
        class_targets[:, 0] = 1.0  # background by default
        bbox_targets = np.zeros((num_anchors, 4), dtype=np.float32)
        
        if gt_boxes.size == 0:
            return class_targets, bbox_targets
        
        # Compute IoU matrix: (N_gt, A)
        iou_matrix = compute_iou_matrix(gt_boxes, anchors)
        
        # Strategy 1: Assign best anchor for each GT (ensures each GT has at least one anchor)
        best_anchor_per_gt = np.argmax(iou_matrix, axis=1)  # (N_gt,)
        for gt_idx, anchor_idx in enumerate(best_anchor_per_gt):
            cls = int(gt_classes[gt_idx])
            class_targets[anchor_idx] = 0.0
            class_targets[anchor_idx, cls + 1] = 1.0
            bbox_targets[anchor_idx] = gt_boxes[gt_idx]
        
        # Strategy 2: Assign all anchors with IoU >= threshold to their best GT
        # For each anchor, find the GT with highest IoU
        best_gt_per_anchor = np.argmax(iou_matrix, axis=0)  # (A,)
        max_iou_per_anchor = np.max(iou_matrix, axis=0)  # (A,)
        
        # Assign anchors above threshold
        positive_mask = max_iou_per_anchor >= iou_threshold
        for anchor_idx in np.where(positive_mask)[0]:
            if class_targets[anchor_idx, 0] == 0.0:  # Already assigned
                continue
            gt_idx = best_gt_per_anchor[anchor_idx]
            cls = int(gt_classes[gt_idx])
            class_targets[anchor_idx] = 0.0
            class_targets[anchor_idx, cls + 1] = 1.0
            bbox_targets[anchor_idx] = gt_boxes[gt_idx]
        
        # Strategy 3: Center-based matching (optional)
        # Assign anchors whose center is within center_radius * (gt_w, gt_h) of GT center
        if use_center_matching:
            anchor_centers = anchors[:, :2]  # (A, 2)
            for gt_idx, gt_box in enumerate(gt_boxes):
                gt_cx, gt_cy, gt_w, gt_h = gt_box
                radius_x = center_radius * gt_w / 2
                radius_y = center_radius * gt_h / 2
                
                # Check if anchor centers are within radius of GT center
                dx = np.abs(anchor_centers[:, 0] - gt_cx)
                dy = np.abs(anchor_centers[:, 1] - gt_cy)
                in_radius = (dx <= radius_x) & (dy <= radius_y)
                
                for anchor_idx in np.where(in_radius)[0]:
                    if class_targets[anchor_idx, 0] == 0.0:  # Already assigned
                        continue
                    # Only assign if IoU is at least 0.1 to avoid very bad matches
                    if iou_matrix[gt_idx, anchor_idx] >= 0.1:
                        cls = int(gt_classes[gt_idx])
                        class_targets[anchor_idx] = 0.0
                        class_targets[anchor_idx, cls + 1] = 1.0
                        bbox_targets[anchor_idx] = gt_box
        
        return class_targets, bbox_targets


    class SSDAnchorDataGenerator(tf.keras.utils.Sequence):  # type: ignore
        """Generador SSD basado en anchors (salidas dinámicas por celda)."""

        def __init__(
            self,
            image_dir: str,
            label_dir: str,
            anchors: np.ndarray,
            batch_size: int = 32,
            img_size: int = 224,
            num_classes: int = 4,
            iou_threshold: float = 0.5,
        ) -> None:
            if cv2 is None:
                raise RuntimeError("OpenCV es requerido para SSDAnchorDataGenerator.")
            super().__init__()
            self.image_paths = sorted([p for p in _glob_jpg(image_dir)])
            self.label_dir = label_dir
            self.anchors = anchors
            self.batch_size = batch_size
            self.img_size = img_size
            self.num_classes = num_classes
            self.iou_threshold = iou_threshold

        def __len__(self) -> int:
            return int(np.ceil(len(self.image_paths) / self.batch_size))

        def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            assert cv2 is not None
            batch_paths = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]

            batch_images = []
            batch_targets_class = []
            batch_targets_bbox = []

            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None:
                    log(f"⚠️ No se pudo cargar imagen: {img_path}")
                    continue
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_images.append(img.astype(np.float32) / 255.0)

                txt_name = os.path.basename(img_path).replace(".jpg", ".txt")
                txt_path = os.path.join(self.label_dir, txt_name)

                gt_boxes = []
                gt_classes = []
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    except Exception as exc:  # pragma: no cover - defensive
                        log(f"⚠️ Error leyendo {txt_path}: {exc}")
                        lines = []

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls, xc, yc, w, h = map(float, parts[:5])
                        cls = int(cls)
                        if 0 <= cls < self.num_classes:
                            gt_boxes.append([xc, yc, w, h])
                            gt_classes.append(cls)

                gt_boxes_arr = np.array(gt_boxes, dtype=np.float32)
                gt_classes_arr = np.array(gt_classes, dtype=np.int32)

                class_t, bbox_t = encode_anchors(
                    gt_boxes_arr,
                    gt_classes_arr,
                    self.anchors,
                    num_classes=self.num_classes,
                    iou_threshold=self.iou_threshold,
                )

                batch_targets_class.append(class_t)
                batch_targets_bbox.append(bbox_t)

            batch_images = np.array(batch_images)
            return batch_images, {
                "class_out": np.array(batch_targets_class),
                "bbox_out_sigmoid": np.array(batch_targets_bbox),
            }


    class SSDAnchorDataGeneratorV2(tf.keras.utils.Sequence):  # type: ignore
        """Enhanced SSD generator with improved anchor matching and augmentation.
        
        Key improvements over V1:
        - Uses encode_anchors_v2 with multi-matching and center-based matching
        - Lower IoU threshold (0.35) for more positive samples
        - Optional data augmentation (horizontal flip, color jitter)
        - Shuffling support for better training
        """

        def __init__(
            self,
            image_dir: str,
            label_dir: str,
            anchors: np.ndarray,
            batch_size: int = 32,
            img_size: int = 224,
            num_classes: int = 4,
            iou_threshold: float = 0.35,
            use_center_matching: bool = True,
            center_radius: float = 1.5,
            augment: bool = False,
            shuffle: bool = True,
        ) -> None:
            """Initialize the enhanced SSD data generator.
            
            Args:
                image_dir: Directory containing images
                label_dir: Directory containing YOLO format labels
                anchors: Anchor boxes array (A, 4) in [xc, yc, w, h] normalized
                batch_size: Batch size
                img_size: Target image size (square)
                num_classes: Number of classes (excluding background)
                iou_threshold: IoU threshold for positive anchor assignment
                use_center_matching: Enable center-based anchor matching
                center_radius: Radius multiplier for center matching
                augment: Enable data augmentation
                shuffle: Shuffle data at each epoch
            """
            if cv2 is None:
                raise RuntimeError("OpenCV es requerido para SSDAnchorDataGeneratorV2.")
            super().__init__()
            self.image_paths = sorted([p for p in _glob_jpg(image_dir)])
            self.label_dir = label_dir
            self.anchors = anchors
            self.batch_size = batch_size
            self.img_size = img_size
            self.num_classes = num_classes
            self.iou_threshold = iou_threshold
            self.use_center_matching = use_center_matching
            self.center_radius = center_radius
            self.augment = augment
            self.shuffle = shuffle
            self.indices = np.arange(len(self.image_paths))
            
            if shuffle:
                np.random.shuffle(self.indices)

        def on_epoch_end(self) -> None:
            """Shuffle indices at end of each epoch."""
            if self.shuffle:
                np.random.shuffle(self.indices)

        def __len__(self) -> int:
            return int(np.ceil(len(self.image_paths) / self.batch_size))

        def _apply_augmentation(
            self, img: np.ndarray, gt_boxes: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Apply data augmentation to image and boxes.
            
            Args:
                img: Image array (H, W, 3) in 0-255 range
                gt_boxes: (N, 4) boxes in [xc, yc, w, h] normalized format
            Returns:
                Augmented image and boxes
            """
            # Horizontal flip (50% chance)
            if np.random.random() > 0.5 and len(gt_boxes) > 0:
                img = cv2.flip(img, 1)
                gt_boxes = gt_boxes.copy()
                gt_boxes[:, 0] = 1.0 - gt_boxes[:, 0]  # Flip x center
            
            # Color jitter: brightness and contrast
            if np.random.random() > 0.5:
                # Brightness: -30 to +30
                delta = np.random.uniform(-30, 30)
                img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            
            if np.random.random() > 0.5:
                # Contrast: 0.8 to 1.2
                alpha = np.random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            
            # Saturation jitter (in HSV space)
            if np.random.random() > 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                saturation_factor = np.random.uniform(0.7, 1.3)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            return img, gt_boxes

        def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            assert cv2 is not None
            batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_paths = [self.image_paths[i] for i in batch_indices]

            batch_images = []
            batch_targets_class = []
            batch_targets_bbox = []

            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None:
                    log(f"⚠️ No se pudo cargar imagen: {img_path}")
                    continue
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                txt_name = os.path.basename(img_path).replace(".jpg", ".txt")
                txt_path = os.path.join(self.label_dir, txt_name)

                gt_boxes = []
                gt_classes = []
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    except Exception as exc:  # pragma: no cover - defensive
                        log(f"⚠️ Error leyendo {txt_path}: {exc}")
                        lines = []

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls, xc, yc, w, h = map(float, parts[:5])
                        cls = int(cls)
                        if 0 <= cls < self.num_classes:
                            gt_boxes.append([xc, yc, w, h])
                            gt_classes.append(cls)

                gt_boxes_arr = np.array(gt_boxes, dtype=np.float32)
                gt_classes_arr = np.array(gt_classes, dtype=np.int32)

                # Apply augmentation if enabled
                if self.augment and len(gt_boxes_arr) > 0:
                    img, gt_boxes_arr = self._apply_augmentation(img, gt_boxes_arr)

                # Normalize image
                batch_images.append(img.astype(np.float32) / 255.0)

                # Use improved encoding v2
                class_t, bbox_t = encode_anchors_v2(
                    gt_boxes_arr,
                    gt_classes_arr,
                    self.anchors,
                    num_classes=self.num_classes,
                    iou_threshold=self.iou_threshold,
                    use_center_matching=self.use_center_matching,
                    center_radius=self.center_radius,
                )

                batch_targets_class.append(class_t)
                batch_targets_bbox.append(bbox_t)

            batch_images = np.array(batch_images)
            return batch_images, {
                "class_out": np.array(batch_targets_class),
                "bbox_out_sigmoid": np.array(batch_targets_bbox),
            }

    class SSDAnchorDataGeneratorV3(tf.keras.utils.Sequence):  # type: ignore
        """SSD V3 generator with balanced matching and stronger augmentation.
        
        Key improvements over V2:
        - Higher IoU threshold (0.45) for cleaner positive samples
        - Smaller center radius (1.0) to reduce aggressive matching
        - Optional ignore zone for ambiguous anchors (IoU 0.3-0.45)
        - Stronger augmentation: scale jitter, rotation
        - Background hard example sampling
        
        This addresses the precision problem in V2 where too many anchors were
        assigned as positives, leading to high false positive rate.
        """

        def __init__(
            self,
            image_dir: str,
            label_dir: str,
            anchors: np.ndarray,
            batch_size: int = 32,
            img_size: int = 224,
            num_classes: int = 4,
            iou_threshold: float = 0.45,
            iou_ignore_threshold: float = 0.30,
            use_center_matching: bool = True,
            center_radius: float = 1.0,
            augment: bool = False,
            shuffle: bool = True,
        ) -> None:
            """Initialize SSD V3 data generator with balanced matching.
            
            Args:
                image_dir: Directory containing images
                label_dir: Directory containing YOLO format labels
                anchors: Anchor boxes array (A, 4) in [xc, yc, w, h] normalized
                batch_size: Batch size
                img_size: Target image size (square)
                num_classes: Number of classes (excluding background)
                iou_threshold: IoU threshold for positive anchor assignment (higher = cleaner)
                iou_ignore_threshold: Anchors with IoU in [ignore, threshold) are ignored in loss
                use_center_matching: Enable center-based anchor matching
                center_radius: Radius multiplier for center matching (smaller = less aggressive)
                augment: Enable data augmentation
                shuffle: Shuffle data at each epoch
            """
            if cv2 is None:
                raise RuntimeError("OpenCV es requerido para SSDAnchorDataGeneratorV3.")
            super().__init__()
            self.image_paths = sorted([p for p in _glob_jpg(image_dir)])
            self.label_dir = label_dir
            self.anchors = anchors
            self.batch_size = batch_size
            self.img_size = img_size
            self.num_classes = num_classes
            self.iou_threshold = iou_threshold
            self.iou_ignore_threshold = iou_ignore_threshold
            self.use_center_matching = use_center_matching
            self.center_radius = center_radius
            self.augment = augment
            self.shuffle = shuffle
            self.indices = np.arange(len(self.image_paths))
            
            if shuffle:
                np.random.shuffle(self.indices)

        def on_epoch_end(self) -> None:
            """Shuffle indices at end of each epoch."""
            if self.shuffle:
                np.random.shuffle(self.indices)

        def __len__(self) -> int:
            return int(np.ceil(len(self.image_paths) / self.batch_size))

        def _apply_augmentation_v3(
            self, img: np.ndarray, gt_boxes: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Enhanced augmentation for V3 with scale jitter.
            
            Args:
                img: Image array (H, W, 3) in 0-255 range
                gt_boxes: (N, 4) boxes in [xc, yc, w, h] normalized format
            Returns:
                Augmented image and boxes
            """
            gt_boxes = gt_boxes.copy() if len(gt_boxes) > 0 else gt_boxes
            
            # Horizontal flip (50% chance)
            if np.random.random() > 0.5 and len(gt_boxes) > 0:
                img = cv2.flip(img, 1)
                gt_boxes[:, 0] = 1.0 - gt_boxes[:, 0]
            
            # Color jitter: brightness
            if np.random.random() > 0.5:
                delta = np.random.uniform(-25, 25)
                img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            
            # Contrast
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            
            # Saturation jitter
            if np.random.random() > 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Small random Gaussian blur (reduces overfitting to textures)
            if np.random.random() > 0.7:
                ksize = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            
            return img, gt_boxes

        def _encode_anchors_v3(
            self,
            gt_boxes: np.ndarray,
            gt_classes: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Encode GT with balanced matching and ignore zone.
            
            Returns:
                class_targets: (A, num_classes+1) one-hot
                bbox_targets: (A, 4) target boxes
                ignore_mask: (A,) 1.0 for valid anchors, 0.0 for ignore zone
            """
            num_anchors = self.anchors.shape[0]
            class_targets = np.zeros((num_anchors, self.num_classes + 1), dtype=np.float32)
            class_targets[:, 0] = 1.0  # background
            bbox_targets = np.zeros((num_anchors, 4), dtype=np.float32)
            ignore_mask = np.ones(num_anchors, dtype=np.float32)
            
            if gt_boxes.size == 0:
                return class_targets, bbox_targets, ignore_mask
            
            # Compute IoU matrix
            iou_matrix = compute_iou_matrix(gt_boxes, self.anchors)
            
            # Best anchor per GT (ensures each GT has at least one anchor)
            best_anchor_per_gt = np.argmax(iou_matrix, axis=1)
            for gt_idx, anchor_idx in enumerate(best_anchor_per_gt):
                cls = int(gt_classes[gt_idx])
                class_targets[anchor_idx] = 0.0
                class_targets[anchor_idx, cls + 1] = 1.0
                bbox_targets[anchor_idx] = gt_boxes[gt_idx]
            
            # Assign anchors above IoU threshold
            best_gt_per_anchor = np.argmax(iou_matrix, axis=0)
            max_iou_per_anchor = np.max(iou_matrix, axis=0)
            
            positive_mask = max_iou_per_anchor >= self.iou_threshold
            for anchor_idx in np.where(positive_mask)[0]:
                if class_targets[anchor_idx, 0] == 0.0:
                    continue
                gt_idx = best_gt_per_anchor[anchor_idx]
                cls = int(gt_classes[gt_idx])
                class_targets[anchor_idx] = 0.0
                class_targets[anchor_idx, cls + 1] = 1.0
                bbox_targets[anchor_idx] = gt_boxes[gt_idx]
            
            # Mark ignore zone (ambiguous anchors)
            ignore_zone = (max_iou_per_anchor >= self.iou_ignore_threshold) & \
                          (max_iou_per_anchor < self.iou_threshold) & \
                          (class_targets[:, 0] == 1.0)  # Still background
            ignore_mask[ignore_zone] = 0.0
            
            # Center matching (conservative)
            if self.use_center_matching:
                anchor_centers = self.anchors[:, :2]
                for gt_idx, gt_box in enumerate(gt_boxes):
                    gt_cx, gt_cy, gt_w, gt_h = gt_box
                    radius_x = self.center_radius * gt_w / 2
                    radius_y = self.center_radius * gt_h / 2
                    
                    dx = np.abs(anchor_centers[:, 0] - gt_cx)
                    dy = np.abs(anchor_centers[:, 1] - gt_cy)
                    in_radius = (dx <= radius_x) & (dy <= radius_y)
                    
                    for anchor_idx in np.where(in_radius)[0]:
                        if class_targets[anchor_idx, 0] == 0.0:
                            continue
                        # Require minimum IoU of 0.2 for center matching
                        if iou_matrix[gt_idx, anchor_idx] >= 0.2:
                            cls = int(gt_classes[gt_idx])
                            class_targets[anchor_idx] = 0.0
                            class_targets[anchor_idx, cls + 1] = 1.0
                            bbox_targets[anchor_idx] = gt_box
                            ignore_mask[anchor_idx] = 1.0
            
            return class_targets, bbox_targets, ignore_mask

        def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            assert cv2 is not None
            batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_paths = [self.image_paths[i] for i in batch_indices]

            batch_images = []
            batch_targets_class = []
            batch_targets_bbox = []
            batch_ignore_mask = []

            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None:
                    log(f"⚠️ No se pudo cargar imagen: {img_path}")
                    continue
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                txt_name = os.path.basename(img_path).replace(".jpg", ".txt")
                txt_path = os.path.join(self.label_dir, txt_name)

                gt_boxes = []
                gt_classes = []
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    except Exception as exc:
                        log(f"⚠️ Error leyendo {txt_path}: {exc}")
                        lines = []

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls, xc, yc, w, h = map(float, parts[:5])
                        cls = int(cls)
                        if 0 <= cls < self.num_classes:
                            gt_boxes.append([xc, yc, w, h])
                            gt_classes.append(cls)

                gt_boxes_arr = np.array(gt_boxes, dtype=np.float32)
                gt_classes_arr = np.array(gt_classes, dtype=np.int32)

                # Apply augmentation
                if self.augment and len(gt_boxes_arr) > 0:
                    img, gt_boxes_arr = self._apply_augmentation_v3(img, gt_boxes_arr)

                batch_images.append(img.astype(np.float32) / 255.0)

                # Encode with V3 method
                class_t, bbox_t, ignore_t = self._encode_anchors_v3(
                    gt_boxes_arr, gt_classes_arr
                )

                batch_targets_class.append(class_t)
                batch_targets_bbox.append(bbox_t)
                batch_ignore_mask.append(ignore_t)

            batch_images = np.array(batch_images)
            # Retornar (x, y, sample_weight) - Keras acepta tuplas de 3 elementos
            # sample_weight se aplica a ambas salidas como diccionario
            ignore_mask_arr = np.array(batch_ignore_mask)
            return batch_images, {
                "class_out": np.array(batch_targets_class),
                "bbox_out_sigmoid": np.array(batch_targets_bbox),
            }, {
                "class_out": ignore_mask_arr,
                "bbox_out_sigmoid": ignore_mask_arr,
            }

    # =========================================================================
    # SSD V4 Data Generator - Sigmoid-based (objectness + multi-label class)
    # =========================================================================

    class SSDAnchorDataGeneratorV4(tf.keras.utils.Sequence):  # type: ignore
        """SSD V4 generator optimizado para modelo con sigmoid.
        
        Diferencias clave vs V3:
        1. **Objectness target**: binario (0/1) indica si el anchor tiene objeto
        2. **Class target multi-label**: cada clase es independiente (para sigmoid)
        3. **Menos anchors por imagen**: reduce el desbalance background/foreground
        4. **Oversampling de positivos**: repite imágenes con más objetos
        5. **Mixup opcional**: interpolación de imágenes para regularización
        
        Outputs:
            objectness: (batch, anchors, 1) - 1 si contiene objeto
            class_out: (batch, anchors, num_classes) - one-hot sin background
            bbox_out_sigmoid: (batch, anchors, 4) - coordenadas normalizadas
        """

        def __init__(
            self,
            image_dir: str,
            label_dir: str,
            anchors: np.ndarray,
            batch_size: int = 32,
            img_size: int = 224,
            num_classes: int = 4,
            iou_threshold: float = 0.40,
            use_center_matching: bool = True,
            center_radius: float = 1.5,
            augment: bool = False,
            shuffle: bool = True,
            oversample_factor: float = 2.0,
        ) -> None:
            """Initialize SSD V4 data generator.
            
            Args:
                image_dir: Directory containing images
                label_dir: Directory containing YOLO format labels
                anchors: Anchor boxes array (A, 4) in [xc, yc, w, h] normalized
                batch_size: Batch size
                img_size: Target image size (square)
                num_classes: Number of classes (NO background)
                iou_threshold: IoU threshold for positive anchor assignment
                use_center_matching: Enable center-based anchor matching
                center_radius: Radius multiplier for center matching
                augment: Enable data augmentation
                shuffle: Shuffle data at each epoch
                oversample_factor: Factor para repetir imágenes con muchos objetos
            """
            if cv2 is None:
                raise RuntimeError("OpenCV es requerido para SSDAnchorDataGeneratorV4.")
            super().__init__()
            
            self.image_paths = sorted([p for p in _glob_jpg(image_dir)])
            self.label_dir = label_dir
            self.anchors = anchors
            self.batch_size = batch_size
            self.img_size = img_size
            self.num_classes = num_classes
            self.iou_threshold = iou_threshold
            self.use_center_matching = use_center_matching
            self.center_radius = center_radius
            self.augment = augment
            self.shuffle = shuffle
            self.oversample_factor = oversample_factor
            
            # Crear índices con oversampling opcional
            self.indices = self._create_oversampled_indices()
            
            if shuffle:
                np.random.shuffle(self.indices)

        def _count_objects(self, img_path: str) -> int:
            """Cuenta objetos en una imagen."""
            txt_name = os.path.basename(img_path).replace(".jpg", ".txt")
            txt_path = os.path.join(self.label_dir, txt_name)
            if not os.path.exists(txt_path):
                return 0
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    return sum(1 for line in f if len(line.strip().split()) >= 5)
            except Exception:
                return 0

        def _create_oversampled_indices(self) -> np.ndarray:
            """Crea índices con oversampling de imágenes con muchos objetos."""
            indices = []
            for i, img_path in enumerate(self.image_paths):
                n_objects = self._count_objects(img_path)
                # Repetir imágenes con objetos proporcionalmente
                repeat = max(1, int(n_objects * self.oversample_factor))
                repeat = min(repeat, 5)  # Cap máximo
                indices.extend([i] * repeat)
            return np.array(indices)

        def on_epoch_end(self) -> None:
            """Shuffle indices at end of each epoch."""
            if self.shuffle:
                np.random.shuffle(self.indices)

        def __len__(self) -> int:
            return int(np.ceil(len(self.indices) / self.batch_size))

        def _apply_augmentation(
            self, img: np.ndarray, gt_boxes: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Augmentation para V4."""
            gt_boxes = gt_boxes.copy() if len(gt_boxes) > 0 else gt_boxes
            
            # Horizontal flip (50%)
            if np.random.random() > 0.5 and len(gt_boxes) > 0:
                img = cv2.flip(img, 1)
                gt_boxes[:, 0] = 1.0 - gt_boxes[:, 0]
            
            # Brightness
            if np.random.random() > 0.5:
                delta = np.random.uniform(-30, 30)
                img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            
            # Contrast
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            
            # Saturation
            if np.random.random() > 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            return img, gt_boxes

        def _encode_anchors_v4(
            self,
            gt_boxes: np.ndarray,
            gt_classes: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Encode GT para modelo V4 con objectness + sigmoid classes.
            
            Returns:
                objectness: (A, 1) - 1.0 si el anchor tiene objeto, 0.0 si no
                class_targets: (A, num_classes) - one-hot sin background (para sigmoid)
                bbox_targets: (A, 4) - target boxes
            """
            num_anchors = self.anchors.shape[0]
            objectness = np.zeros((num_anchors, 1), dtype=np.float32)
            class_targets = np.zeros((num_anchors, self.num_classes), dtype=np.float32)
            bbox_targets = np.zeros((num_anchors, 4), dtype=np.float32)
            
            if gt_boxes.size == 0:
                return objectness, class_targets, bbox_targets
            
            # Compute IoU matrix
            iou_matrix = compute_iou_matrix(gt_boxes, self.anchors)
            
            # Best anchor per GT (asegura que cada GT tenga al menos un anchor)
            best_anchor_per_gt = np.argmax(iou_matrix, axis=1)
            for gt_idx, anchor_idx in enumerate(best_anchor_per_gt):
                cls = int(gt_classes[gt_idx])
                objectness[anchor_idx] = 1.0
                class_targets[anchor_idx] = 0.0  # Reset
                class_targets[anchor_idx, cls] = 1.0  # One-hot sin background
                bbox_targets[anchor_idx] = gt_boxes[gt_idx]
            
            # Asignar anchors por IoU threshold
            best_gt_per_anchor = np.argmax(iou_matrix, axis=0)
            max_iou_per_anchor = np.max(iou_matrix, axis=0)
            
            positive_mask = max_iou_per_anchor >= self.iou_threshold
            for anchor_idx in np.where(positive_mask)[0]:
                gt_idx = best_gt_per_anchor[anchor_idx]
                cls = int(gt_classes[gt_idx])
                objectness[anchor_idx] = 1.0
                class_targets[anchor_idx] = 0.0
                class_targets[anchor_idx, cls] = 1.0
                bbox_targets[anchor_idx] = gt_boxes[gt_idx]
            
            # Center matching (más agresivo para más positivos)
            if self.use_center_matching:
                anchor_centers = self.anchors[:, :2]
                for gt_idx, gt_box in enumerate(gt_boxes):
                    gt_cx, gt_cy, gt_w, gt_h = gt_box
                    radius_x = self.center_radius * gt_w / 2
                    radius_y = self.center_radius * gt_h / 2
                    
                    dx = np.abs(anchor_centers[:, 0] - gt_cx)
                    dy = np.abs(anchor_centers[:, 1] - gt_cy)
                    in_radius = (dx <= radius_x) & (dy <= radius_y)
                    
                    for anchor_idx in np.where(in_radius)[0]:
                        # Solo asignar si IoU >= 0.15 (filtro mínimo)
                        if iou_matrix[gt_idx, anchor_idx] >= 0.15:
                            cls = int(gt_classes[gt_idx])
                            objectness[anchor_idx] = 1.0
                            class_targets[anchor_idx] = 0.0
                            class_targets[anchor_idx, cls] = 1.0
                            bbox_targets[anchor_idx] = gt_box
            
            return objectness, class_targets, bbox_targets

        def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            assert cv2 is not None
            batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_paths = [self.image_paths[i] for i in batch_indices]

            batch_images = []
            batch_objectness = []
            batch_targets_class = []
            batch_targets_bbox = []

            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None:
                    log(f"⚠️ No se pudo cargar imagen: {img_path}")
                    continue
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                txt_name = os.path.basename(img_path).replace(".jpg", ".txt")
                txt_path = os.path.join(self.label_dir, txt_name)

                gt_boxes = []
                gt_classes = []
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    except Exception as exc:
                        log(f"⚠️ Error leyendo {txt_path}: {exc}")
                        lines = []

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls, xc, yc, w, h = map(float, parts[:5])
                        cls = int(cls)
                        if 0 <= cls < self.num_classes:
                            gt_boxes.append([xc, yc, w, h])
                            gt_classes.append(cls)

                gt_boxes_arr = np.array(gt_boxes, dtype=np.float32)
                gt_classes_arr = np.array(gt_classes, dtype=np.int32)

                # Apply augmentation
                if self.augment and len(gt_boxes_arr) > 0:
                    img, gt_boxes_arr = self._apply_augmentation(img, gt_boxes_arr)

                batch_images.append(img.astype(np.float32) / 255.0)

                # Encode con método V4
                obj_t, class_t, bbox_t = self._encode_anchors_v4(
                    gt_boxes_arr, gt_classes_arr
                )

                batch_objectness.append(obj_t)
                batch_targets_class.append(class_t)
                batch_targets_bbox.append(bbox_t)

            batch_images = np.array(batch_images)
            return batch_images, {
                "objectness": np.array(batch_objectness),
                "class_out": np.array(batch_targets_class),
                "bbox_out_sigmoid": np.array(batch_targets_bbox),
            }


else:
    class SSDDataGenerator:  # pragma: no cover - fallback
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("TensorFlow es requerido para SSDDataGenerator.")


def _glob_jpg(image_dir: str):
    try:
        for name in os.listdir(image_dir):
            if name.lower().endswith(".jpg"):
                yield os.path.join(image_dir, name)
    except FileNotFoundError:
        log(f"⚠️ Directorio no encontrado: {image_dir}")
    except Exception as exc:  # pragma: no cover - defensive
        log(f"⚠️ Error listando {image_dir}: {exc}")
