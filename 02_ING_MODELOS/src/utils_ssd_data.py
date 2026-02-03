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


    def encode_anchors(
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        anchors: np.ndarray,
        num_classes: int,
        iou_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode GT boxes to anchor targets.

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
