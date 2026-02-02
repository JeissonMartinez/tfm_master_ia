"""SSD data generator utilities (YOLO txt -> SSD targets)."""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from .utils_io import log

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
                "bbox_out": np.array(batch_targets_bbox),
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
