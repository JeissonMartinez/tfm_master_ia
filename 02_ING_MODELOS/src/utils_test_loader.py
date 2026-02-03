"""COCO test loader utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import json
import os

import cv2
import numpy as np

try:
    from .utils_io import log
    from .utils_eval import BoundingBox
except ImportError:  # fallback for notebooks
    from utils_io import log
    from utils_eval import BoundingBox


@dataclass
class ImageInfo:
    image_id: int
    file_name: str
    width: int
    height: int


class COCOTargets:
    def __init__(
        self,
        coco_json_path: str,
        images_dir: str,
        class_names: List[str],
        image_size: Tuple[int, int] = (224, 224),
        category_id_map: Dict[int, int] | None = None,
    ) -> None:
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.class_names = class_names
        self.image_size = image_size
        # Allow custom category_id mapping; default builds from COCO file
        self._custom_cat_map = category_id_map
        self.category_id_to_class: Dict[int, int] = {}
        self.class_id_to_name = {i: name for i, name in enumerate(class_names)}
        self.images: Dict[int, ImageInfo] = {}
        self.targets: Dict[int, List[BoundingBox]] = {}

        self._load_annotations()

    def _load_annotations(self) -> None:
        with open(self.coco_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Build category_id -> class_id map from COCO categories
        if self._custom_cat_map is not None:
            self.category_id_to_class = self._custom_cat_map
        else:
            # Auto-build from categories in COCO file, matching by name
            name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            for cat in coco.get("categories", []):
                cat_name = cat.get("name", "")
                if cat_name in name_to_idx:
                    self.category_id_to_class[cat["id"]] = name_to_idx[cat_name]
            if not self.category_id_to_class:
                # Fallback: assume category_ids are 1-indexed
                self.category_id_to_class = {i + 1: i for i in range(len(self.class_names))}
                log("⚠️ No matching categories found in COCO file, using default 1-indexed mapping.")

        for img in coco.get("images", []):
            self.images[img["id"]] = ImageInfo(
                image_id=img["id"],
                file_name=img["file_name"],
                width=img["width"],
                height=img["height"],
            )

        targets: Dict[int, List[BoundingBox]] = {img_id: [] for img_id in self.images}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.images:
                continue
            category_id = ann["category_id"]
            class_id = self.category_id_to_class.get(category_id, None)
            if class_id is None:
                continue
            class_name = self.class_id_to_name[class_id]
            x, y, w, h = ann["bbox"]
            targets[img_id].append(
                BoundingBox(x=x, y=y, w=w, h=h, class_id=class_id, class_name=class_name)
            )

        self.targets = targets
        log(f"Loaded {len(self.images)} images and {sum(len(v) for v in targets.values())} annotations.")

    def get_image_path(self, image_id: int) -> str:
        info = self.images[image_id]
        return os.path.join(self.images_dir, info.file_name)

    def get_ground_truth(self, image_id: int) -> List[BoundingBox]:
        return list(self.targets.get(image_id, []))

    def get_scaled_ground_truth(self, image_id: int) -> List[BoundingBox]:
        info = self.images[image_id]
        dst_h, dst_w = self.image_size
        scaled = [
            box.scale_to(info.width, info.height, dst_w, dst_h)
            for box in self.get_ground_truth(image_id)
        ]
        return scaled

    def iter_images(self) -> List[int]:
        return list(self.images.keys())

    def load_image(self, image_id: int) -> np.ndarray:
        path = self.get_image_path(image_id)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_resized_image(self, image_id: int) -> np.ndarray:
        img = self.load_image(image_id)
        dst_h, dst_w = self.image_size
        img = cv2.resize(img, (dst_w, dst_h))
        return img

    def iter_batches(self, batch_size: int = 8) -> Tuple[np.ndarray, List[List[BoundingBox]], List[int]]:
        image_ids = self.iter_images()
        for i in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[i : i + batch_size]
            imgs = [self.load_resized_image(img_id) for img_id in batch_ids]
            gts = [self.get_scaled_ground_truth(img_id) for img_id in batch_ids]
            yield np.array(imgs), gts, batch_ids
