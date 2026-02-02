import json
import os
import random
import shutil
from typing import Dict, List, Tuple, Any

import albumentations as A
import cv2
import numpy as np


def load_coco(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def save_coco(data: dict, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(data, f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_class_counts(data: dict) -> Dict[str, int]:
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    counts = {name: 0 for name in cat_id_to_name.values()}
    for ann in data["annotations"]:
        name = cat_id_to_name.get(ann["category_id"])
        if name in counts:
            counts[name] += 1
    return counts


def filter_coco_by_classes(data: dict, keep_names: List[str]) -> dict:
    cat_name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    keep_ids = {cat_name_to_id[n] for n in keep_names if n in cat_name_to_id}

    new_annotations = []
    kept_image_ids = set()
    for ann in data["annotations"]:
        if ann["category_id"] in keep_ids:
            new_annotations.append(ann)
            kept_image_ids.add(ann["image_id"])

    new_images = [img for img in data["images"] if img["id"] in kept_image_ids]
    new_categories = [c for c in data["categories"] if c["id"] in keep_ids]

    return {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": new_categories,
        "images": new_images,
        "annotations": new_annotations,
    }


def build_anns_map(annotations: List[dict]) -> Dict[int, List[dict]]:
    anns_map: Dict[int, List[dict]] = {}
    for ann in annotations:
        anns_map.setdefault(ann["image_id"], []).append(ann)
    return anns_map


def _build_transforms() -> Tuple[A.Compose, A.Compose]:
    mild = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.15, contrast_limit=0.15),
            A.OneOf(
                [
                    A.GaussNoise(p=0.6, var_limit=(10.0, 30.0)),
                    A.ISONoise(p=0.4, intensity=(0.1, 0.4)),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(p=0.5, blur_limit=(3, 5)),
                    A.MotionBlur(p=0.5, blur_limit=(3, 5)),
                ],
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], min_visibility=0.2),
    )

    aggressive = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=12,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            A.RandomBrightnessContrast(p=0.6, brightness_limit=0.2, contrast_limit=0.2),
            A.ColorJitter(p=0.4, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            A.OneOf(
                [
                    A.GaussNoise(p=0.7, var_limit=(15.0, 50.0)),
                    A.ISONoise(p=0.5, intensity=(0.1, 0.6)),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(p=0.6, blur_limit=(3, 7)),
                    A.MotionBlur(p=0.6, blur_limit=(3, 7)),
                ],
                p=0.5,
            ),
            A.RandomGamma(p=0.4),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"], min_visibility=0.2),
    )

    return mild, aggressive


def _copy_image(src_path: str, dst_path: str) -> None:
    if not os.path.exists(dst_path):
        ensure_dir(os.path.dirname(dst_path))
        shutil.copy2(src_path, dst_path)


def _load_image_rgb(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_image_rgb(path: str, image: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def paste_augmentation(
    data: dict,
    images_dir: str,
    output_dir: str,
    start_img_id: int,
    start_ann_id: int,
    majority_class_id: int,
    donor_class_ids: List[int],
    paste_per_image: int = 1,
    max_objects: int = 2,
    seed: int = 42,
    filename_prefix: str = "paste2",
) -> Tuple[List[dict], List[dict], int, int]:
    random.seed(seed)

    img_map = {img["id"]: img for img in data["images"]}
    anns_map = build_anns_map(data["annotations"])

    donor_pool: List[Tuple[str, List[float], int]] = []
    for ann in data["annotations"]:
        if ann["category_id"] in donor_class_ids:
            img_info = img_map.get(ann["image_id"])
            if not img_info:
                continue
            donor_pool.append((img_info["file_name"], ann["bbox"], ann["category_id"]))

    if not donor_pool:
        return [], [], start_img_id, start_ann_id

    majority_only_images = []
    for img in data["images"]:
        anns = anns_map.get(img["id"], [])
        if not anns:
            continue
        if all(a["category_id"] == majority_class_id for a in anns):
            majority_only_images.append(img)

    if not majority_only_images:
        return [], [], start_img_id, start_ann_id

    new_images: List[dict] = []
    new_annotations: List[dict] = []
    img_counter = 0

    for img_info in majority_only_images:
        if paste_per_image <= 0:
            break
        for _ in range(paste_per_image):
            base_path = os.path.join(images_dir, img_info["file_name"])
            base_img = _load_image_rgb(base_path)
            if base_img is None:
                continue

            h, w = base_img.shape[:2]
            objects_to_paste = random.randint(1, max_objects)
            pasted_bboxes: List[Tuple[float, float, float, float, int]] = []

            for _ in range(objects_to_paste):
                donor_file, donor_bbox, donor_cat = random.choice(donor_pool)
                donor_path = os.path.join(images_dir, donor_file)
                donor_img = _load_image_rgb(donor_path)
                if donor_img is None:
                    continue

                x, y, bw, bh = [int(v) for v in donor_bbox]
                x = max(0, x)
                y = max(0, y)
                bw = max(1, bw)
                bh = max(1, bh)

                donor_crop = donor_img[y : y + bh, x : x + bw]
                if donor_crop.size == 0:
                    continue

                scale = random.uniform(0.8, 1.2)
                new_w = max(1, int(bw * scale))
                new_h = max(1, int(bh * scale))
                donor_crop = cv2.resize(donor_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                max_x = max(0, w - new_w)
                max_y = max(0, h - new_h)
                if max_x == 0 or max_y == 0:
                    continue

                paste_x = random.randint(0, max_x)
                paste_y = random.randint(0, max_y)

                base_img[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = donor_crop
                pasted_bboxes.append((paste_x, paste_y, new_w, new_h, donor_cat))

            if not pasted_bboxes:
                continue

            start_img_id += 1
            img_counter += 1
            new_filename = f"{filename_prefix}_{img_counter}_{img_info['file_name']}"
            save_path = os.path.join(output_dir, new_filename)
            _save_image_rgb(save_path, base_img)

            new_images.append(
                {
                    "id": start_img_id,
                    "file_name": new_filename,
                    "height": img_info["height"],
                    "width": img_info["width"],
                    "date_captured": "2026-PasteAug",
                }
            )

            for ann in anns_map.get(img_info["id"], []):
                start_ann_id += 1
                new_ann = dict(ann)
                new_ann["id"] = start_ann_id
                new_ann["image_id"] = start_img_id
                new_annotations.append(new_ann)

            for pb in pasted_bboxes:
                start_ann_id += 1
                px, py, pw, ph, pcat = pb
                new_annotations.append(
                    {
                        "id": start_ann_id,
                        "image_id": start_img_id,
                        "category_id": pcat,
                        "bbox": [float(px), float(py), float(pw), float(ph)],
                        "area": float(pw * ph),
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )

    return new_images, new_annotations, start_img_id, start_ann_id


def targeted_augmentation(
    input_json: str,
    images_dir: str,
    output_dir: str,
    output_json: str,
    keep_names: List[str],
    mild_multiplier: int = 1,
    aggressive_multiplier: int = 3,
    enable_paste: bool = True,
    paste_per_image: int = 1,
    paste_max_objects: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    random.seed(seed)

    data = load_coco(input_json)
    filtered = filter_coco_by_classes(data, keep_names)

    counts = compute_class_counts(filtered)
    majority_class = max(counts.items(), key=lambda x: x[1])[0] if counts else None
    minority_classes = [n for n in keep_names if n != majority_class]

    cat_name_to_id = {c["name"]: c["id"] for c in filtered["categories"]}
    majority_class_id = cat_name_to_id.get(majority_class)
    minority_class_ids = [cat_name_to_id[n] for n in minority_classes if n in cat_name_to_id]

    ensure_dir(output_dir)

    anns_map = build_anns_map(filtered["annotations"])

    max_img_id = max([img["id"] for img in filtered["images"]]) if filtered["images"] else 0
    max_ann_id = max([ann["id"] for ann in filtered["annotations"]]) if filtered["annotations"] else 0

    new_images: List[dict] = []
    new_annotations: List[dict] = []

    mild_tf, aggressive_tf = _build_transforms()

    for img_info in filtered["images"]:
        file_name = img_info["file_name"]
        img_path = os.path.join(images_dir, file_name)
        image = _load_image_rgb(img_path)
        if image is None:
            continue

        _copy_image(img_path, os.path.join(output_dir, file_name))

        new_images.append(img_info)
        anns = anns_map.get(img_info["id"], [])
        new_annotations.extend(anns)

        bboxes = [a["bbox"] for a in anns]
        cat_ids = [a["category_id"] for a in anns]

        has_minority = any(c in minority_class_ids for c in cat_ids)
        multiplier = aggressive_multiplier if has_minority else mild_multiplier

        if multiplier <= 0:
            continue

        transform = aggressive_tf if has_minority else mild_tf

        for i in range(multiplier):
            if bboxes:
                try:
                    augmented = transform(image=image, bboxes=bboxes, category_ids=cat_ids)
                except ValueError:
                    continue
            else:
                augmented = transform(image=image, bboxes=[], category_ids=[])

            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_cats = augmented["category_ids"]

            max_img_id += 1
            new_filename = f"aug2_{i}_{file_name}"
            _save_image_rgb(os.path.join(output_dir, new_filename), aug_img)

            new_images.append(
                {
                    "id": max_img_id,
                    "file_name": new_filename,
                    "height": img_info["height"],
                    "width": img_info["width"],
                    "date_captured": "2026-TargetedAug",
                }
            )

            for bbox, cat_id in zip(aug_bboxes, aug_cats):
                max_ann_id += 1
                w, h = bbox[2], bbox[3]
                new_annotations.append(
                    {
                        "id": max_ann_id,
                        "image_id": max_img_id,
                        "category_id": cat_id,
                        "bbox": [float(v) for v in bbox],
                        "area": float(w * h),
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )

    paste_imgs: List[dict] = []
    paste_anns: List[dict] = []
    if enable_paste and majority_class_id is not None and minority_class_ids:
        paste_imgs, paste_anns, max_img_id, max_ann_id = paste_augmentation(
            filtered,
            images_dir,
            output_dir,
            max_img_id,
            max_ann_id,
            majority_class_id,
            minority_class_ids,
            paste_per_image=paste_per_image,
            max_objects=paste_max_objects,
            seed=seed,
        )

    final_data = {
        "info": {"description": "Targeted Augmentation Dataset"},
        "licenses": filtered.get("licenses", []),
        "categories": filtered["categories"],
        "images": new_images + paste_imgs,
        "annotations": new_annotations + paste_anns,
    }

    save_coco(final_data, output_json)

    return {
        "majority_class": majority_class or "",
        "minority_classes": ",".join(minority_classes),
        "original_images": len(filtered["images"]),
        "final_images": len(final_data["images"]),
        "final_annotations": len(final_data["annotations"]),
    }
