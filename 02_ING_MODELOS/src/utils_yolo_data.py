"""COCO -> YOLO dataset conversion utilities (SSD/YOLO focus)."""
from __future__ import annotations

import os
import shutil
from collections import defaultdict
from typing import List

from .utils_io import safe_read_json, safe_mkdir, safe_write_text, log


def _normalize_bbox(bbox, img_w, img_h):
    x_min, y_min, w, h = bbox
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    x_center = (x_min + w / 2.0) * dw
    y_center = (y_min + h / 2.0) * dh
    return x_center, y_center, w * dw, h * dh


def convert_single_set(
    json_path: str,
    img_source_dir: str,
    yolo_root: str,
    subset_name: str,
) -> List[str]:
    """Convert one COCO subset to YOLO folder structure.

    Returns class_names used for data.yaml.
    """
    log(f"\n🔄 Procesando subset: {subset_name.upper()}...")

    data = safe_read_json(json_path)
    if data is None:
        log(f"⚠️ Saltando {subset_name}: no se pudo leer {json_path}")
        return []

    dest_img_dir = os.path.join(yolo_root, "images", subset_name)
    dest_lbl_dir = os.path.join(yolo_root, "labels", subset_name)
    safe_mkdir(dest_img_dir)
    safe_mkdir(dest_lbl_dir)

    cats = sorted(data.get("categories", []), key=lambda x: x.get("id", 0))
    if not cats:
        log("⚠️ No hay categorías en el JSON.")
        return []

    cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(cats)}
    class_names = [cat["name"] for cat in cats]

    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    images = data.get("images", [])
    for img_info in images:
        file_name = img_info.get("file_name")
        img_id = img_info.get("id")
        if not file_name or img_id is None:
            continue

        src = os.path.join(img_source_dir, file_name)
        dst = os.path.join(dest_img_dir, file_name)
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
            except Exception as exc:  # pragma: no cover - defensive
                log(f"⚠️ Error copiando {src}: {exc}")
                continue
        else:
            continue

        txt_path = os.path.join(dest_lbl_dir, os.path.splitext(file_name)[0] + ".txt")
        lines = []
        for ann in anns_by_img.get(img_id, []):
            if ann.get("category_id") not in cat_id_to_idx:
                continue
            x_c, y_c, w_n, h_n = _normalize_bbox(ann["bbox"], img_info["width"], img_info["height"])
            cls_idx = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{cls_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

        safe_write_text(txt_path, "".join(lines))

    return class_names


def create_yolo_dataset(
    train_json: str,
    train_img_dir: str,
    val_json: str,
    val_img_dir: str,
    test_json: str,
    test_img_dir: str,
    yolo_root: str,
) -> List[str]:
    """Rebuild YOLO dataset folders and data.yaml.

    Returns class names.
    """
    if os.path.exists(yolo_root):
        try:
            shutil.rmtree(yolo_root)
        except Exception as exc:  # pragma: no cover - defensive
            log(f"⚠️ No se pudo limpiar {yolo_root}: {exc}")

    class_names = convert_single_set(train_json, train_img_dir, yolo_root, "train")
    convert_single_set(val_json, val_img_dir, yolo_root, "val")
    convert_single_set(test_json, test_img_dir, yolo_root, "test")

    yaml_content = (
        f"path: {yolo_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    safe_write_text(os.path.join(yolo_root, "data.yaml"), yaml_content)
    log("✅ Dataset YOLO reestructurado correctamente.")
    return class_names
