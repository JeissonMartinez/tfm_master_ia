"""Unified data pipeline for YOLO (txt) and MobileNet (TFRecord) datasets.

Provides:
- YOLO dataset verification, data.yaml generation, class distribution
- TFRecord write/read/verify pipelines with tf.data augmentation
- Anchor generation and target encoding for SSD models
- Shared utilities: class distribution, class weights, visualization
"""
from __future__ import annotations

import glob
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .utils_io import log, safe_mkdir, write_json, write_yaml, read_json, file_exists


# =====================================================================
#  YOLO FORMAT UTILITIES
# =====================================================================

def verify_yolo_dataset(dataset_dir: str | Path) -> Dict[str, Any]:
    """Verify a YOLO-format dataset has correct structure.

    Checks for images/{train,val,test} and labels/{train,val,test}.

    Returns:
        Dict with 'valid' flag and per-split statistics.
    """
    dataset_dir = Path(dataset_dir)
    result: Dict[str, Any] = {"valid": True, "splits": {}, "issues": []}

    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        info: Dict[str, Any] = {"images": 0, "labels": 0, "missing_labels": 0}

        if not img_dir.exists():
            result["issues"].append(f"Falta directorio: {img_dir}")
            result["valid"] = False
        else:
            imgs = set(
                p.stem
                for p in img_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
            info["images"] = len(imgs)

            if lbl_dir.exists():
                lbls = set(p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt")
                info["labels"] = len(lbls)
                info["missing_labels"] = len(imgs - lbls)
            else:
                result["issues"].append(f"Falta directorio: {lbl_dir}")
                result["valid"] = False

        result["splits"][split] = info

    # Print report
    log(f"\n📂 Dataset YOLO: {dataset_dir.name}")
    for split, info in result["splits"].items():
        status = "✅" if info["images"] > 0 else "❌"
        log(f"  {status} {split:>5s}: {info['images']:>5d} imgs | "
            f"{info['labels']:>5d} labels | "
            f"{info['missing_labels']} sin label")
    if result["issues"]:
        log(f"\n  ⚠️  Problemas encontrados:")
        for issue in result["issues"]:
            log(f"      - {issue}")

    return result


def generate_data_yaml(
    dataset_dir: str | Path,
    class_names: List[str],
) -> str:
    """Generate/overwrite data.yaml with absolute paths.

    Returns:
        Path to the generated data.yaml.
    """
    dataset_dir = Path(dataset_dir).resolve()
    yaml_path = dataset_dir / "data.yaml"
    content = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    write_yaml(yaml_path, content)
    log(f"✅ data.yaml generado: {yaml_path}")
    return str(yaml_path)


def get_class_distribution_yolo(
    labels_dir: str | Path,
    class_names: List[str],
) -> Dict[str, int]:
    """Count annotations per class in YOLO label files."""
    labels_dir = Path(labels_dir)
    counts = Counter()
    for txt in labels_dir.glob("*.txt"):
        for line in txt.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                cls_idx = int(parts[0])
                if 0 <= cls_idx < len(class_names):
                    counts[class_names[cls_idx]] += 1
    return {name: counts.get(name, 0) for name in class_names}


def _resolve_yolo_split_dirs(
    dataset_dir: Path, split: str,
) -> Tuple[Path, Path]:
    """Return (images_dir, labels_dir) detecting Roboflow vs Ultralytics layout.

    Roboflow:     {split}/images, {split}/labels
    Ultralytics:  images/{split}, labels/{split}
    """
    # Roboflow format (preferred)
    if (dataset_dir / split / "images").exists():
        return dataset_dir / split / "images", dataset_dir / split / "labels"
    # Ultralytics format
    return dataset_dir / "images" / split, dataset_dir / "labels" / split


def create_yolo_working_copy(
    original_dir: str | Path,
    master_classes: List[str],
    selected_classes: List[str],
    work_root: str | Path | None = None,
    negative_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[str, str, Dict[str, Any]]:
    """Create a lightweight working copy of a YOLO dataset for a class subset.

    The **original dataset is never modified**.  Instead a new directory is
    created with:

    * ``images/{split}/`` — **symlinks** to the original images (zero copy)
    * ``labels/{split}/`` — filtered + remapped ``.txt`` files
    * ``data.yaml`` — with ``nc`` = len(selected_classes)

    Only images that have at least one annotation in the selected classes are
    included ("positives").  A controlled fraction of images with **no**
    matching annotations is added as negative / background samples.

    Re-running this function with different classes automatically creates
    (or replaces) a separate working copy — the original stays intact.

    Parameters
    ----------
    original_dir : path
        Root of the original YOLO dataset (``images/`` + ``labels/``).
    master_classes : list[str]
        Full ordered class list matching the original label IDs.
    selected_classes : list[str]
        Class names to keep.  Order defines new IDs (index 0, 1, …).
    work_root : path, optional
        Parent directory for working copies.  Defaults to
        ``{original_dir.parent}/_work``.
    negative_ratio : float
        Max fraction of negative samples relative to positives (default 10 %).
    seed : int
        Random seed for reproducible negative sampling.

    Returns
    -------
    (work_dir, data_yaml_path, stats)  where *stats* is a per-split dict.
    """
    original_dir = Path(original_dir).resolve()
    if work_root is None:
        work_root = original_dir.parent / "_work"
    work_root = Path(work_root)

    # Deterministic directory name based on sorted class selection
    slug = "_".join(sorted(selected_classes))
    work_dir = work_root / f"{original_dir.name}__{slug}"

    # Clean previous working copy (idempotent)
    if work_dir.exists():
        shutil.rmtree(work_dir)

    # Build ID mapping: original_id → new_id
    old_to_new: Dict[int, int] = {}
    for new_idx, name in enumerate(selected_classes):
        if name in master_classes:
            old_to_new[master_classes.index(name)] = new_idx

    if not old_to_new:
        raise ValueError(
            f"Ninguna clase seleccionada ({selected_classes}) existe en "
            f"master_classes ({master_classes})."
        )

    log(f"\n🔄 Creando working copy: {len(selected_classes)} de "
        f"{len(master_classes)} clases")
    mapeo_str = {selected_classes[v]: f"{k}→{v}"
                 for k, v in old_to_new.items()}
    log(f"   Mapeo: {mapeo_str}")
    log(f"   Directorio: {work_dir}")

    stats: Dict[str, Any] = {}
    rng = random.Random(seed)

    for split in ("train", "valid", "test"):
        orig_img_dir, orig_lbl_dir = _resolve_yolo_split_dirs(original_dir, split)
        if not orig_lbl_dir.exists() or not orig_img_dir.exists():
            continue

        # Always output in Roboflow format: {split}/images, {split}/labels
        out_lbl_dir = work_dir / split / "labels"
        out_img_dir = work_dir / split / "images"
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        out_img_dir.mkdir(parents=True, exist_ok=True)

        positives = 0      # images with ≥1 selected-class annotation
        negatives_pool: List[str] = []  # stems with 0 matching annotations
        kept_annot = 0
        dropped_annot = 0

        for txt in sorted(orig_lbl_dir.glob("*.txt")):
            stem = txt.stem
            lines = txt.read_text().strip().splitlines()
            new_lines: List[str] = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_idx = int(parts[0])
                if cls_idx in old_to_new:
                    parts[0] = str(old_to_new[cls_idx])
                    new_lines.append(" ".join(parts))
                    kept_annot += 1
                else:
                    dropped_annot += 1

            if new_lines:
                # Write remapped label
                (out_lbl_dir / txt.name).write_text(
                    "\n".join(new_lines) + "\n"
                )
                # Symlink the image
                _symlink_image(orig_img_dir, out_img_dir, stem)
                positives += 1
            else:
                negatives_pool.append(stem)

        # Add controlled negative samples (empty label → background)
        max_neg = max(1, int(positives * negative_ratio))
        if len(negatives_pool) > max_neg:
            chosen_neg = rng.sample(negatives_pool, max_neg)
        else:
            chosen_neg = negatives_pool

        for stem in chosen_neg:
            # Empty label file = "no objects" (background image)
            (out_lbl_dir / f"{stem}.txt").write_text("")
            _symlink_image(orig_img_dir, out_img_dir, stem)

        stats[split] = {
            "positives": positives,
            "negatives_included": len(chosen_neg),
            "negatives_excluded": len(negatives_pool) - len(chosen_neg),
            "total_images": positives + len(chosen_neg),
            "annotations_kept": kept_annot,
            "annotations_dropped": dropped_annot,
        }
        s = stats[split]
        log(f"  {split:>5s}: {s['total_images']:>5d} imgs "
            f"({s['positives']} pos + {s['negatives_included']} neg), "
            f"{s['annotations_kept']} anotaciones, "
            f"{s['negatives_excluded']} neg excluidos")

    # Generate data.yaml inside working copy
    data_yaml_path = generate_data_yaml(str(work_dir), selected_classes)

    log(f"\n✅ Working copy lista: {work_dir}")
    return str(work_dir), data_yaml_path, stats


def _symlink_image(
    src_img_dir: Path, dst_img_dir: Path, stem: str,
) -> None:
    """Create a symlink for an image file (tries common extensions)."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        src = src_img_dir / f"{stem}{ext}"
        if src.exists():
            dst = dst_img_dir / f"{stem}{ext}"
            if not dst.exists():
                dst.symlink_to(src)
            return


def split_yolo_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    class_names: List[str],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> str:
    """Split a flat YOLO dataset into train/val/test.

    Expects ``source_dir`` to contain ``images/`` and ``labels/``.
    Creates the standard split structure under ``output_dir``.

    Returns:
        Path to the generated data.yaml.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    img_src = source_dir / "images"
    lbl_src = source_dir / "labels"

    # Collect stems
    all_stems = sorted(
        p.stem
        for p in img_src.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
    random.seed(seed)
    random.shuffle(all_stems)

    n = len(all_stems)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    splits = {
        "train": all_stems[:n_train],
        "val": all_stems[n_train : n_train + n_val],
        "test": all_stems[n_train + n_val :],
    }

    for split, stems in splits.items():
        for sub in ("images", "labels"):
            safe_mkdir(output_dir / sub / split)

        for stem in stems:
            # Find the actual image extension
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                src_img = img_src / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, output_dir / "images" / split / src_img.name)
                    break
            # Copy label
            src_lbl = lbl_src / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, output_dir / "labels" / split / src_lbl.name)

        log(f"  {split}: {len(stems)} imágenes")

    return generate_data_yaml(output_dir, class_names)


def delete_yolo_cache(dataset_dir: str | Path) -> None:
    """Remove *.cache files so Ultralytics regenerates them."""
    for cache in Path(dataset_dir).rglob("*.cache"):
        cache.unlink()
        log(f"  🗑️ Cache eliminado: {cache}")


# =====================================================================
#  TFRECORD FORMAT UTILITIES (for MobileNet + SSD)
# =====================================================================

def _bytes_feature(value: bytes):
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    import tensorflow as tf
    if not isinstance(value, (list, np.ndarray)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    import tensorflow as tf
    if not isinstance(value, (list, np.ndarray)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecord(
    images_dir: str | Path,
    annotations: List[Dict[str, Any]],
    output_path: str | Path,
    img_size: int = 224,
) -> int:
    """Write a TFRecord file from a list of image annotations.

    Each ``annotation`` dict must have:
      - ``file_name``: relative path to image
      - ``boxes``: list of [xc, yc, w, h] (normalised)
      - ``class_ids``: list of int class indices

    Returns:
        Number of examples written.
    """
    import tensorflow as tf
    import cv2

    safe_mkdir(Path(output_path).parent)
    images_dir = Path(images_dir)
    count = 0

    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for ann in annotations:
            img_path = images_dir / ann["file_name"]
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img_encoded = cv2.imencode(".jpg", img)[1].tobytes()

            boxes = np.array(ann.get("boxes", []), dtype=np.float32).flatten()
            class_ids = np.array(ann.get("class_ids", []), dtype=np.int64).flatten()
            num_objects = len(ann.get("class_ids", []))

            feature = {
                "image/encoded": _bytes_feature(img_encoded),
                "image/height": _int64_feature(img_size),
                "image/width": _int64_feature(img_size),
                "image/num_objects": _int64_feature(num_objects),
                "image/boxes": _float_feature(boxes.tolist() if boxes.size else []),
                "image/class_ids": _int64_feature(class_ids.tolist() if class_ids.size else []),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(example.SerializeToString())
            count += 1

    log(f"  ✅ TFRecord escrito: {output_path} ({count} ejemplos)")
    return count


def _parse_tfrecord_fn(example_proto, img_size: int, max_objects: int = 50):
    """Parse a single TFRecord example."""
    import tensorflow as tf

    feature_desc = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/num_objects": tf.io.FixedLenFeature([], tf.int64),
        "image/boxes": tf.io.VarLenFeature(tf.float32),
        "image/class_ids": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_desc)

    # Decode image
    image = tf.io.decode_jpeg(parsed["image/encoded"], channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0

    # Decode boxes and classes
    num_obj = tf.cast(parsed["image/num_objects"], tf.int32)
    boxes = tf.sparse.to_dense(parsed["image/boxes"])
    boxes = tf.reshape(boxes, [-1, 4])  # [N, 4]
    class_ids = tf.cast(tf.sparse.to_dense(parsed["image/class_ids"]), tf.int32)

    # Pad to max_objects
    pad_n = max_objects - tf.shape(boxes)[0]
    boxes = tf.pad(boxes, [[0, tf.maximum(pad_n, 0)], [0, 0]])[:max_objects]
    class_ids = tf.pad(class_ids, [[0, tf.maximum(pad_n, 0)]])[:max_objects]

    return image, boxes, class_ids, num_obj


def read_tfrecord_dataset(
    tfrecord_path: str | Path,
    img_size: int = 224,
    max_objects: int = 50,
):
    """Read a TFRecord file into a tf.data.Dataset.

    Returns dataset of (image, boxes, class_ids, num_objects).
    """
    import tensorflow as tf

    ds = tf.data.TFRecordDataset(str(tfrecord_path))
    ds = ds.map(
        lambda x: _parse_tfrecord_fn(x, img_size, max_objects),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds


def verify_tfrecord_dataset(dataset_dir: str | Path) -> Dict[str, Any]:
    """Verify a TFRecord dataset has the expected structure.

    Expects {train,val,test}.tfrecord + metadata.json in ``dataset_dir``.
    """
    dataset_dir = Path(dataset_dir)
    result: Dict[str, Any] = {"valid": True, "splits": {}, "issues": []}

    # Check metadata
    meta_path = dataset_dir / "metadata.json"
    if meta_path.exists():
        meta = read_json(str(meta_path))
        result["metadata"] = meta
    else:
        result["issues"].append(f"Falta metadata.json en {dataset_dir}")
        result["valid"] = False
        meta = None

    for split in ("train", "val", "test"):
        tfr = dataset_dir / f"{split}.tfrecord"
        info: Dict[str, Any] = {"exists": False, "num_samples": 0}
        if tfr.exists():
            info["exists"] = True
            # Count records
            try:
                import tensorflow as tf
                count = sum(1 for _ in tf.data.TFRecordDataset(str(tfr)))
                info["num_samples"] = count
            except Exception:
                info["num_samples"] = -1
        else:
            result["issues"].append(f"Falta {split}.tfrecord")
            result["valid"] = False
        result["splits"][split] = info

    # Print report
    log(f"\n📂 Dataset TFRecord: {dataset_dir.name}")
    if meta:
        log(f"   Clases: {meta.get('class_names', '?')}")
    for split, info in result["splits"].items():
        status = "✅" if info["exists"] else "❌"
        log(f"  {status} {split:>5s}: {info['num_samples']:>5d} ejemplos")
    if result["issues"]:
        log(f"\n  ⚠️  Problemas:")
        for issue in result["issues"]:
            log(f"      - {issue}")

    return result


# ── SSD Augmentation in tf.data ──────────────────────────────────────

def _augment_image(image, level: str = "medium"):
    """Apply augmentation to a single image tensor (graph-mode safe)."""
    import tensorflow as tf

    if level == "none":
        return image

    # Light: horizontal flip
    image = tf.image.random_flip_left_right(image)

    if level in ("medium", "heavy"):
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)

    if level == "heavy":
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.02)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def create_mobilenet_pipeline(
    tfrecord_path: str | Path,
    anchors: np.ndarray,
    num_classes: int,
    batch_size: int = 32,
    img_size: int = 224,
    augmentation_level: str = "none",
    shuffle: bool = True,
    buffer_size: int = 1000,
    max_objects: int = 50,
):
    """Build a complete tf.data pipeline from TFRecord → SSD targets.

    1. Read TFRecord
    2. Augment images (if training)
    3. Encode targets (objectness, class, bbox per anchor)
    4. Batch and prefetch

    Returns:
        tf.data.Dataset yielding (image_batch, {objectness, class_out, bbox_out}).
    """
    import tensorflow as tf

    ds = read_tfrecord_dataset(tfrecord_path, img_size, max_objects)

    # Augmentation (only makes sense for training)
    if augmentation_level != "none":
        ds = ds.map(
            lambda img, boxes, cls, n: (_augment_image(img, augmentation_level), boxes, cls, n),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Encode targets using numpy (via tf.py_function)
    anchors_tf = tf.constant(anchors, dtype=tf.float32)

    def _encode_fn(image, boxes, class_ids, num_obj):
        def _np_encode(boxes_np, class_ids_np, num_obj_np):
            n = int(num_obj_np)
            gt_boxes = boxes_np[:n]
            gt_cls = class_ids_np[:n]
            obj_t, cls_t, bbox_t = encode_targets(
                gt_boxes, gt_cls, anchors, num_classes
            )
            return (
                obj_t.astype(np.float32),
                cls_t.astype(np.float32),
                bbox_t.astype(np.float32),
            )

        obj, cls, bbox = tf.numpy_function(
            _np_encode,
            [boxes, class_ids, num_obj],
            [tf.float32, tf.float32, tf.float32],
        )
        n_anchors = anchors.shape[0]
        obj.set_shape([n_anchors, 1])
        cls.set_shape([n_anchors, num_classes])
        bbox.set_shape([n_anchors, 4])
        return image, {"objectness": obj, "class_out": cls, "bbox_out": bbox}

    ds = ds.map(_encode_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =====================================================================
#  ANCHOR GENERATION
# =====================================================================

def generate_anchors(
    feature_map_size: int = 7,
    scales: Optional[List[float]] = None,
    aspect_ratios: Optional[List[float]] = None,
) -> np.ndarray:
    """Generate SSD anchor boxes on a regular grid.

    Returns:
        Array of shape (H*W*S*A, 4) in normalised [cx, cy, w, h].
    """
    if scales is None:
        scales = [0.1, 0.2, 0.4]
    if aspect_ratios is None:
        aspect_ratios = [0.5, 1.0, 2.0]

    step = 1.0 / feature_map_size
    anchors = []
    for i in range(feature_map_size):
        for j in range(feature_map_size):
            cx = (j + 0.5) * step
            cy = (i + 0.5) * step
            for s in scales:
                for ar in aspect_ratios:
                    w = s * np.sqrt(ar)
                    h = s / np.sqrt(ar)
                    anchors.append([cx, cy, w, h])
    return np.array(anchors, dtype=np.float32)


def compute_anchor_statistics(
    anchors: np.ndarray,
) -> Dict[str, Any]:
    """Print and return summary statistics for generated anchors."""
    stats = {
        "total_anchors": anchors.shape[0],
        "min_w": float(anchors[:, 2].min()),
        "max_w": float(anchors[:, 2].max()),
        "min_h": float(anchors[:, 3].min()),
        "max_h": float(anchors[:, 3].max()),
    }
    log(f"⚓ Anchors: {stats['total_anchors']}  |  "
        f"W: [{stats['min_w']:.3f}, {stats['max_w']:.3f}]  |  "
        f"H: [{stats['min_h']:.3f}, {stats['max_h']:.3f}]")
    return stats


# =====================================================================
#  TARGET ENCODING (SSD)
# =====================================================================

def compute_iou_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    """IoU between two sets of [xc, yc, w, h] boxes.  Returns (N, M)."""
    b1_x1 = boxes1[:, 0:1] - boxes1[:, 2:3] / 2
    b1_y1 = boxes1[:, 1:2] - boxes1[:, 3:4] / 2
    b1_x2 = boxes1[:, 0:1] + boxes1[:, 2:3] / 2
    b1_y2 = boxes1[:, 1:2] + boxes1[:, 3:4] / 2
    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
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
    """Encode GT boxes to SSD anchor targets.

    Returns:
        (objectness (A,1), class (A,C), bbox (A,4))
    """
    A = anchors.shape[0]
    obj = np.zeros((A, 1), dtype=np.float32)
    cls = np.zeros((A, num_classes), dtype=np.float32)
    bbox = np.zeros((A, 4), dtype=np.float32)

    if gt_boxes.size == 0:
        return obj, cls, bbox

    iou_mat = compute_iou_matrix(gt_boxes, anchors)

    # Best anchor per GT
    best_a = np.argmax(iou_mat, axis=1)
    for gi, ai in enumerate(best_a):
        c = int(gt_classes[gi])
        obj[ai, 0] = 1.0
        cls[ai] = 0.0
        cls[ai, c] = 1.0
        bbox[ai] = gt_boxes[gi]

    # All anchors above threshold
    best_gt = np.argmax(iou_mat, axis=0)
    max_iou = np.max(iou_mat, axis=0)
    for ai in np.where(max_iou >= iou_threshold)[0]:
        gi = best_gt[ai]
        c = int(gt_classes[gi])
        obj[ai, 0] = 1.0
        cls[ai] = 0.0
        cls[ai, c] = 1.0
        bbox[ai] = gt_boxes[gi]

    return obj, cls, bbox


# =====================================================================
#  SHARED UTILITIES
# =====================================================================

def verify_dataset(
    dataset_dir: str | Path,
    model_family: str,
) -> Dict[str, Any]:
    """Dispatch to the correct verification function."""
    from .config import is_yolo_family, is_mobilenet_family

    if is_yolo_family(model_family):
        return verify_yolo_dataset(dataset_dir)
    elif is_mobilenet_family(model_family):
        return verify_tfrecord_dataset(dataset_dir)
    else:
        raise ValueError(f"Familia de modelo no soportada: {model_family}")


def get_class_distribution(
    dataset_dir: str | Path,
    model_family: str,
    class_names: List[str],
    split: str = "train",
) -> Dict[str, int]:
    """Get annotation counts per class."""
    from .config import is_yolo_family

    dataset_dir = Path(dataset_dir)
    if is_yolo_family(model_family):
        _, lbl_dir = _resolve_yolo_split_dirs(dataset_dir, split)
        return get_class_distribution_yolo(lbl_dir, class_names)
    else:
        # TFRecord: read metadata or iterate
        meta_path = dataset_dir / "metadata.json"
        if meta_path.exists():
            meta = read_json(str(meta_path))
            dist = meta.get("splits", {}).get(split, {}).get("class_distribution", {})
            if dist:
                return {n: dist.get(n, 0) for n in class_names}
        # Fallback: iterate TFRecord
        return _count_classes_tfrecord(
            dataset_dir / f"{split}.tfrecord", class_names
        )


def _count_classes_tfrecord(
    tfrecord_path: str | Path,
    class_names: List[str],
) -> Dict[str, int]:
    """Count per-class annotations in a TFRecord file."""
    counts = Counter()
    if not Path(tfrecord_path).exists():
        return {n: 0 for n in class_names}
    try:
        import tensorflow as tf
        ds = tf.data.TFRecordDataset(str(tfrecord_path))
        feat_desc = {
            "image/class_ids": tf.io.VarLenFeature(tf.int64),
        }
        for raw in ds:
            parsed = tf.io.parse_single_example(raw, feat_desc)
            ids = tf.sparse.to_dense(parsed["image/class_ids"]).numpy()
            for i in ids:
                if 0 <= i < len(class_names):
                    counts[class_names[int(i)]] += 1
    except Exception as exc:
        log(f"⚠️ Error contando clases en TFRecord: {exc}")
    return {n: counts.get(n, 0) for n in class_names}


def calculate_class_weights(
    distribution: Dict[str, int],
    method: str = "effective_samples",
) -> np.ndarray:
    """Compute class weights from distribution.

    Methods: inverse_freq, sqrt_inverse, effective_samples.
    """
    counts = np.array(list(distribution.values()), dtype=np.float32)
    num_classes = len(counts)
    weights = np.ones(num_classes, dtype=np.float32)

    if counts.sum() == 0:
        return weights

    if method == "inverse_freq":
        max_c = counts.max()
        for i, c in enumerate(counts):
            weights[i] = max_c / c if c > 0 else 1.0

    elif method == "sqrt_inverse":
        max_c = counts.max()
        for i, c in enumerate(counts):
            weights[i] = np.sqrt(max_c / c) if c > 0 else 1.0

    elif method == "effective_samples":
        beta = 0.9999
        for i, c in enumerate(counts):
            if c > 0:
                eff = (1 - beta ** c) / (1 - beta)
                weights[i] = 1.0 / eff
        weights = weights / weights.sum() * num_classes

    log(f"\n📊 Class weights ({method}):")
    for name, w in zip(distribution.keys(), weights):
        log(f"   {name}: {distribution[name]:>5d} samples → weight={w:.3f}")
    return weights


def plot_class_distribution(
    distribution: Dict[str, int],
    save_path: Optional[str | Path] = None,
    title: str = "Distribución de Clases",
) -> None:
    """Bar chart of class counts."""
    import matplotlib.pyplot as plt

    names = list(distribution.keys())
    values = list(distribution.values())

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    bars = ax.bar(names, values, color="#4C72B0", edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(v), ha="center", va="bottom", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Anotaciones")
    ax.set_xlabel("Clase")
    try:
        plt.tight_layout()
    except Exception:
        pass  # bbox_inches="tight" in savefig handles layout
    if save_path:
        safe_mkdir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"  💾 Gráfico guardado: {save_path}")
    plt.show()
