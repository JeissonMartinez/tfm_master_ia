"""COCO -> YOLO dataset conversion utilities for YOLO26.

Handles conversion of COCO JSON format to YOLO txt format,
with proper directory structure for Ultralytics training.
"""
from __future__ import annotations

import os
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .utils_io import safe_read_json, safe_mkdir, safe_write_text, log


def _normalize_bbox(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert COCO bbox [x_min, y_min, w, h] to YOLO format [x_center, y_center, w, h] normalized."""
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
    filter_classes: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Convert one COCO subset to YOLO folder structure.

    Args:
        json_path: Path to COCO JSON annotation file
        img_source_dir: Directory containing source images
        yolo_root: Root directory for YOLO dataset output
        subset_name: Name of subset (train, val, test)
        filter_classes: Optional list of class names to include (filters out others)

    Returns:
        Tuple of (class_names, class_counts)
    """
    log(f"\n🔄 Procesando subset: {subset_name.upper()}...")

    data = safe_read_json(json_path)
    if data is None:
        log(f"⚠️ Saltando {subset_name}: no se pudo leer {json_path}")
        return [], {}

    dest_img_dir = os.path.join(yolo_root, "images", subset_name)
    dest_lbl_dir = os.path.join(yolo_root, "labels", subset_name)
    safe_mkdir(dest_img_dir)
    safe_mkdir(dest_lbl_dir)

    # Extract categories from JSON
    cats = sorted(data.get("categories", []), key=lambda x: x.get("id", 0))
    if not cats:
        log("⚠️ No hay categorías en el JSON.")
        return [], {}

    # Build mapping from original category to filtered index
    # If filter_classes is provided, only include those classes
    if filter_classes:
        # Create mapping: original cat_id -> new index (0, 1, 2, ...)
        cat_id_to_new_idx = {}
        cat_id_to_name = {}
        for cat in cats:
            if cat["name"] in filter_classes:
                new_idx = filter_classes.index(cat["name"])  # Use order from filter_classes
                cat_id_to_new_idx[cat["id"]] = new_idx
                cat_id_to_name[cat["id"]] = cat["name"]
        class_names = filter_classes.copy()  # Use the provided order
        
        # Print detailed mapping info (like MobileNet does)
        log(f"   🔍 Filtrando a {len(filter_classes)} clases: {filter_classes}")
        log(f"   📋 Categorías originales en JSON: {[(c['id'], c['name']) for c in cats]}")
        log(f"   📋 Categorías seleccionadas: {cat_id_to_name}")
        log(f"   🔗 Mapeo de IDs: {cat_id_to_new_idx}")
        log(f"   ✅ Nombres de clase (ordenados): {class_names}")
    else:
        # No filtering - use all classes from JSON
        cat_id_to_new_idx = {cat["id"]: i for i, cat in enumerate(cats)}
        class_names = [cat["name"] for cat in cats]
        log(f"   📋 Usando todas las clases: {class_names}")

    # Group annotations by image
    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    # Track class distribution
    class_counts: Dict[str, int] = {name: 0 for name in class_names}
    images_processed = 0
    images_skipped = 0
    annotations_filtered = 0

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
            except Exception as exc:
                log(f"⚠️ Error copiando {src}: {exc}")
                images_skipped += 1
                continue
        else:
            images_skipped += 1
            continue

        # Create label file
        txt_path = os.path.join(dest_lbl_dir, os.path.splitext(file_name)[0] + ".txt")
        lines = []
        for ann in anns_by_img.get(img_id, []):
            cat_id = ann.get("category_id")
            
            # Skip if category is not in our filtered list
            if cat_id not in cat_id_to_new_idx:
                annotations_filtered += 1
                continue
            
            x_c, y_c, w_n, h_n = _normalize_bbox(
                ann["bbox"], 
                img_info["width"], 
                img_info["height"]
            )
            cls_idx = cat_id_to_new_idx[cat_id]
            lines.append(f"{cls_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
            
            # Update class counts
            class_name = class_names[cls_idx]
            class_counts[class_name] += 1

        safe_write_text(txt_path, "".join(lines))
        images_processed += 1

    log(f"   ✅ {images_processed} imágenes procesadas, {images_skipped} omitidas")
    if annotations_filtered > 0:
        log(f"   🔍 {annotations_filtered} anotaciones filtradas (clases excluidas)")
    log(f"   📊 Distribución: {class_counts}")

    return class_names, class_counts


def create_yolo_dataset(
    coco_train_json: str,
    coco_val_json: str,
    coco_test_json: str,
    output_dir: str,
    class_names: Optional[List[str]] = None,
    copy_images: bool = True,
    verbose: bool = True,
) -> Optional[str]:
    """Create complete YOLO dataset from COCO JSON files.

    Args:
        coco_train_json: Path to training COCO JSON (annotations file)
        coco_val_json: Path to validation COCO JSON
        coco_test_json: Path to test COCO JSON
        output_dir: Output directory for YOLO dataset
        class_names: Optional list of class names to filter (if None, use all from JSON)
        copy_images: Whether to copy images to output directory
        verbose: Whether to print progress

    Returns:
        Path to data.yaml file, or None if failed
    """
    if verbose:
        log(f"\n{'='*60}")
        log("🔧 CREANDO DATASET YOLO26")
        log(f"{'='*60}")
        log(f"📁 Destino: {output_dir}")

    # Infer image directories from JSON paths (images are in same folder as JSON)
    train_img_dir = os.path.dirname(coco_train_json)
    val_img_dir = os.path.dirname(coco_val_json)
    test_img_dir = os.path.dirname(coco_test_json)

    # Clean existing directory
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            if verbose:
                log("🗑️ Directorio existente eliminado")
        except Exception as exc:
            log(f"⚠️ No se pudo limpiar {output_dir}: {exc}")

    # Convert each split (pass filter_classes to filter annotations)
    detected_names, train_counts = convert_single_set(
        coco_train_json, train_img_dir, output_dir, "train", filter_classes=class_names
    )
    _, val_counts = convert_single_set(
        coco_val_json, val_img_dir, output_dir, "val", filter_classes=class_names
    )
    _, test_counts = convert_single_set(
        coco_test_json, test_img_dir, output_dir, "test", filter_classes=class_names
    )

    # Use provided class_names or detected ones
    final_class_names = class_names if class_names else detected_names

    # Create data.yaml
    yaml_content = (
        f"# YOLO26 Dataset Configuration\n"
        f"# Generated for ESP32-S3 deployment\n\n"
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: {len(final_class_names)}\n"
        f"names: {final_class_names}\n"
    )
    yaml_path = os.path.join(output_dir, "data.yaml")
    safe_write_text(yaml_path, yaml_content)
    
    if verbose:
        log(f"\n{'='*60}")
        log(f"✅ Dataset YOLO26 creado correctamente")
        log(f"📄 data.yaml: {yaml_path}")
        log(f"📦 Clases: {final_class_names}")
        log(f"{'='*60}\n")

    return yaml_path


def get_class_distribution(coco_json_path: str) -> Dict[str, int]:
    """Get class distribution from a COCO JSON annotation file.

    Args:
        coco_json_path: Path to COCO JSON annotation file

    Returns:
        Dictionary with {class_name: count}
    """
    data = safe_read_json(coco_json_path)
    if data is None:
        log(f"⚠️ No se pudo leer: {coco_json_path}")
        return {}

    # Build category mapping
    cats = data.get("categories", [])
    cat_id_to_name = {cat["id"]: cat["name"] for cat in cats}
    
    # Count annotations per class
    class_counts: Dict[str, int] = {cat["name"]: 0 for cat in cats}
    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id")
        if cat_id in cat_id_to_name:
            class_counts[cat_id_to_name[cat_id]] += 1

    return class_counts


def get_class_distribution_yolo(yolo_root: str) -> Dict[str, Dict[str, int]]:
    """Get class distribution from an existing YOLO dataset.

    Args:
        yolo_root: Root directory of YOLO dataset

    Returns:
        Dictionary with distribution per split {split: {class_name: count}}
    """
    import yaml
    
    yaml_path = os.path.join(yolo_root, "data.yaml")
    if not os.path.exists(yaml_path):
        log(f"⚠️ data.yaml no encontrado: {yaml_path}")
        return {}

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    class_names = config.get("names", [])
    if isinstance(class_names, dict):
        class_names = list(class_names.values())

    distribution = {}
    for split in ["train", "val", "test"]:
        labels_dir = os.path.join(yolo_root, "labels", split)
        if not os.path.exists(labels_dir):
            continue

        counts = {name: 0 for name in class_names}
        for txt_file in os.listdir(labels_dir):
            if not txt_file.endswith(".txt"):
                continue
            with open(os.path.join(labels_dir, txt_file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_idx = int(parts[0])
                        if 0 <= cls_idx < len(class_names):
                            counts[class_names[cls_idx]] += 1

        distribution[split] = counts

    return distribution


def calculate_class_weights(
    distribution: Dict[str, int], 
    class_names: Optional[List[str]] = None,
    method: str = "inverse",
) -> Dict[str, float]:
    """Calculate class weights for handling imbalanced datasets.

    Args:
        distribution: Dictionary of {class_name: count}
        class_names: Optional list to filter/order classes (if None, use all from distribution)
        method: 'inverse' for inverse frequency, 'effective' for effective number

    Returns:
        Dictionary of {class_name: weight}
    """
    if not distribution:
        return {}

    # Filter by class_names if provided
    if class_names:
        distribution = {k: distribution.get(k, 0) for k in class_names}

    total = sum(distribution.values())
    num_classes = len(distribution)

    if total == 0:
        return {cls_name: 1.0 for cls_name in distribution}

    weights = {}
    if method == "inverse":
        # Inverse frequency weighting
        for cls_name, count in distribution.items():
            if count > 0:
                weights[cls_name] = total / (num_classes * count)
            else:
                weights[cls_name] = 1.0
    elif method == "effective":
        # Effective number of samples (beta = 0.9999)
        beta = 0.9999
        for cls_name, count in distribution.items():
            if count > 0:
                effective_num = (1.0 - beta ** count) / (1.0 - beta)
                weights[cls_name] = 1.0 / effective_num
            else:
                weights[cls_name] = 1.0
    else:
        weights = {cls_name: 1.0 for cls_name in distribution}

    # Normalize weights so mean = 1.0
    mean_weight = sum(weights.values()) / len(weights)
    weights = {k: v / mean_weight for k, v in weights.items()}

    return weights


def verify_yolo_labels(
    yolo_root: str,
    class_names: List[str],
    num_samples: int = 5,
    verbose: bool = True,
) -> Dict:
    """Verify YOLO labels are correctly formatted and mapped.
    
    Reads sample label files and verifies:
    - Class indices are within valid range
    - Bounding box coordinates are normalized (0-1)
    - Distribution matches expected classes
    
    Args:
        yolo_root: Root directory of YOLO dataset
        class_names: Expected class names (ordered by index)
        num_samples: Number of sample files to inspect per split
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with verification results
    """
    import random
    
    results = {
        "valid": True,
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    
    if verbose:
        log(f"\n{'='*60}")
        log("🔍 VERIFICACIÓN DE LABELS YOLO")
        log(f"{'='*60}")
        log(f"   📂 Dataset: {yolo_root}")
        log(f"   🏷️ Clases esperadas ({len(class_names)}): {class_names}")
    
    for split in ["train", "val", "test"]:
        labels_dir = os.path.join(yolo_root, "labels", split)
        images_dir = os.path.join(yolo_root, "images", split)
        
        if not os.path.exists(labels_dir):
            if verbose:
                log(f"\n   ⚠️ {split}: directorio de labels no existe")
            results["warnings"].append(f"{split}: labels directory not found")
            continue
        
        # Get label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
        
        if not label_files:
            if verbose:
                log(f"\n   ⚠️ {split}: sin archivos de labels")
            results["warnings"].append(f"{split}: no label files found")
            continue
        
        # Count class distribution
        class_counts = {i: 0 for i in range(len(class_names))}
        total_boxes = 0
        invalid_indices = []
        invalid_coords = []
        
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_idx = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])
                        
                        # Verify class index
                        if cls_idx < 0 or cls_idx >= len(class_names):
                            invalid_indices.append((label_file, line_num, cls_idx))
                            results["valid"] = False
                        else:
                            class_counts[cls_idx] += 1
                        
                        # Verify normalized coordinates
                        if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            invalid_coords.append((label_file, line_num, (x_c, y_c, w, h)))
                        
                        total_boxes += 1
        
        # Store results for this split
        split_results = {
            "num_images": len(label_files),
            "num_boxes": total_boxes,
            "class_distribution": {class_names[i]: class_counts[i] for i in range(len(class_names))},
            "invalid_indices": len(invalid_indices),
            "invalid_coords": len(invalid_coords),
        }
        results["splits"][split] = split_results
        
        if verbose:
            log(f"\n   📁 {split.upper()}:")
            log(f"      Imágenes: {len(label_files)}")
            log(f"      Bounding boxes: {total_boxes}")
            log(f"      Distribución por clase:")
            for i, name in enumerate(class_names):
                count = class_counts[i]
                pct = (count / total_boxes * 100) if total_boxes > 0 else 0
                log(f"         [{i}] {name}: {count} ({pct:.1f}%)")
            
            if invalid_indices:
                log(f"      ❌ Índices inválidos: {len(invalid_indices)}")
                for file, line, idx in invalid_indices[:3]:
                    log(f"         {file}:{line} → índice {idx} (máx: {len(class_names)-1})")
                results["errors"].append(f"{split}: {len(invalid_indices)} invalid class indices")
            
            if invalid_coords:
                log(f"      ⚠️ Coordenadas fuera de rango: {len(invalid_coords)}")
        
        # Show sample labels
        if verbose and num_samples > 0:
            log(f"\n      📋 Muestra de {min(num_samples, len(label_files))} labels:")
            sample_files = random.sample(label_files, min(num_samples, len(label_files)))
            
            for label_file in sample_files:
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, "r") as f:
                    lines = f.readlines()
                
                log(f"         {label_file}:")
                for line in lines[:3]:  # Max 3 lines per file
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_idx = int(parts[0])
                        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else "???"
                        log(f"            idx={cls_idx} → '{cls_name}' | bbox=[{parts[1][:6]}, {parts[2][:6]}, {parts[3][:6]}, {parts[4][:6]}]")
                if len(lines) > 3:
                    log(f"            ... ({len(lines)-3} más)")
    
    # Summary
    if verbose:
        log(f"\n{'='*60}")
        if results["valid"]:
            log("✅ VERIFICACIÓN EXITOSA - Labels correctamente formateados")
        else:
            log("❌ VERIFICACIÓN FALLIDA - Se encontraron errores")
            for error in results["errors"]:
                log(f"   • {error}")
        log(f"{'='*60}\n")
    
    return results
