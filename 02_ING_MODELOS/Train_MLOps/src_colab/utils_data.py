"""Unified data pipeline for YOLO-format datasets — Cycle 2 (PyTorch only).

Provides ``IODCDataset``, a PyTorch Dataset class with:
- YOLO-format label loading (class_id cx cy w h)
- Albumentations-based augmentation
- Progressive resizing via ``set_image_size()``
- Optional Mosaic / Mixup

Also retains YOLO dataset verification utilities from Cycle 1.
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

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUM_AVAILABLE = True
except ImportError:
    ALBUM_AVAILABLE = False

from .utils_io import log, safe_mkdir, write_json, write_yaml, read_json, file_exists


# =====================================================================
#  YOLO FORMAT VERIFICATION
# =====================================================================

def verify_yolo_dataset(dataset_dir: str | Path) -> Dict[str, Any]:
    """Verify structure and integrity of a YOLO-format dataset."""
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
                p.stem for p in img_dir.iterdir()
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


def verify_dataset(dataset_dir: str | Path, family: str) -> Dict[str, Any]:
    """Unified verification dispatcher (all families use YOLO format in Cycle 2)."""
    return verify_yolo_dataset(dataset_dir)


# =====================================================================
#  DATA YAML GENERATION
# =====================================================================

def generate_data_yaml(
    dataset_dir: str | Path,
    class_names: List[str],
    output_path: Optional[str | Path] = None,
) -> str:
    """Generate a data.yaml file in YOLO format."""
    dataset_dir = Path(dataset_dir)
    if output_path is None:
        output_path = dataset_dir / "data.yaml"
    else:
        output_path = Path(output_path)

    data = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    write_yaml(data, str(output_path))
    log(f"📄 data.yaml generado: {output_path}")
    return str(output_path)


def delete_yolo_cache(dataset_dir: str | Path) -> None:
    """Remove .cache files from a YOLO dataset directory."""
    dataset_dir = Path(dataset_dir)
    for cache_file in dataset_dir.rglob("*.cache"):
        cache_file.unlink()
        log(f"  🗑️  Eliminado cache: {cache_file}")


# =====================================================================
#  CLASS DISTRIBUTION & WEIGHTS
# =====================================================================

def get_class_distribution_yolo(
    dataset_dir: str | Path,
    class_names: List[str],
) -> Dict[str, Dict[str, int]]:
    """Get per-class annotation counts for each YOLO split."""
    dataset_dir = Path(dataset_dir)
    dist: Dict[str, Dict[str, int]] = {}

    for split in ("train", "valid", "test"):
        lbl_dir = dataset_dir / split / "labels"
        counts: Dict[str, int] = {name: 0 for name in class_names}
        if lbl_dir.exists():
            for txt_file in lbl_dir.glob("*.txt"):
                with open(txt_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if 0 <= cls_id < len(class_names):
                                counts[class_names[cls_id]] += 1
        dist[split] = counts

    return dist


def get_class_distribution(
    dataset_dir: str | Path,
    family: str,
    class_names: List[str],
) -> Dict[str, Dict[str, int]]:
    """Get class distribution (all families use YOLO format)."""
    return get_class_distribution_yolo(dataset_dir, class_names)


def calculate_class_weights(
    distribution: Dict[str, Dict[str, int]],
    method: str = "inverse_freq",
) -> List[float]:
    """Calculate class weights from the training split distribution."""
    train_dist = distribution.get("train", {})
    counts = list(train_dist.values())
    total = sum(counts) or 1

    if method == "inverse_freq":
        n_classes = len(counts)
        weights = [total / (n_classes * max(c, 1)) for c in counts]
    elif method == "sqrt_inverse":
        weights = [1.0 / max(np.sqrt(c), 1.0) for c in counts]
    elif method == "effective_samples":
        beta = 0.9999
        weights = [(1.0 - beta) / max(1.0 - beta ** c, 1e-8) for c in counts]
    else:
        weights = [1.0] * len(counts)

    # Normalize so max weight = 1.0
    max_w = max(weights) if weights else 1.0
    weights = [w / max_w for w in weights]
    return weights


def plot_class_distribution(
    distribution: Dict[str, Dict[str, int]],
    save_path: Optional[str] = None,
    title: str = "Distribución de Clases",
) -> None:
    """Plot a grouped bar chart of class distribution per split."""
    import matplotlib.pyplot as plt

    classes = list(distribution.get("train", {}).keys())
    splits = [s for s in ("train", "valid", "test") if s in distribution]

    x = np.arange(len(classes))
    width = 0.8 / max(len(splits), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, split in enumerate(splits):
        counts = [distribution[split].get(c, 0) for c in classes]
        ax.bar(x + i * width, counts, width, label=split, alpha=0.85)

    ax.set_xlabel("Clase")
    ax.set_ylabel("Anotaciones")
    ax.set_title(title)
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        log(f"  📊 Guardado: {save_path}")
    plt.close(fig)


# =====================================================================
#  YOLO WORKING COPY (class subsetting)
# =====================================================================

def create_yolo_working_copy(
    original_dir: str | Path,
    master_classes: List[str],
    selected_classes: List[str],
) -> Tuple[str, str, Dict]:
    """Create a filtered copy of a YOLO dataset for a subset of classes."""
    original_dir = Path(original_dir)
    work_dir = original_dir.parent / f"{original_dir.name}_working"

    # Build class mapping: master_id → new_id (or -1 if not selected)
    class_map = {}
    for i, name in enumerate(master_classes):
        if name in selected_classes:
            class_map[i] = selected_classes.index(name)
        else:
            class_map[i] = -1

    stats = {"kept": 0, "removed": 0, "empty_images": 0}

    for split in ("train", "valid", "test"):
        src_img = original_dir / split / "images"
        src_lbl = original_dir / split / "labels"
        dst_img = work_dir / split / "images"
        dst_lbl = work_dir / split / "labels"
        safe_mkdir(str(dst_img))
        safe_mkdir(str(dst_lbl))

        if not src_lbl.exists():
            continue

        for txt_file in src_lbl.glob("*.txt"):
            new_lines = []
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    new_id = class_map.get(cls_id, -1)
                    if new_id >= 0:
                        new_lines.append(f"{new_id} " + " ".join(parts[1:]))
                        stats["kept"] += 1
                    else:
                        stats["removed"] += 1

            # Copy image + write filtered labels
            img_stem = txt_file.stem
            for ext in (".jpg", ".jpeg", ".png"):
                img_src = src_img / f"{img_stem}{ext}"
                if img_src.exists():
                    shutil.copy2(str(img_src), str(dst_img / img_src.name))
                    break

            dst_txt = dst_lbl / txt_file.name
            with open(dst_txt, "w") as f:
                f.write("\n".join(new_lines))

            if not new_lines:
                stats["empty_images"] += 1

    data_yaml_path = generate_data_yaml(work_dir, selected_classes)
    log(f"📂 Working copy: {work_dir}")
    log(f"   Kept: {stats['kept']} | Removed: {stats['removed']} | "
        f"Empty: {stats['empty_images']}")
    return str(work_dir), data_yaml_path, stats


def split_yolo_dataset(
    dataset_dir: str | Path,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> None:
    """Split images into train/valid/test (in-place, from a flat folder)."""
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        log(f"⚠️ No images/ dir found in {dataset_dir}")
        return

    all_imgs = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    random.seed(seed)
    random.shuffle(all_imgs)

    n = len(all_imgs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        "train": all_imgs[:n_train],
        "valid": all_imgs[n_train : n_train + n_val],
        "test": all_imgs[n_train + n_val :],
    }

    for split_name, imgs in splits.items():
        split_img = dataset_dir / split_name / "images"
        split_lbl = dataset_dir / split_name / "labels"
        safe_mkdir(str(split_img))
        safe_mkdir(str(split_lbl))
        for img_path in imgs:
            shutil.move(str(img_path), str(split_img / img_path.name))
            lbl_src = img_path.with_suffix(".txt")
            if lbl_src.exists():
                shutil.move(str(lbl_src), str(split_lbl / lbl_src.name))

    log(f"✅ Split: train={len(splits['train'])} | val={len(splits['valid'])} | "
        f"test={len(splits['test'])}")


# =====================================================================
#  IODC PYTORCH DATASET
# =====================================================================

class IODCDataset(Dataset):
    """PyTorch Dataset for YOLO-format object detection.

    Supports:
    - Dynamic image resizing via ``set_image_size()``
    - Albumentations augmentation pipeline
    - YOLO label format (class_id cx cy w h, normalized)

    Args:
        dataset_dir: Root of the YOLO dataset.
        split: 'train', 'valid', or 'test'.
        img_size: Initial image size (square).
        class_names: List of class names.
        augment: Whether to apply augmentation (train only).
        aug_config: Dict of augmentation parameters.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        img_size: int = 640,
        class_names: Optional[List[str]] = None,
        augment: bool = False,
        aug_config: Optional[Dict[str, Any]] = None,
    ):
        assert TORCH_AVAILABLE, "PyTorch is required for IODCDataset"
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.img_size = img_size
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)
        self.augment = augment
        self.aug_config = aug_config or {}

        # Locate images
        img_dir = self.dataset_dir / split / "images"
        self.image_paths = sorted([
            str(p) for p in img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ])

        self.label_dir = str(self.dataset_dir / split / "labels")

        # Build augmentation pipeline
        self._build_transforms()

    def _build_transforms(self) -> None:
        """Build Albumentations transform pipeline."""
        if not ALBUM_AVAILABLE:
            self._transform = None
            return

        cfg = self.aug_config
        transforms = []

        if self.augment:
            transforms.extend([
                A.HorizontalFlip(p=cfg.get("aug_hflip_prob", 0.5)),
                A.RandomBrightnessContrast(
                    brightness_limit=cfg.get("aug_brightness_limit", 0.2),
                    contrast_limit=cfg.get("aug_contrast_limit", 0.2),
                    p=0.5,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=cfg.get("aug_hue_shift_limit", 20),
                    sat_shift_limit=cfg.get("aug_sat_shift_limit", 30),
                    val_shift_limit=cfg.get("aug_val_shift_limit", 20),
                    p=0.5,
                ),
                A.ShiftScaleRotate(
                    shift_limit=cfg.get("aug_shift_limit", 0.1),
                    scale_limit=cfg.get("aug_scale_limit", 0.2),
                    rotate_limit=cfg.get("aug_rotate_limit", 15),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
            ])

        # Always resize + normalize + convert to tensor
        transforms.extend([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),  # [0,1]
            ToTensorV2(),
        ])

        self._transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2,
            ),
        )

    def set_image_size(self, new_size: int) -> None:
        """Change the target image size (for progressive resizing)."""
        self.img_size = new_size
        self._build_transforms()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        """Return (image_tensor, targets) where targets is dict with boxes/labels."""
        # Load image (BGR → RGB)
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load YOLO labels
        stem = Path(img_path).stem
        label_path = os.path.join(self.label_dir, f"{stem}.txt")

        bboxes = []
        class_labels = []
        if os.path.isfile(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), \
                            float(parts[3]), float(parts[4])
                        # Clamp to [0, 1] for safety
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        w = max(0.001, min(1.0, w))
                        h = max(0.001, min(1.0, h))
                        bboxes.append([cx, cy, w, h])
                        class_labels.append(cls_id)

        # Apply transforms
        if self._transform is not None and bboxes:
            transformed = self._transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            img_tensor = transformed["image"]  # (C, H, W) float32
            bboxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]
        elif self._transform is not None:
            transformed = self._transform(
                image=img, bboxes=[], class_labels=[],
            )
            img_tensor = transformed["image"]
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Build targets dict
        if bboxes:
            boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(class_labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        targets = {
            "boxes": boxes_tensor,      # (N, 4) YOLO format (cx, cy, w, h)
            "labels": labels_tensor,     # (N,) class indices
            "image_path": img_path,
        }

        return img_tensor, targets


def iodc_collate_fn(
    batch: List[Tuple],
) -> Tuple:
    """Custom collate function for IODCDataset with variable-length targets."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def create_dataloader(
    dataset_dir: str | Path,
    split: str,
    img_size: int,
    class_names: List[str],
    batch_size: int = 32,
    augment: bool = False,
    aug_config: Optional[Dict[str, Any]] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Factory function to create a DataLoader from a YOLO dataset."""
    ds = IODCDataset(
        dataset_dir=dataset_dir,
        split=split,
        img_size=img_size,
        class_names=class_names,
        augment=augment,
        aug_config=aug_config,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and split == "train",
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=iodc_collate_fn,
        drop_last=(split == "train"),
    )


# =====================================================================
#  GT VISUALIZATION
# =====================================================================

def visualize_gt_samples_per_class(
    dataset_dir: str | Path,
    class_names: List[str],
    split: str = "train",
    samples_per_class: int = 3,
    title: str = "GT Samples",
    save_path: Optional[str] = None,
) -> None:
    """Visualize ground truth bounding boxes for each class."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"

    # Collect samples per class
    class_samples: Dict[int, List[Tuple[str, List]]] = {
        i: [] for i in range(len(class_names))
    }

    for txt_file in sorted(lbl_dir.glob("*.txt")):
        with open(txt_file) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id < len(class_names):
                    # Find corresponding image
                    stem = txt_file.stem
                    for ext in (".jpg", ".jpeg", ".png"):
                        img_path = img_dir / f"{stem}{ext}"
                        if img_path.exists():
                            bbox = [float(p) for p in parts[1:5]]
                            class_samples[cls_id].append((str(img_path), bbox))
                            break

    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes, samples_per_class,
        figsize=(4 * samples_per_class, 4 * n_classes),
    )
    if n_classes == 1:
        axes = [axes]

    for cls_id in range(n_classes):
        samples = class_samples[cls_id][:samples_per_class]
        for j in range(samples_per_class):
            ax = axes[cls_id][j] if samples_per_class > 1 else axes[cls_id]
            if j < len(samples):
                img_path, bbox = samples[j]
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                ax.imshow(img)

                cx, cy, bw, bh = bbox
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                rect = patches.Rectangle(
                    (x1, y1), bw * w, bh * h,
                    linewidth=2, edgecolor="lime", facecolor="none",
                )
                ax.add_patch(rect)
            ax.set_title(
                f"{class_names[cls_id]}" if j == 0 else "",
                fontsize=10,
            )
            ax.axis("off")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"  🖼️  Guardado: {save_path}")
    plt.close(fig)
