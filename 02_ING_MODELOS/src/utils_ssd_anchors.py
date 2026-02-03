"""Anchor utilities for SSD-style models."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

try:
    from .utils_io import log
except ImportError:  # fallback when running as a script/notebook
    from utils_io import log


def compute_kmeans_anchors(json_path: str, n_clusters: int = 6) -> np.ndarray:
    """Compute anchors (w, h) via KMeans from COCO json.

    Returns normalized (w, h) in [0, 1].
    """
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn es requerido para KMeans anchors.") from exc

    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_dims = {img["id"]: (img["width"], img["height"]) for img in data["images"]}
    wh_list = []
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_dims:
            continue
        img_w, img_h = img_dims[img_id]
        w, h = ann["bbox"][2], ann["bbox"][3]
        wh_list.append([w / img_w, h / img_h])

    if not wh_list:
        raise RuntimeError("No se encontraron bboxes en el JSON.")

    wh_array = np.array(wh_list, dtype=np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(wh_array)
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    return anchors


def derive_aspect_ratios(anchors: np.ndarray) -> List[float]:
    """Derive aspect ratios (w/h) from anchors."""
    ratios = anchors[:, 0] / np.clip(anchors[:, 1], 1e-6, None)
    ratios = np.clip(ratios, 0.1, 10.0)
    return sorted(ratios.tolist())


def generate_anchors(
    feature_map_size: int,
    scales: Iterable[float],
    aspect_ratios: Iterable[float],
) -> np.ndarray:
    """Generate anchors for a single feature map.

    Returns anchors in (xc, yc, w, h) normalized to [0,1].
    """
    scales = list(scales)
    aspect_ratios = list(aspect_ratios)
    anchors = []

    step = 1.0 / feature_map_size
    for i in range(feature_map_size):
        for j in range(feature_map_size):
            cx = (j + 0.5) * step
            cy = (i + 0.5) * step
            for s in scales:
                for ar in aspect_ratios:
                    w = s * np.sqrt(ar)
                    h = s / np.sqrt(ar)
                    anchors.append([cx, cy, w, h])

    anchors_arr = np.array(anchors, dtype=np.float32)
    return anchors_arr


def log_anchor_summary(anchors: np.ndarray) -> None:
    sizes = anchors[:, 2] * anchors[:, 3]
    log(f"📦 Anchors: {len(anchors)}")
    log(f"   Area min/max: {sizes.min():.4f} / {sizes.max():.4f}")
    log(f"   w min/max: {anchors[:, 2].min():.4f} / {anchors[:, 2].max():.4f}")
    log(f"   h min/max: {anchors[:, 3].min():.4f} / {anchors[:, 3].max():.4f}")
