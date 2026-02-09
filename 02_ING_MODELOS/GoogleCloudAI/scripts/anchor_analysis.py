#!/usr/bin/env python3
"""Anchor Analysis for TFM Dataset — K-means with IoU distance.

Loads COCO annotations, extracts all GT bbox (w,h) pairs normalised to
[0,1], runs k-means clustering using 1-IoU as the distance metric, and
compares the resulting centroids with the current hand-crafted anchors.

Output:
  - Descriptive statistics (per class + global)
  - Optimal anchor (w,h) centroids for k=5..8
  - Coverage comparison: mean/median maxIoU per GT with current vs proposed
  - Recommended anchor_sizes & anchor_ratios for YAML configs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
COCO_ROOT = Path("/Users/admin/Documents/TFM_UNIR/01_ING_DATOS/Datasets/"
                  "TFM_Dataset.v1-v1_2026-02-06_5-48pm.coco")

# Current anchor config from YAML
CURRENT_SIZES = [0.03, 0.06, 0.12, 0.24, 0.38, 0.55, 0.75]
CURRENT_RATIOS = [0.33, 0.5, 1.0, 2.0, 3.0]


# ── Helper: IoU between (w,h) pairs ────────────────────────────────
def _wh_iou(wh1: np.ndarray, wh2: np.ndarray) -> np.ndarray:
    """IoU between two sets of (w,h) anchors centred at the origin.

    Args:
        wh1: (N, 2) — widths & heights
        wh2: (M, 2) — widths & heights

    Returns:
        (N, M) IoU matrix where boxes are centred at (0.5, 0.5).
    """
    inter_w = np.minimum(wh1[:, 0:1], wh2[:, 0])  # (N, M)
    inter_h = np.minimum(wh1[:, 1:2], wh2[:, 1])  # (N, M)
    inter = inter_w * inter_h
    area1 = (wh1[:, 0] * wh1[:, 1])[:, None]  # (N, 1)
    area2 = (wh2[:, 0] * wh2[:, 1])[None, :]  # (1, M)
    union = area1 + area2 - inter
    return np.where(union > 0, inter / union, 0.0)


def kmeans_iou(
    wh: np.ndarray,
    k: int,
    max_iter: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """K-means clustering with 1-IoU distance on (w,h) pairs.

    Returns:
        (centroids (k, 2), mean_iou)
    """
    rng = np.random.RandomState(seed)
    n = wh.shape[0]
    # Initialise with k-means++ style
    indices = [rng.randint(n)]
    for _ in range(1, k):
        iou_mat = _wh_iou(wh, wh[indices])  # (N, current_k)
        max_iou = iou_mat.max(axis=1)
        dist = 1.0 - max_iou
        prob = dist / (dist.sum() + 1e-8)
        indices.append(rng.choice(n, p=prob))

    centroids = wh[indices].copy()

    for _ in range(max_iter):
        iou_mat = _wh_iou(wh, centroids)  # (N, k)
        assignments = iou_mat.argmax(axis=1)  # (N,)

        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                new_centroids[j] = wh[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Final mean IoU
    iou_mat = _wh_iou(wh, centroids)
    mean_iou = iou_mat.max(axis=1).mean()
    return centroids, mean_iou


def generate_current_anchors_wh() -> np.ndarray:
    """Generate (w,h) pairs from current YAML config."""
    wh_pairs = []
    for s in CURRENT_SIZES:
        for ar in CURRENT_RATIOS:
            w = s * np.sqrt(ar)
            h = s / np.sqrt(ar)
            wh_pairs.append([w, h])
    return np.array(wh_pairs, dtype=np.float32)


def extract_wh_from_coco(json_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Extract normalised (w, h) and class_ids from COCO annotations.

    Returns:
        wh: (N, 2) normalised widths and heights
        class_ids: (N,) class id per annotation
        cat_map: {id: name}
    """
    with open(json_path) as f:
        data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    img_sizes = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    wh_list = []
    cls_list = []
    for ann in data["annotations"]:
        img_w, img_h = img_sizes[ann["image_id"]]
        _, _, bw, bh = ann["bbox"]  # COCO format: x, y, w, h (absolute)
        wh_list.append([bw / img_w, bh / img_h])
        cls_list.append(ann["category_id"])

    return (
        np.array(wh_list, dtype=np.float32),
        np.array(cls_list, dtype=np.int32),
        cat_map,
    )


def coverage_analysis(
    gt_wh: np.ndarray,
    anchor_wh: np.ndarray,
    label: str,
) -> Dict[str, float]:
    """Compute max IoU per GT box with a set of anchor (w,h) pairs."""
    iou_mat = _wh_iou(gt_wh, anchor_wh)  # (N_gt, N_anchors)
    max_iou_per_gt = iou_mat.max(axis=1)
    stats = {
        "label": label,
        "mean_maxIoU": float(max_iou_per_gt.mean()),
        "median_maxIoU": float(np.median(max_iou_per_gt)),
        "pct_above_0.5": float((max_iou_per_gt > 0.5).mean() * 100),
        "pct_above_0.35": float((max_iou_per_gt > 0.35).mean() * 100),
        "min_maxIoU": float(max_iou_per_gt.min()),
    }
    return stats


def centroids_to_sizes_ratios(
    centroids: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """Convert (w,h) centroids to approximate (sizes, ratios) parameters.

    Strategy: extract unique *scales* as sqrt(w*h) and unique *aspect ratios*
    as w/h, then return the de-duplicated sets.
    """
    scales = np.sqrt(centroids[:, 0] * centroids[:, 1])
    ratios = centroids[:, 0] / (centroids[:, 1] + 1e-8)

    # Sort and round
    scales = sorted(set(round(float(s), 3) for s in scales))
    ratios = sorted(set(round(float(r), 2) for r in ratios))
    return scales, ratios


def main():
    print("=" * 70)
    print("  ANCHOR ANALYSIS FOR TFM DATASET")
    print("=" * 70)

    # ── 1. Load annotations ──
    splits = ["train", "valid", "test"]
    all_wh = []
    all_cls = []
    cat_map = {}

    for split in splits:
        json_path = COCO_ROOT / split / "_annotations.coco.json"
        if not json_path.exists():
            print(f"⚠️ Missing: {json_path}")
            continue
        wh, cls_ids, cmap = extract_wh_from_coco(json_path)
        all_wh.append(wh)
        all_cls.append(cls_ids)
        cat_map.update(cmap)
        print(f"  {split}: {len(wh)} annotations loaded")

    wh = np.concatenate(all_wh, axis=0)
    cls_ids = np.concatenate(all_cls, axis=0)
    print(f"\n  Total annotations: {len(wh)}")
    print(f"  Categories: {cat_map}")

    # ── 2. Descriptive statistics ──
    print("\n" + "=" * 70)
    print("  BBOX SIZE STATISTICS (normalised w,h)")
    print("=" * 70)

    for cid, cname in sorted(cat_map.items()):
        mask = cls_ids == cid
        if mask.sum() == 0:
            continue
        cw, ch = wh[mask, 0], wh[mask, 1]
        areas = cw * ch
        ars = cw / (ch + 1e-8)
        print(f"\n  [{cid}] {cname} ({mask.sum()} boxes):")
        print(f"    Width:  min={cw.min():.4f}  mean={cw.mean():.4f}  "
              f"median={np.median(cw):.4f}  max={cw.max():.4f}")
        print(f"    Height: min={ch.min():.4f}  mean={ch.mean():.4f}  "
              f"median={np.median(ch):.4f}  max={ch.max():.4f}")
        print(f"    Area:   min={areas.min():.6f}  mean={areas.mean():.4f}  "
              f"max={areas.max():.4f}")
        print(f"    AR:     min={ars.min():.3f}  mean={ars.mean():.3f}  "
              f"median={np.median(ars):.3f}  max={ars.max():.3f}")

    # Global
    areas = wh[:, 0] * wh[:, 1]
    ars = wh[:, 0] / (wh[:, 1] + 1e-8)
    print(f"\n  GLOBAL ({len(wh)} boxes):")
    print(f"    Width:  min={wh[:, 0].min():.4f}  mean={wh[:, 0].mean():.4f}  "
          f"median={np.median(wh[:, 0]):.4f}  max={wh[:, 0].max():.4f}")
    print(f"    Height: min={wh[:, 1].min():.4f}  mean={wh[:, 1].mean():.4f}  "
          f"median={np.median(wh[:, 1]):.4f}  max={wh[:, 1].max():.4f}")
    print(f"    Area:   min={areas.min():.6f}  mean={areas.mean():.4f}  "
          f"max={areas.max():.4f}")
    print(f"    AR:     min={ars.min():.3f}  mean={ars.mean():.3f}  "
          f"median={np.median(ars):.3f}  max={ars.max():.3f}")

    # ── 3. K-means clustering ──
    print("\n" + "=" * 70)
    print("  K-MEANS CLUSTERING (1-IoU distance)")
    print("=" * 70)

    best_k = 0
    best_centroids = None
    best_mean_iou = 0.0

    for k in [5, 6, 7, 8, 9, 10, 12, 15]:
        centroids, mean_iou = kmeans_iou(wh, k)
        # Sort by area
        areas_c = centroids[:, 0] * centroids[:, 1]
        order = np.argsort(areas_c)
        centroids = centroids[order]
        print(f"\n  k={k:2d}  |  mean IoU = {mean_iou:.4f}")
        for i, (cw, ch) in enumerate(centroids):
            print(f"    [{i}] w={cw:.4f} h={ch:.4f}  "
                  f"area={cw*ch:.6f}  AR={cw/(ch+1e-8):.3f}")

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_k = k
            best_centroids = centroids.copy()

    # ── 4. Coverage comparison ──
    print("\n" + "=" * 70)
    print("  COVERAGE COMPARISON")
    print("=" * 70)

    current_wh = generate_current_anchors_wh()
    cov_current = coverage_analysis(wh, current_wh, "Current YAML anchors")
    print(f"\n  Current anchors ({len(CURRENT_SIZES)} sizes × {len(CURRENT_RATIOS)} ratios = {len(current_wh)}):")
    for key, val in cov_current.items():
        if key != "label":
            print(f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}")

    # Compare with k-means results for a few k values
    for k in [7, 9, 12, 15]:
        centroids, mean_iou = kmeans_iou(wh, k)
        cov = coverage_analysis(wh, centroids, f"K-means k={k}")
        print(f"\n  K-means k={k} (mean_iou={mean_iou:.4f}):")
        for key, val in cov.items():
            if key != "label":
                print(f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}")

    # ── 5. Derive recommended sizes & ratios ──
    print("\n" + "=" * 70)
    print("  RECOMMENDED ANCHOR CONFIGURATION")
    print("=" * 70)

    # Use k=9 as a good balance (current uses 7×5=35 anchors per cell)
    for target_k in [7, 9, 12]:
        centroids, mean_iou = kmeans_iou(wh, target_k)
        areas_c = centroids[:, 0] * centroids[:, 1]
        order = np.argsort(areas_c)
        centroids = centroids[order]

        sizes, ratios = centroids_to_sizes_ratios(centroids)
        print(f"\n  --- k={target_k} (mean IoU={mean_iou:.4f}) ---")
        print(f"  Raw centroids (w,h):")
        for cw, ch in centroids:
            print(f"    ({cw:.4f}, {ch:.4f})")
        print(f"  Suggested sizes:  {sizes}")
        print(f"  Suggested ratios: {ratios}")
        n_total = len(sizes) * len(ratios)
        print(f"  Total anchors per cell: {n_total}  "
              f"(grid 7×7 → {7*7*n_total} total)")

    # ── 6. Final recommendation ──
    # Use k=9 centroids directly as anchor priors (SSD-style direct use)
    centroids_9, miou_9 = kmeans_iou(wh, 9)
    areas_c = centroids_9[:, 0] * centroids_9[:, 1]
    order = np.argsort(areas_c)
    centroids_9 = centroids_9[order]

    print("\n" + "=" * 70)
    print("  FINAL RECOMMENDATION: Use k-means centroids directly")
    print("=" * 70)
    print(f"  Use these 9 (w,h) pairs as anchor_priors (not sizes×ratios):")
    anchor_priors = []
    for cw, ch in centroids_9:
        anchor_priors.append([round(float(cw), 4), round(float(ch), 4)])
        print(f"    [{cw:.4f}, {ch:.4f}]")
    print(f"\n  Mean anchor IoU with GT: {miou_9:.4f}")
    cov_final = coverage_analysis(wh, centroids_9, "Final k=9")
    print(f"  Coverage above 0.35 IoU: {cov_final['pct_above_0.35']:.1f}%")
    print(f"  Coverage above 0.50 IoU: {cov_final['pct_above_0.5']:.1f}%")

    # Also test with current-style decomposition for comparison
    print("\n  --- For YAML (sizes × ratios decomposition) ---")
    sizes_9, ratios_9 = centroids_to_sizes_ratios(centroids_9)
    # Regenerate anchor wh from decomposition and test coverage
    decomposed_wh = []
    for s in sizes_9:
        for r in ratios_9:
            w = s * np.sqrt(r)
            h = s / np.sqrt(r)
            decomposed_wh.append([w, h])
    decomposed_wh = np.array(decomposed_wh, dtype=np.float32)
    cov_decomposed = coverage_analysis(wh, decomposed_wh, "Decomposed sizes×ratios")
    print(f"  anchor_sizes: {sizes_9}")
    print(f"  anchor_ratios: {ratios_9}")
    print(f"  Anchors per cell: {len(decomposed_wh)}")
    print(f"  Coverage above 0.35 IoU: {cov_decomposed['pct_above_0.35']:.1f}%")
    print(f"  Coverage above 0.50 IoU: {cov_decomposed['pct_above_0.5']:.1f}%")


if __name__ == "__main__":
    main()
