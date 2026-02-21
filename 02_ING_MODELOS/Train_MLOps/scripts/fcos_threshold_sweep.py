#!/usr/bin/env python3
"""FCOS Post-NMS Threshold Sweep.

Loads the best FCOS checkpoint and evaluates on val and test splits
at multiple conf_threshold values, keeping NMS and all other scoring
params fixed.  Produces:

1. P-R curve (val + test) with F1 iso-lines.
2. F1 vs threshold bar chart.
3. Detections & FP vs threshold chart.
4. Per-class AP@50 heatmap across thresholds.
5. CSV with all numeric results.

Usage:
    cd 02_ING_MODELOS/Train_MLOps
    python scripts/fcos_threshold_sweep.py [--output-dir outputs/fcos_v3s_v1-1771695807]

All plots are saved to ``<output-dir>/threshold_sweep/``.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize as mplNorm
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
#  Project imports (run from Train_MLOps/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent       # Train_MLOps/
sys.path.insert(0, str(ROOT))

from src_colab.utils_model import build_fcos_model          # noqa: E402
from src_colab.utils_data import IODCDataset, iodc_collate_fn  # noqa: E402
from src_colab.utils_infer import predict_fcos               # noqa: E402


# =====================================================================
#  Evaluation helpers (self-contained, no side effects)
# =====================================================================

def _compute_iou(box1: Tuple, box2: Tuple) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def _collect_detections(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    conf_threshold: float,
    nms_threshold: float,
    class_names: List[str],
    strides: List[int],
    ctr_power: float,
    iou_aware: bool,
) -> Tuple[list, list]:
    """Run inference and collect raw detections + ground truths."""
    all_dets: list = []    # (img_idx, cls_id, conf, bbox)
    all_gts: list = []     # (img_idx, cls_id, bbox)
    model.eval()
    img_idx = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            batch_dets = predict_fcos(
                model, images,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                class_names=class_names,
                strides=strides,
                ctr_power=ctr_power,
                iou_aware=iou_aware,
            )
            for b in range(images.shape[0]):
                dets = batch_dets[b] if b < len(batch_dets) else []
                for d in dets:
                    all_dets.append((img_idx, d.class_id, d.confidence, d.bbox))
                tgt = targets[b]
                gt_boxes = tgt["boxes"].cpu().numpy()
                gt_labels = tgt["labels"].cpu().numpy()
                for j in range(len(gt_labels)):
                    cx, cy, bw, bh = gt_boxes[j]
                    x1 = max(0, cx - bw / 2); y1 = max(0, cy - bh / 2)
                    x2 = min(1, cx + bw / 2); y2 = min(1, cy + bh / 2)
                    all_gts.append((img_idx, int(gt_labels[j]), (x1, y1, x2, y2)))
                img_idx += 1
    return all_dets, all_gts


def _match_and_score(
    detections: list,
    ground_truths: list,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict:
    """Compute P, R, F1, mAP@50, per-class AP, TP/FP/FN counts."""
    gt_by_img_cls: Dict[tuple, list] = defaultdict(list)
    for img_idx, cls_id, bbox in ground_truths:
        gt_by_img_cls[(img_idx, cls_id)].append({"bbox": bbox, "matched": False})

    per_class: Dict[str, Dict] = {}
    aps = []

    for c in range(num_classes):
        dets_c = [(d[0], d[2], d[3]) for d in detections if d[1] == c]
        dets_c.sort(key=lambda x: x[1], reverse=True)
        n_gt_c = sum(1 for gt in ground_truths if gt[1] == c)

        tp = np.zeros(len(dets_c))
        fp = np.zeros(len(dets_c))

        # Reset matched flags
        for key in gt_by_img_cls:
            for gt in gt_by_img_cls[key]:
                gt["matched"] = False

        for i, (img_idx, conf, bbox) in enumerate(dets_c):
            gts = gt_by_img_cls.get((img_idx, c), [])
            best_iou, best_g = 0.0, -1
            for g_idx, gt in enumerate(gts):
                iou = _compute_iou(bbox, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou; best_g = g_idx
            if best_iou >= iou_threshold and best_g >= 0 and not gts[best_g]["matched"]:
                tp[i] = 1; gts[best_g]["matched"] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / (n_gt_c + 1e-8)
        prec = tp_cum / (tp_cum + fp_cum + 1e-8)

        # 101-point interpolation AP
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ap = sum(mpre[np.where(mrec >= t)[0][0]] for t in np.linspace(0, 1, 101)
                 if len(np.where(mrec >= t)[0]) > 0) / 101.0
        aps.append(ap)

        total_tp = int(tp.sum())
        total_fp = int(fp.sum())
        total_fn = n_gt_c - total_tp
        p_final = float(prec[-1]) if len(prec) > 0 else 0.0
        r_final = float(rec[-1]) if len(rec) > 0 else 0.0
        f1_final = 2 * p_final * r_final / (p_final + r_final + 1e-8)

        per_class[c] = {
            "ap50": ap, "precision": p_final, "recall": r_final,
            "f1": f1_final, "tp": total_tp, "fp": total_fp, "fn": total_fn,
        }

    mAP50 = float(np.mean(aps)) if aps else 0.0
    global_p = float(np.mean([v["precision"] for v in per_class.values()]))
    global_r = float(np.mean([v["recall"] for v in per_class.values()]))
    global_f1 = 2 * global_p * global_r / (global_p + global_r + 1e-8)
    total_dets = len(detections)
    total_tp = sum(v["tp"] for v in per_class.values())
    total_fp = sum(v["fp"] for v in per_class.values())
    total_fn = sum(v["fn"] for v in per_class.values())

    return {
        "mAP50": mAP50, "precision": global_p, "recall": global_r,
        "f1": global_f1, "n_dets": total_dets,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "per_class": per_class,
    }


# =====================================================================
#  Plotting (magma palette)
# =====================================================================

def _magma_colors(n: int) -> list:
    """Return *n* colors from the magma colormap, avoiding extremes."""
    cmap = plt.cm.magma  # type: ignore[attr-defined]
    return [cmap(0.15 + 0.70 * i / max(n - 1, 1)) for i in range(n)]


def plot_pr_curve(
    results: Dict[str, Dict[float, Dict]],
    thresholds: List[float],
    save_path: Path,
) -> None:
    """Precision-Recall scatter + F1 iso-lines, one series per split."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = _magma_colors(len(results))

    # F1 iso-lines
    for f1_val in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        r_pts = np.linspace(0.01, 1.0, 200)
        p_pts = f1_val * r_pts / (2 * r_pts - f1_val + 1e-8)
        valid = (p_pts > 0) & (p_pts <= 1)
        ax.plot(r_pts[valid], p_pts[valid], "--", color="grey", alpha=0.25, lw=0.8)
        # Label F1 iso at the rightmost valid point
        if valid.any():
            idx = np.max(np.where(valid))
            ax.annotate(f"F1={f1_val:.1f}", (r_pts[idx], p_pts[idx]),
                        fontsize=7, color="grey", alpha=0.6,
                        ha="right", va="bottom")

    for (split, thr_results), color in zip(results.items(), colors):
        recalls = [thr_results[t]["recall"] for t in thresholds]
        precisions = [thr_results[t]["precision"] for t in thresholds]
        f1s = [thr_results[t]["f1"] for t in thresholds]

        ax.plot(recalls, precisions, "-o", color=color, lw=2, ms=8,
                label=split.capitalize(), zorder=5)

        # Annotate each point with threshold and F1
        for t, r, p, f1 in zip(thresholds, recalls, precisions, f1s):
            ax.annotate(
                f"thr={t:.2f}\nF1={f1:.3f}",
                (r, p), textcoords="offset points", xytext=(8, 8),
                fontsize=7, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
            )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("FCOS — Threshold Sweep: Precision vs Recall", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ P-R curve → {save_path}")


def plot_f1_vs_threshold(
    results: Dict[str, Dict[float, Dict]],
    thresholds: List[float],
    save_path: Path,
) -> None:
    """Grouped bar chart: F1 at each threshold, per split."""
    fig, ax = plt.subplots(figsize=(9, 5))
    n_splits = len(results)
    bar_w = 0.25
    x = np.arange(len(thresholds))
    colors = _magma_colors(n_splits)

    for i, (split, thr_results) in enumerate(results.items()):
        f1s = [thr_results[t]["f1"] for t in thresholds]
        bars = ax.bar(x + i * bar_w, f1s, bar_w, label=split.capitalize(),
                      color=colors[i], edgecolor="white", lw=0.5)
        for bar, f1 in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{f1:.3f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=colors[i])

    ax.set_xlabel("conf_threshold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("FCOS — F1 vs conf_threshold", fontsize=13, fontweight="bold")
    ax.set_xticks(x + bar_w * (n_splits - 1) / 2)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, min(1.0, max(
        max(r[t]["f1"] for t in thresholds) for r in results.values()
    ) * 1.2))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ F1 chart → {save_path}")


def plot_dets_fp(
    results: Dict[str, Dict[float, Dict]],
    thresholds: List[float],
    save_path: Path,
) -> None:
    """Line plot: total detections, TP, FP vs threshold per split."""
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5),
                             sharey=False)
    if len(results) == 1:
        axes = [axes]
    colors = _magma_colors(3)  # dets, TP, FP

    for ax, (split, thr_results) in zip(axes, results.items()):
        dets = [thr_results[t]["n_dets"] for t in thresholds]
        tps = [thr_results[t]["tp"] for t in thresholds]
        fps = [thr_results[t]["fp"] for t in thresholds]

        ax.plot(thresholds, dets, "-s", color=colors[0], lw=2, label="Detections")
        ax.plot(thresholds, tps, "-o", color=colors[1], lw=2, label="TP")
        ax.plot(thresholds, fps, "-^", color=colors[2], lw=2, label="FP")

        for t, d, tp, fp in zip(thresholds, dets, tps, fps):
            ax.annotate(f"{d}", (t, d), fontsize=7, ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 5), color=colors[0])
            ax.annotate(f"{fp}", (t, fp), fontsize=7, ha="center", va="top",
                        textcoords="offset points", xytext=(0, -7), color=colors[2])

        ax.set_xlabel("conf_threshold", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{split.capitalize()}: Detections / TP / FP", fontsize=12,
                      fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("FCOS — Detection Counts vs Threshold", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Dets/FP chart → {save_path}")


def plot_per_class_ap_heatmap(
    results: Dict[str, Dict[float, Dict]],
    thresholds: List[float],
    class_names: List[str],
    save_path: Path,
) -> None:
    """Heatmap: per-class AP@50 across thresholds, one subplot per split."""
    n_splits = len(results)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits + 1, 4))
    if n_splits == 1:
        axes = [axes]

    for ax, (split, thr_results) in zip(axes, results.items()):
        matrix = np.zeros((len(class_names), len(thresholds)))
        for j, t in enumerate(thresholds):
            pc = thr_results[t]["per_class"]
            for c in range(len(class_names)):
                matrix[c, j] = pc[c]["ap50"]

        im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("conf_threshold", fontsize=10)
        ax.set_title(f"{split.capitalize()} — Per-class AP@50", fontsize=11,
                      fontweight="bold")

        # Annotate cells
        for c in range(len(class_names)):
            for j in range(len(thresholds)):
                val = matrix[c, j]
                text_color = "white" if val < 0.55 else "black"
                ax.text(j, c, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.8, label="AP@50")
    fig.suptitle("FCOS — Per-class AP@50 Heatmap", fontsize=13,
                 fontweight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.15)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ AP heatmap → {save_path}")


def save_csv(
    results: Dict[str, Dict[float, Dict]],
    thresholds: List[float],
    class_names: List[str],
    save_path: Path,
) -> None:
    """Write all numeric results to CSV."""
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "split", "conf_threshold", "mAP50", "precision", "recall", "f1",
            "n_dets", "tp", "fp", "fn",
        ]
        for name in class_names:
            header.extend([f"ap50_{name}", f"p_{name}", f"r_{name}", f"f1_{name}",
                           f"tp_{name}", f"fp_{name}", f"fn_{name}"])
        writer.writerow(header)

        for split, thr_results in results.items():
            for t in thresholds:
                r = thr_results[t]
                row = [
                    split, f"{t:.2f}", f"{r['mAP50']:.4f}",
                    f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                    r["n_dets"], r["tp"], r["fp"], r["fn"],
                ]
                for c, name in enumerate(class_names):
                    pc = r["per_class"][c]
                    row.extend([
                        f"{pc['ap50']:.4f}", f"{pc['precision']:.4f}",
                        f"{pc['recall']:.4f}", f"{pc['f1']:.4f}",
                        pc["tp"], pc["fp"], pc["fn"],
                    ])
                writer.writerow(row)
    print(f"  ✅ CSV → {save_path}")


# =====================================================================
#  Main sweep
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="FCOS Post-NMS Threshold Sweep")
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/fcos_v3s_v1-1771695807",
        help="Path to the FCOS training output directory",
    )
    parser.add_argument(
        "--dataset-dir", type=str,
        default=None,
        help="Path to IODC YOLO dataset (auto-detected if omitted)",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30],
        help="Confidence thresholds to sweep",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+",
        default=["valid", "test"],
        help="Dataset splits to evaluate",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # ----- Paths -----
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    ckpt_path = output_dir / "checkpoints" / "best_fcos.pt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    sweep_dir = output_dir / "threshold_sweep"
    sweep_dir.mkdir(exist_ok=True)

    # Auto-detect dataset
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        candidate = ROOT.parent / "datasets" / "IODC" / "yolo"
        assert candidate.exists(), f"Dataset not found at {candidate}"
        dataset_dir = candidate

    # ----- Config (matches Train 4 — T4) -----
    CLASS_NAMES = ["dog", "door", "obstacle", "person", "stair"]
    IMG_SIZE = 224
    BATCH_SIZE = 32
    STRIDES = [8, 16, 32]
    NMS_THR = 0.45
    CTR_POWER = 0.5
    IOU_AWARE = True

    # ----- Device -----
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"📱 Device: {device}")

    # ----- Build model -----
    print(f"\n🏗️  Loading FCOS model from {ckpt_path}")
    model = build_fcos_model(
        num_classes=len(CLASS_NAMES), fpn_channels=64,
        pretrained_backbone=False, device=device,
    )
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ----- Build dataloaders -----
    dataloaders = {}
    for split in args.splits:
        ds = IODCDataset(
            dataset_dir=str(dataset_dir), split=split,
            img_size=IMG_SIZE, class_names=CLASS_NAMES,
            augment=False,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=iodc_collate_fn, num_workers=2,
            pin_memory=(device != "cpu"),
        )
        dataloaders[split] = dl
        print(f"  📂 {split}: {len(ds)} images, {len(dl)} batches")

    # ----- Sweep -----
    thresholds = sorted(args.thresholds)
    results: Dict[str, Dict[float, Dict]] = {}

    for split, dl in dataloaders.items():
        print(f"\n{'='*60}")
        print(f"  Split: {split.upper()}")
        print(f"{'='*60}")
        results[split] = {}

        for thr in thresholds:
            t0 = time.perf_counter()
            all_dets, all_gts = _collect_detections(
                model, dl, device,
                conf_threshold=thr,
                nms_threshold=NMS_THR,
                class_names=CLASS_NAMES,
                strides=STRIDES,
                ctr_power=CTR_POWER,
                iou_aware=IOU_AWARE,
            )
            metrics = _match_and_score(
                all_dets, all_gts, num_classes=len(CLASS_NAMES),
            )
            elapsed = time.perf_counter() - t0
            results[split][thr] = metrics

            print(f"  thr={thr:.2f} | mAP50={metrics['mAP50']:.4f} "
                  f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                  f"F1={metrics['f1']:.4f} | dets={metrics['n_dets']} "
                  f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
                  f"| {elapsed:.1f}s")

    # ----- Find optimal operating point -----
    print(f"\n{'='*60}")
    print("  🎯 OPTIMAL OPERATING POINTS (max F1)")
    print(f"{'='*60}")
    for split in results:
        best_thr = max(thresholds, key=lambda t: results[split][t]["f1"])
        best = results[split][best_thr]
        print(f"  {split:>5s}: thr={best_thr:.2f} → F1={best['f1']:.4f} "
              f"(P={best['precision']:.4f}, R={best['recall']:.4f}, "
              f"mAP50={best['mAP50']:.4f})")

    # ----- Plots -----
    print(f"\n📊 Generating plots (magma palette) → {sweep_dir}/")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    plot_pr_curve(results, thresholds, sweep_dir / "pr_curve_sweep.png")
    plot_f1_vs_threshold(results, thresholds, sweep_dir / "f1_vs_threshold.png")
    plot_dets_fp(results, thresholds, sweep_dir / "dets_fp_vs_threshold.png")
    plot_per_class_ap_heatmap(results, thresholds, CLASS_NAMES,
                              sweep_dir / "per_class_ap_heatmap.png")
    save_csv(results, thresholds, CLASS_NAMES, sweep_dir / "threshold_sweep.csv")

    print("\n✅ Threshold sweep complete.")


if __name__ == "__main__":
    main()
