"""Visualization utilities for detections."""
from __future__ import annotations

from typing import List
import random

import matplotlib.pyplot as plt
import numpy as np

try:
    from .utils_eval import BoundingBox
except ImportError:
    from utils_eval import BoundingBox


def _draw_boxes(ax, boxes: List[BoundingBox], color: str, label: str) -> None:
    for box in boxes:
        rect = plt.Rectangle(
            (box.x, box.y),
            box.w,
            box.h,
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            box.x,
            box.y - 2,
            f"{label}:{box.class_name} {box.confidence:.2f}",
            color=color,
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )


def plot_detections(
    image: np.ndarray,
    ground_truth: List[BoundingBox],
    predictions: List[BoundingBox],
    title: str = "Detections",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    _draw_boxes(ax, ground_truth, color="lime", label="GT")
    _draw_boxes(ax, predictions, color="red", label="P")
    ax.set_title(title)
    ax.axis("off")
    plt.show()


def show_random_samples(
    images: List[np.ndarray],
    ground_truths: List[List[BoundingBox]],
    predictions: List[List[BoundingBox]],
    max_samples: int = 4,
    title: str = "Random samples",
) -> None:
    num_samples = min(max_samples, len(images))
    idxs = random.sample(range(len(images)), num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    for ax, idx in zip(axes, idxs):
        ax.imshow(images[idx])
        _draw_boxes(ax, ground_truths[idx], color="lime", label="GT")
        _draw_boxes(ax, predictions[idx], color="red", label="P")
        ax.axis("off")
    fig.suptitle(title)
    plt.show()


try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    pd = None


def plot_training_history(
    history_df: "pd.DataFrame",
    title: str = "Training History",
    loss_title: str = "Loss",
    acc_title: str = "Accuracy",
    secondary_title: str = "Secondary Loss",
    loss_cols: tuple = ("loss", "val_loss"),
    acc_pattern: str = "accuracy",
    secondary_pattern: str = "bbox",
    figsize: tuple = (15, 4),
    save_path: str | None = None,
) -> None:
    """Plot training history curves (loss, accuracy, secondary metric).
    
    Generic function that works for both SSD and YOLO training histories.
    
    Args:
        history_df: DataFrame with training history (columns: loss, val_loss, etc.)
        title: Main figure title
        loss_title: Title for loss subplot
        acc_title: Title for accuracy subplot
        secondary_title: Title for secondary metric subplot
        loss_cols: Tuple of (train_loss_col, val_loss_col) names
        acc_pattern: Pattern to match accuracy columns
        secondary_pattern: Pattern to match secondary metric columns (e.g., 'bbox', 'class_out_loss')
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required for plot_training_history")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # Plot 1: Loss
    train_loss, val_loss = loss_cols
    if train_loss in history_df.columns:
        axes[0].plot(history_df[train_loss], label='train', color='blue')
    if val_loss in history_df.columns:
        axes[0].plot(history_df[val_loss], label='val', color='orange')
    axes[0].set_title(loss_title)
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    acc_cols_found = [c for c in history_df.columns if acc_pattern in c and not c.startswith('val')]
    if acc_cols_found:
        axes[1].plot(history_df[acc_cols_found[0]], label='train', color='blue')
    val_acc_cols = [c for c in history_df.columns if acc_pattern in c and c.startswith('val')]
    if val_acc_cols:
        axes[1].plot(history_df[val_acc_cols[0]], label='val', color='orange')
    axes[1].set_title(acc_title)
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Secondary metric (bbox loss, class loss, etc.)
    sec_cols = [c for c in history_df.columns if secondary_pattern in c and 'loss' in c and not c.startswith('val')]
    if sec_cols:
        axes[2].plot(history_df[sec_cols[0]], label='train', color='blue')
    val_sec_cols = [c for c in history_df.columns if secondary_pattern in c and 'loss' in c and c.startswith('val')]
    if val_sec_cols:
        axes[2].plot(history_df[val_sec_cols[0]], label='val', color='orange')
    axes[2].set_title(secondary_title)
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_ssd_history(
    history_df: "pd.DataFrame",
    title: str = "SSD Training",
    save_path: str | None = None,
) -> None:
    """Convenience wrapper for SSD training history."""
    plot_training_history(
        history_df=history_df,
        title=title,
        loss_title="Total Loss",
        acc_title="Class Accuracy",
        secondary_title="BBox Loss",
        loss_cols=("loss", "val_loss"),
        acc_pattern="class_out_accuracy",
        secondary_pattern="bbox_out",
        save_path=save_path,
    )


def plot_ssd_v2_history(
    history_df: "pd.DataFrame",
    title: str = "SSD Anchor V2 - Focal Loss + HNM",
    save_path: str | None = None,
) -> None:
    """Convenience wrapper for SSD V2 (Focal Loss) training history."""
    plot_training_history(
        history_df=history_df,
        title=title,
        loss_title="Total Loss (Focal + BBox)",
        acc_title="Class Accuracy",
        secondary_title="Focal Classification Loss",
        loss_cols=("loss", "val_loss"),
        acc_pattern="class_out_accuracy",
        secondary_pattern="class_out_loss",
        save_path=save_path,
    )

