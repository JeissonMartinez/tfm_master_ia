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
