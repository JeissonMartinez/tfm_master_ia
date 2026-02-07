"""Unified training metrics extraction and visualization.

Provides a common ``TrainingHistory`` schema and standardized
multi-panel plots for both YOLO (Ultralytics CSV) and MobileNet
(Keras CSVLogger) training histories.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils_io import log


@dataclass
class TrainingHistory:
    """Unified per-epoch training history — works for every family."""
    epoch: List[int] = field(default_factory=list)
    # --- losses ---
    train_box_loss: List[float] = field(default_factory=list)
    train_cls_loss: List[float] = field(default_factory=list)
    train_obj_loss: List[float] = field(default_factory=list)
    train_total_loss: List[float] = field(default_factory=list)
    val_box_loss: List[float] = field(default_factory=list)
    val_cls_loss: List[float] = field(default_factory=list)
    val_obj_loss: List[float] = field(default_factory=list)
    val_total_loss: List[float] = field(default_factory=list)
    # --- metrics ---
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    mAP50: List[float] = field(default_factory=list)
    mAP50_95: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    # --- meta ---
    family: str = ""
    model_name: str = ""
    phase: List[str] = field(default_factory=list)

    @property
    def n_epochs(self) -> int:
        return len(self.epoch)

    @property
    def best_epoch_by_val_loss(self) -> int:
        if not self.val_total_loss:
            return -1
        return int(np.argmin(self.val_total_loss))

    @property
    def best_epoch_by_mAP50(self) -> int:
        if not self.mAP50:
            return -1
        return int(np.argmax(self.mAP50))


# =====================================================================
#  Extractors
# =====================================================================

def extract_yolo_history(results_csv: str) -> TrainingHistory:
    """Parse ``results.csv`` produced by Ultralytics training.

    Column names vary slightly between YOLO versions, so we match
    by partial name.
    """
    import pandas as pd

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    h = TrainingHistory(family="yolo")

    def _col(pattern: str) -> Optional[str]:
        for c in df.columns:
            if pattern in c.lower():
                return c
        return None

    h.epoch = list(range(len(df)))

    for attr, pat in [
        ("train_box_loss", "train/box"),
        ("train_cls_loss", "train/cls"),
        ("train_obj_loss", "train/dfl"),
        ("val_box_loss", "val/box"),
        ("val_cls_loss", "val/cls"),
        ("val_obj_loss", "val/dfl"),
    ]:
        col = _col(pat)
        if col is not None:
            setattr(h, attr, df[col].tolist())

    # total losses
    if h.train_box_loss:
        h.train_total_loss = [
            sum(x) for x in zip(
                h.train_box_loss,
                h.train_cls_loss or [0] * len(h.epoch),
                h.train_obj_loss or [0] * len(h.epoch),
            )
        ]
    if h.val_box_loss:
        h.val_total_loss = [
            sum(x) for x in zip(
                h.val_box_loss,
                h.val_cls_loss or [0] * len(h.epoch),
                h.val_obj_loss or [0] * len(h.epoch),
            )
        ]

    for attr, pat in [
        ("precision", "precision"),
        ("recall", "recall"),
        ("mAP50", "map50(b)"),
        ("mAP50_95", "map50-95"),
    ]:
        col = _col(pat)
        if col is None and pat == "map50(b)":
            col = _col("map50")
        if col is not None:
            setattr(h, attr, df[col].tolist())

    lr_col = _col("lr/pg0") or _col("lr")
    if lr_col is not None:
        h.lr = df[lr_col].tolist()

    return h


def extract_mobilenet_history(
    csv_path: str,
    phase1_csv: Optional[str] = None,
    phase2_csv: Optional[str] = None,
) -> TrainingHistory:
    """Parse Keras CSVLogger output.

    If *phase1_csv* and *phase2_csv* are given, concatenates both.
    Otherwise reads *csv_path* (which may already include a ``phase``
    column from :func:`combine_histories`).
    """
    import pandas as pd

    if phase1_csv and phase2_csv:
        dfs = []
        for p, label in [(phase1_csv, "phase1"), (phase2_csv, "phase2")]:
            if os.path.exists(p):
                df = pd.read_csv(p)
                df["phase"] = label
                dfs.append(df)
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

    if df.empty:
        return TrainingHistory(family="mobilenet")

    h = TrainingHistory(family="mobilenet")
    h.epoch = list(range(len(df)))
    if "phase" in df.columns:
        h.phase = df["phase"].tolist()

    # Keras logs individual head losses
    for attr, col in [
        ("train_box_loss", "bbox_out_loss"),
        ("train_cls_loss", "class_out_loss"),
        ("train_obj_loss", "objectness_loss"),
        ("val_box_loss", "val_bbox_out_loss"),
        ("val_cls_loss", "val_class_out_loss"),
        ("val_obj_loss", "val_objectness_loss"),
    ]:
        if col in df.columns:
            setattr(h, attr, df[col].tolist())

    if "loss" in df.columns:
        h.train_total_loss = df["loss"].tolist()
    if "val_loss" in df.columns:
        h.val_total_loss = df["val_loss"].tolist()

    if "lr" in df.columns:
        h.lr = df["lr"].tolist()
    elif "learning_rate" in df.columns:
        h.lr = df["learning_rate"].tolist()

    return h


# =====================================================================
#  Plots
# =====================================================================

def plot_training_curves(
    history: TrainingHistory,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12),
    title_prefix: str = "",
) -> None:
    """Standardized 6-panel training curves.

    Layout (2×3):
    [Total Loss] [Box Loss]    [Cls Loss]
    [Obj Loss]   [LR Schedule] [Metrics (P/R/mAP)]
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        f"{title_prefix}{history.family.upper()} – {history.model_name}",
        fontsize=14, fontweight="bold",
    )

    epochs = history.epoch or list(range(history.n_epochs))
    has_phase = len(history.phase) > 0

    def _maybe_phase_bg(ax):
        """Draw a subtle background shade for Phase 2."""
        if has_phase and "phase2" in history.phase:
            idx = history.phase.index("phase2")
            ax.axvspan(idx, len(epochs), alpha=0.06, color="orange", label="Phase 2")

    def _plot_pair(ax, train, val, label):
        _maybe_phase_bg(ax)
        if train:
            ax.plot(epochs[: len(train)], train, label=f"Train {label}", marker='o', markersize=4)
        if val:
            ax.plot(epochs[: len(val)], val, label=f"Val {label}", linestyle="--", marker='s', markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot_pair(axes[0, 0], history.train_total_loss, history.val_total_loss, "Total Loss")
    _plot_pair(axes[0, 1], history.train_box_loss, history.val_box_loss, "Box Loss")
    _plot_pair(axes[0, 2], history.train_cls_loss, history.val_cls_loss, "Cls Loss")
    _plot_pair(axes[1, 0], history.train_obj_loss, history.val_obj_loss, "Obj/DFL Loss")

    # LR
    ax_lr = axes[1, 1]
    _maybe_phase_bg(ax_lr)
    if history.lr:
        ax_lr.plot(epochs[: len(history.lr)], history.lr, color="tab:green", marker='o', markersize=4)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_title("LR Schedule")
    ax_lr.grid(True, alpha=0.3)

    # Metrics
    ax_m = axes[1, 2]
    _maybe_phase_bg(ax_m)
    for vals, label, color in [
        (history.precision, "Precision", "tab:blue"),
        (history.recall, "Recall", "tab:orange"),
        (history.mAP50, "mAP@50", "tab:green"),
        (history.mAP50_95, "mAP@50-95", "tab:red"),
    ]:
        if vals:
            ax_m.plot(epochs[: len(vals)], vals, label=label, color=color, marker='o', markersize=4)
    ax_m.set_xlabel("Epoch")
    ax_m.set_ylabel("Score")
    ax_m.set_title("Métricas Val")
    ax_m.legend(fontsize=8)
    ax_m.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"📊 Curvas guardadas: {save_path}")
    plt.show()


def plot_loss_comparison(
    histories: List[TrainingHistory],
    metric: str = "val_total_loss",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Overlay a single metric from multiple experiments."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for h in histories:
        vals = getattr(h, metric, [])
        if vals:
            label = h.model_name or h.family
            ax.plot(vals, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Comparativa: {metric}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_training_summary(history: TrainingHistory) -> None:
    """Print a compact text summary of training."""
    log(f"\n📈 Resumen – {history.family.upper()} {history.model_name}")
    log(f"  Épocas: {history.n_epochs}")
    if history.val_total_loss:
        best_e = history.best_epoch_by_val_loss
        log(f"  Mejor val_loss: {history.val_total_loss[best_e]:.4f} (epoch {best_e})")
    if history.mAP50:
        best_m = history.best_epoch_by_mAP50
        log(f"  Mejor mAP@50: {history.mAP50[best_m]:.4f} (epoch {best_m})")
    if history.precision:
        log(f"  Última P: {history.precision[-1]:.4f}")
    if history.recall:
        log(f"  Última R: {history.recall[-1]:.4f}")
