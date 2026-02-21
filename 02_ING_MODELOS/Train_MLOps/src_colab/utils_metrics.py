"""Unified training metrics extraction and visualization — Cycle 2.

Provides ``TrainingHistory`` and standardized multi-panel plots for
both two-phase PyTorch training and YOLO26 Ultralytics CSV.
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
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_box_loss: List[float] = field(default_factory=list)
    train_cls_loss: List[float] = field(default_factory=list)
    val_box_loss: List[float] = field(default_factory=list)
    val_cls_loss: List[float] = field(default_factory=list)
    train_ctr_loss: List[float] = field(default_factory=list)
    val_ctr_loss: List[float] = field(default_factory=list)
    # --- metrics (YOLO CSV) ---
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    mAP50: List[float] = field(default_factory=list)
    mAP50_95: List[float] = field(default_factory=list)
    # --- LR & meta ---
    lr: List[float] = field(default_factory=list)
    img_size: List[int] = field(default_factory=list)
    phase: List[str] = field(default_factory=list)
    family: str = ""
    model_name: str = ""

    @property
    def n_epochs(self) -> int:
        return len(self.epoch)

    @property
    def best_epoch_by_val_loss(self) -> int:
        if not self.val_loss:
            return -1
        return int(np.argmin(self.val_loss))

    @property
    def best_epoch_by_mAP50(self) -> int:
        if not self.mAP50:
            return -1
        return int(np.argmax(self.mAP50))


# =====================================================================
#  Extractors
# =====================================================================

def extract_two_phase_history(csv_path: str) -> TrainingHistory:
    """Parse CSV from ``save_two_phase_history()``."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    h = TrainingHistory(family="pytorch")
    h.epoch = df["epoch"].tolist()
    h.train_loss = df["train_loss"].tolist()
    h.val_loss = df["val_loss"].tolist()
    if "lr" in df.columns:
        h.lr = df["lr"].tolist()
    if "img_size" in df.columns:
        h.img_size = df["img_size"].astype(int).tolist()
    if "phase" in df.columns:
        h.phase = df["phase"].tolist()
    # Loss component breakdown (reg → box for TrainingHistory naming)
    if "train_cls_loss" in df.columns:
        h.train_cls_loss = df["train_cls_loss"].tolist()
    if "train_reg_loss" in df.columns:
        h.train_box_loss = df["train_reg_loss"].tolist()
    if "val_cls_loss" in df.columns:
        h.val_cls_loss = df["val_cls_loss"].tolist()
    if "val_reg_loss" in df.columns:
        h.val_box_loss = df["val_reg_loss"].tolist()
    if "train_ctr_loss" in df.columns:
        h.train_ctr_loss = df["train_ctr_loss"].tolist()
    if "val_ctr_loss" in df.columns:
        h.val_ctr_loss = df["val_ctr_loss"].tolist()
    return h


def extract_yolo26_history(results_csv: str) -> TrainingHistory:
    """Parse ``results.csv`` produced by Ultralytics training."""
    import pandas as pd

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    h = TrainingHistory(family="yolo26_custom")
    h.epoch = list(range(len(df)))

    def _col(pattern: str) -> Optional[str]:
        for c in df.columns:
            if pattern in c.lower():
                return c
        return None

    for attr, pat in [
        ("train_box_loss", "train/box"),
        ("train_cls_loss", "train/cls"),
        ("val_box_loss", "val/box"),
        ("val_cls_loss", "val/cls"),
    ]:
        col = _col(pat)
        if col is not None:
            setattr(h, attr, df[col].tolist())

    # total losses (sum of components)
    if h.train_box_loss:
        dfl_col = _col("train/dfl")
        dfl_vals = df[dfl_col].tolist() if dfl_col else [0] * len(h.epoch)
        h.train_loss = [
            b + c + d for b, c, d in zip(
                h.train_box_loss,
                h.train_cls_loss or [0] * len(h.epoch),
                dfl_vals,
            )
        ]
    if h.val_box_loss:
        dfl_col = _col("val/dfl")
        dfl_vals = df[dfl_col].tolist() if dfl_col else [0] * len(h.epoch)
        h.val_loss = [
            b + c + d for b, c, d in zip(
                h.val_box_loss,
                h.val_cls_loss or [0] * len(h.epoch),
                dfl_vals,
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

    Layout (2x3):
    [Total Loss] [Box Loss]     [Cls Loss]
    [Img Size]   [LR Schedule]  [Metrics (P/R/mAP)]
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
        if has_phase and "phase2" in history.phase:
            idx = history.phase.index("phase2")
            ax.axvspan(idx, len(epochs), alpha=0.06, color="orange",
                       label="Phase 2")

    def _plot_pair(ax, train, val, label):
        _maybe_phase_bg(ax)
        if train:
            ax.plot(epochs[:len(train)], train, label=f"Train {label}",
                    marker="o", markersize=3)
        if val:
            ax.plot(epochs[:len(val)], val, label=f"Val {label}",
                    linestyle="--", marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot_pair(axes[0, 0], history.train_loss, history.val_loss, "Total Loss")
    _plot_pair(axes[0, 1], history.train_box_loss, history.val_box_loss, "Box Loss")
    _plot_pair(axes[0, 2], history.train_cls_loss, history.val_cls_loss, "Cls Loss")

    # Image size schedule
    ax_sz = axes[1, 0]
    _maybe_phase_bg(ax_sz)
    if history.img_size:
        ax_sz.step(epochs[:len(history.img_size)], history.img_size,
                   where="post", color="tab:purple")
    ax_sz.set_xlabel("Epoch")
    ax_sz.set_ylabel("Image Size (px)")
    ax_sz.set_title("Progressive Resizing")
    ax_sz.grid(True, alpha=0.3)

    # LR
    ax_lr = axes[1, 1]
    _maybe_phase_bg(ax_lr)
    if history.lr:
        ax_lr.plot(epochs[:len(history.lr)], history.lr, color="tab:green",
                   marker="o", markersize=3)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_title("LR Schedule")
    ax_lr.grid(True, alpha=0.3)

    # Panel 6: Metrics (YOLO) or Centerness Loss (PyTorch FCOS/ESPDet)
    ax_m = axes[1, 2]
    _maybe_phase_bg(ax_m)
    has_yolo_metrics = any([
        history.precision, history.recall, history.mAP50, history.mAP50_95,
    ])
    if has_yolo_metrics:
        for vals, label, color in [
            (history.precision, "Precision", "tab:blue"),
            (history.recall, "Recall", "tab:orange"),
            (history.mAP50, "mAP@50", "tab:green"),
            (history.mAP50_95, "mAP@50-95", "tab:red"),
        ]:
            if vals:
                ax_m.plot(epochs[:len(vals)], vals, label=label, color=color,
                          marker="o", markersize=3)
        ax_m.set_xlabel("Epoch")
        ax_m.set_ylabel("Score")
        ax_m.set_title("Métricas Val")
    else:
        # PyTorch FCOS/ESPDet: show centerness loss
        if history.train_ctr_loss:
            ax_m.plot(epochs[:len(history.train_ctr_loss)],
                      history.train_ctr_loss, label="Train Ctr",
                      marker="o", markersize=3, color="tab:blue")
        if history.val_ctr_loss:
            ax_m.plot(epochs[:len(history.val_ctr_loss)],
                      history.val_ctr_loss, label="Val Ctr",
                      linestyle="--", marker="s", markersize=3,
                      color="tab:orange")
        ax_m.set_xlabel("Epoch")
        ax_m.set_ylabel("Centerness Loss")
        ax_m.set_title("Centerness Loss")
    ax_m.legend(fontsize=8)
    ax_m.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"📊 Curvas guardadas: {save_path}")
    plt.close(fig)


def plot_loss_comparison(
    histories: List[TrainingHistory],
    metric: str = "val_loss",
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
    plt.close(fig)


def print_training_summary(history: TrainingHistory) -> None:
    """Print a compact text summary of training."""
    log(f"\n📈 Resumen – {history.family.upper()} {history.model_name}")
    log(f"  Épocas: {history.n_epochs}")
    if history.val_loss:
        best_e = history.best_epoch_by_val_loss
        log(f"  Mejor val_loss: {history.val_loss[best_e]:.4f} (epoch {best_e})")
    if history.mAP50:
        best_m = history.best_epoch_by_mAP50
        log(f"  Mejor mAP@50: {history.mAP50[best_m]:.4f} (epoch {best_m})")
    if history.img_size:
        log(f"  Resoluciones: {sorted(set(history.img_size), reverse=True)}")
