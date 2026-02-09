"""YOLO26 metrics extraction and visualization utilities.

Handles extraction of training metrics from Ultralytics results
and visualization of training progress.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .utils_io import log, safe_copy, safe_exists, safe_filesize_mb, safe_mkdir


def extract_yolo_metrics(
    results_dir: str,
    logs_dir: str,
    version_tag: str,
    copy_best_model: bool = True,
) -> Optional[pd.DataFrame]:
    """Extract Ultralytics results.csv and optionally copy best.pt.

    Args:
        results_dir: Directory containing Ultralytics training results
        logs_dir: Directory to save extracted metrics
        version_tag: Version tag for file naming
        copy_best_model: Whether to copy best.pt to models directory

    Returns:
        DataFrame with training history or None on failure
    """
    csv_src = os.path.join(results_dir, "results.csv")
    csv_dst = os.path.join(logs_dir, f"{version_tag}_history.csv")

    if not safe_exists(csv_src):
        log(f"⚠️ No se encontró: {csv_src}")
        return None

    try:
        df = pd.read_csv(csv_src)
        df.columns = df.columns.str.strip()
        
        # Ensure epoch column exists
        if "epoch" not in df.columns:
            df.insert(0, "epoch", range(1, len(df) + 1))
        
        safe_mkdir(logs_dir)
        df.to_csv(csv_dst, index=False)
        log(f"✅ Historial guardado en: {csv_dst}")

    except Exception as exc:
        log(f"⚠️ Error procesando {csv_src}: {exc}")
        return None

    # Copy best model
    if copy_best_model:
        best_src = os.path.join(results_dir, "weights", "best.pt")
        models_dir = os.path.join(os.path.dirname(logs_dir), "models", "checkpoints")
        best_dst = os.path.join(models_dir, f"{version_tag}_best.pt")
        
        if safe_copy(best_src, best_dst):
            size_mb = safe_filesize_mb(best_dst)
            if size_mb is not None:
                log(f"✅ Mejor modelo: {best_dst}")
                log(f"   📦 Tamaño: {size_mb:.2f} MB | INT8 estimado: ~{size_mb * 0.25:.2f} MB")

    return df


def plot_yolo_history(
    df: pd.DataFrame,
    save_path: str,
    title: str = "YOLO26 - Evolución del Entrenamiento",
    show: bool = True,
) -> None:
    """Plot training curves for a YOLO run.

    Args:
        df: DataFrame with training history
        save_path: Path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    # Check for YOLO26 columns (no DFL loss)
    has_dfl = "train/dfl_loss" in df.columns and "val/dfl_loss" in df.columns
    
    if has_dfl:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Box Loss
    ax = axes[0, 0]
    if "train/box_loss" in df.columns:
        ax.plot(df["epoch"], df["train/box_loss"], label="Train", alpha=0.7)
    if "val/box_loss" in df.columns:
        ax.plot(df["epoch"], df["val/box_loss"], label="Val", linewidth=2)
    ax.set_title("Box Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Classification Loss
    ax = axes[0, 1]
    if "train/cls_loss" in df.columns:
        ax.plot(df["epoch"], df["train/cls_loss"], label="Train", alpha=0.7)
    if "val/cls_loss" in df.columns:
        ax.plot(df["epoch"], df["val/cls_loss"], label="Val", linewidth=2)
    ax.set_title("Cls Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # DFL Loss or Learning Rate
    ax = axes[0, 2]
    if has_dfl:
        ax.plot(df["epoch"], df["train/dfl_loss"], label="Train", alpha=0.7)
        ax.plot(df["epoch"], df["val/dfl_loss"], label="Val", linewidth=2)
        ax.set_title("DFL Loss")
    else:
        # YOLO26 doesn't have DFL, show LR instead
        if "lr/pg0" in df.columns:
            ax.plot(df["epoch"], df["lr/pg0"], label="LR pg0", linewidth=2)
        if "lr/pg1" in df.columns:
            ax.plot(df["epoch"], df["lr/pg1"], label="LR pg1", linewidth=2, alpha=0.7)
        ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mAP@50
    ax = axes[1, 0]
    if "metrics/mAP50(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50(B)"], linewidth=2, color="green")
        best_idx = df["metrics/mAP50(B)"].idxmax()
        best_val = df["metrics/mAP50(B)"].max()
        ax.axhline(y=best_val, color="green", linestyle="--", alpha=0.5)
        ax.annotate(f"Best: {best_val:.3f}", xy=(best_idx, best_val), fontsize=9)
    ax.set_title("mAP@50")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # mAP@50-95
    ax = axes[1, 1]
    if "metrics/mAP50-95(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], linewidth=2, color="blue")
        best_idx = df["metrics/mAP50-95(B)"].idxmax()
        best_val = df["metrics/mAP50-95(B)"].max()
        ax.axhline(y=best_val, color="blue", linestyle="--", alpha=0.5)
        ax.annotate(f"Best: {best_val:.3f}", xy=(best_idx, best_val), fontsize=9)
    ax.set_title("mAP@50-95")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Precision/Recall
    ax = axes[1, 2]
    if "metrics/precision(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", linewidth=2)
    if "metrics/recall(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", linewidth=2)
    ax.set_title("Precision / Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_mkdir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    log(f"✅ Gráfico guardado en: {save_path}")


def compare_yolo_versions(
    histories: Dict[str, pd.DataFrame],
    save_path: str,
    title: str = "YOLO26 - Comparativa de Versiones",
    show: bool = True,
) -> None:
    """Compare metrics across multiple YOLO training runs.

    Args:
        histories: Dictionary of {version_name: DataFrame}
        save_path: Path to save the comparison plot
        title: Plot title
        show: Whether to display the plot
    """
    if not histories:
        log("⚠️ No hay historiales para comparar.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    metrics = [
        ("val/box_loss", "Val Box Loss", axes[0, 0]),
        ("val/cls_loss", "Val Cls Loss", axes[0, 1]),
        ("metrics/mAP50(B)", "mAP@50", axes[0, 2]),
        ("metrics/mAP50-95(B)", "mAP@50-95", axes[1, 0]),
        ("metrics/precision(B)", "Precision", axes[1, 1]),
        ("metrics/recall(B)", "Recall", axes[1, 2]),
    ]

    for col, metric_title, ax in metrics:
        for version, df in histories.items():
            if col in df.columns and "epoch" in df.columns:
                ax.plot(df["epoch"], df[col], label=version, linewidth=2)
        
        ax.set_title(metric_title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set y limits for metrics
        if "mAP" in metric_title or "Precision" in metric_title or "Recall" in metric_title:
            ax.set_ylim(0, 1)

    plt.tight_layout()
    safe_mkdir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    log(f"✅ Comparativa guardada en: {save_path}")


def get_best_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Extract best metrics from training history.

    Args:
        df: DataFrame with training history

    Returns:
        Dictionary with best values for each metric
    """
    best_metrics = {}
    
    metric_columns = [
        ("metrics/mAP50(B)", "best_mAP50"),
        ("metrics/mAP50-95(B)", "best_mAP50-95"),
        ("metrics/precision(B)", "best_precision"),
        ("metrics/recall(B)", "best_recall"),
    ]
    
    for col, key in metric_columns:
        if col in df.columns:
            best_metrics[key] = float(df[col].max())
            best_metrics[f"{key}_epoch"] = int(df[col].idxmax()) + 1
    
    # Best validation loss
    loss_columns = ["val/box_loss", "val/cls_loss"]
    for col in loss_columns:
        if col in df.columns:
            key = col.replace("/", "_").replace("val_", "best_")
            best_metrics[key] = float(df[col].min())

    return best_metrics


def load_history_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """Load training history from CSV file.

    Args:
        csv_path: Path to history CSV

    Returns:
        DataFrame or None on failure
    """
    if not safe_exists(csv_path):
        log(f"⚠️ Archivo no encontrado: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as exc:
        log(f"⚠️ Error leyendo {csv_path}: {exc}")
        return None


def plot_per_class_metrics(
    per_class_ap: Dict[str, float],
    save_path: str,
    title: str = "mAP@50 por Clase",
    show: bool = True,
) -> None:
    """Plot per-class mAP as bar chart.

    Args:
        per_class_ap: Dictionary of {class_name: ap_value}
        save_path: Path to save the plot
        title: Plot title
        show: Whether to display the plot
    """
    if not per_class_ap:
        log("⚠️ No hay métricas por clase para graficar")
        return

    classes = list(per_class_ap.keys())
    values = list(per_class_ap.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(classes, values, color=plt.cm.tab10.colors[:len(classes)])
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Clase")
    ax.set_ylabel("mAP@50")
    ax.set_ylim(0, 1)
    ax.axhline(y=np.mean(values), color="red", linestyle="--", alpha=0.7, label=f"Media: {np.mean(values):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    safe_mkdir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    log(f"✅ Gráfico por clase guardado en: {save_path}")
