"""YOLO metrics extraction and visualization utilities."""
from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt

from .utils_io import log, safe_copy, safe_exists, safe_filesize_mb


def extract_yolo_metrics(results_dir: str, logs_dir: str, version_tag: str) -> Optional[pd.DataFrame]:
    """Extract Ultralytics results.csv and copy best.pt.

    Returns cleaned DataFrame or None on failure.
    """
    csv_src = os.path.join(results_dir, "results.csv")
    csv_dst = os.path.join(logs_dir, f"{version_tag}_history.csv")
    best_src = os.path.join(results_dir, "weights", "best.pt")
    best_dst = os.path.join(logs_dir, "..", "models", "checkpoints", f"{version_tag}_best.pt")

    if not safe_exists(csv_src):
        log(f"⚠️ No se encontró: {csv_src}")
        return None

    try:
        df = pd.read_csv(csv_src)
        df.columns = df.columns.str.strip()
        if "epoch" not in df.columns:
            df.insert(0, "epoch", range(1, len(df) + 1))
        df.to_csv(csv_dst, index=False)
        log(f"✅ Historial guardado en: {csv_dst}")
    except Exception as exc:  # pragma: no cover - defensive
        log(f"⚠️ Error procesando {csv_src}: {exc}")
        return None

    if safe_copy(best_src, best_dst):
        size_mb = safe_filesize_mb(best_dst)
        if size_mb is not None:
            log(f"✅ Mejor modelo copiado: {best_dst} ({size_mb:.2f} MB, int8~{size_mb*0.25:.2f} MB)")

    return df


def plot_yolo_history(df: pd.DataFrame, save_path: str) -> None:
    """Plot training curves for a YOLO run."""
    required = ["epoch", "train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss"]
    if not all(col in df.columns for col in required):
        log("⚠️ Columnas insuficientes para graficar historial YOLO.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("YOLO - Evolución del Entrenamiento", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train/box_loss"], label="Train", alpha=0.7)
    ax.plot(df["epoch"], df["val/box_loss"], label="Val", linewidth=2)
    ax.set_title("Box Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train/cls_loss"], label="Train", alpha=0.7)
    ax.plot(df["epoch"], df["val/cls_loss"], label="Val", linewidth=2)
    ax.set_title("Cls Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if "train/dfl_loss" in df.columns and "val/dfl_loss" in df.columns:
        ax.plot(df["epoch"], df["train/dfl_loss"], label="Train", alpha=0.7)
        ax.plot(df["epoch"], df["val/dfl_loss"], label="Val", linewidth=2)
    ax.set_title("DFL Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if "metrics/mAP50(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50(B)"], linewidth=2, color="green")
    ax.set_title("mAP@50")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "metrics/mAP50-95(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], linewidth=2, color="blue")
    ax.set_title("mAP@50-95")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if "metrics/precision(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", linewidth=2)
    if "metrics/recall(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", linewidth=2)
    ax.set_title("Precision/Recall")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    log(f"✅ Gráfico guardado en: {save_path}")


def compare_yolo_versions(histories: Dict[str, pd.DataFrame], save_path: str) -> None:
    """Compare mAP and losses across multiple YOLO versions."""
    if not histories:
        log("⚠️ No hay historiales para comparar.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("YOLO - Comparativa de Versiones", fontsize=14, fontweight="bold")

    metrics = [
        ("val/box_loss", "Val Box Loss", axes[0, 0]),
        ("val/cls_loss", "Val Cls Loss", axes[0, 1]),
        ("metrics/mAP50(B)", "mAP@50", axes[0, 2]),
        ("metrics/mAP50-95(B)", "mAP@50-95", axes[1, 0]),
        ("metrics/precision(B)", "Precision", axes[1, 1]),
        ("metrics/recall(B)", "Recall", axes[1, 2]),
    ]

    for col, title, ax in metrics:
        for version, df in histories.items():
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label=version)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    log(f"✅ Gráfico guardado en: {save_path}")
