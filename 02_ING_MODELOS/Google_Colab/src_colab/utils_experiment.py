"""Unified experiment tracking schema and cross-experiment comparison.

Every run (YOLO11, YOLO26, MobileNetV2-SSD, MobileNetV3-SSD) produces
a single JSON file with the same schema, making it trivial to load all
experiments and compare them on the same axes.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .utils_io import log, safe_mkdir, write_json, read_json


# =====================================================================
#  Unified schema
# =====================================================================

@dataclass
class UnifiedExperimentConfig:
    """Common configuration — every family fills the applicable fields."""
    experiment_name: str = ""
    family: str = ""           # yolo11 | yolo26 | mobilenet_v2 | mobilenet_v3
    model_variant: str = ""    # n | s | m | l | x  (YOLO) / small | large (MBN)
    base_model: str = ""       # yolo11n.pt | mobilenet_v3
    imgsz: int = 224
    batch_size: int = 16
    # classes
    class_names: List[str] = field(default_factory=list)
    num_classes: int = 0
    # YOLO-specific
    yolo_epochs: int = 0
    yolo_patience: int = 0
    yolo_optimizer: str = ""
    yolo_lr0: float = 0.0
    yolo_lrf: float = 0.0
    yolo_mosaic: float = 0.0
    yolo_mixup: float = 0.0
    yolo_augment_params: Dict[str, Any] = field(default_factory=dict)
    # MobileNet-specific
    mbn_phase1_epochs: int = 0
    mbn_phase1_lr: float = 0.0
    mbn_phase2_epochs: int = 0
    mbn_phase2_lr: float = 0.0
    mbn_unfreeze_layers: int = 0
    mbn_augment_level: str = ""
    mbn_anchor_sizes: List[float] = field(default_factory=list)
    mbn_anchor_ratios: List[float] = field(default_factory=list)
    # paths
    dataset_path: str = ""
    output_dir: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedExperimentResults:
    """Common results — filled post-training + post-evaluation."""
    # training metrics (final epoch)
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = 0.0
    best_epoch: int = 0
    total_epochs_run: int = 0
    training_time_min: float = 0.0
    # eval on val set
    val_mAP50: float = 0.0
    val_mAP50_95: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    val_per_class_ap50: Dict[str, float] = field(default_factory=dict)
    # eval on test set
    test_mAP50: float = 0.0
    test_mAP50_95: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_f1: float = 0.0
    test_per_class_ap50: Dict[str, float] = field(default_factory=dict)
    # export
    tflite_size_mb: float = 0.0
    tflite_esp32_ok: bool = False
    tflite_agreement: float = 0.0
    tflite_avg_latency_ms: float = 0.0
    # framework inference
    framework_avg_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedExperiment:
    """Complete experiment record."""
    config: UnifiedExperimentConfig = field(default_factory=UnifiedExperimentConfig)
    results: UnifiedExperimentResults = field(default_factory=UnifiedExperimentResults)
    notes: str = ""
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "config": self.config.to_dict(),
            "results": self.results.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnifiedExperiment":
        exp = cls()
        exp.version = d.get("version", "1.0")
        exp.notes = d.get("notes", "")
        if "config" in d:
            for k, v in d["config"].items():
                if hasattr(exp.config, k):
                    setattr(exp.config, k, v)
        if "results" in d:
            for k, v in d["results"].items():
                if hasattr(exp.results, k):
                    setattr(exp.results, k, v)
        return exp


# =====================================================================
#  Save / Load
# =====================================================================

def save_experiment(exp: UnifiedExperiment, output_dir: str) -> str:
    """Save experiment JSON to ``output_dir/experiment.json``."""
    safe_mkdir(output_dir)
    path = os.path.join(output_dir, "experiment.json")
    write_json(path, exp.to_dict())
    log(f"💾 Experimento guardado: {path}")
    return path


def load_experiment(path: str) -> UnifiedExperiment:
    """Load experiment from JSON."""
    d = read_json(path)
    return UnifiedExperiment.from_dict(d)


def load_all_experiments(base_dir: str) -> List[UnifiedExperiment]:
    """Recursively find and load all ``experiment.json`` files."""
    experiments = []
    for root, _, files in os.walk(base_dir):
        if "experiment.json" in files:
            try:
                exp = load_experiment(os.path.join(root, "experiment.json"))
                experiments.append(exp)
            except Exception as exc:
                log(f"⚠️  Error loading {root}: {exc}")
    log(f"📂 Cargados {len(experiments)} experimentos de {base_dir}")
    return experiments


# =====================================================================
#  Create experiment from widget setup
# =====================================================================

def create_experiment_from_setup(setup) -> UnifiedExperiment:
    """Create a :class:`UnifiedExperiment` from an ``ExperimentSetup``.

    The ``ExperimentSetup`` comes from ``utils_widgets.create_model_selector()``.
    """
    exp = UnifiedExperiment()
    c = exp.config

    c.experiment_name = setup.experiment_name
    c.family = setup.model_family
    c.model_variant = setup.model_variant
    c.imgsz = setup.img_size
    c.batch_size = setup.batch_size
    c.class_names = setup.class_names
    c.num_classes = len(setup.class_names)
    c.dataset_path = setup.dataset_name  # dataset_name, not path
    c.timestamp = datetime.now().isoformat()

    if "yolo" in setup.model_family.lower():
        yc = setup.yolo_config
        c.base_model = f"{setup.model_variant}.pt"
        c.yolo_epochs = yc.get("epochs", 100)
        c.yolo_patience = setup.patience
        c.yolo_optimizer = yc.get("optimizer", "auto")
        c.yolo_lr0 = yc.get("lr0", 0.01)
        c.yolo_lrf = yc.get("lrf", 0.01)
        c.yolo_mosaic = yc.get("mosaic", 1.0)
        c.yolo_mixup = yc.get("mixup", 0.0)
    else:
        mc = setup.mobilenet_config
        c.base_model = setup.model_family
        c.mbn_phase1_epochs = mc.get("phase1_epochs", 20)
        c.mbn_phase1_lr = mc.get("phase1_lr", 1e-3)
        c.mbn_phase2_epochs = mc.get("phase2_epochs", 50)
        c.mbn_phase2_lr = mc.get("phase2_lr", 1e-4)
        c.mbn_unfreeze_layers = mc.get("phase2_unfreeze_layers", 20)
        c.mbn_augment_level = mc.get("augmentation_level", "medium")

    return exp


# =====================================================================
#  Cross-experiment comparison
# =====================================================================

def experiments_to_dataframe(experiments: List[UnifiedExperiment]):
    """Convert list of experiments to a Pandas DataFrame for analysis."""
    import pandas as pd

    rows = []
    for exp in experiments:
        row = {
            "name": exp.config.experiment_name,
            "family": exp.config.family,
            "variant": exp.config.model_variant,
            "imgsz": exp.config.imgsz,
            "batch": exp.config.batch_size,
            "num_classes": exp.config.num_classes,
            "val_mAP50": exp.results.val_mAP50,
            "val_mAP50_95": exp.results.val_mAP50_95,
            "val_precision": exp.results.val_precision,
            "val_recall": exp.results.val_recall,
            "val_f1": exp.results.val_f1,
            "test_mAP50": exp.results.test_mAP50,
            "test_precision": exp.results.test_precision,
            "test_recall": exp.results.test_recall,
            "test_f1": exp.results.test_f1,
            "tflite_mb": exp.results.tflite_size_mb,
            "tflite_ok": exp.results.tflite_esp32_ok,
            "tflite_agreement": exp.results.tflite_agreement,
            "tflite_ms": exp.results.tflite_avg_latency_ms,
            "train_time_min": exp.results.training_time_min,
            "total_epochs": exp.results.total_epochs_run,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_experiments_comparison(
    experiments: List[UnifiedExperiment],
    save_path: Optional[str] = None,
    figsize: tuple = (22, 14),
) -> None:
    """Six-panel comparison across all experiments.

    Layout:
    [val_mAP50 bar]        [val P/R/F1 grouped bar]  [TFLite size bar]
    [per-class AP heatmap] [train time vs mAP]        [Summary table]
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = experiments_to_dataframe(experiments)
    if df.empty:
        log("⚠️ No hay experimentos para comparar")
        return

    names = df["name"].tolist()
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Comparativa de Experimentos", fontsize=16, fontweight="bold")

    # (0,0) val mAP@50
    ax = axes[0, 0]
    colors = _family_colors(df["family"].tolist())
    ax.bar(x, df["val_mAP50"], color=colors, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mAP@50"); ax.set_title("Val mAP@50")
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)

    # (0,1) P/R/F1
    ax = axes[0, 1]
    w = 0.25
    ax.bar(x - w, df["val_precision"], w, label="Precision", color="tab:blue", alpha=0.8)
    ax.bar(x, df["val_recall"], w, label="Recall", color="tab:orange", alpha=0.8)
    ax.bar(x + w, df["val_f1"], w, label="F1", color="tab:green", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score"); ax.set_title("Val P / R / F1")
    ax.legend(fontsize=8); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)

    # (0,2) TFLite size
    ax = axes[0, 2]
    bars = ax.bar(x, df["tflite_mb"], color=colors, alpha=0.8)
    ax.axhline(8.0, color="red", linestyle="--", label="ESP32-S3 limit (8 MB)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Size (MB)"); ax.set_title("TFLite INT8 Size")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # (1,0) per-class AP heatmap
    ax = axes[1, 0]
    all_classes = sorted({
        c for exp in experiments for c in exp.results.val_per_class_ap50
    })
    if all_classes:
        heatmap = np.zeros((len(experiments), len(all_classes)))
        for i, exp in enumerate(experiments):
            for j, cls in enumerate(all_classes):
                heatmap[i, j] = exp.results.val_per_class_ap50.get(cls, 0.0)
        im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(all_classes)))
        ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                ax.text(j, i, f"{heatmap[i,j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Per-Class AP@50")

    # (1,1) train time vs mAP scatter
    ax = axes[1, 1]
    ax.scatter(df["train_time_min"], df["val_mAP50"], c=colors, s=100, alpha=0.8, edgecolors="black")
    for i, name in enumerate(names):
        ax.annotate(name, (df["train_time_min"].iloc[i], df["val_mAP50"].iloc[i]),
                    fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Training Time (min)"); ax.set_ylabel("Val mAP@50")
    ax.set_title("Eficiencia: Tiempo vs Calidad"); ax.grid(alpha=0.3)

    # (1,2) summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for _, row in df.iterrows():
        esp = "✅" if row["tflite_ok"] else "❌"
        table_data.append([
            row["name"], row["family"],
            f"{row['val_mAP50']:.3f}", f"{row['val_f1']:.3f}",
            f"{row['tflite_mb']:.1f}", esp,
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["Exp", "Family", "mAP50", "F1", "TFLite MB", "ESP"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title("Resumen", pad=20)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log(f"📊 Comparativa guardada: {save_path}")
    plt.show()


def print_experiments_table(experiments: List[UnifiedExperiment]) -> None:
    """Print a compact text table of all experiments."""
    import pandas as pd

    df = experiments_to_dataframe(experiments)
    cols = ["name", "family", "variant", "val_mAP50", "val_f1",
            "tflite_mb", "tflite_ok", "train_time_min"]
    available = [c for c in cols if c in df.columns]
    log("\n" + df[available].to_string(index=False))


def save_comparison_csv(
    experiments: List[UnifiedExperiment],
    output_path: str,
) -> None:
    """Save all experiments as a comparison CSV."""
    df = experiments_to_dataframe(experiments)
    safe_mkdir(Path(output_path).parent)
    df.to_csv(output_path, index=False)
    log(f"💾 CSV comparativo: {output_path}")


# =====================================================================
#  Internal helpers
# =====================================================================

_FAMILY_COLOR_MAP = {
    "yolo11": "tab:blue",
    "yolo26": "tab:cyan",
    "mobilenet_v2": "tab:orange",
    "mobilenet_v3": "tab:green",
}


def _family_colors(families: List[str]) -> List[str]:
    return [_FAMILY_COLOR_MAP.get(f, "tab:gray") for f in families]
