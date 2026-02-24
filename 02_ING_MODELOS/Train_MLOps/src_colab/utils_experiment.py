"""Experiment tracking & comparison — Cycle 2.

Provides ``UnifiedExperimentConfig`` and ``UnifiedExperiment`` for
persisting training parameters / results and comparing across runs.
"""
from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils_io import log, safe_mkdir


# =====================================================================
#  Experiment configuration
# =====================================================================

@dataclass
class TwoPhaseTrainConf:
    """Training hyper-parameters for a single two-phase schedule."""
    phase1_epochs: int = 30
    phase2_epochs: int = 60
    phase1_lr: float = 1e-3
    phase2_lr: float = 1e-4
    phase1_wd: float = 1e-4
    phase2_wd: float = 1e-5
    resize_schedule: Dict[int, int] = field(default_factory=lambda: {
        0: 640, 10: 416, 20: 320, 30: 224
    })
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    amp: bool = True
    grad_clip: float = 10.0
    batch_size: int = 16


@dataclass
class Yolo26TrainConf:
    """Ultralytics-managed training config for YOLO26 Custom."""
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.01
    lrf: float = 0.01
    freeze_layers: int = 10
    pretrained_weights: str = "yolo11n.pt"
    mosaic: float = 1.0
    mixup: float = 0.1


@dataclass
class UnifiedExperimentConfig:
    """Unified experiment configuration for any model family."""
    # Identifiers
    experiment_name: str = ""
    family: str = ""           # FCOS | YOLO26_CUSTOM | ESPDet
    model_variant: str = ""    # e.g. "v3s_v1", "yolo26n_custom_v1"
    cycle: int = 2

    # Dataset
    dataset_name: str = "IODC"
    dataset_format: str = "yolo"
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: [
        "dog", "door", "obstacle", "person", "stair"
    ])

    # Device
    accelerator: str = "T4"
    machine_type: str = "n1-standard-8"

    # Family-specific training config
    two_phase: Optional[TwoPhaseTrainConf] = None    # FCOS / ESPDet
    yolo26: Optional[Yolo26TrainConf] = None          # YOLO26_CUSTOM

    # Export
    export_imgsz: int = 224
    export_opset: int = 13

    # Augmentation (aligned with IODCDataset aug_config keys since v2.6.2)
    aug_hflip_prob: float = 0.5
    aug_brightness_limit: float = 0.3
    aug_contrast_limit: float = 0.3
    aug_gaussian_noise: float = 0.2
    aug_rotate_limit: int = 15

    # Notes
    notes: str = ""
    tags: List[str] = field(default_factory=list)


# =====================================================================
#  Experiment result + lifecycle
# =====================================================================

@dataclass
class UnifiedExperiment:
    """Full experiment record — config + results."""
    config: UnifiedExperimentConfig = field(
        default_factory=UnifiedExperimentConfig
    )
    # Timestamps
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0

    # Status
    status: str = "created"  # created | running | completed | failed

    # Training results
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0

    # Evaluation results
    val_map50: float = 0.0
    val_map50_per_class: Dict[str, float] = field(default_factory=dict)
    test_map50: float = 0.0
    test_map50_per_class: Dict[str, float] = field(default_factory=dict)

    # Export results
    onnx_size_mb: float = 0.0
    onnx_latency_ms: float = 0.0

    # Artifact paths (relative to experiment dir)
    model_path: str = ""
    onnx_path: str = ""
    history_csv: str = ""
    curves_png: str = ""

    # Free-form logs
    log_lines: List[str] = field(default_factory=list)
    error_msg: str = ""

    def mark_running(self):
        self.status = "running"
        self.started_at = datetime.now().isoformat()

    def mark_completed(self):
        self.status = "completed"
        self.finished_at = datetime.now().isoformat()
        if self.started_at:
            t0 = datetime.fromisoformat(self.started_at)
            t1 = datetime.fromisoformat(self.finished_at)
            self.duration_s = (t1 - t0).total_seconds()

    def mark_failed(self, msg: str):
        self.status = "failed"
        self.error_msg = msg
        self.finished_at = datetime.now().isoformat()

    def add_log(self, line: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {line}")


# =====================================================================
#  Factory from ExperimentSetup (YAML → ExperimentSetup → Config)
# =====================================================================

def create_experiment_from_setup(setup) -> UnifiedExperiment:
    """Build a ``UnifiedExperiment`` from ``ExperimentSetup``.

    Args:
        setup: An ExperimentSetup instance (from utils_widgets).

    Returns:
        Initialized UnifiedExperiment ready to ``mark_running()``.
    """
    family = setup.model_family.upper()
    fc = setup.family_config  # dict from YAML for the selected family

    cfg = UnifiedExperimentConfig(
        experiment_name=setup.experiment_name,
        family=family,
        model_variant=fc.get("variant", ""),
        dataset_name=setup.dataset_name,
        num_classes=setup.num_classes,
        class_names=setup.class_names,
        accelerator=getattr(setup, "accelerator_type", "T4"),
    )

    if family in ("FCOS", "ESPDET"):
        cfg.two_phase = TwoPhaseTrainConf(
            phase1_epochs=fc.get("phase1_epochs", 30),
            phase2_epochs=fc.get("phase2_epochs", 60),
            phase1_lr=fc.get("phase1_lr", 1e-3),
            phase2_lr=fc.get("phase2_lr", 1e-4),
            batch_size=fc.get("batch_size", setup.batch_size),
            resize_schedule=fc.get("resize_schedule", {0: 640, 10: 416, 20: 320, 30: 224}),
            amp=fc.get("amp", True),
            grad_clip=fc.get("grad_clip", 10.0),
        )
    elif family == "YOLO26_CUSTOM":
        cfg.yolo26 = Yolo26TrainConf(
            epochs=fc.get("epochs", 100),
            imgsz=fc.get("imgsz", 640),
            batch=fc.get("batch", 16),
            lr0=fc.get("lr0", 0.01),
            lrf=fc.get("lrf", 0.01),
            freeze_layers=fc.get("freeze_layers", 10),
            pretrained_weights=fc.get("pretrained_weights", "yolo11n.pt"),
        )

    cfg.export_imgsz = fc.get("export_imgsz", 224)
    cfg.notes = fc.get("notes", "")
    cfg.tags = fc.get("tags", [])

    # Populate augmentation fields from family config (v2.6.2)
    cfg.aug_hflip_prob = fc.get("aug_hflip_prob", 0.5)
    cfg.aug_brightness_limit = fc.get("aug_brightness_limit", 0.3)
    cfg.aug_contrast_limit = fc.get("aug_contrast_limit", 0.3)
    cfg.aug_gaussian_noise = fc.get("aug_gaussian_noise", 0.2)
    cfg.aug_rotate_limit = fc.get("aug_rotate_limit", 15)

    exp = UnifiedExperiment(config=cfg)
    exp.created_at = datetime.now().isoformat()
    return exp


# =====================================================================
#  Persistence
# =====================================================================

def save_experiment(exp: UnifiedExperiment, path: str) -> str:
    """Save experiment to JSON."""
    safe_mkdir(os.path.dirname(path))
    data = _to_serializable(asdict(exp))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log(f"💾 Experimento guardado: {path}")
    return path


def load_experiment(path: str) -> UnifiedExperiment:
    """Load experiment from JSON."""
    with open(path) as f:
        data = json.load(f)

    cfg_data = data.pop("config", {})
    two_phase_data = cfg_data.pop("two_phase", None)
    yolo26_data = cfg_data.pop("yolo26", None)

    cfg = UnifiedExperimentConfig(**cfg_data)
    if two_phase_data:
        cfg.two_phase = TwoPhaseTrainConf(**two_phase_data)
    if yolo26_data:
        cfg.yolo26 = Yolo26TrainConf(**yolo26_data)

    exp = UnifiedExperiment(config=cfg, **data)
    return exp


def load_experiments(directory: str) -> List[UnifiedExperiment]:
    """Load all experiment JSONs from a directory."""
    exps = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            try:
                exps.append(load_experiment(os.path.join(directory, fname)))
            except Exception as e:
                log(f"⚠️ No se pudo cargar {fname}: {e}")
    return exps


def _to_serializable(obj):
    """Recursively convert dataclass fields to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, float) and (obj != obj):  # NaN
        return None
    elif isinstance(obj, float) and obj == float("inf"):
        return "inf"
    return obj


# =====================================================================
#  Comparison plots
# =====================================================================

def plot_experiments_comparison(
    experiments: List[UnifiedExperiment],
    metric: str = "val_map50",
    save_path: Optional[str] = None,
    title: str = "",
) -> None:
    """Bar chart comparing a metric across experiments."""
    import matplotlib.pyplot as plt

    names = [e.config.experiment_name or e.config.family for e in experiments]
    values = [getattr(e, metric, 0.0) for e in experiments]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    colors = plt.cm.Set2(range(len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Comparación: {metric}")
    ax.set_ylim(0, max(values) * 1.2 if values and max(values) > 0 else 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        safe_mkdir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_experiments_radar(
    experiments: List[UnifiedExperiment],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Radar / spider chart comparing multiple metrics across experiments."""
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = ["val_map50", "test_map50", "best_val_loss",
                   "onnx_size_mb", "onnx_latency_ms"]

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for exp in experiments:
        vals = []
        for m in metrics:
            v = getattr(exp, m, 0.0)
            if v == float("inf"):
                v = 0.0
            vals.append(v)
        vals.append(vals[0])
        label = exp.config.experiment_name or exp.config.family
        ax.plot(angles, vals, "o-", label=label, linewidth=2)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]],
        [m.replace("_", "\n") for m in metrics],
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_experiments_table(experiments: List[UnifiedExperiment]) -> None:
    """Print a summary table of experiments."""
    header = (
        f"{'Name':<25} {'Family':<15} {'Status':<10} "
        f"{'Val mAP50':>10} {'Test mAP50':>10} {'ONNX MB':>8} {'Time':>8}"
    )
    log(header)
    log("-" * len(header))
    for e in experiments:
        dur = f"{e.duration_s / 60:.0f}m" if e.duration_s > 0 else "–"
        log(
            f"{(e.config.experiment_name or '?'):<25} "
            f"{e.config.family:<15} {e.status:<10} "
            f"{e.val_map50:>10.4f} {e.test_map50:>10.4f} "
            f"{e.onnx_size_mb:>8.2f} {dur:>8}"
        )


# Ensure numpy is importable for radar plot
try:
    import numpy as np  # noqa: F811
except ImportError:
    pass
