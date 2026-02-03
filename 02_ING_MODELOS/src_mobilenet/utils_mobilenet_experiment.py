"""Experiment management for object detection models.

This module provides:
- Experiment configuration dataclass
- Saving/loading experiments
- Experiment comparison
- Visualization of multiple experiments
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""
    
    # Experiment identification
    name: str
    description: str = ""
    
    # Model configuration
    backbone: str = "MobileNetV3Small"
    minimalistic: bool = True
    num_anchors: int = 9
    num_classes: int = 4
    
    # Data configuration
    selected_classes: List[str] = field(default_factory=lambda: ["door", "footpath", "obstacle", "person"])
    img_size: int = 224
    
    # Augmentation configuration
    augmentation_level: str = "medium"  # "none", "light", "medium", "heavy"
    
    # Training configuration - Phase 1 (frozen backbone)
    phase1_epochs: int = 15
    phase1_lr: float = 1e-3
    
    # Training configuration - Phase 2 (unfrozen)
    phase2_epochs: int = 30
    phase2_lr: float = 1e-4
    
    # Common training params
    batch_size: int = 16
    warmup_epochs: int = 3
    
    # Loss configuration
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Regularization
    dropout_rate: float = 0.2
    l2_reg: float = 1e-4
    
    # TFLite configuration
    quantization: str = "int8"  # "float32", "float16", "int8"

    # Inference configuration
    score_threshold: float = 0.4

    # Número de canales en las características extraídas por el backbone
    feature_channels: int = 128
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**data)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"💾 Config saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def print_summary(self):
        """Print formatted configuration summary."""
        print("\n" + "="*60)
        print(f"🔬 EXPERIMENT: {self.name}")
        print("="*60)
        
        if self.description:
            print(f"📝 {self.description}")
        
        print(f"\n🏗️ Model:")
        print(f"   Backbone: {self.backbone} (minimalistic={self.minimalistic})")
        print(f"   Anchors: {self.num_anchors}")
        print(f"   Classes: {self.num_classes} - {self.selected_classes}")
        
        print(f"\n📊 Data:")
        print(f"   Image size: {self.img_size}x{self.img_size}")
        print(f"   Augmentation: {self.augmentation_level}")
        
        print(f"\n🎯 Training:")
        print(f"   Phase 1: {self.phase1_epochs} epochs @ lr={self.phase1_lr}")
        print(f"   Phase 2: {self.phase2_epochs} epochs @ lr={self.phase2_lr}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Focal loss: {self.use_focal_loss} (α={self.focal_alpha}, γ={self.focal_gamma})")
        
        print(f"\n📦 TFLite:")
        print(f"   Quantization: {self.quantization}")
        
        print("="*60)


@dataclass
class ExperimentResults:
    """Results from a completed experiment."""
    
    # Metrics
    map50_keras: float = 0.0
    map50_tflite: float = 0.0
    precision_keras: float = 0.0
    precision_tflite: float = 0.0
    recall_keras: float = 0.0
    recall_tflite: float = 0.0
    f1_keras: float = 0.0
    f1_tflite: float = 0.0
    
    # Per-class mAP
    ap_per_class_keras: Dict[str, float] = field(default_factory=dict)
    ap_per_class_tflite: Dict[str, float] = field(default_factory=dict)
    
    # Training metrics
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = 0.0
    total_epochs: int = 0
    training_time_minutes: float = 0.0
    
    # Model info
    keras_model_size_mb: float = 0.0
    tflite_model_size_kb: float = 0.0
    
    # Paths
    keras_model_path: str = ""
    tflite_model_path: str = ""
    history_csv_path: str = ""
    confusion_matrix_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResults":
        return cls(**data)


@dataclass
class Experiment:
    """Complete experiment with config and results."""
    config: ExperimentConfig
    results: Optional[ExperimentResults] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "results": self.results.to_dict() if self.results else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        config = ExperimentConfig.from_dict(data["config"])
        results = ExperimentResults.from_dict(data["results"]) if data.get("results") else None
        return cls(config=config, results=results)


def save_experiment(
    experiment: Experiment,
    base_dir: str = "logs/experiments",
) -> str:
    """Save complete experiment to JSON.
    
    Args:
        experiment: Experiment to save
        base_dir: Base directory for experiments
        
    Returns:
        Path to saved experiment file
    """
    # Create directory
    exp_dir = Path(base_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from experiment name
    safe_name = experiment.config.name.replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.json"
    filepath = exp_dir / filename
    
    # Save
    with open(filepath, 'w') as f:
        json.dump(experiment.to_dict(), f, indent=2)
    
    print(f"💾 Experiment saved to: {filepath}")
    return str(filepath)


def load_experiment(path: str) -> Experiment:
    """Load experiment from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return Experiment.from_dict(data)


def load_all_experiments(base_dir: str = "logs/experiments") -> List[Experiment]:
    """Load all experiments from directory."""
    exp_dir = Path(base_dir)
    experiments = []
    
    if not exp_dir.exists():
        return experiments
    
    for filepath in sorted(exp_dir.glob("*.json")):
        try:
            exp = load_experiment(str(filepath))
            experiments.append(exp)
        except Exception as e:
            print(f"⚠️ Failed to load {filepath}: {e}")
    
    print(f"📂 Loaded {len(experiments)} experiments from {base_dir}")
    return experiments


def save_experiment_history(
    history: dict,
    experiment_name: str,
    output_dir: str = "logs",
) -> str:
    """Save training history to CSV (alternative to utils_mobilenet_train version).
    
    Args:
        history: Keras history.history dict or combined dict (NOT the History object)
        experiment_name: Name of experiment
        output_dir: Output directory
        
    Returns:
        Path to saved CSV
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    safe_name = experiment_name.replace(" ", "_").lower()
    csv_path = output_path / f"{safe_name}_history.csv"
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    df.index.name = "epoch"
    df.to_csv(csv_path)
    
    print(f"💾 Training history saved to: {csv_path}")
    return str(csv_path)


def load_training_history(path: str) -> pd.DataFrame:
    """Load training history from CSV."""
    return pd.read_csv(path, index_col="epoch")


def experiments_to_dataframe(experiments: List[Experiment]) -> pd.DataFrame:
    """Convert list of experiments to comparison DataFrame."""
    rows = []
    
    for exp in experiments:
        if exp.results is None:
            continue
        
        row = {
            "name": exp.config.name,
            "description": exp.config.description,
            "created_at": exp.config.created_at,
            
            # Config
            "backbone": exp.config.backbone,
            "num_classes": exp.config.num_classes,
            "img_size": exp.config.img_size,
            "augmentation": exp.config.augmentation_level,
            "phase1_epochs": exp.config.phase1_epochs,
            "phase2_epochs": exp.config.phase2_epochs,
            "batch_size": exp.config.batch_size,
            "quantization": exp.config.quantization,
            
            # Results
            "mAP@50_keras": exp.results.map50_keras,
            "mAP@50_tflite": exp.results.map50_tflite,
            "precision_keras": exp.results.precision_keras,
            "precision_tflite": exp.results.precision_tflite,
            "recall_keras": exp.results.recall_keras,
            "recall_tflite": exp.results.recall_tflite,
            "f1_keras": exp.results.f1_keras,
            "f1_tflite": exp.results.f1_tflite,
            "final_val_loss": exp.results.final_val_loss,
            "best_val_loss": exp.results.best_val_loss,
            "training_time_min": exp.results.training_time_minutes,
            "keras_size_mb": exp.results.keras_model_size_mb,
            "tflite_size_kb": exp.results.tflite_model_size_kb,
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def compare_experiments(
    experiments: List[Experiment],
    metrics: List[str] = None,
    sort_by: str = "mAP@50_keras",
    ascending: bool = False,
) -> pd.DataFrame:
    """Compare multiple experiments in a formatted table.
    
    Args:
        experiments: List of experiments to compare
        metrics: List of metrics to show (None = all)
        sort_by: Column to sort by
        ascending: Sort order
        
    Returns:
        DataFrame with comparison
    """
    df = experiments_to_dataframe(experiments)
    
    if df.empty:
        print("⚠️ No completed experiments to compare")
        return df
    
    if metrics:
        cols = ["name"] + [c for c in metrics if c in df.columns]
        df = df[cols]
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    
    return df


def plot_experiments_comparison(
    experiments: List[Experiment],
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    esp32_limit_mb: float = 2.0,
) -> plt.Figure:
    """Create visual comparison of experiments.
    
    Args:
        experiments: List of experiments
        metrics: Metrics to plot (default: mAP, precision, recall, F1)
        figsize: Figure size
        save_path: Path to save figure
        title: Optional title for the entire figure
        esp32_limit_mb: ESP32-S3 memory limit in MB (default: 2.0)
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ["mAP@50_keras", "mAP@50_tflite", "precision_keras", "recall_keras", "f1_keras"]
    
    df = experiments_to_dataframe(experiments)
    
    if df.empty:
        print("⚠️ No completed experiments to plot")
        return None
    
    # Filter to available metrics
    metrics = [m for m in metrics if m in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    # Plot 1: mAP comparison
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35
    
    if "mAP@50_keras" in df.columns:
        ax.bar(x - width/2, df["mAP@50_keras"], width, label="Keras", color='steelblue')
    if "mAP@50_tflite" in df.columns:
        ax.bar(x + width/2, df["mAP@50_tflite"], width, label="TFLite", color='coral')
    
    ax.set_ylabel("mAP@50")
    ax.set_title("mAP@50: Keras vs TFLite")
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Precision/Recall/F1
    ax = axes[1]
    x = np.arange(len(df))
    width = 0.25
    
    if "precision_keras" in df.columns:
        ax.bar(x - width, df["precision_keras"], width, label="Precision", color='forestgreen')
    if "recall_keras" in df.columns:
        ax.bar(x, df["recall_keras"], width, label="Recall", color='royalblue')
    if "f1_keras" in df.columns:
        ax.bar(x + width, df["f1_keras"], width, label="F1", color='darkorange')
    
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 (Keras)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Model size
    ax = axes[2]
    if "tflite_size_kb" in df.columns:
        esp32_limit_kb = esp32_limit_mb * 1024
        bars = ax.bar(df["name"], df["tflite_size_kb"], color='purple', alpha=0.7)
        ax.axhline(y=esp32_limit_kb, color='red', linestyle='--', 
                   label=f"ESP32-S3 limit ({esp32_limit_mb:.1f}MB)")
        ax.set_ylabel("TFLite Size (KB)")
        ax.set_title("TFLite Model Size")
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, df["tflite_size_kb"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Training time
    ax = axes[3]
    if "training_time_min" in df.columns and df["training_time_min"].sum() > 0:
        ax.bar(df["name"], df["training_time_min"], color='teal', alpha=0.7)
        ax.set_ylabel("Training Time (minutes)")
        ax.set_title("Training Time")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    else:
        # Plot val loss instead
        if "best_val_loss" in df.columns:
            ax.bar(df["name"], df["best_val_loss"], color='indianred', alpha=0.7)
            ax.set_ylabel("Validation Loss")
            ax.set_title("Best Validation Loss")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
    
    # Add figure title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Comparison plot saved to: {save_path}")
    
    return fig


def plot_training_histories(
    experiment_names: List[str],
    history_dir: str = "logs",
    metric: str = "val_loss",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training histories for multiple experiments.
    
    Args:
        experiment_names: List of experiment names
        history_dir: Directory containing history CSVs
        metric: Metric to plot
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_names)))
    
    for i, name in enumerate(experiment_names):
        safe_name = name.replace(" ", "_").lower()
        csv_path = Path(history_dir) / f"{safe_name}_history.csv"
        
        if not csv_path.exists():
            print(f"⚠️ History not found: {csv_path}")
            continue
        
        df = load_training_history(str(csv_path))
        
        # Plot loss
        if "loss" in df.columns:
            axes[0].plot(df.index, df["loss"], label=f"{name} (train)", 
                        color=colors[i], linestyle='-')
        if "val_loss" in df.columns:
            axes[0].plot(df.index, df["val_loss"], label=f"{name} (val)", 
                        color=colors[i], linestyle='--')
    
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Comparison")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot metric (e.g., val_loss zoomed or other metric)
    for i, name in enumerate(experiment_names):
        safe_name = name.replace(" ", "_").lower()
        csv_path = Path(history_dir) / f"{safe_name}_history.csv"
        
        if not csv_path.exists():
            continue
        
        df = load_training_history(str(csv_path))
        
        if metric in df.columns:
            axes[1].plot(df.index, df[metric], label=name, color=colors[i])
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric)
    axes[1].set_title(f"{metric} Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 History comparison saved to: {save_path}")
    
    return fig


def create_experiment_summary_table(
    experiments: List[Experiment],
    output_path: Optional[str] = None,
) -> str:
    """Create formatted Markdown table of experiments.
    
    Args:
        experiments: List of experiments
        output_path: Optional path to save markdown
        
    Returns:
        Markdown string
    """
    df = experiments_to_dataframe(experiments)
    
    if df.empty:
        return "No experiments to display."
    
    # Select key columns
    cols = ["name", "mAP@50_keras", "mAP@50_tflite", "precision_keras", 
            "recall_keras", "f1_keras", "tflite_size_kb"]
    cols = [c for c in cols if c in df.columns]
    
    # Sort by mAP
    df = df.sort_values("mAP@50_keras", ascending=False)
    
    # Format table
    lines = [
        "# Experiment Comparison Summary\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    
    for _, row in df.iterrows():
        values = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    
    markdown = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(markdown)
        print(f"💾 Summary table saved to: {output_path}")
    
    return markdown


# Convenience function for quick experiment setup
def create_default_config(
    name: str,
    selected_classes: List[str] = None,
    **overrides: Any,
) -> ExperimentConfig:
    """Create experiment config with defaults.
    
    Args:
        name: Experiment name
        selected_classes: Classes to use (default: all 4)
        **overrides: Any config overrides
        
    Returns:
        ExperimentConfig
    """
    if selected_classes is None:
        selected_classes = ["door", "footpath", "obstacle", "person"]
    
    config = ExperimentConfig(
        name=name,
        selected_classes=selected_classes,
        num_classes=len(selected_classes),
    )
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
