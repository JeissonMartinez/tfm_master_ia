"""YOLO26 experiment tracking and serialization utilities.

Handles saving and loading experiment configurations and results
for reproducibility and comparison.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils_io import log, safe_exists, safe_mkdir, safe_read_json, safe_write_json


@dataclass
class Yolo26ExperimentConfig:
    """Configuration for a YOLO26 training experiment."""
    
    # Experiment identification
    name: str
    description: str = ""
    
    # Model configuration
    model: str = "yolo26n.pt"
    imgsz: int = 224
    
    # Training parameters
    epochs: int = 100
    patience: int = 50
    batch: int = 16
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    cos_lr: bool = True
    
    # Augmentation
    mosaic: float = 1.0
    mixup: float = 0.1
    close_mosaic: int = 10
    scale: float = 0.5
    fliplr: float = 0.5
    
    # Loss weights
    box: float = 7.5
    cls: float = 0.5
    
    # Dataset
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: ["door", "footpath", "obstacle", "person"])
    
    # YOLO26 specific
    end2end: bool = True  # NMS-free inference
    
    # Evaluation/Inference thresholds
    conf_threshold: float = 0.25  # Confidence threshold for detections
    iou_threshold: float = 0.5   # IoU threshold for NMS and matching
    
    # Paths (relative to project root)
    data_yaml: str = ""
    project_dir: str = "logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Yolo26ExperimentResults:
    """Results from a YOLO26 training experiment."""
    
    # Training metrics
    best_epoch: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    
    # Evaluation metrics (Keras/PyTorch)
    map50_keras: float = 0.0
    map50_95_keras: float = 0.0
    precision_keras: float = 0.0
    recall_keras: float = 0.0
    f1_keras: float = 0.0
    per_class_ap50_keras: Dict[str, float] = field(default_factory=dict)
    per_class_precision_keras: Dict[str, float] = field(default_factory=dict)
    per_class_recall_keras: Dict[str, float] = field(default_factory=dict)
    
    # Evaluation metrics (TFLite)
    map50_tflite: float = 0.0
    map50_95_tflite: float = 0.0
    precision_tflite: float = 0.0
    recall_tflite: float = 0.0
    f1_tflite: float = 0.0
    per_class_ap50_tflite: Dict[str, float] = field(default_factory=dict)
    
    # Confusion matrix (stored as list of lists for JSON serialization)
    confusion_matrix: List[List[float]] = field(default_factory=list)
    
    # Evaluation stats
    total_images: int = 0
    total_predictions: int = 0
    total_ground_truth: int = 0
    
    # Model sizes
    pytorch_model_size_mb: float = 0.0
    tflite_model_size_mb: float = 0.0
    tflite_int8_size_mb: float = 0.0
    
    # Inference times
    inference_time_pytorch_ms: float = 0.0
    inference_time_tflite_ms: float = 0.0
    
    # PyTorch vs TFLite comparison - Basic
    comparison_speedup: float = 0.0
    comparison_keras_detections: int = 0
    comparison_tflite_detections: int = 0
    comparison_keras_detections_per_image: float = 0.0
    comparison_tflite_detections_per_image: float = 0.0
    
    # PyTorch vs TFLite comparison - Agreement metrics
    comparison_agreement_rate: float = 0.0
    comparison_matched_detections: int = 0
    comparison_unmatched_keras: int = 0
    comparison_unmatched_tflite: int = 0
    
    # PyTorch vs TFLite comparison - IoU statistics
    comparison_mean_iou: float = 0.0
    comparison_min_iou: float = 0.0
    comparison_max_iou: float = 0.0
    comparison_std_iou: float = 0.0
    
    # PyTorch vs TFLite comparison - Confidence comparison
    comparison_mean_conf_keras: float = 0.0
    comparison_mean_conf_tflite: float = 0.0
    comparison_mean_conf_diff: float = 0.0
    comparison_conf_correlation: float = 0.0
    
    # PyTorch vs TFLite comparison - Per-class
    comparison_per_class_agreement: Dict[str, float] = field(default_factory=dict)
    comparison_per_class_detections_keras: Dict[str, int] = field(default_factory=dict)
    comparison_per_class_detections_tflite: Dict[str, int] = field(default_factory=dict)
    
    # Paths to saved artifacts
    best_model_path: str = ""
    tflite_model_path: str = ""
    history_csv_path: str = ""
    confusion_matrix_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def gap_keras_tflite(self) -> float:
        """Calculate mAP@50 gap between Keras and TFLite."""
        return abs(self.map50_keras - self.map50_tflite)


@dataclass
class Yolo26Experiment:
    """Complete experiment with config and results."""
    
    config: Yolo26ExperimentConfig
    results: Optional[Yolo26ExperimentResults] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    status: str = "pending"  # pending, running, completed, failed
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary for JSON serialization."""
        return {
            "config": self.config.to_dict(),
            "results": self.results.to_dict() if self.results else None,
            "timestamp": self.timestamp,
            "status": self.status,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Yolo26Experiment":
        """Create experiment from dictionary.
        
        Handles backward compatibility by ignoring unknown fields
        from older experiment versions.
        """
        # Get valid field names for each dataclass
        config_fields = {f.name for f in fields(Yolo26ExperimentConfig)}
        results_fields = {f.name for f in fields(Yolo26ExperimentResults)}
        
        # Filter config data to only include known fields
        config_data = data.get("config", {})
        filtered_config = {k: v for k, v in config_data.items() if k in config_fields}
        config = Yolo26ExperimentConfig(**filtered_config)
        
        # Filter results data to only include known fields
        results_data = data.get("results")
        if results_data:
            filtered_results = {k: v for k, v in results_data.items() if k in results_fields}
            results = Yolo26ExperimentResults(**filtered_results)
        else:
            results = None
        
        return cls(
            config=config,
            results=results,
            timestamp=data.get("timestamp", ""),
            status=data.get("status", "pending"),
            notes=data.get("notes", ""),
        )
    
    def summary(self) -> str:
        """Return formatted summary of experiment."""
        lines = [
            "=" * 70,
            f"🔬 EXPERIMENTO: {self.config.name}",
            "=" * 70,
            f"📝 Descripción: {self.config.description}",
            f"📅 Timestamp: {self.timestamp}",
            f"📊 Status: {self.status}",
            "",
            "⚙️ Configuración:",
            f"   Modelo: {self.config.model} | Tamaño: {self.config.imgsz}x{self.config.imgsz}",
            f"   Épocas: {self.config.epochs} | Batch: {self.config.batch}",
            f"   Optimizer: {self.config.optimizer} | LR: {self.config.lr0}",
            f"   Mosaic: {self.config.mosaic} | Mixup: {self.config.mixup}",
            f"   End-to-end: {self.config.end2end}",
        ]
        
        if self.results:
            lines.extend([
                "",
                "📊 Resultados:",
                f"   mAP@50 (PyTorch): {self.results.map50_keras:.4f}",
                f"   mAP@50 (TFLite):  {self.results.map50_tflite:.4f}",
                f"   Gap:              {self.results.gap_keras_tflite():.4f}",
                "",
                f"   Precision: {self.results.precision_keras:.4f}",
                f"   Recall:    {self.results.recall_keras:.4f}",
                f"   F1-Score:  {self.results.f1_keras:.4f}",
                "",
                f"   Tamaño TFLite INT8: {self.results.tflite_int8_size_mb:.2f} MB",
                f"   Tiempo inferencia:  {self.results.inference_time_tflite_ms:.2f} ms",
            ])
            
            if self.results.per_class_ap50_keras:
                lines.append("")
                lines.append("   mAP@50 por clase:")
                for cls_name, ap in self.results.per_class_ap50_keras.items():
                    lines.append(f"      {cls_name}: {ap:.4f}")
        
        if self.notes:
            lines.extend(["", f"📝 Notas: {self.notes}"])
        
        lines.append("=" * 70)
        return "\n".join(lines)


def save_experiment(
    experiment: Yolo26Experiment,
    experiments_dir: str,
) -> str:
    """Save experiment to JSON file.

    Args:
        experiment: Experiment to save
        experiments_dir: Directory for experiment files

    Returns:
        Path to saved file
    """
    safe_mkdir(experiments_dir)
    
    filename = f"{experiment.config.name}_{experiment.timestamp}.json"
    filepath = os.path.join(experiments_dir, filename)
    
    if safe_write_json(filepath, experiment.to_dict()):
        log(f"✅ Experimento guardado: {filepath}")
        return filepath
    else:
        log(f"❌ Error guardando experimento: {filepath}")
        return ""


def load_experiment(filepath: str) -> Optional[Yolo26Experiment]:
    """Load experiment from JSON file.

    Args:
        filepath: Path to experiment JSON file

    Returns:
        Experiment object or None on failure
    """
    data = safe_read_json(filepath)
    if data is None:
        return None
    
    try:
        return Yolo26Experiment.from_dict(data)
    except Exception as exc:
        log(f"❌ Error cargando experimento: {exc}")
        return None


def load_all_experiments(experiments_dir: str) -> List[Yolo26Experiment]:
    """Load all experiments from a directory.

    Args:
        experiments_dir: Directory containing experiment JSON files

    Returns:
        List of Experiment objects
    """
    if not safe_exists(experiments_dir):
        log(f"⚠️ Directorio no encontrado: {experiments_dir}")
        return []

    experiments = []
    for filename in sorted(os.listdir(experiments_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(experiments_dir, filename)
            exp = load_experiment(filepath)
            if exp:
                experiments.append(exp)

    log(f"📦 Cargados {len(experiments)} experimentos de {experiments_dir}")
    return experiments


def compare_experiments(
    experiments: List[Yolo26Experiment],
    sort_by: str = "map50_keras",
    ascending: bool = False,
) -> pd.DataFrame:
    """Create comparison DataFrame from experiments.

    Args:
        experiments: List of experiments to compare
        sort_by: Column to sort by
        ascending: Sort order

    Returns:
        DataFrame with comparison
    """
    rows = []
    for exp in experiments:
        if exp.results is None:
            continue
            
        row = {
            "name": exp.config.name,
            "timestamp": exp.timestamp,
            "status": exp.status,
            "model": exp.config.model,
            "imgsz": exp.config.imgsz,
            "epochs": exp.config.epochs,
            "batch": exp.config.batch,
            "mosaic": exp.config.mosaic,
            "mixup": exp.config.mixup,
            "map50_keras": exp.results.map50_keras,
            "map50_95_keras": exp.results.map50_95_keras,
            "precision_keras": exp.results.precision_keras,
            "recall_keras": exp.results.recall_keras,
            "f1_keras": exp.results.f1_keras,
            "map50_tflite": exp.results.map50_tflite,
            "gap_keras_tflite": exp.results.gap_keras_tflite(),
            "tflite_size_mb": exp.results.tflite_int8_size_mb,
            "inference_ms": exp.results.inference_time_tflite_ms,
        }
        
        # Add per-class AP
        for cls_name, ap in exp.results.per_class_ap50_keras.items():
            row[f"ap50_{cls_name}"] = ap
        
        rows.append(row)

    if not rows:
        log("⚠️ No hay experimentos con resultados para comparar")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    
    return df


def print_experiments_table(
    experiments: List[Yolo26Experiment],
    max_rows: int = 20,
) -> None:
    """Print formatted comparison table of experiments.

    Args:
        experiments: List of experiments
        max_rows: Maximum rows to display
    """
    df = compare_experiments(experiments)
    
    if df.empty:
        return

    # Select key columns for display
    display_cols = [
        "name", "map50_keras", "map50_tflite", "gap_keras_tflite",
        "precision_keras", "recall_keras", "tflite_size_mb"
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    log("\n" + "=" * 100)
    log("📊 COMPARACIÓN DE EXPERIMENTOS YOLO26")
    log("=" * 100)
    
    # Print header
    header = f"{'Name':<25} {'mAP@50 PT':>10} {'mAP@50 TF':>10} {'Gap':>8} {'Prec':>8} {'Rec':>8} {'Size MB':>8}"
    log(header)
    log("-" * 100)
    
    # Print rows
    for _, row in df.head(max_rows).iterrows():
        line = f"{row['name']:<25} "
        line += f"{row.get('map50_keras', 0):>10.4f} "
        line += f"{row.get('map50_tflite', 0):>10.4f} "
        line += f"{row.get('gap_keras_tflite', 0):>8.4f} "
        line += f"{row.get('precision_keras', 0):>8.4f} "
        line += f"{row.get('recall_keras', 0):>8.4f} "
        line += f"{row.get('tflite_size_mb', 0):>8.2f}"
        log(line)
    
    log("=" * 100)
    
    if len(df) > max_rows:
        log(f"... y {len(df) - max_rows} experimentos más")


def export_experiments_csv(
    experiments: List[Yolo26Experiment],
    output_path: str,
) -> bool:
    """Export experiments comparison to CSV.

    Args:
        experiments: List of experiments
        output_path: Path to save CSV

    Returns:
        True if successful
    """
    df = compare_experiments(experiments)
    
    if df.empty:
        return False

    try:
        safe_mkdir(os.path.dirname(output_path))
        df.to_csv(output_path, index=False)
        log(f"✅ Comparación exportada: {output_path}")
        return True
    except Exception as exc:
        log(f"❌ Error exportando CSV: {exc}")
        return False
