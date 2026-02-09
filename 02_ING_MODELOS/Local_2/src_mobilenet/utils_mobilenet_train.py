"""Training utilities for MobileNet + SSD-Lite.

This module provides:
- Two-phase training (warm-up + fine-tuning)
- Backbone freezing/unfreezing utilities
- Custom callbacks (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping)
- Training history logging
"""
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf


def freeze_backbone(model: tf.keras.Model, backbone_prefix: str = "") -> int:
    """Freeze all backbone layers (keep only SSD head trainable).
    
    Used in Phase 1 (warm-up) to train only the detection head.
    
    Args:
        model: The full model
        backbone_prefix: Optional prefix to identify backbone layers
    
    Returns:
        Number of layers frozen
    """
    frozen_count = 0
    for layer in model.layers:
        name = layer.name.lower()
        # Freeze if not part of SSD head
        if "ssd" not in name and "objectness" not in name and "class_out" not in name and "bbox_out" not in name:
            layer.trainable = False
            frozen_count += 1
        else:
            layer.trainable = True
    
    # Recompile needed after changing trainable
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    total_params = model.count_params()
    
    print(f"🔒 Frozen {frozen_count} backbone layers")
    print(f"   Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    return frozen_count


def unfreeze_backbone_layers(
    model: tf.keras.Model,
    num_layers_to_unfreeze: int = -1,
    unfreeze_from: str = "",
) -> int:
    """Unfreeze backbone layers for fine-tuning.
    
    Used in Phase 2 to fine-tune the last blocks of the backbone.
    
    Args:
        model: The full model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end.
                                -1 means unfreeze all.
        unfreeze_from: Layer name from which to start unfreezing
    
    Returns:
        Number of layers unfrozen
    """
    if num_layers_to_unfreeze == -1:
        # Unfreeze all
        for layer in model.layers:
            layer.trainable = True
        print(f"🔓 Unfroze all {len(model.layers)} layers")
        return len(model.layers)
    
    # Find layers to unfreeze
    all_layers = list(model.layers)
    
    if unfreeze_from:
        # Unfreeze from specific layer
        start_idx = 0
        for i, layer in enumerate(all_layers):
            if unfreeze_from in layer.name:
                start_idx = i
                break
        
        unfrozen = 0
        for layer in all_layers[start_idx:]:
            layer.trainable = True
            unfrozen += 1
        print(f"🔓 Unfroze {unfrozen} layers from '{unfreeze_from}'")
        return unfrozen
    
    # Unfreeze last N layers
    layers_to_unfreeze = all_layers[-num_layers_to_unfreeze:]
    for layer in layers_to_unfreeze:
        layer.trainable = True
    
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    total_params = model.count_params()
    
    print(f"🔓 Unfroze last {num_layers_to_unfreeze} layers")
    print(f"   Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    return num_layers_to_unfreeze


def create_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    model_name: str = "mobilenet_ssd",
    monitor: str = "val_loss",
    patience_reduce_lr: int = 5,
    patience_early_stop: int = 15,
    reduce_lr_factor: float = 0.2,
    min_lr: float = 1e-7,
    save_best_only: bool = True,
) -> List[tf.keras.callbacks.Callback]:
    """Create standard callbacks for training.
    
    Includes:
    - ModelCheckpoint: Save best model
    - ReduceLROnPlateau: Reduce LR when stuck
    - EarlyStopping: Stop when no improvement
    - TensorBoard: Visualization
    - CSVLogger: Log metrics to CSV
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory for logs (TensorBoard, CSV)
        model_name: Name prefix for saved files
        monitor: Metric to monitor
        patience_reduce_lr: Epochs before reducing LR
        patience_early_stop: Epochs before stopping (should be > patience_reduce_lr)
        reduce_lr_factor: Factor to reduce LR by
        min_lr: Minimum learning rate
        save_best_only: Only save best model
    
    Returns:
        List of Keras callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.keras"),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode="min" if "loss" in monitor else "max",
            verbose=1,
        ),
        
        # Reduce LR when stuck
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=patience_reduce_lr,
            min_lr=min_lr,
            verbose=1,
            mode="min" if "loss" in monitor else "max",
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
            mode="min" if "loss" in monitor else "max",
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, f"{model_name}_{timestamp}"),
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
        ),
        
        # CSV logging
        tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, f"{model_name}_history.csv"),
            append=True,
        ),
    ]
    
    print(f"📋 Created {len(callbacks)} callbacks:")
    print(f"   Checkpoint: {checkpoint_dir}/{model_name}_best.keras")
    print(f"   Logs: {log_dir}/{model_name}_history.csv")
    print(f"   Monitor: {monitor}")
    print(f"   ReduceLR patience: {patience_reduce_lr}, factor: {reduce_lr_factor}")
    print(f"   EarlyStopping patience: {patience_early_stop}")
    
    return callbacks


def train_two_phase(
    model: tf.keras.Model,
    train_gen,
    val_gen,
    losses: Dict[str, Any],
    loss_weights: Dict[str, float],
    checkpoint_dir: str,
    log_dir: str,
    model_name: str = "mobilenet_ssd",
    # Phase 1 params
    phase1_epochs: int = 15,
    phase1_lr: float = 1e-3,
    # Phase 2 params  
    phase2_epochs: int = 100,
    phase2_lr: float = 1e-5,
    num_layers_to_unfreeze: int = 30,
    # Common params
    batch_size: int = 32,
    patience_reduce_lr: int = 5,
    patience_early_stop: int = 15,
) -> Tuple[tf.keras.callbacks.History, tf.keras.callbacks.History]:
    """Two-phase training: warm-up + fine-tuning.
    
    Phase 1 (Warm-up):
    - Freeze backbone
    - Train only SSD head with higher LR
    - Short training (10-20 epochs)
    
    Phase 2 (Fine-tuning):
    - Unfreeze last N layers of backbone
    - Train with very low LR
    - Long training with early stopping
    
    Args:
        model: The model to train
        train_gen: Training data generator
        val_gen: Validation data generator
        losses: Dictionary of loss functions
        loss_weights: Dictionary of loss weights
        checkpoint_dir: Where to save checkpoints
        log_dir: Where to save logs
        model_name: Name for saved files
        phase1_epochs: Epochs for phase 1
        phase1_lr: Learning rate for phase 1
        phase2_epochs: Max epochs for phase 2
        phase2_lr: Learning rate for phase 2
        num_layers_to_unfreeze: Layers to unfreeze in phase 2
        batch_size: Batch size
        patience_reduce_lr: LR reduction patience
        patience_early_stop: Early stopping patience
    
    Returns:
        Tuple of (phase1_history, phase2_history)
    """
    print("\n" + "="*60)
    print("PHASE 1: Warm-up (Frozen Backbone)")
    print("="*60)
    
    # Phase 1: Freeze backbone
    freeze_backbone(model)
    
    # Compile for phase 1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=phase1_lr),
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "objectness": ["binary_accuracy"],
            "class_out": ["categorical_accuracy"],
        }
    )
    
    # Callbacks for phase 1
    phase1_callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        model_name=f"{model_name}_phase1",
        monitor="val_loss",
        patience_reduce_lr=patience_reduce_lr,
        patience_early_stop=phase1_epochs,  # No early stopping in phase 1
    )
    
    # Train phase 1
    print(f"\n🚀 Starting Phase 1: {phase1_epochs} epochs, LR={phase1_lr}")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase1_epochs,
        callbacks=phase1_callbacks,
        verbose=1,
    )
    
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning (Unfrozen Backbone)")
    print("="*60)
    
    # Phase 2: Unfreeze backbone layers
    unfreeze_backbone_layers(model, num_layers_to_unfreeze=num_layers_to_unfreeze)
    
    # Compile for phase 2 with lower LR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=phase2_lr),
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "objectness": ["binary_accuracy"],
            "class_out": ["categorical_accuracy"],
        }
    )
    
    # Callbacks for phase 2
    phase2_callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        model_name=f"{model_name}_phase2",
        monitor="val_loss",
        patience_reduce_lr=patience_reduce_lr,
        patience_early_stop=patience_early_stop,
    )
    
    # Train phase 2
    print(f"\n🚀 Starting Phase 2: up to {phase2_epochs} epochs, LR={phase2_lr}")
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase2_epochs,
        callbacks=phase2_callbacks,
        verbose=1,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return history1, history2


def save_training_history(
    history: tf.keras.callbacks.History,
    filepath: str,
) -> pd.DataFrame:
    """Save training history to CSV.
    
    Args:
        history: Keras History object
        filepath: Output CSV path
    
    Returns:
        DataFrame with history
    """
    df = pd.DataFrame(history.history)
    df["epoch"] = range(1, len(df) + 1)
    df.to_csv(filepath, index=False)
    print(f"💾 Saved training history to {filepath}")
    return df


def plot_training_history(
    history: tf.keras.callbacks.History,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None,
) -> None:
    """Plot training history with loss and metrics.
    
    Args:
        history: Keras History object or dict
        output_path: Optional path to save figure
        figsize: Figure size
        title: Optional title for the entire figure
    """
    import matplotlib.pyplot as plt
    
    if isinstance(history, tf.keras.callbacks.History):
        history = history.history
    
    # Find all metrics
    metrics = set()
    for key in history.keys():
        # Remove val_ prefix and get base metric name
        base_name = key.replace("val_", "")
        metrics.add(base_name)
    
    # Filter out loss-related for separate plotting
    loss_metrics = [m for m in metrics if "loss" in m.lower()]
    other_metrics = [m for m in metrics if "loss" not in m.lower()]
    
    n_plots = len(loss_metrics) + len(other_metrics)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    plot_idx = 0
    
    # Plot losses
    for metric in sorted(loss_metrics):
        ax = axes[plot_idx]
        if metric in history:
            ax.plot(history[metric], label=f"Train {metric}")
        if f"val_{metric}" in history:
            ax.plot(history[f"val_{metric}"], label=f"Val {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot other metrics
    for metric in sorted(other_metrics):
        if plot_idx >= len(axes):
            break
        ax = axes[plot_idx]
        if metric in history:
            ax.plot(history[metric], label=f"Train {metric}")
        if f"val_{metric}" in history:
            ax.plot(history[f"val_{metric}"], label=f"Val {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    # Add figure title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved training plot to {output_path}")
    
    plt.show()


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Callback to log learning rate at each epoch."""
    
    def __init__(self):
        super().__init__()
        self.lrs = []
    
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lrs.append(lr)
        if logs is not None:
            logs["lr"] = lr


class PositiveAnchorMonitor(tf.keras.callbacks.Callback):
    """Monitor the number of positive anchors during training.
    
    Useful for debugging anchor matching issues.
    """
    
    def __init__(self, log_every: int = 5):
        super().__init__()
        self.log_every = log_every
        self.positive_counts = []
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every == 0:
            # This would need access to a batch to compute
            # Left as placeholder for custom implementation
            pass
