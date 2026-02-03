"""Shared training utilities (callbacks, logging)."""
from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

import tensorflow as tf


class EarlyStoppingAccuracyRange(tf.keras.callbacks.Callback):
    """
    Detiene el entrenamiento si la métrica de accuracy está dentro de un rango dado durante N épocas consecutivas.
    Guarda el último checkpoint alcanzado.
    """
    def __init__(self, monitor="val_objectness_binary_accuracy", min_acc=0.85, max_acc=1.0, patience=10, verbose=1):
        super().__init__()
        self.monitor = monitor
        self.min_acc = min_acc
        self.max_acc = max_acc
        self.patience = patience
        self.verbose = verbose
        self._in_range_count = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None and self.min_acc <= acc <= self.max_acc:
            self._in_range_count += 1
            if self.verbose:
                print(f"[EarlyStoppingAccuracyRange] {self._in_range_count}/{self.patience} épocas en rango [{self.min_acc}, {self.max_acc}] ({acc:.4f})")
            if self._in_range_count >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"[EarlyStoppingAccuracyRange] Deteniendo entrenamiento en la época {epoch+1} (accuracy en rango {self.patience} épocas)")
                self.model.stop_training = True
        else:
            self._in_range_count = 0


class MultiConditionEarlyStopping(tf.keras.callbacks.Callback):
    """
    Early Stopping con múltiples condiciones (OR lógico).
    
    Detiene el entrenamiento si CUALQUIERA de las siguientes condiciones se cumple:
    1. val_loss no mejora durante `patience_loss` épocas (detecta overfitting)
    2. val_accuracy en rango objetivo durante `patience_accuracy` épocas (detecta convergencia)
    3. val_loss empeora por encima de un factor respecto al mejor (detecta divergencia)
    
    Args:
        monitor_loss: Métrica de loss a monitorear (ej: "val_loss")
        monitor_accuracy: Métrica de accuracy a monitorear (ej: "val_objectness_binary_accuracy")
        patience_loss: Épocas sin mejora en loss para detener
        patience_accuracy: Épocas con accuracy en rango para detener
        min_accuracy: Límite inferior del rango de accuracy objetivo
        max_accuracy: Límite superior del rango de accuracy objetivo
        divergence_factor: Factor de empeoramiento de loss para detener (ej: 3.0 = 3x peor que el mejor)
        restore_best_weights: Restaurar pesos del mejor modelo al detener
        verbose: Nivel de verbosidad (0=silencioso, 1=mensajes)
    """
    
    def __init__(
        self,
        monitor_loss: str = "val_loss",
        monitor_accuracy: str = "val_objectness_binary_accuracy",
        patience_loss: int = 12,
        patience_accuracy: int = 8,
        min_accuracy: float = 0.85,
        max_accuracy: float = 1.0,
        divergence_factor: float = 3.0,
        restore_best_weights: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        self.monitor_loss = monitor_loss
        self.monitor_accuracy = monitor_accuracy
        self.patience_loss = patience_loss
        self.patience_accuracy = patience_accuracy
        self.min_accuracy = min_accuracy
        self.max_accuracy = max_accuracy
        self.divergence_factor = divergence_factor
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Estado interno
        self.best_loss = float("inf")
        self.best_weights = None
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.epochs_in_accuracy_range = 0
        self.stopped_epoch = 0
        self.stop_reason = ""
    
    def on_train_begin(self, logs=None):
        self.best_loss = float("inf")
        self.best_weights = None
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.epochs_in_accuracy_range = 0
        self.stopped_epoch = 0
        self.stop_reason = ""
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get(self.monitor_loss)
        current_accuracy = logs.get(self.monitor_accuracy)
        
        should_stop = False
        
        # --- Condición 1: Mejora en loss ---
        if current_loss is not None:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.epochs_no_improve += 1
            
            # Verificar patience de loss
            if self.epochs_no_improve >= self.patience_loss:
                should_stop = True
                self.stop_reason = f"val_loss sin mejora durante {self.patience_loss} épocas"
            
            # Verificar divergencia
            if current_loss > self.best_loss * self.divergence_factor:
                should_stop = True
                self.stop_reason = f"val_loss divergió ({current_loss:.4f} > {self.best_loss:.4f} x {self.divergence_factor})"
        
        # --- Condición 2: Accuracy en rango objetivo ---
        if current_accuracy is not None:
            if self.min_accuracy <= current_accuracy <= self.max_accuracy:
                self.epochs_in_accuracy_range += 1
                if self.verbose:
                    print(f"[MultiConditionES] Accuracy en rango: {self.epochs_in_accuracy_range}/{self.patience_accuracy} ({current_accuracy:.4f})")
                
                if self.epochs_in_accuracy_range >= self.patience_accuracy:
                    should_stop = True
                    self.stop_reason = f"val_accuracy en rango [{self.min_accuracy}, {self.max_accuracy}] durante {self.patience_accuracy} épocas"
            else:
                self.epochs_in_accuracy_range = 0
        
        # --- Detener si alguna condición se cumple ---
        if should_stop:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose:
                    print(f"\n[MultiConditionES] Restaurando pesos de época {self.best_epoch + 1} (best val_loss: {self.best_loss:.4f})")
                self.model.set_weights(self.best_weights)
            
            if self.verbose:
                print(f"[MultiConditionES] Deteniendo en época {epoch + 1}: {self.stop_reason}")
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            print(f"\n[MultiConditionES] Entrenamiento terminado en época {self.stopped_epoch + 1}")
            print(f"   Razón: {self.stop_reason}")
            print(f"   Mejor época: {self.best_epoch + 1} (val_loss: {self.best_loss:.4f})")

try:
    from .utils_io import log, safe_mkdir
except ImportError:  # fallback when running as a script/notebook
    from utils_io import log, safe_mkdir


def create_callbacks(
    model_name: str,
    logs_dir: str,
    checkpoints_dir: str,
    monitor: str = "val_loss",
    mode: str = "min",
    patience_early: int = 8,
    patience_lr: int = 4,
) -> List:
    """Create a robust callback list.

    Returns empty list if TensorFlow is unavailable.
    """
    if tf is None:
        log("⚠️ TensorFlow no disponible: callbacks vacíos.")
        return []

    safe_mkdir(logs_dir)
    safe_mkdir(checkpoints_dir)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(  # type: ignore
            filepath=os.path.join(checkpoints_dir, f"{model_name}_best.keras"),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(  # type: ignore
            monitor=monitor,
            mode=mode,
            patience=patience_early,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(  # type: ignore
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(  # type: ignore
            log_dir=os.path.join(logs_dir, model_name),
            histogram_freq=1,
            write_graph=True,
        ),
        tf.keras.callbacks.CSVLogger(  # type: ignore
            os.path.join(logs_dir, f"{model_name}_history.csv"),
            separator=",",
            append=False,
        ),
    ]

    log(f"✅ Callbacks configurados para '{model_name}'")
    return callbacks
