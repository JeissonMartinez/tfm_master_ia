"""Shared training utilities (callbacks, logging)."""
from __future__ import annotations

import os
from typing import List

try:
    from .utils_io import log, safe_mkdir
except ImportError:  # fallback when running as a script/notebook
    from utils_io import log, safe_mkdir

try:
    import tensorflow as tf  # type: ignore
except Exception as exc:  # pragma: no cover - defensive
    tf = None
    log(f"⚠️ TensorFlow no disponible: {exc}")


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
