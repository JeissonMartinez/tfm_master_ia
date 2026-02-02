"""Loss functions for SSD heads with padding masks."""
from __future__ import annotations

from .utils_io import log

try:
    import tensorflow as tf  # type: ignore
except Exception as exc:  # pragma: no cover - defensive
    tf = None
    log(f"⚠️ TensorFlow no disponible: {exc}")


def masked_classification_loss(y_true, y_pred):
    """Cross-entropy with mask for valid objects only."""
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para masked_classification_loss.")
    valid_mask = tf.reduce_sum(y_true, axis=-1)
    valid_mask = tf.cast(valid_mask > 0.5, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    masked_loss = ce_loss * valid_mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(valid_mask) + 1e-7)


def masked_bbox_smooth_l1_loss(y_true, y_pred):
    """Smooth L1 loss with mask for valid boxes only."""
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para masked_bbox_smooth_l1_loss.")
    valid_mask = tf.reduce_max(tf.abs(y_true), axis=-1)
    valid_mask = tf.cast(valid_mask > 0.001, tf.float32)

    diff = tf.abs(y_true - y_pred)
    smooth_l1 = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    box_loss = tf.reduce_mean(smooth_l1, axis=-1)
    masked_loss = box_loss * valid_mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(valid_mask) + 1e-7)


def masked_huber_loss(y_true, y_pred, delta: float = 1.0):
    """Huber loss with mask for valid boxes only."""
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para masked_huber_loss.")
    valid_mask = tf.reduce_max(tf.abs(y_true), axis=-1)
    valid_mask = tf.cast(valid_mask > 0.001, tf.float32)

    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear
    huber_per_slot = tf.reduce_mean(huber, axis=-1)

    masked_loss = huber_per_slot * valid_mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(valid_mask) + 1e-7)
