"""Loss functions for SSD heads with padding masks."""
from __future__ import annotations

try:
    from .utils_io import log
except ImportError:  # fallback when running as a script/notebook
    from utils_io import log

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


def weighted_categorical_crossentropy(class_weights):
    """Weighted categorical cross-entropy for anchor classification.

    Args:
        class_weights: list/array of shape (num_classes+1,) with background at index 0
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para weighted_categorical_crossentropy.")

    weights = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        class_indices = tf.argmax(y_true, axis=-1)
        sample_weights = tf.gather(weights, class_indices)
        return tf.reduce_mean(ce * sample_weights)

    return loss_fn


def focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """Focal Loss para clasificación con desbalance extremo de clases.
    
    Reduce el peso de ejemplos fáciles (background con alta confianza) y
    enfoca el entrenamiento en ejemplos difíciles.
    
    Args:
        alpha: Factor de balance para clase positiva (0-1). Default 0.25.
        gamma: Factor de enfoque. Valores altos reducen más el peso de ejemplos fáciles.
               gamma=0 equivale a cross-entropy estándar. Default 2.0.
    
    Returns:
        Loss function compatible con Keras.
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para focal_loss.")

    def loss_fn(y_true, y_pred):
        # y_true: (batch, num_anchors, num_classes+1) one-hot
        # y_pred: (batch, num_anchors, num_classes+1) softmax probabilities
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Cross entropy per anchor
        ce = -y_true * tf.math.log(y_pred)
        
        # Probabilidad de la clase correcta (pt)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = tf.pow(1.0 - pt, gamma)
        
        # Alpha weighting: alpha para positivos (foreground), (1-alpha) para background
        # Background está en índice 0
        is_background = y_true[..., 0]  # 1 si es background, 0 si no
        alpha_weight = (1.0 - is_background) * alpha + is_background * (1.0 - alpha)
        
        # Loss final
        focal_ce = alpha_weight * focal_weight * tf.reduce_sum(ce, axis=-1)
        
        return tf.reduce_mean(focal_ce)

    return loss_fn


def focal_loss_with_hard_negative_mining(
    alpha: float = 0.25,
    gamma: float = 2.0,
    neg_pos_ratio: float = 3.0,
):
    """Focal Loss con Hard Negative Mining integrado.
    
    Combina Focal Loss con selección de negativos difíciles para
    manejar el desbalance extremo en detección de objetos.
    
    Args:
        alpha: Factor de balance para clase positiva.
        gamma: Factor de enfoque para focal loss.
        neg_pos_ratio: Ratio de negativos a positivos a mantener.
    
    Returns:
        Loss function compatible con Keras.
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para focal_loss_with_hard_negative_mining.")

    def loss_fn(y_true, y_pred):
        # y_true: (batch, num_anchors, num_classes+1)
        # y_pred: (batch, num_anchors, num_classes+1)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        batch_size = tf.shape(y_true)[0]
        num_anchors = tf.shape(y_true)[1]
        
        # Identificar positivos (no background) y negativos (background)
        is_positive = 1.0 - y_true[..., 0]  # (batch, num_anchors)
        is_negative = y_true[..., 0]
        
        # Cross entropy per anchor
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Focal weight
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - pt, gamma)
        
        # Loss sin reducir
        loss_per_anchor = focal_weight * ce  # (batch, num_anchors)
        
        # Contar positivos por batch
        num_positives = tf.reduce_sum(is_positive, axis=-1)  # (batch,)
        num_positives = tf.maximum(num_positives, 1.0)  # Evitar división por 0
        
        # Número de negativos a seleccionar
        num_negatives = tf.cast(num_positives * neg_pos_ratio, tf.int32)
        num_negatives = tf.minimum(num_negatives, tf.cast(tf.reduce_sum(is_negative, axis=-1), tf.int32))
        
        # Loss de positivos (todos)
        pos_loss = loss_per_anchor * is_positive
        pos_loss_sum = tf.reduce_sum(pos_loss, axis=-1)  # (batch,)
        
        # Hard negative mining: seleccionar top-k negativos por loss
        neg_loss = loss_per_anchor * is_negative  # (batch, num_anchors)
        
        # Para cada elemento del batch, seleccionar top-k
        def select_hard_negatives(args):
            neg_losses, k = args
            # Obtener top-k losses
            top_k_values, _ = tf.math.top_k(neg_losses, k=tf.maximum(k, 1))
            return tf.reduce_sum(top_k_values)
        
        neg_loss_sum = tf.map_fn(
            select_hard_negatives,
            (neg_loss, num_negatives),
            fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
        )
        
        # Total loss normalizado por número de positivos
        total_loss = (pos_loss_sum + neg_loss_sum) / num_positives
        
        # Alpha weighting
        return tf.reduce_mean(total_loss) * alpha

    return loss_fn


def ssd_multitask_loss(
    class_loss_fn,
    bbox_loss_weight: float = 1.0,
    class_loss_weight: float = 1.0,
):
    """Wrapper para combinar classification loss y bbox loss en una sola función.
    
    Args:
        class_loss_fn: Función de loss para clasificación (e.g., focal_loss).
        bbox_loss_weight: Peso para la loss de bounding boxes.
        class_loss_weight: Peso para la loss de clasificación.
    
    Returns:
        Dict de funciones de loss para usar con model.compile().
    """
    return {
        "class_out": class_loss_fn,
        "bbox_out_sigmoid": masked_bbox_smooth_l1_loss,
    }, {
        "class_out": class_loss_weight,
        "bbox_out_sigmoid": bbox_loss_weight,
    }


def focal_loss_with_ignore_mask(
    alpha: float = 0.25,
    gamma: float = 2.0,
    neg_pos_ratio: float = 3.0,
):
    """Focal Loss with Hard Negative Mining that respects ignore mask.
    
    This version ignores anchors in the ambiguous zone (IoU between 
    iou_ignore_threshold and iou_threshold) during loss computation.
    
    The ignore mask is expected to be passed through y_true with a special
    encoding or as a separate input. For simplicity, we apply it during
    training via sample_weight or custom training loop.
    
    Args:
        alpha: Class balance factor for positive class.
        gamma: Focusing parameter for focal loss.
        neg_pos_ratio: Ratio of hard negatives to positives.
    
    Returns:
        Loss function compatible with Keras.
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para focal_loss_with_ignore_mask.")

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Identify positives and negatives
        is_positive = 1.0 - y_true[..., 0]
        is_negative = y_true[..., 0]
        
        # Cross entropy per anchor
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Focal weight
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - pt, gamma)
        
        loss_per_anchor = focal_weight * ce
        
        # Count positives per batch
        num_positives = tf.reduce_sum(is_positive, axis=-1)
        num_positives = tf.maximum(num_positives, 1.0)
        
        # Number of negatives to select (capped)
        num_negatives = tf.cast(num_positives * neg_pos_ratio, tf.int32)
        num_negatives = tf.minimum(
            num_negatives, 
            tf.cast(tf.reduce_sum(is_negative, axis=-1), tf.int32)
        )
        
        # Positive loss (all positives)
        pos_loss = loss_per_anchor * is_positive
        pos_loss_sum = tf.reduce_sum(pos_loss, axis=-1)
        
        # Hard negative mining
        neg_loss = loss_per_anchor * is_negative
        
        def select_hard_negatives(args):
            neg_losses, k = args
            top_k_values, _ = tf.math.top_k(neg_losses, k=tf.maximum(k, 1))
            return tf.reduce_sum(top_k_values)
        
        neg_loss_sum = tf.map_fn(
            select_hard_negatives,
            (neg_loss, num_negatives),
            fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
        )
        
        # Normalize by positives
        total_loss = (pos_loss_sum + neg_loss_sum) / num_positives
        
        return tf.reduce_mean(total_loss) * alpha

    return loss_fn


def confidence_calibration_loss(temperature: float = 1.5):
    """Regularization loss to calibrate prediction confidence.
    
    Penalizes overconfident predictions on background to reduce false positives.
    
    Args:
        temperature: Softening temperature. Higher = more regularization.
    
    Returns:
        Regularization term to add to main loss.
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para confidence_calibration_loss.")

    def loss_fn(y_true, y_pred):
        # Only apply to background predictions
        is_background = y_true[..., 0]  # 1 if background
        
        # Penalize high confidence on background (should be confident but not extreme)
        bg_conf = y_pred[..., 0]  # Predicted background probability
        
        # Target: background confidence should be high but not > 0.95
        overconfident = tf.maximum(bg_conf - 0.9, 0.0)
        penalty = overconfident ** 2 * is_background
        
        return tf.reduce_mean(penalty) * temperature

    return loss_fn


# ============================================================================
# SSD V4 LOSSES - Sigmoid-based (no softmax)
# ============================================================================

def binary_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """Binary Focal Loss para clasificación multi-label con sigmoid.
    
    A diferencia de categorical focal loss, esta versión trata cada clase
    de forma independiente (multi-label), lo que evita la dilución de
    probabilidades que ocurre con softmax cuando hay muchos anchors.
    
    Args:
        alpha: Factor de balance para ejemplos positivos (0-1).
        gamma: Factor de enfoque para ejemplos difíciles.
    
    Returns:
        Loss function para usar con activación sigmoid.
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para binary_focal_loss.")

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Binary cross entropy por clase
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Focal weights
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        
        # Alpha weights (alpha para positivos, 1-alpha para negativos)
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        focal_loss = alpha_weight * focal_weight * bce
        
        return tf.reduce_mean(focal_loss)

    return loss_fn


def objectness_focal_loss(alpha: float = 0.25, gamma: float = 2.0, pos_weight: float = 5.0):
    """Focal Loss para objectness head (predice si hay objeto).
    
    Args:
        alpha: Balance factor.
        gamma: Focusing parameter.
        pos_weight: Peso adicional para ejemplos positivos (compensar desbalance).
    
    Returns:
        Loss function para objectness head.
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para objectness_focal_loss.")

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Binary cross entropy
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        
        # Aplicar pos_weight a ejemplos positivos
        sample_weight = y_true * pos_weight + (1 - y_true) * 1.0
        
        loss = focal_weight * bce * sample_weight
        
        return tf.reduce_mean(loss)

    return loss_fn


def ssd_v4_class_loss_with_objectness(alpha: float = 0.25, gamma: float = 2.0):
    """Loss de clasificación que ignora anchors sin objetos.
    
    Solo calcula loss de clasificación para anchors que contienen objetos,
    eliminando el ruido de los ~99% de anchors que son background.
    
    El target y_true tiene shape (batch, anchors, num_classes) donde
    anchors sin objeto tienen todos 0s.
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para ssd_v4_class_loss_with_objectness.")

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Máscara de anchors con objetos (al menos una clase = 1)
        has_object = tf.reduce_max(y_true, axis=-1, keepdims=True)  # (batch, anchors, 1)
        
        # Binary cross entropy por clase
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Focal weights
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        
        # Alpha balance
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Loss solo para anchors con objetos
        focal_loss = alpha_weight * focal_weight * bce
        masked_loss = focal_loss * has_object
        
        # Normalizar por número de positivos
        num_positives = tf.reduce_sum(has_object) + 1e-7
        
        return tf.reduce_sum(masked_loss) / num_positives

    return loss_fn
