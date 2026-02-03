from __future__ import annotations

"""MobileNetV2 SSD model builders (parameterized)."""
def squeeze_excite_block(input_tensor, ratio=8):
    import tensorflow as tf
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    return tf.keras.layers.Multiply()([input_tensor, se])

def build_mobilenet_ssd_v5(
    input_shape=(224, 224, 3),
    num_classes: int = 4,
    anchors_per_cell: int = 18,
    alpha: float = 0.50,
    feature_channels: int = 256,
    use_batchnorm: bool = True,
    dropout_rate: float = 0.2,
    model_name: str = "MobileNetV2_SSD_V5",
):
    """SSD V5: Igual que V4 pero añade bloque Squeeze-and-Excitation tras la cabeza de features."""
    import tensorflow as tf
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )

    x = base_model.output  # (batch, 7, 7, channels)
    x = tf.keras.layers.Conv2D(feature_channels, (3, 3), padding="same", use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(feature_channels, (3, 3), padding="same", use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
    # --- Bloque SE ---
    x = squeeze_excite_block(x, ratio=8)

    # === OBJECTNESS HEAD ===
    obj = tf.keras.layers.Conv2D(anchors_per_cell * 1, (1, 1), padding="same")(x)
    obj = tf.keras.layers.Reshape((-1, 1))(obj)
    obj = tf.keras.layers.Activation("sigmoid", name="objectness")(obj)

    # === CLASSIFICATION HEAD ===
    cls_channels = anchors_per_cell * num_classes
    cls = tf.keras.layers.Conv2D(cls_channels, (1, 1), padding="same")(x)
    cls = tf.keras.layers.Reshape((-1, num_classes))(cls)
    cls = tf.keras.layers.Activation("sigmoid", name="class_out")(cls)

    # === BBOX HEAD ===
    box_channels = anchors_per_cell * 4
    box = tf.keras.layers.Conv2D(box_channels, (1, 1), padding="same")(x)
    box = tf.keras.layers.Reshape((-1, 4))(box)
    box = tf.keras.layers.Activation("sigmoid", name="bbox_out_sigmoid")(box)

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs={
            "objectness": obj,
            "class_out": cls,
            "bbox_out_sigmoid": box,
        },
        name=model_name,
    )
    return model

from typing import Iterable, List, Optional

try:
    from .utils_io import log
except ImportError:  # fallback when running as a script/notebook
    from utils_io import log

try:
    import tensorflow as tf  # type: ignore
except Exception as exc:  # pragma: no cover - defensive
    tf = None
    log(f"⚠️ TensorFlow no disponible: {exc}")


def build_mobilenet_ssd(
    input_shape=(224, 224, 3),
    num_classes: int = 4,
    max_objects: int = 2,
    alpha: float = 0.50,
    conv1x1_filters: int = 128,
    dense_units: Iterable[int] = (128, 64),
    dropout_rates: Iterable[float] = (0.4, 0.3),
    l2_reg: float = 0.001,
    use_batchnorm: bool = True,
    model_name: str = "MobileNetV2_SSD_Custom",
):
    """Build a lightweight SSD-style head on MobileNetV2.

    Designed to match the SSD V5/V6 style used in the notebook.
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para construir el modelo SSD.")

    l2 = tf.keras.regularizers.l2(l2_reg) if l2_reg else None  # type: ignore

    base_model = tf.keras.applications.MobileNetV2(  # type: ignore
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )

    x = base_model.output
    x = tf.keras.layers.Conv2D(conv1x1_filters, (1, 1), padding="same", use_bias=not use_batchnorm)(x)  # type: ignore
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)  # type: ignore
        x = tf.keras.layers.ReLU()(x)  # type: ignore

    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # type: ignore

    dense_units_list: List[int] = list(dense_units)
    dropout_list: List[float] = list(dropout_rates)
    while len(dropout_list) < len(dense_units_list):
        dropout_list.append(dropout_list[-1] if dropout_list else 0.0)

    for units, dr in zip(dense_units_list, dropout_list):
        if dr > 0:
            x = tf.keras.layers.Dropout(dr)(x)  # type: ignore
        x = tf.keras.layers.Dense(units, use_bias=not use_batchnorm, kernel_regularizer=l2)(x)  # type: ignore
        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)  # type: ignore
            x = tf.keras.layers.ReLU()(x)  # type: ignore

    class_logits = tf.keras.layers.Dense(max_objects * num_classes, kernel_regularizer=l2)(x)  # type: ignore
    class_logits = tf.keras.layers.Reshape((max_objects, num_classes))(class_logits)  # type: ignore
    class_output = tf.keras.layers.Softmax(axis=-1, name="class_out")(class_logits)  # type: ignore

    bbox_output = tf.keras.layers.Dense(max_objects * 4, activation="sigmoid", kernel_regularizer=l2)(x)  # type: ignore
    bbox_output = tf.keras.layers.Reshape((max_objects, 4), name="bbox_out")(bbox_output)  # type: ignore

    model = tf.keras.Model(inputs=base_model.input, outputs=[class_output, bbox_output], name=model_name)  # type: ignore
    return model


def build_mobilenet_ssd_anchor_head(
    input_shape=(224, 224, 3),
    num_classes: int = 4,
    anchors_per_cell: int = 5,
    alpha: float = 0.50,
    feature_channels: int = 256,
    use_batchnorm: bool = True,
    model_name: str = "MobileNetV2_SSD_Anchors",
):
    """Build an SSD-style head with anchor-based dynamic outputs.

    Outputs:
        class_out: (batch, H*W*A, num_classes+1) with background at index 0
        bbox_out:  (batch, H*W*A, 4) normalized (xc, yc, w, h)
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para construir el modelo SSD con anchors.")

    base_model = tf.keras.applications.MobileNetV2(  # type: ignore
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )

    x = base_model.output
    x = tf.keras.layers.Conv2D(feature_channels, (3, 3), padding="same", use_bias=not use_batchnorm)(x)  # type: ignore
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)  # type: ignore
        x = tf.keras.layers.ReLU()(x)  # type: ignore

    # Classification head
    cls_channels = anchors_per_cell * (num_classes + 1)
    cls = tf.keras.layers.Conv2D(cls_channels, (1, 1), padding="same")(x)  # type: ignore

    # BBox head
    box_channels = anchors_per_cell * 4
    box = tf.keras.layers.Conv2D(box_channels, (1, 1), padding="same")(x)  # type: ignore

    # Reshape to (batch, H*W*A, num_classes+1) and (batch, H*W*A, 4)
    cls = tf.keras.layers.Reshape((-1, num_classes + 1))(cls)  # type: ignore
    cls = tf.keras.layers.Softmax(axis=-1, name="class_out")(cls)  # type: ignore

    box = tf.keras.layers.Reshape((-1, 4), name="bbox_out")(box)  # type: ignore
    box = tf.keras.layers.Activation("sigmoid", name="bbox_out_sigmoid")(box)  # type: ignore

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs={"class_out": cls, "bbox_out_sigmoid": box},
        name=model_name,
    )  # type: ignore
    return model


def build_mobilenet_ssd_v4(
    input_shape=(224, 224, 3),
    num_classes: int = 4,
    anchors_per_cell: int = 18,
    alpha: float = 0.50,
    feature_channels: int = 256,
    use_batchnorm: bool = True,
    dropout_rate: float = 0.2,
    model_name: str = "MobileNetV2_SSD_V4",
):
    """SSD V4: Mejoras para confianzas más altas y mejor calibración.
    
    Cambios clave vs V3:
    1. **Sigmoid por clase** en lugar de softmax: evita dilución de probabilidades
       cuando hay muchos anchors. Cada clase se predice independientemente.
    2. **Objectness head separado**: predice si el anchor contiene un objeto
       antes de clasificar la clase específica.
    3. **Dropout en cabezas**: regularización adicional para evitar overfitting.
    4. **Feature pyramid simplificado**: combina features de diferentes escalas.
    
    Outputs:
        objectness: (batch, H*W*A, 1) - probabilidad de contener objeto
        class_out: (batch, H*W*A, num_classes) - prob. de cada clase (sigmoid)
        bbox_out_sigmoid: (batch, H*W*A, 4) - coordenadas normalizadas
    """
    if tf is None:
        raise RuntimeError("TensorFlow es requerido para construir el modelo SSD V4.")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )

    # Feature extraction con más capacidad
    x = base_model.output  # (batch, 7, 7, channels)
    
    # Cabeza de features compartida
    x = tf.keras.layers.Conv2D(feature_channels, (3, 3), padding="same", use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    
    # Segunda capa convolucional para más capacidad
    x = tf.keras.layers.Conv2D(feature_channels, (3, 3), padding="same", use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    
    if dropout_rate > 0:
        x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
    
    # === OBJECTNESS HEAD ===
    # Predice si el anchor contiene un objeto (binario)
    obj = tf.keras.layers.Conv2D(anchors_per_cell * 1, (1, 1), padding="same")(x)
    obj = tf.keras.layers.Reshape((-1, 1))(obj)
    obj = tf.keras.layers.Activation("sigmoid", name="objectness")(obj)
    
    # === CLASSIFICATION HEAD ===
    # Usa SIGMOID en lugar de softmax: cada clase es independiente
    # Esto evita la dilución de probabilidades con muchos anchors
    cls_channels = anchors_per_cell * num_classes
    cls = tf.keras.layers.Conv2D(cls_channels, (1, 1), padding="same")(x)
    cls = tf.keras.layers.Reshape((-1, num_classes))(cls)
    cls = tf.keras.layers.Activation("sigmoid", name="class_out")(cls)  # SIGMOID, no softmax
    
    # === BBOX HEAD ===
    box_channels = anchors_per_cell * 4
    box = tf.keras.layers.Conv2D(box_channels, (1, 1), padding="same")(x)
    box = tf.keras.layers.Reshape((-1, 4))(box)
    box = tf.keras.layers.Activation("sigmoid", name="bbox_out_sigmoid")(box)

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs={
            "objectness": obj,
            "class_out": cls,
            "bbox_out_sigmoid": box,
        },
        name=model_name,
    )
    return model
