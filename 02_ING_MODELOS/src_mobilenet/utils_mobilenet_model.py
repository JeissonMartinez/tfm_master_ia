"""MobileNetV3 Small + SSD-Lite model builder for ESP32-S3.

This module provides lightweight object detection models using:
- MobileNetV3 Small or MobileNetV2 as backbone
- SSD-Lite detection head with depthwise separable convolutions
- Optimized for TFLite conversion and ESP32-S3 deployment

Reference:
    - MobileNetV3: Howard et al., "Searching for MobileNetV3", ICCV 2019
    - SSD-Lite: Sandler et al., "MobileNetV2", CVPR 2018
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import tensorflow as tf


def ssd_lite_conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    stride: int = 1,
    use_batchnorm: bool = True,
    name: str = "",
) -> tf.Tensor:
    """SSD-Lite convolution block using depthwise separable convolutions.
    
    Replaces standard Conv2D with DepthwiseConv2D + Conv2D 1x1,
    reducing parameters and computation significantly.
    
    Args:
        x: Input tensor
        filters: Number of output filters
        kernel_size: Kernel size for depthwise conv
        stride: Stride for depthwise conv
        use_batchnorm: Whether to use batch normalization
        name: Layer name prefix
    
    Returns:
        Output tensor
    """
    # Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=not use_batchnorm,
        name=f"{name}_depthwise" if name else None,
    )(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1" if name else None)(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu1" if name else None)(x)
    
    # Pointwise convolution (1x1)
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding="same",
        use_bias=not use_batchnorm,
        name=f"{name}_pointwise" if name else None,
    )(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2" if name else None)(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu2" if name else None)(x)
    
    return x


def ssd_lite_head(
    features: tf.Tensor,
    num_anchors: int,
    num_classes: int,
    feature_channels: int = 128,
    use_batchnorm: bool = True,
    name_prefix: str = "ssd_head",
) -> Dict[str, tf.Tensor]:
    """SSD-Lite detection head with depthwise separable convolutions.
    
    Produces three outputs:
    - objectness: Binary classification (object vs background)
    - class_out: Multi-class classification (excluding background)
    - bbox_out: Bounding box regression (xc, yc, w, h normalized)
    
    Args:
        features: Feature map from backbone (B, H, W, C)
        num_anchors: Number of anchors per cell
        num_classes: Number of object classes (excluding background)
        feature_channels: Intermediate feature channels
        use_batchnorm: Whether to use batch normalization
        name_prefix: Prefix for layer names
    
    Returns:
        Dictionary with 'objectness', 'class_out', 'bbox_out' tensors
    """
    # Shared feature extraction with SSD-Lite blocks
    x = ssd_lite_conv_block(
        features, 
        feature_channels, 
        use_batchnorm=use_batchnorm,
        name=f"{name_prefix}_shared1"
    )
    x = ssd_lite_conv_block(
        x, 
        feature_channels, 
        use_batchnorm=use_batchnorm,
        name=f"{name_prefix}_shared2"
    )
    
    # Get spatial dimensions for reshape
    # Shape: (batch, height, width, channels)
    h, w = x.shape[1], x.shape[2]
    total_anchors = h * w * num_anchors
    
    # === OBJECTNESS HEAD (binary: object vs background) ===
    obj = tf.keras.layers.Conv2D(
        num_anchors * 1, 
        kernel_size=1, 
        padding="same",
        name=f"{name_prefix}_obj_conv"
    )(x)
    obj = tf.keras.layers.Reshape((total_anchors, 1), name=f"{name_prefix}_obj_reshape")(obj)
    obj = tf.keras.layers.Activation("sigmoid", name="objectness")(obj)
    
    # === CLASSIFICATION HEAD (multi-class with sigmoid) ===
    cls = tf.keras.layers.Conv2D(
        num_anchors * num_classes,
        kernel_size=1,
        padding="same",
        name=f"{name_prefix}_cls_conv"
    )(x)
    cls = tf.keras.layers.Reshape((total_anchors, num_classes), name=f"{name_prefix}_cls_reshape")(cls)
    cls = tf.keras.layers.Activation("sigmoid", name="class_out")(cls)
    
    # === BBOX REGRESSION HEAD ===
    bbox = tf.keras.layers.Conv2D(
        num_anchors * 4,
        kernel_size=1,
        padding="same",
        name=f"{name_prefix}_bbox_conv"
    )(x)
    bbox = tf.keras.layers.Reshape((total_anchors, 4), name=f"{name_prefix}_bbox_reshape")(bbox)
    bbox = tf.keras.layers.Activation("sigmoid", name="bbox_out")(bbox)
    
    return {
        "objectness": obj,
        "class_out": cls,
        "bbox_out": bbox,
    }


def build_mobilenetv3_ssd_lite(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    num_anchors_per_cell: int = 6,
    alpha: float = 1.0,
    minimalistic: bool = True,
    feature_channels: int = 128,
    use_batchnorm: bool = True,
    dropout_rate: float = 0.2,
    model_name: str = "MobileNetV3Small_SSDLite",
) -> tf.keras.Model:
    """Build MobileNetV3 Small + SSD-Lite for object detection.
    
    Designed for deployment on ESP32-S3 with TFLite INT8 quantization.
    
    IMPORTANT: Use `minimalistic=True` for ESP32-S3 compatibility.
    This replaces hard-swish with ReLU and removes Squeeze-Excitation blocks,
    which are slow without proper INT8 support.
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of object classes (excluding background)
        num_anchors_per_cell: Number of anchors per spatial location
        alpha: Width multiplier (0.75 or 1.0 recommended)
        minimalistic: Use ReLU instead of hard-swish (recommended for ESP32)
        feature_channels: Channels in SSD-Lite head
        use_batchnorm: Whether to use batch normalization
        dropout_rate: Dropout rate before detection heads
        model_name: Model name
    
    Returns:
        Keras Model with outputs: objectness, class_out, bbox_out
    
    Example:
        >>> model = build_mobilenetv3_ssd_lite(
        ...     input_shape=(224, 224, 3),
        ...     num_classes=4,
        ...     num_anchors_per_cell=6,
        ...     minimalistic=True,  # For ESP32-S3
        ... )
        >>> model.summary()
    """
    # Load MobileNetV3 Small backbone
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        alpha=alpha,
        minimalistic=minimalistic,  # ReLU instead of hard-swish
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,  # We handle preprocessing
    )
    
    # Get feature map from backbone
    # MobileNetV3 Small output: 7x7 for 224x224 input
    features = base_model.output
    
    # Optional dropout for regularization
    if dropout_rate > 0:
        features = tf.keras.layers.SpatialDropout2D(
            dropout_rate, 
            name="feature_dropout"
        )(features)
    
    # SSD-Lite detection head
    outputs = ssd_lite_head(
        features,
        num_anchors=num_anchors_per_cell,
        num_classes=num_classes,
        feature_channels=feature_channels,
        use_batchnorm=use_batchnorm,
        name_prefix="ssd_lite",
    )
    
    # Build model
    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name=model_name,
    )
    
    return model


def build_mobilenetv2_ssd_lite(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    num_anchors_per_cell: int = 6,
    alpha: float = 0.35,
    feature_channels: int = 128,
    use_batchnorm: bool = True,
    dropout_rate: float = 0.2,
    model_name: str = "MobileNetV2_SSDLite",
) -> tf.keras.Model:
    """Build MobileNetV2 + SSD-Lite for object detection.
    
    Alternative to MobileNetV3, proven to work on ESP32-S3.
    Uses lower alpha (0.35) by default for smaller model size.
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of object classes (excluding background)
        num_anchors_per_cell: Number of anchors per spatial location
        alpha: Width multiplier (0.35 for ESP32-S3, ~250KB model)
        feature_channels: Channels in SSD-Lite head
        use_batchnorm: Whether to use batch normalization
        dropout_rate: Dropout rate before detection heads
        model_name: Model name
    
    Returns:
        Keras Model with outputs: objectness, class_out, bbox_out
    
    Example:
        >>> model = build_mobilenetv2_ssd_lite(
        ...     input_shape=(224, 224, 3),
        ...     num_classes=4,
        ...     alpha=0.35,  # Small model for ESP32-S3
        ... )
    """
    # Load MobileNetV2 backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )
    
    # Get feature map from backbone
    # MobileNetV2 output: 7x7 for 224x224 input
    features = base_model.output
    
    # Optional dropout
    if dropout_rate > 0:
        features = tf.keras.layers.SpatialDropout2D(
            dropout_rate,
            name="feature_dropout"
        )(features)
    
    # SSD-Lite detection head
    outputs = ssd_lite_head(
        features,
        num_anchors=num_anchors_per_cell,
        num_classes=num_classes,
        feature_channels=feature_channels,
        use_batchnorm=use_batchnorm,
        name_prefix="ssd_lite",
    )
    
    # Build model
    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name=model_name,
    )
    
    return model


def get_backbone_layer_names(model: tf.keras.Model, backbone_name: str = "mobilenetv3") -> List[str]:
    """Get layer names belonging to the backbone.
    
    Useful for selective freezing/unfreezing during fine-tuning.
    
    Args:
        model: The full model
        backbone_name: 'mobilenetv3' or 'mobilenetv2'
    
    Returns:
        List of layer names in the backbone
    """
    backbone_layers = []
    for layer in model.layers:
        # MobileNetV3 layers typically start with "expanded_conv" or "Conv"
        # MobileNetV2 layers typically start with "block_" or "Conv"
        name = layer.name.lower()
        if any(prefix in name for prefix in [
            "expanded_conv", "block_", "conv", "bn", "re_lu", 
            "multiply", "add", "rescaling", "normalization"
        ]):
            if "ssd" not in name:  # Exclude SSD head layers
                backbone_layers.append(layer.name)
    return backbone_layers


def print_model_summary(model: tf.keras.Model) -> None:
    """Print detailed model summary with trainable parameter count.
    
    Args:
        model: Keras model
    """
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"\n{'='*60}")
    print(f"Model: {model.name}")
    print(f"{'='*60}")
    print(f"Total params: {model.count_params():,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")
    print(f"{'='*60}")
    
    # Estimate model size
    total_bytes = model.count_params() * 4  # float32
    quantized_bytes = model.count_params()  # int8
    print(f"Estimated size (float32): {total_bytes / 1024 / 1024:.2f} MB")
    print(f"Estimated size (int8): {quantized_bytes / 1024 / 1024:.2f} MB")
    print(f"{'='*60}\n")
