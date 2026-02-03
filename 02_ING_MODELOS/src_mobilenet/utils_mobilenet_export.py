"""TFLite export utilities for ESP32-S3 deployment.

This module provides:
- TFLite conversion (float32, float16, int8)
- Quantization-aware training (QAT)
- Representative dataset for calibration
- Model size estimation
"""
from __future__ import annotations

import os
from typing import Optional, Callable, Iterator, List, Dict, Any

import numpy as np
import tensorflow as tf


def export_tflite(
    model: tf.keras.Model,
    output_path: str,
    quantization: str = "none",
    representative_dataset: Optional[Callable[[], Iterator]] = None,
) -> str:
    """Export Keras model to TFLite format.
    
    Args:
        model: Trained Keras model
        output_path: Path to save .tflite file
        quantization: Quantization type:
            - 'none': No quantization (float32)
            - 'float16': Float16 quantization
            - 'dynamic': Dynamic range quantization
            - 'int8': Full integer quantization (requires representative_dataset)
        representative_dataset: Generator function for INT8 calibration
    
    Returns:
        Path to saved TFLite file
    
    Example:
        >>> export_tflite(model, "model.tflite", quantization="float16")
        >>> # For INT8:
        >>> def rep_dataset():
        ...     for img, _ in val_gen:
        ...         yield [img]
        >>> export_tflite(model, "model_int8.tflite", "int8", rep_dataset)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization
    if quantization == "none":
        print("📦 Exporting TFLite (float32, no quantization)")
        
    elif quantization == "float16":
        print("📦 Exporting TFLite (float16 quantization)")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
    elif quantization == "dynamic":
        print("📦 Exporting TFLite (dynamic range quantization)")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    elif quantization == "int8":
        if representative_dataset is None:
            raise ValueError("INT8 quantization requires representative_dataset")
        
        print("📦 Exporting TFLite (INT8 full integer quantization)")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        
        # Ensure full integer ops (required for ESP32)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32  # Keep output as float for easier post-processing
    
    else:
        raise ValueError(f"Unknown quantization type: {quantization}")
    
    # Convert
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"⚠️ Conversion failed: {e}")
        print("   Trying with experimental flags...")
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True
        tflite_model = converter.convert()
    
    # Save
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    # Report size
    size_bytes = os.path.getsize(output_path)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    
    print(f"✅ Saved TFLite model to {output_path}")
    print(f"   Size: {size_kb:.1f} KB ({size_mb:.2f} MB)")
    
    return output_path


def export_tflite_int8(
    model: tf.keras.Model,
    output_path: str,
    calibration_images: np.ndarray,
    num_calibration_samples: int = 100,
) -> str:
    """Export model with INT8 quantization for ESP32-S3.
    
    Convenience function for INT8 export with automatic
    representative dataset creation.
    
    Args:
        model: Trained Keras model
        output_path: Path to save .tflite file
        calibration_images: Array of images for calibration (N, H, W, C)
        num_calibration_samples: Number of samples to use for calibration
    
    Returns:
        Path to saved TFLite file
    """
    # Limit samples
    if len(calibration_images) > num_calibration_samples:
        indices = np.random.choice(len(calibration_images), num_calibration_samples, replace=False)
        calibration_images = calibration_images[indices]
    
    def representative_dataset():
        for i in range(len(calibration_images)):
            # Expand dims if needed
            sample = calibration_images[i:i+1].astype(np.float32)
            yield [sample]
    
    return export_tflite(
        model=model,
        output_path=output_path,
        quantization="int8",
        representative_dataset=representative_dataset,
    )


def create_representative_dataset_from_generator(
    data_generator,
    num_samples: int = 100,
) -> Callable[[], Iterator]:
    """Create representative dataset function from Keras generator.
    
    Args:
        data_generator: Keras Sequence or generator
        num_samples: Number of samples to use
    
    Returns:
        Callable that yields samples for calibration
    """
    # Collect samples
    samples = []
    for batch_x, _ in data_generator:
        for i in range(len(batch_x)):
            samples.append(batch_x[i:i+1].astype(np.float32))
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    
    def representative_dataset():
        for sample in samples:
            yield [sample]
    
    return representative_dataset


def apply_quantization_aware_training(
    model: tf.keras.Model,
    train_gen,
    val_gen,
    losses: Dict[str, Any],
    loss_weights: Dict[str, float],
    epochs: int = 10,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """Apply Quantization-Aware Training (QAT) to fine-tune model.
    
    QAT simulates quantization during training, allowing the model
    to adapt to precision loss. This typically improves INT8 accuracy.
    
    IMPORTANT: Requires tensorflow_model_optimization package.
    
    Args:
        model: Pre-trained Keras model
        train_gen: Training data generator
        val_gen: Validation data generator
        losses: Loss functions dict
        loss_weights: Loss weights dict
        epochs: Number of QAT fine-tuning epochs
        learning_rate: Learning rate (should be very low)
    
    Returns:
        Quantization-aware model
    
    Example:
        >>> qat_model = apply_quantization_aware_training(
        ...     model, train_gen, val_gen, losses, loss_weights, epochs=10
        ... )
        >>> export_tflite(qat_model, "model_qat.tflite", "int8", rep_dataset)
    """
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        raise ImportError(
            "tensorflow_model_optimization is required for QAT. "
            "Install with: pip install tensorflow-model-optimization"
        )
    
    print("\n" + "="*60)
    print("QUANTIZATION-AWARE TRAINING")
    print("="*60)
    
    # Apply QAT to model
    print("🔧 Applying quantization-aware training wrappers...")
    
    # Use default quantization config
    quantize_model = tfmot.quantization.keras.quantize_model
    
    try:
        qat_model = quantize_model(model)
    except Exception as e:
        print(f"⚠️ Full model QAT failed: {e}")
        print("   Trying to quantize only compatible layers...")
        
        # Annotate specific layers for quantization
        def apply_quantization_to_layer(layer):
            # Skip layers that don't quantize well
            if isinstance(layer, (
                tf.keras.layers.BatchNormalization,
                tf.keras.layers.Dropout,
                tf.keras.layers.SpatialDropout2D,
            )):
                return layer
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        
        # Clone with quantization annotations
        annotated_model = tf.keras.models.clone_model(
            model,
            clone_function=apply_quantization_to_layer,
        )
        
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    
    # Compile QAT model
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "objectness": ["binary_accuracy"],
            "class_out": ["categorical_accuracy"],
        }
    )
    
    print(f"✅ QAT model created")
    print(f"   Training for {epochs} epochs with LR={learning_rate}")
    
    # Fine-tune with QAT
    qat_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        verbose=1,
    )
    
    print("✅ QAT fine-tuning complete")
    
    return qat_model


def estimate_model_size(model: tf.keras.Model) -> Dict[str, float]:
    """Estimate model size for different quantization levels.
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with size estimates in KB
    """
    total_params = model.count_params()
    
    sizes = {
        "float32_kb": total_params * 4 / 1024,
        "float16_kb": total_params * 2 / 1024,
        "int8_kb": total_params * 1 / 1024,
        "total_params": total_params,
    }
    
    print(f"\n📊 Model Size Estimates:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Float32: {sizes['float32_kb']:.1f} KB ({sizes['float32_kb']/1024:.2f} MB)")
    print(f"   Float16: {sizes['float16_kb']:.1f} KB ({sizes['float16_kb']/1024:.2f} MB)")
    print(f"   INT8:    {sizes['int8_kb']:.1f} KB ({sizes['int8_kb']/1024:.2f} MB)")
    
    # ESP32-S3 constraints
    esp32_flash_limit_kb = 1024  # 1MB recommended max
    fits_esp32 = sizes["int8_kb"] < esp32_flash_limit_kb
    
    if fits_esp32:
        print(f"   ✅ INT8 model fits ESP32-S3 flash limit ({esp32_flash_limit_kb} KB)")
    else:
        print(f"   ⚠️ INT8 model exceeds ESP32-S3 limit ({sizes['int8_kb']:.1f} > {esp32_flash_limit_kb} KB)")
    
    return sizes


def verify_tflite_model(
    tflite_path: str,
    test_input: Optional[np.ndarray] = None,
    input_shape: tuple = (1, 224, 224, 3),
) -> Dict[str, Any]:
    """Verify TFLite model by running inference.
    
    Args:
        tflite_path: Path to TFLite model
        test_input: Optional test input array
        input_shape: Input shape if test_input not provided
    
    Returns:
        Dictionary with verification results
    """
    # Load model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n🔍 TFLite Model Verification: {tflite_path}")
    print(f"   Input: {input_details[0]['name']}, shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    
    for out in output_details:
        print(f"   Output: {out['name']}, shape={out['shape']}, dtype={out['dtype']}")
    
    # Create test input if not provided
    if test_input is None:
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.uint8:
            test_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        else:
            test_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    outputs = {}
    for out in output_details:
        outputs[out['name']] = interpreter.get_tensor(out['index'])
    
    print(f"   ✅ Inference successful")
    for name, tensor in outputs.items():
        print(f"      {name}: shape={tensor.shape}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Model size
    size_kb = os.path.getsize(tflite_path) / 1024
    
    return {
        "input_details": input_details,
        "output_details": output_details,
        "outputs": outputs,
        "size_kb": size_kb,
    }


def export_for_esp32(
    model: tf.keras.Model,
    output_dir: str,
    model_name: str,
    calibration_data: np.ndarray,
    num_calibration_samples: int = 100,
    skip_float32: bool = False,
    skip_float16: bool = False,
) -> Dict[str, str]:
    """Export model in multiple formats for ESP32-S3 testing.
    
    Creates:
    - Float32 TFLite (for accuracy baseline)
    - Float16 TFLite (for reduced size)
    - INT8 TFLite (for ESP32-S3 deployment)
    
    Args:
        model: Trained Keras model
        output_dir: Directory for output files
        model_name: Base name for files
        calibration_data: Data for INT8 calibration
        num_calibration_samples: Number of calibration samples
        skip_float32: Skip float32 export to save memory
        skip_float16: Skip float16 export to save memory
    
    Returns:
        Dictionary with paths to exported models
    """
    import gc
    
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # Float32
    if not skip_float32:
        print("\n--- Float32 Export ---")
        try:
            paths["float32"] = export_tflite(
                model,
                os.path.join(output_dir, f"{model_name}_float32.tflite"),
                quantization="none",
            )
        except Exception as e:
            print(f"⚠️ Float32 export failed: {e}")
            paths["float32"] = None
        # Clear memory
        gc.collect()
        tf.keras.backend.clear_session()
    
    # Float16
    if not skip_float16:
        print("\n--- Float16 Export ---")
        try:
            paths["float16"] = export_tflite(
                model,
                os.path.join(output_dir, f"{model_name}_float16.tflite"),
                quantization="float16",
            )
        except Exception as e:
            print(f"⚠️ Float16 export failed: {e}")
            paths["float16"] = None
        # Clear memory
        gc.collect()
        tf.keras.backend.clear_session()
    
    # INT8 - most important for ESP32
    print("\n--- INT8 Export ---")
    try:
        paths["int8"] = export_tflite_int8(
            model,
            os.path.join(output_dir, f"{model_name}_int8.tflite"),
            calibration_data,
            num_calibration_samples,
        )
    except Exception as e:
        print(f"⚠️ INT8 export failed: {e}")
        paths["int8"] = None
    
    # Clear memory
    gc.collect()
    
    # Summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    for variant, path in paths.items():
        if path and os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"   {variant}: {size_kb:.1f} KB - {path}")
        else:
            print(f"   {variant}: ❌ Export failed")
    
    return paths
