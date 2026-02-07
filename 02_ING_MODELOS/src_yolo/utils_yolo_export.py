"""YOLO26 TFLite export utilities for ESP32-S3 deployment.

Handles exporting trained YOLO26 models to TFLite format
with INT8 quantization for edge deployment.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils_io import log, safe_exists, safe_filesize_mb, safe_mkdir

try:
    from ultralytics import YOLO  # type: ignore
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    ULTRALYTICS_AVAILABLE = False


def export_tflite(
    model_path: str,
    output_dir: str,
    imgsz: int = 224,
    half: bool = False,
    int8: bool = False,
    data_yaml: Optional[str] = None,
    end2end: bool = True,
) -> Optional[str]:
    """Export YOLO26 model to TFLite format.

    Args:
        model_path: Path to YOLO model weights (.pt)
        output_dir: Directory to save exported model
        imgsz: Input image size
        half: Use FP16 quantization
        int8: Use INT8 quantization
        data_yaml: Path to data.yaml for INT8 calibration
        end2end: Export with end-to-end inference (NMS-free)

    Returns:
        Path to exported TFLite model or None on failure
    """
    if not ULTRALYTICS_AVAILABLE:
        log("❌ Ultralytics no disponible")
        return None

    if not safe_exists(model_path):
        log(f"❌ Modelo no encontrado: {model_path}")
        return None

    safe_mkdir(output_dir)

    try:
        log(f"\n🔄 Exportando modelo a TFLite...")
        log(f"   📥 Entrada: {model_path}")
        log(f"   📐 Tamaño: {imgsz}x{imgsz}")
        log(f"   🔢 Cuantización: {'INT8' if int8 else 'FP16' if half else 'FP32'}")
        log(f"   🎯 End-to-end (NMS-free): {end2end}")

        model = YOLO(model_path)
        
        # Build export arguments
        export_args = {
            "format": "tflite",
            "imgsz": imgsz,
            "half": half,
            "int8": int8,
            "end2end": end2end,
        }
        
        if int8 and data_yaml and safe_exists(data_yaml):
            export_args["data"] = data_yaml
            log(f"   📊 Datos calibración: {data_yaml}")

        # Export
        start_time = time.time()
        export_path = model.export(**export_args)
        export_time = time.time() - start_time

        if export_path and safe_exists(export_path):
            size_mb = safe_filesize_mb(export_path)
            log(f"\n✅ Exportación completada en {export_time:.1f}s")
            log(f"   📁 Modelo: {export_path}")
            log(f"   📦 Tamaño: {size_mb:.2f} MB")
            
            # ESP32-S3 compatibility check
            if size_mb and size_mb <= 2.0:
                log(f"   ✅ Compatible con ESP32-S3 (8MB PSRAM)")
            elif size_mb and size_mb <= 4.0:
                log(f"   ⚠️ Ajustado para ESP32-S3 - verificar memoria")
            else:
                log(f"   ❌ Puede ser demasiado grande para ESP32-S3")

            return export_path
        else:
            log("❌ Exportación falló - archivo no creado")
            return None

    except Exception as exc:
        log(f"❌ Error durante exportación: {exc}")
        import traceback
        traceback.print_exc()
        return None


def export_tflite_int8(
    model_path: str,
    output_dir: str,
    data_yaml: str,
    imgsz: int = 224,
    end2end: bool = True,
) -> Optional[str]:
    """Export YOLO26 model to TFLite INT8 format.

    Wrapper for export_tflite with INT8 quantization enabled.

    Args:
        model_path: Path to YOLO model weights
        output_dir: Directory to save exported model
        data_yaml: Path to data.yaml for calibration
        imgsz: Input image size
        end2end: Export with end-to-end inference

    Returns:
        Path to exported model or None
    """
    return export_tflite(
        model_path=model_path,
        output_dir=output_dir,
        imgsz=imgsz,
        half=False,
        int8=True,
        data_yaml=data_yaml,
        end2end=end2end,
    )


def verify_tflite_model(
    tflite_path: str,
    test_images: Optional[List[np.ndarray]] = None,
    imgsz: int = 224,
) -> Dict[str, Any]:
    """Verify TFLite model can load and run inference.

    Args:
        tflite_path: Path to TFLite model
        test_images: Optional list of test images
        imgsz: Expected input size

    Returns:
        Dictionary with verification results
    """
    results = {
        "valid": False,
        "path": tflite_path,
        "size_mb": None,
        "input_shape": None,
        "output_shapes": None,
        "inference_time_ms": None,
        "error": None,
    }

    if not safe_exists(tflite_path):
        results["error"] = f"Archivo no encontrado: {tflite_path}"
        return results

    results["size_mb"] = safe_filesize_mb(tflite_path)

    try:
        import tensorflow as tf
        
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        results["input_shape"] = input_details[0]["shape"].tolist()
        results["output_shapes"] = [out["shape"].tolist() for out in output_details]
        results["input_dtype"] = str(input_details[0]["dtype"])
        results["num_outputs"] = len(output_details)

        # Run test inference
        if test_images is not None and len(test_images) > 0:
            test_img = test_images[0]
        else:
            # Create dummy input
            input_shape = input_details[0]["shape"]
            if input_details[0]["dtype"] == np.uint8:
                test_img = np.random.randint(0, 255, input_shape, dtype=np.uint8)
            else:
                test_img = np.random.randn(*input_shape).astype(np.float32)

        # Measure inference time
        times = []
        for _ in range(10):
            start = time.time()
            interpreter.set_tensor(input_details[0]["index"], test_img)
            interpreter.invoke()
            times.append((time.time() - start) * 1000)

        results["inference_time_ms"] = np.mean(times[1:])  # Skip first (warmup)
        results["valid"] = True

        log(f"\n✅ Verificación TFLite exitosa:")
        log(f"   📁 Modelo: {tflite_path}")
        log(f"   📦 Tamaño: {results['size_mb']:.2f} MB")
        log(f"   📐 Input: {results['input_shape']} ({results['input_dtype']})")
        log(f"   📤 Outputs: {results['num_outputs']}")
        log(f"   ⏱️ Inferencia: {results['inference_time_ms']:.2f} ms")

    except ImportError:
        results["error"] = "TensorFlow no disponible"
        log("⚠️ TensorFlow no instalado - no se puede verificar TFLite")
    except Exception as exc:
        results["error"] = str(exc)
        log(f"❌ Error verificando TFLite: {exc}")

    return results


def estimate_model_size(
    model_path: str,
    target_format: str = "int8",
) -> Dict[str, float]:
    """Estimate model size in different formats.

    Args:
        model_path: Path to PyTorch model
        target_format: Target format for estimation

    Returns:
        Dictionary with estimated sizes
    """
    estimates = {
        "pytorch_mb": None,
        "fp32_mb": None,
        "fp16_mb": None,
        "int8_mb": None,
        "esp32_compatible": False,
    }

    if not safe_exists(model_path):
        log(f"⚠️ Modelo no encontrado: {model_path}")
        return estimates

    pytorch_size = safe_filesize_mb(model_path)
    if pytorch_size is None:
        return estimates

    estimates["pytorch_mb"] = pytorch_size
    
    # Rough estimations based on typical compression ratios
    # PyTorch models include optimizer state, so actual model is smaller
    model_params_mb = pytorch_size * 0.5  # Approximate model-only size
    
    estimates["fp32_mb"] = model_params_mb
    estimates["fp16_mb"] = model_params_mb / 2
    estimates["int8_mb"] = model_params_mb / 4

    # YOLO26n specific adjustments (2.4M params → ~2.4MB FP32)
    if "yolo26n" in model_path.lower():
        estimates["fp32_mb"] = 2.4
        estimates["fp16_mb"] = 1.2
        estimates["int8_mb"] = 0.6

    estimates["esp32_compatible"] = estimates["int8_mb"] < 2.0 if estimates["int8_mb"] else False

    log(f"\n📊 Estimación de tamaño del modelo:")
    log(f"   PyTorch: {estimates['pytorch_mb']:.2f} MB")
    log(f"   FP32 TFLite: ~{estimates['fp32_mb']:.2f} MB")
    log(f"   FP16 TFLite: ~{estimates['fp16_mb']:.2f} MB")
    log(f"   INT8 TFLite: ~{estimates['int8_mb']:.2f} MB")
    log(f"   ESP32-S3: {'✅ Compatible' if estimates['esp32_compatible'] else '⚠️ Verificar'}")

    return estimates


def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def _parse_tflite_detections(
    output: np.ndarray,
    conf_threshold: float = 0.25,
    num_classes: int = 2,
    imgsz: int = 224,
) -> List[Dict[str, Any]]:
    """Parse TFLite output to get detections.
    
    Handles different output formats from YOLO TFLite models:
      - End-to-end (NMS-free): [batch, max_det, 6] → [x1, y1, x2, y2, conf, class_id]
      - Standard YOLO grid:    [batch, 4+num_classes, num_anchors] → xywh + class_scores
    
    Args:
        output: Raw TFLite output tensor
        conf_threshold: Confidence threshold for filtering
        num_classes: Number of classes
        imgsz: Input image size (for coordinate scaling)
    
    Returns:
        List of detections with keys: box, confidence, class_id
    """
    detections = []
    
    # Remove batch dimension
    if len(output.shape) == 3:
        output = output[0]  # [num_det, features] or [features, num_det]
    
    if len(output.shape) != 2:
        return detections
    
    num_features = output.shape[1] if output.shape[0] >= output.shape[1] else output.shape[0]
    
    # -------------------------------------------------------------------------
    # Format 1: End-to-end (NMS-free) → [max_det, 6] = [x1, y1, x2, y2, conf, class_id]
    # This is the output when model is exported with end2end=True
    # -------------------------------------------------------------------------
    if num_features == 6:
        # Ensure shape is [num_det, 6]
        if output.shape[1] != 6:
            output = output.T
        
        for i in range(output.shape[0]):
            x1, y1, x2, y2, confidence, class_id = output[i]
            confidence = float(confidence)
            class_id = int(round(class_id))
            
            if confidence >= conf_threshold and 0 <= class_id < num_classes:
                # Coordinates are normalized (0-1) from end2end export;
                # scale to pixel coordinates to match PyTorch output
                detections.append({
                    "box": np.array([
                        x1 * imgsz, y1 * imgsz,
                        x2 * imgsz, y2 * imgsz,
                    ]),
                    "confidence": confidence,
                    "class_id": class_id,
                })
        
        return detections
    
    # -------------------------------------------------------------------------
    # Format 2: Standard YOLO grid → [4+num_classes, num_anchors] or transposed
    # -------------------------------------------------------------------------
    # Determine format: [num_det, features] or [features, num_det]
    if output.shape[0] < output.shape[1]:
        output = output.T
    
    num_features = output.shape[1]
    
    for i in range(output.shape[0]):
        row = output[i]
        
        if num_features >= 4 + num_classes:
            # Format: [x, y, w, h, class_scores...] or [x, y, w, h, obj, class_scores...]
            if num_features == 4 + num_classes:
                box = row[:4]
                class_scores = row[4:]
                obj_score = 1.0
            else:
                box = row[:4]
                obj_score = row[4]
                class_scores = row[5:5+num_classes]
            
            class_id = int(np.argmax(class_scores))
            confidence = float(obj_score * class_scores[class_id])
            
            if confidence >= conf_threshold:
                x, y, w, h = box
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                
                detections.append({
                    "box": np.array([x1, y1, x2, y2]),
                    "confidence": confidence,
                    "class_id": class_id,
                })
    
    return detections


def _draw_detections_on_ax(
    ax: Any,
    image_rgb: np.ndarray,
    detections: List[Dict[str, Any]],
    class_names: List[str],
    title: str,
) -> None:
    """Draw bounding boxes with labels on a matplotlib axis.

    Args:
        ax: Matplotlib axis
        image_rgb: Image in RGB format
        detections: List of dicts with keys: box (xyxy), confidence, class_id
        class_names: List of class names
        title: Subplot title
    """
    import matplotlib.patches as patches

    COLORS = ["green", "blue", "red", "purple", "orange", "cyan", "magenta", "yellow"]

    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class_id"]
        conf = det["confidence"]
        color = COLORS[cls_id % len(COLORS)]
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls{cls_id}"

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        label = f"{cls_name} {conf:.2f}"
        ax.text(
            x1, max(y1 - 3, 0), label,
            fontsize=7, fontweight="bold", color="white",
            bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
        )


def compare_keras_vs_tflite(
    keras_model: Any,
    tflite_path: str,
    test_images: List[np.ndarray],
    class_names: List[str],
    imgsz: int = 224,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    save_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    num_visual_samples: int = 3,
) -> Dict[str, Any]:
    """Compare inference results between Keras/PyTorch and TFLite models.
    
    Computes comprehensive quality metrics including:
    - Timing comparison (speedup)
    - Detection counts (filtered by confidence)
    - Agreement rate (matching detections via IoU)
    - IoU statistics between matched detections
    - Confidence comparison (correlation, mean difference)
    - Side-by-side visual comparison of sample images

    Args:
        keras_model: YOLO model instance
        tflite_path: Path to TFLite model
        test_images: List of test images (BGR format, resized to imgsz)
        class_names: List of class names
        imgsz: Image size
        conf_threshold: Confidence threshold for filtering detections
        iou_threshold: IoU threshold for matching detections
        save_path: Optional path to save visual comparison figure
        experiment_name: Optional experiment name for figure title
        num_visual_samples: Number of random images for visual comparison (default: 3)

    Returns:
        Dictionary with comprehensive comparison results
    """
    import tensorflow as tf

    num_classes = len(class_names)
    
    results = {
        # Basic info
        "num_images": len(test_images),
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        
        # Timing
        "keras_time_ms": 0.0,
        "tflite_time_ms": 0.0,
        "speedup": 0.0,
        
        # Detection counts (filtered by confidence)
        "keras_detections": 0,
        "tflite_detections": 0,
        "keras_detections_per_image": 0.0,
        "tflite_detections_per_image": 0.0,
        
        # Agreement metrics
        "agreement_rate": 0.0,  # % of keras detections matched in tflite
        "matched_detections": 0,
        "unmatched_keras": 0,
        "unmatched_tflite": 0,
        
        # IoU statistics (between matched detections)
        "mean_iou": 0.0,
        "min_iou": 0.0,
        "max_iou": 0.0,
        "std_iou": 0.0,
        
        # Confidence comparison (between matched detections)
        "mean_conf_keras": 0.0,
        "mean_conf_tflite": 0.0,
        "mean_conf_diff": 0.0,  # keras - tflite (positive = keras more confident)
        "conf_correlation": 0.0,  # Pearson correlation
        
        # Per-class agreement
        "per_class_agreement": {},
        "per_class_detections_keras": {},
        "per_class_detections_tflite": {},
    }

    if not test_images:
        log("⚠️ No hay imágenes de prueba")
        return results

    # Storage for all detections
    all_keras_dets = []  # List of lists (per image)
    all_tflite_dets = []
    
    # =========================================================================
    # Run Keras/PyTorch inference
    # =========================================================================
    log("\n🔄 Ejecutando inferencia PyTorch...")
    keras_times = []
    
    for img in test_images:
        start = time.time()
        res = keras_model.predict(source=img, imgsz=imgsz, conf=conf_threshold, verbose=False)
        keras_times.append((time.time() - start) * 1000)
        
        img_dets = []
        if res[0].boxes is not None and len(res[0].boxes) > 0:
            boxes = res[0].boxes
            for j in range(len(boxes)):
                img_dets.append({
                    "box": boxes.xyxy[j].cpu().numpy(),
                    "confidence": float(boxes.conf[j].cpu()),
                    "class_id": int(boxes.cls[j].cpu()),
                })
        all_keras_dets.append(img_dets)

    results["keras_time_ms"] = float(np.mean(keras_times[1:]) if len(keras_times) > 1 else keras_times[0])
    
    # =========================================================================
    # Run TFLite inference
    # =========================================================================
    log("🔄 Ejecutando inferencia TFLite...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    tflite_times = []
    
    for img in test_images:
        # Prepare input
        if len(img.shape) == 3:
            img_input = np.expand_dims(img, axis=0)
        else:
            img_input = img
            
        # Resize if needed
        if img_input.shape[1:3] != (imgsz, imgsz):
            import cv2
            img_input = cv2.resize(img_input[0], (imgsz, imgsz))
            img_input = np.expand_dims(img_input, axis=0)
        
        # Handle input type
        if input_details[0]["dtype"] == np.uint8:
            img_input = img_input.astype(np.uint8)
        else:
            img_input = img_input.astype(np.float32) / 255.0
        
        start = time.time()
        interpreter.set_tensor(input_details[0]["index"], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        tflite_times.append((time.time() - start) * 1000)
        
        # Parse TFLite output
        img_dets = _parse_tflite_detections(output, conf_threshold, num_classes, imgsz)
        all_tflite_dets.append(img_dets)

    results["tflite_time_ms"] = float(np.mean(tflite_times[1:]) if len(tflite_times) > 1 else tflite_times[0])
    
    # =========================================================================
    # Calculate speedup
    # =========================================================================
    if results["keras_time_ms"] > 0 and results["tflite_time_ms"] > 0:
        results["speedup"] = results["keras_time_ms"] / results["tflite_time_ms"]
    
    # =========================================================================
    # Count detections
    # =========================================================================
    total_keras = sum(len(dets) for dets in all_keras_dets)
    total_tflite = sum(len(dets) for dets in all_tflite_dets)
    
    results["keras_detections"] = total_keras
    results["tflite_detections"] = total_tflite
    results["keras_detections_per_image"] = total_keras / len(test_images) if test_images else 0
    results["tflite_detections_per_image"] = total_tflite / len(test_images) if test_images else 0
    
    # Per-class counts
    for cls_name in class_names:
        results["per_class_detections_keras"][cls_name] = 0
        results["per_class_detections_tflite"][cls_name] = 0
        results["per_class_agreement"][cls_name] = 0.0
    
    for img_dets in all_keras_dets:
        for det in img_dets:
            cls_id = det["class_id"]
            if 0 <= cls_id < len(class_names):
                results["per_class_detections_keras"][class_names[cls_id]] += 1
    
    for img_dets in all_tflite_dets:
        for det in img_dets:
            cls_id = det["class_id"]
            if 0 <= cls_id < len(class_names):
                results["per_class_detections_tflite"][class_names[cls_id]] += 1
    
    # =========================================================================
    # Match detections between models (agreement analysis)
    # =========================================================================
    all_ious = []
    all_keras_confs = []
    all_tflite_confs = []
    matched_count = 0
    per_class_matched = {cls: 0 for cls in class_names}
    per_class_keras_total = {cls: 0 for cls in class_names}
    
    for keras_dets, tflite_dets in zip(all_keras_dets, all_tflite_dets):
        # For each keras detection, find best matching tflite detection
        tflite_matched = [False] * len(tflite_dets)
        
        for k_det in keras_dets:
            k_cls = k_det["class_id"]
            if 0 <= k_cls < len(class_names):
                per_class_keras_total[class_names[k_cls]] += 1
            
            best_iou = 0.0
            best_idx = -1
            
            for t_idx, t_det in enumerate(tflite_dets):
                # Must be same class
                if t_det["class_id"] != k_det["class_id"]:
                    continue
                if tflite_matched[t_idx]:
                    continue
                    
                iou = _compute_iou(k_det["box"], t_det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = t_idx
            
            # Check if match found
            if best_iou >= iou_threshold and best_idx >= 0:
                tflite_matched[best_idx] = True
                matched_count += 1
                all_ious.append(best_iou)
                all_keras_confs.append(k_det["confidence"])
                all_tflite_confs.append(tflite_dets[best_idx]["confidence"])
                
                if 0 <= k_cls < len(class_names):
                    per_class_matched[class_names[k_cls]] += 1
    
    results["matched_detections"] = matched_count
    results["unmatched_keras"] = total_keras - matched_count
    results["unmatched_tflite"] = total_tflite - matched_count
    
    # Agreement rate
    if total_keras > 0:
        results["agreement_rate"] = matched_count / total_keras
    
    # Per-class agreement
    for cls_name in class_names:
        if per_class_keras_total[cls_name] > 0:
            results["per_class_agreement"][cls_name] = (
                per_class_matched[cls_name] / per_class_keras_total[cls_name]
            )
    
    # =========================================================================
    # IoU statistics
    # =========================================================================
    if all_ious:
        results["mean_iou"] = float(np.mean(all_ious))
        results["min_iou"] = float(np.min(all_ious))
        results["max_iou"] = float(np.max(all_ious))
        results["std_iou"] = float(np.std(all_ious))
    
    # =========================================================================
    # Confidence comparison
    # =========================================================================
    if all_keras_confs and all_tflite_confs:
        results["mean_conf_keras"] = float(np.mean(all_keras_confs))
        results["mean_conf_tflite"] = float(np.mean(all_tflite_confs))
        results["mean_conf_diff"] = float(np.mean(np.array(all_keras_confs) - np.array(all_tflite_confs)))
        
        # Pearson correlation
        if len(all_keras_confs) > 1:
            corr_matrix = np.corrcoef(all_keras_confs, all_tflite_confs)
            results["conf_correlation"] = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    
    # =========================================================================
    # Visual comparison (side-by-side)
    # =========================================================================
    if num_visual_samples > 0 and len(test_images) > 0:
        try:
            import matplotlib.pyplot as plt
            import cv2

            # Prefer images with detections in at least one model
            candidates = [
                i for i in range(len(test_images))
                if len(all_keras_dets[i]) > 0 or len(all_tflite_dets[i]) > 0
            ]
            if len(candidates) < num_visual_samples:
                candidates = list(range(len(test_images)))

            rng = np.random.default_rng()
            sample_indices = rng.choice(
                candidates,
                size=min(num_visual_samples, len(candidates)),
                replace=False,
            )

            n_rows = len(sample_indices)
            fig_vis, axes = plt.subplots(
                n_rows, 2,
                figsize=(10, 4.5 * n_rows),
                squeeze=False,
            )

            exp_label = experiment_name or "Experiment"
            fig_vis.suptitle(
                f"Comparación Visual PyTorch vs TFLite INT8 — {exp_label}",
                fontsize=14, fontweight="bold", y=1.0,
            )

            for row, idx in enumerate(sample_indices):
                img_rgb = cv2.cvtColor(test_images[idx], cv2.COLOR_BGR2RGB)

                k_dets = all_keras_dets[idx]
                t_dets = all_tflite_dets[idx]

                _draw_detections_on_ax(
                    axes[row, 0], img_rgb, k_dets, class_names,
                    title=f"PyTorch ({len(k_dets)} det)",
                )
                _draw_detections_on_ax(
                    axes[row, 1], img_rgb, t_dets, class_names,
                    title=f"TFLite INT8 ({len(t_dets)} det)",
                )

            fig_vis.tight_layout()

            if save_path:
                safe_mkdir(os.path.dirname(save_path))
                fig_vis.savefig(save_path, dpi=150, bbox_inches="tight")
                log(f"\n📸 Comparación visual guardada: {save_path}")

            results["visual_comparison_fig"] = fig_vis

        except Exception as exc:
            log(f"⚠️ No se pudo generar comparación visual: {exc}")

    # =========================================================================
    # Log results
    # =========================================================================
    log(f"\n" + "=" * 60)
    log(f"📊 COMPARACIÓN PYTORCH vs TFLITE")
    log(f"=" * 60)
    log(f"\n⏱️ Tiempos de inferencia:")
    log(f"   PyTorch: {results['keras_time_ms']:.2f} ms/img")
    log(f"   TFLite:  {results['tflite_time_ms']:.2f} ms/img")
    log(f"   Speedup: {results['speedup']:.2f}x")
    
    log(f"\n🔢 Detecciones (conf >= {conf_threshold}):")
    log(f"   PyTorch: {results['keras_detections']} ({results['keras_detections_per_image']:.1f}/img)")
    log(f"   TFLite:  {results['tflite_detections']} ({results['tflite_detections_per_image']:.1f}/img)")
    
    log(f"\n🎯 Concordancia (IoU >= {iou_threshold}):")
    log(f"   Agreement rate: {results['agreement_rate']*100:.1f}%")
    log(f"   Matched: {results['matched_detections']}")
    log(f"   Unmatched PyTorch: {results['unmatched_keras']}")
    log(f"   Unmatched TFLite: {results['unmatched_tflite']}")
    
    if all_ious:
        log(f"\n📐 IoU entre detecciones coincidentes:")
        log(f"   Mean: {results['mean_iou']:.3f}")
        log(f"   Min:  {results['min_iou']:.3f}")
        log(f"   Max:  {results['max_iou']:.3f}")
        log(f"   Std:  {results['std_iou']:.3f}")
    
    if all_keras_confs:
        log(f"\n🔬 Comparación de confianzas:")
        log(f"   Mean PyTorch: {results['mean_conf_keras']:.3f}")
        log(f"   Mean TFLite:  {results['mean_conf_tflite']:.3f}")
        log(f"   Diff (Py-TF): {results['mean_conf_diff']:+.3f}")
        log(f"   Correlation:  {results['conf_correlation']:.3f}")
    
    log(f"\n📋 Concordancia por clase:")
    for cls_name in class_names:
        k_count = results["per_class_detections_keras"].get(cls_name, 0)
        t_count = results["per_class_detections_tflite"].get(cls_name, 0)
        agreement = results["per_class_agreement"].get(cls_name, 0.0)
        log(f"   {cls_name}: Py={k_count}, TF={t_count}, Agreement={agreement*100:.1f}%")
    
    log(f"=" * 60)

    return results
