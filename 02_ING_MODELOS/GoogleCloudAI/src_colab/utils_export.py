"""Unified TFLite INT8 export and verification.

Handles both YOLO (Ultralytics ``model.export()``) and MobileNet-SSD
(Keras → SavedModel → TFLite converter with full-integer quantisation).

Target: ESP32-S3 (N16R8) — models must be < 8 MB.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils_io import log, safe_mkdir, write_json, get_file_size_mb


# =====================================================================
#  Result structs
# =====================================================================

@dataclass
class TFLiteExportResult:
    """Standardised export result for every family."""
    model_name: str = ""
    family: str = ""
    tflite_path: str = ""
    size_mb: float = 0.0
    input_shape: Tuple[int, ...] = ()
    output_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    quantization: str = "int8"
    export_time_s: float = 0.0
    esp32_compatible: bool = False
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        ok = "✅" if self.esp32_compatible else "❌"
        lines = [
            f"\n📦 Export: {self.model_name} ({self.family})",
            f"  Archivo: {self.tflite_path}",
            f"  Tamaño: {self.size_mb:.2f} MB  {ok} ESP32-S3 (<8 MB)",
            f"  Input:  {self.input_shape}",
            f"  Quant:  {self.quantization}",
            f"  Tiempo: {self.export_time_s:.1f}s",
        ]
        if self.output_shapes:
            lines.append(f"  Outputs:")
            for k, v in self.output_shapes.items():
                lines.append(f"    {k}: {v}")
        if self.errors:
            lines.append(f"  ⚠️ Errores: {self.errors}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "family": self.family,
            "tflite_path": self.tflite_path,
            "size_mb": self.size_mb,
            "input_shape": list(self.input_shape),
            "output_shapes": {k: list(v) for k, v in self.output_shapes.items()},
            "quantization": self.quantization,
            "export_time_s": self.export_time_s,
            "esp32_compatible": self.esp32_compatible,
            "errors": self.errors,
        }


@dataclass
class TFLiteVerificationResult:
    """Result of model-vs-TFLite comparison."""
    tflite_path: str = ""
    n_samples: int = 0
    agreement_rate: float = 0.0
    avg_iou: float = 0.0
    avg_conf_diff: float = 0.0
    avg_inference_ms: float = 0.0
    passed: bool = False

    def summary(self) -> str:
        ok = "✅" if self.passed else "❌"
        return (
            f"\n{ok} Verificación TFLite ({self.n_samples} muestras)\n"
            f"  Agreement: {self.agreement_rate:.1%}\n"
            f"  Avg IoU: {self.avg_iou:.4f}\n"
            f"  Avg Δconf: {self.avg_conf_diff:.4f}\n"
            f"  Avg latency: {self.avg_inference_ms:.1f} ms"
        )


# =====================================================================
#  YOLO export
# =====================================================================

def export_yolo_tflite(
    model_path: str,
    imgsz: int = 224,
    output_dir: Optional[str] = None,
    int8: bool = True,
    data: Optional[str] = None,
) -> TFLiteExportResult:
    """Export YOLO model to TFLite INT8 via Ultralytics.

    Ultralytics handles SavedModel → TFLite → quantization internally.
    """
    from ultralytics import YOLO  # type: ignore

    res = TFLiteExportResult(family="yolo")
    res.model_name = Path(model_path).stem

    try:
        model = YOLO(model_path)
        log(f"\n📦 Exportando YOLO → TFLite (INT8={int8})")

        t0 = time.time()
        export_kwargs: Dict[str, Any] = dict(
            format="tflite", imgsz=imgsz, int8=int8,
        )
        if data:
            export_kwargs["data"] = data

        export_path = model.export(**export_kwargs)
        res.export_time_s = time.time() - t0

        # Ultralytics may return folder or file path
        if isinstance(export_path, str):
            if os.path.isdir(export_path):
                tflite_files = list(Path(export_path).glob("*.tflite"))
                if tflite_files:
                    res.tflite_path = str(tflite_files[0])
                else:
                    res.tflite_path = export_path
            else:
                res.tflite_path = export_path
        else:
            res.tflite_path = str(export_path)

        # try to find actual .tflite
        if not res.tflite_path.endswith(".tflite"):
            candidates = list(Path(res.tflite_path).parent.glob("**/*.tflite"))
            if candidates:
                res.tflite_path = str(candidates[0])

        if os.path.exists(res.tflite_path):
            res.size_mb = get_file_size_mb(res.tflite_path)
            res.esp32_compatible = res.size_mb < 8.0
            res.input_shape, res.output_shapes = _inspect_tflite(res.tflite_path)

            # copy to output dir if specified
            if output_dir:
                safe_mkdir(output_dir)
                dest = os.path.join(output_dir, Path(res.tflite_path).name)
                import shutil
                shutil.copy2(res.tflite_path, dest)
                res.tflite_path = dest

        res.quantization = "int8" if int8 else "float32"
        log(res.summary())

    except Exception as exc:
        res.errors.append(str(exc))
        log(f"❌ Error export YOLO: {exc}")
        import traceback; traceback.print_exc()

    return res


# =====================================================================
#  MobileNet export
# =====================================================================

def export_mobilenet_tflite(
    model,
    output_path: str,
    representative_dataset=None,
    imgsz: int = 224,
    model_name: str = "mobilenet_ssd",
) -> TFLiteExportResult:
    """Export Keras MobileNet-SSD to TFLite INT8.

    Steps:
      1. Save as SavedModel
      2. TFLiteConverter with full integer quantization
      3. Write .tflite
    """
    import tensorflow as tf

    res = TFLiteExportResult(family="mobilenet")
    res.model_name = model_name
    res.quantization = "int8"

    try:
        saved_model_dir = output_path.replace(".tflite", "_saved_model")
        safe_mkdir(saved_model_dir)

        log(f"\n📦 Exportando MobileNet → TFLite INT8")
        log(f"  1/3 Guardando SavedModel...")
        t0 = time.time()
        # Keras 3: model.export() genera SavedModel directamente
        model.export(saved_model_dir, format="tf_saved_model")

        log(f"  2/3 Convirtiendo a TFLite con INT8...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.float32
        else:
            log("  ⚠️  Sin dataset representativo — usando dynamic range quantization")

        tflite_model = converter.convert()

        log(f"  3/3 Guardando {output_path}...")
        safe_mkdir(Path(output_path).parent)
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        res.export_time_s = time.time() - t0
        res.tflite_path = output_path
        res.size_mb = get_file_size_mb(output_path)
        res.esp32_compatible = res.size_mb < 8.0
        res.input_shape, res.output_shapes = _inspect_tflite(output_path)

        log(res.summary())

    except Exception as exc:
        res.errors.append(str(exc))
        log(f"❌ Error export MobileNet: {exc}")
        import traceback; traceback.print_exc()

    return res


def create_representative_dataset(
    dataset,
    n_samples: int = 100,
    imgsz: int = 224,
):
    """Create representative dataset generator for INT8 calibration.

    Accepts a tf.data.Dataset (unbatched or batched).
    """
    import tensorflow as tf

    samples = []
    count = 0
    for batch in dataset:
        images = batch[0] if isinstance(batch, tuple) else batch
        if images.ndim == 4:
            for i in range(images.shape[0]):
                samples.append(images[i].numpy())
                count += 1
                if count >= n_samples:
                    break
        else:
            samples.append(images.numpy())
            count += 1
        if count >= n_samples:
            break

    def gen():
        for s in samples:
            yield [np.expand_dims(s.astype(np.float32), 0)]

    log(f"📊 Dataset representativo: {len(samples)} muestras")
    return gen


# =====================================================================
#  Unified export entry-point
# =====================================================================

def export_tflite_int8(
    model,
    family: str,
    output_dir: str,
    model_name: str = "model",
    imgsz: int = 224,
    data_yaml: Optional[str] = None,
    representative_dataset=None,
    **kwargs,
) -> TFLiteExportResult:
    """Unified export dispatch — works for all families."""
    from .config import is_yolo_family

    safe_mkdir(output_dir)
    output_path = os.path.join(output_dir, f"{model_name}_int8.tflite")

    if is_yolo_family(family):
        # For YOLO, model is a path string (best.pt)
        return export_yolo_tflite(
            model_path=str(model),
            imgsz=imgsz,
            output_dir=output_dir,
            int8=True,
            data=data_yaml,
        )
    else:
        return export_mobilenet_tflite(
            model=model,
            output_path=output_path,
            representative_dataset=representative_dataset,
            imgsz=imgsz,
            model_name=model_name,
        )


# =====================================================================
#  Export report
# =====================================================================

def print_export_report(result: TFLiteExportResult) -> None:
    """Print a standardised export report."""
    log(result.summary())


def save_export_result(result: TFLiteExportResult, output_path: str) -> None:
    """Save export result as JSON."""
    write_json(output_path, result.to_dict())
    log(f"💾 Export result guardado: {output_path}")


# =====================================================================
#  Helpers
# =====================================================================

def _inspect_tflite(path: str) -> Tuple[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """Read input/output shapes from TFLite file."""
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()

        inp = interp.get_input_details()[0]
        input_shape = tuple(inp["shape"])

        output_shapes = {}
        for out in interp.get_output_details():
            output_shapes[out["name"]] = tuple(out["shape"])

        return input_shape, output_shapes
    except Exception:
        return (), {}
