"""Model export utilities — Cycle 2 (PyTorch → ONNX).

TFLite conversion is done in a separate pipeline (task_export.py)
after ONNX export.  This module handles the PyTorch→ONNX step.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils_io import log, safe_mkdir


# =====================================================================
#  Export result containers
# =====================================================================

@dataclass
class ExportResult:
    """Container for model export metadata."""
    family: str
    source_path: str
    export_path: str
    export_format: str  # "onnx", "tflite", "espdl"
    input_shape: Tuple[int, ...]
    output_names: List[str]
    file_size_bytes: int = 0
    export_time_s: float = 0.0
    opset_version: int = 13
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)


@dataclass
class OnnxVerificationResult:
    """Result of ONNX model verification (inference test)."""
    valid: bool = False
    output_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    error_msg: str = ""
    num_detections_sample: int = 0


# =====================================================================
#  PyTorch  →  ONNX export
# =====================================================================

def export_pytorch_to_onnx(
    model: torch.nn.Module,
    export_dir: str,
    model_name: str,
    family: str,
    imgsz: int = 224,
    batch_size: int = 1,
    opset: int = 13,
    simplify: bool = True,
    dynamic_batch: bool = False,
) -> ExportResult:
    """Export a PyTorch model (FCOS / ESPDet) to ONNX.

    Args:
        model: Trained nn.Module.
        export_dir: Directory where the ONNX file is written.
        model_name: Base name for the file (e.g. ``fcos_v3s``).
        family: "FCOS" | "ESPDet" — determines output names.
        imgsz: Spatial resolution.
        opset: ONNX opset version.
        simplify: Run ``onnxsim.simplify`` if available.
        dynamic_batch: Allow variable batch dim.

    Returns:
        ExportResult with metadata.
    """
    import onnx  # type: ignore

    safe_mkdir(export_dir)
    out_path = os.path.join(export_dir, f"{model_name}.onnx")

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(batch_size, 3, imgsz, imgsz, device=device)

    # Determine output names from a forward pass
    with torch.no_grad():
        sample_out = model(dummy)

    # Check if model supports esp-ppq interleaved ONNX export
    # ESPDetPico exposes set_export_mode() which switches forward()
    # to return interleaved (box0, score0, box1, score1, ...) tuples.
    has_espdet_export = hasattr(model, "set_export_mode")

    # Build dynamic axes
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes["input"] = {0: "batch"}

    # Generate output names
    output_names = []
    if has_espdet_export:
        # ESPDet interleaved format: (box0, score0, box1, score1, box2, score2)
        # for esp-ppq / ESPDetPostProcessor compatibility
        model.set_export_mode(True)
        with torch.no_grad():
            espdet_out = model(dummy)
        model.set_export_mode(False)
        n_levels = len(espdet_out) // 2
        for i in range(n_levels):
            box_name = f"box{i}"
            score_name = f"score{i}"
            output_names.extend([box_name, score_name])
            if dynamic_batch:
                dynamic_axes[box_name] = {0: "batch"}
                dynamic_axes[score_name] = {0: "batch"}
    elif isinstance(sample_out, dict):
        for key in ["cls", "reg", "centerness"]:
            if key in sample_out:
                for i, t in enumerate(sample_out[key]):
                    name = f"{key}_lvl{i}"
                    output_names.append(name)
                    if dynamic_batch:
                        dynamic_axes[name] = {0: "batch"}
    elif isinstance(sample_out, (tuple, list)):
        for i, t in enumerate(sample_out):
            name = f"output_{i}"
            output_names.append(name)
            if dynamic_batch:
                dynamic_axes[name] = {0: "batch"}
    else:
        output_names = ["output"]
        if dynamic_batch:
            dynamic_axes["output"] = {0: "batch"}

    log(f"📦 Exportando {family} → ONNX (opset={opset}, "
        f"shape={list(dummy.shape)}, outputs={len(output_names)})")
    t0 = time.time()

    # Flatten dict output for tracing
    class _Wrapper(torch.nn.Module):
        def __init__(self, m, use_export_mode: bool = False):
            super().__init__()
            self.m = m
            self.use_export_mode = use_export_mode

        def forward(self, x):
            if self.use_export_mode and hasattr(self.m, "set_export_mode"):
                self.m.set_export_mode(True)
                out = self.m(x)
                self.m.set_export_mode(False)
                return out
            out = self.m(x)
            if isinstance(out, dict):
                flat = []
                for key in ["cls", "reg", "centerness"]:
                    if key in out:
                        for t in out[key]:
                            flat.append(t)
                return tuple(flat)
            return out

    wrapper = _Wrapper(model, use_export_mode=has_espdet_export)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["input"],
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        do_constant_folding=True,
    )
    export_time = time.time() - t0

    # Simplify
    if simplify:
        try:
            import onnxsim  # type: ignore
            onnx_model = onnx.load(out_path)
            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified, out_path)
                log("  ✅ onnx-simplifier applied")
            else:
                log("  ⚠️ onnx-simplifier failed, keeping original")
        except ImportError:
            log("  ℹ️ onnxsim not installed, skipping simplification")

    file_size = os.path.getsize(out_path)
    log(f"  ✅ Exportado: {out_path} ({file_size / 1024 / 1024:.2f} MB, "
        f"{export_time:.1f}s)")

    return ExportResult(
        family=family,
        source_path="pytorch_state_dict",
        export_path=out_path,
        export_format="onnx",
        input_shape=(batch_size, 3, imgsz, imgsz),
        output_names=output_names,
        file_size_bytes=file_size,
        export_time_s=export_time,
        opset_version=opset,
    )


def export_yolo26_to_onnx(
    model_path: str,
    export_dir: str,
    imgsz: int = 224,
    opset: int = 13,
    simplify: bool = True,
    half: bool = False,
) -> ExportResult:
    """Export a YOLO26 Custom model via Ultralytics API."""
    from ultralytics import YOLO  # type: ignore

    safe_mkdir(export_dir)
    log(f"📦 Exportando YOLO26_CUSTOM → ONNX (imgsz={imgsz})")
    t0 = time.time()

    model = YOLO(model_path)
    out_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        half=half,
    )
    export_time = time.time() - t0
    out_path = str(out_path)

    # Move to export_dir if needed
    final_path = os.path.join(export_dir, os.path.basename(out_path))
    if out_path != final_path:
        import shutil
        shutil.move(out_path, final_path)
        out_path = final_path

    file_size = os.path.getsize(out_path)
    log(f"  ✅ Exportado: {out_path} ({file_size / 1024 / 1024:.2f} MB, "
        f"{export_time:.1f}s)")

    return ExportResult(
        family="YOLO26_CUSTOM",
        source_path=model_path,
        export_path=out_path,
        export_format="onnx",
        input_shape=(1, 3, imgsz, imgsz),
        output_names=["output0"],
        file_size_bytes=file_size,
        export_time_s=export_time,
        opset_version=opset,
    )


# =====================================================================
#  ONNX verification
# =====================================================================

def verify_onnx_model(
    onnx_path: str,
    imgsz: int = 224,
    batch_size: int = 1,
    num_runs: int = 5,
) -> OnnxVerificationResult:
    """Verify ONNX model loads and runs inference correctly."""
    try:
        import onnx
        import onnxruntime as ort  # type: ignore
    except ImportError as e:
        return OnnxVerificationResult(
            valid=False, error_msg=f"Missing dependency: {e}"
        )

    result = OnnxVerificationResult()

    try:
        # Structural check
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Inference check
        sess = ort.InferenceSession(onnx_path)
        dummy = np.random.randn(batch_size, 3, imgsz, imgsz).astype(np.float32)
        input_name = sess.get_inputs()[0].name

        # Warmup
        sess.run(None, {input_name: dummy})

        # Timed runs
        latencies = []
        for _ in range(num_runs):
            t0 = time.time()
            outputs = sess.run(None, {input_name: dummy})
            latencies.append((time.time() - t0) * 1000)

        result.valid = True
        result.inference_time_ms = np.median(latencies)

        for out, meta in zip(outputs, sess.get_outputs()):
            result.output_shapes[meta.name] = tuple(out.shape)

        log(f"  ✅ ONNX verificado: {onnx_path}")
        log(f"     Latencia mediana: {result.inference_time_ms:.1f}ms")
        for name, shape in result.output_shapes.items():
            log(f"     {name}: {shape}")

    except Exception as e:
        result.valid = False
        result.error_msg = str(e)
        log(f"  ❌ ONNX verificación fallida: {e}")

    return result


def inspect_onnx_model(onnx_path: str) -> Dict[str, Any]:
    """Return metadata about an ONNX model."""
    import onnx

    model = onnx.load(onnx_path)
    info: Dict[str, Any] = {
        "ir_version": model.ir_version,
        "opset": model.opset_import[0].version,
        "producer": model.producer_name,
        "file_size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
        "inputs": [],
        "outputs": [],
    }

    graph = model.graph
    for inp in graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        info["inputs"].append({"name": inp.name, "shape": dims})
    for out in graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        info["outputs"].append({"name": out.name, "shape": dims})

    return info
