#!/usr/bin/env python3
"""
convert_fcos_mixed_precision.py — Cuantización FCOS T3 con precisión mixta
==========================================================================
Fuerza TODO el head FCOS (96 ops) y el FPN (10 ops) a FP32,
manteniendo el backbone en INT8.

Estrategia "full-head FP32": el intento previo de forzar solo los 12
InstanceNorm + ops vecinos (60 ops) fracasó (100% degradación), porque
las Conv del tower ya cuantizan las features antes de InstanceNorm.
La solución es que toda la ruta FPN→Head permanezca en FP32.

Referencia: Sección 9.5.2 del Registro_Cuantizacion_Modelos.md

Uso:
    cd 02_ING_MODELOS/Train_MLOps
    python scripts/convert_fcos_mixed_precision.py \
        --calib-dir ../datasets/IODC/coco/train/images \
        --n-samples 500
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch

try:
    from esp_ppq import TargetPlatform
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.api.setting import QuantizationSettingFactory
except ImportError:
    print("ERROR: esp-ppq no instalado. Ejecutar: pip install esp-ppq")
    sys.exit(1)


# ─── Configuración ───────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_MLOPS_DIR = SCRIPT_DIR.parent

ONNX_PATH = TRAIN_MLOPS_DIR / "outputs/fcos_v3s_v1-1771690809/export/fcos_v3s.onnx"
OUTPUT_DIR = TRAIN_MLOPS_DIR / "outputs/espdl/fcos_v3s_t3_mixed"
ESPDL_FILE = OUTPUT_DIR / "fcos_v3s_t3_mixed.espdl"
INPUT_SHAPE = [1, 3, 224, 224]
IMGSZ = 224

# ─── Operadores a forzar a FP32: FULL HEAD + FPN ────────────────────
# 96 head ops + 10 FPN ops = 106 total → FP32
# Backbone permanece en INT8

FPN_OPS = [
    "/m/fpn/lateral_convs.0/Conv",
    "/m/fpn/lateral_convs.1/Conv",
    "/m/fpn/lateral_convs.2/Conv",
    "/m/fpn/Resize",
    "/m/fpn/Add",
    "/m/fpn/Resize_1",
    "/m/fpn/Add_1",
    "/m/fpn/smooth_convs.0/Conv",
    "/m/fpn/smooth_convs.1/Conv",
    "/m/fpn/smooth_convs.2/Conv",
]

HEAD_OPS = [
    # ── Level 0 (P3) ──
    "/m/head/cls_tower/cls_tower.0/Conv",
    "/m/head/cls_tower/cls_tower.1/Reshape",
    "/m/head/cls_tower/cls_tower.1/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.1/Reshape_1",
    "/m/head/cls_tower/cls_tower.1/Mul",
    "/m/head/cls_tower/cls_tower.1/Add",
    "/m/head/cls_tower/cls_tower.2/Relu",
    "/m/head/cls_tower/cls_tower.3/Conv",
    "/m/head/cls_tower/cls_tower.4/Reshape",
    "/m/head/cls_tower/cls_tower.4/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.4/Reshape_1",
    "/m/head/cls_tower/cls_tower.4/Mul",
    "/m/head/cls_tower/cls_tower.4/Add",
    "/m/head/cls_tower/cls_tower.5/Relu",
    "/m/head/reg_tower/reg_tower.0/Conv",
    "/m/head/reg_tower/reg_tower.1/Reshape",
    "/m/head/reg_tower/reg_tower.1/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.1/Reshape_1",
    "/m/head/reg_tower/reg_tower.1/Mul",
    "/m/head/reg_tower/reg_tower.1/Add",
    "/m/head/reg_tower/reg_tower.2/Relu",
    "/m/head/reg_tower/reg_tower.3/Conv",
    "/m/head/reg_tower/reg_tower.4/Reshape",
    "/m/head/reg_tower/reg_tower.4/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.4/Reshape_1",
    "/m/head/reg_tower/reg_tower.4/Mul",
    "/m/head/reg_tower/reg_tower.4/Add",
    "/m/head/reg_tower/reg_tower.5/Relu",
    "/m/head/cls_logits/Conv",
    "/m/head/bbox_pred/Conv",
    "/m/head/Relu",
    "/m/head/centerness/Conv",
    # ── Level 1 (P4) ──
    "/m/head/cls_tower/cls_tower.0_1/Conv",
    "/m/head/cls_tower/cls_tower.1_1/Reshape",
    "/m/head/cls_tower/cls_tower.1_1/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.1_1/Reshape_1",
    "/m/head/cls_tower/cls_tower.1_1/Mul",
    "/m/head/cls_tower/cls_tower.1_1/Add",
    "/m/head/cls_tower/cls_tower.2_1/Relu",
    "/m/head/cls_tower/cls_tower.3_1/Conv",
    "/m/head/cls_tower/cls_tower.4_1/Reshape",
    "/m/head/cls_tower/cls_tower.4_1/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.4_1/Reshape_1",
    "/m/head/cls_tower/cls_tower.4_1/Mul",
    "/m/head/cls_tower/cls_tower.4_1/Add",
    "/m/head/cls_tower/cls_tower.5_1/Relu",
    "/m/head/reg_tower/reg_tower.0_1/Conv",
    "/m/head/reg_tower/reg_tower.1_1/Reshape",
    "/m/head/reg_tower/reg_tower.1_1/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.1_1/Reshape_1",
    "/m/head/reg_tower/reg_tower.1_1/Mul",
    "/m/head/reg_tower/reg_tower.1_1/Add",
    "/m/head/reg_tower/reg_tower.2_1/Relu",
    "/m/head/reg_tower/reg_tower.3_1/Conv",
    "/m/head/reg_tower/reg_tower.4_1/Reshape",
    "/m/head/reg_tower/reg_tower.4_1/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.4_1/Reshape_1",
    "/m/head/reg_tower/reg_tower.4_1/Mul",
    "/m/head/reg_tower/reg_tower.4_1/Add",
    "/m/head/reg_tower/reg_tower.5_1/Relu",
    "/m/head/cls_logits_1/Conv",
    "/m/head/bbox_pred_1/Conv",
    "/m/head/Relu_1",
    "/m/head/centerness_1/Conv",
    # ── Level 2 (P5) ──
    "/m/head/cls_tower/cls_tower.0_2/Conv",
    "/m/head/cls_tower/cls_tower.1_2/Reshape",
    "/m/head/cls_tower/cls_tower.1_2/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.1_2/Reshape_1",
    "/m/head/cls_tower/cls_tower.1_2/Mul",
    "/m/head/cls_tower/cls_tower.1_2/Add",
    "/m/head/cls_tower/cls_tower.2_2/Relu",
    "/m/head/cls_tower/cls_tower.3_2/Conv",
    "/m/head/cls_tower/cls_tower.4_2/Reshape",
    "/m/head/cls_tower/cls_tower.4_2/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.4_2/Reshape_1",
    "/m/head/cls_tower/cls_tower.4_2/Mul",
    "/m/head/cls_tower/cls_tower.4_2/Add",
    "/m/head/cls_tower/cls_tower.5_2/Relu",
    "/m/head/reg_tower/reg_tower.0_2/Conv",
    "/m/head/reg_tower/reg_tower.1_2/Reshape",
    "/m/head/reg_tower/reg_tower.1_2/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.1_2/Reshape_1",
    "/m/head/reg_tower/reg_tower.1_2/Mul",
    "/m/head/reg_tower/reg_tower.1_2/Add",
    "/m/head/reg_tower/reg_tower.2_2/Relu",
    "/m/head/reg_tower/reg_tower.3_2/Conv",
    "/m/head/reg_tower/reg_tower.4_2/Reshape",
    "/m/head/reg_tower/reg_tower.4_2/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.4_2/Reshape_1",
    "/m/head/reg_tower/reg_tower.4_2/Mul",
    "/m/head/reg_tower/reg_tower.4_2/Add",
    "/m/head/reg_tower/reg_tower.5_2/Relu",
    "/m/head/cls_logits_2/Conv",
    "/m/head/bbox_pred_2/Conv",
    "/m/head/Relu_2",
    "/m/head/centerness_2/Conv",
]

# All ops to dispatch to FP32
FP32_OPS = FPN_OPS + HEAD_OPS


def fix_negative_axes(onnx_path: str) -> str:
    """Fix negative axis attributes in ONNX model for esp-ppq compatibility."""
    model = onnx.load(onnx_path)
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=False)

    fixed = False
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "axis" and attr.i < 0:
                rank = _get_tensor_rank(model, node.input[0])
                if rank is not None and rank > 0:
                    new_axis = attr.i + rank
                    print(f"    fix: {node.op_type} '{node.name}' axis={attr.i} → axis={new_axis}")
                    attr.i = new_axis
                    fixed = True

    if fixed:
        fixed_path = onnx_path.replace(".onnx", "_fixed.onnx")
        onnx.save(model, fixed_path)
        return fixed_path
    return onnx_path


def _get_tensor_rank(model, tensor_name: str):
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            shape = vi.type.tensor_type.shape
            if shape and shape.dim:
                return len(shape.dim)
    for inp in model.graph.input:
        if inp.name == tensor_name:
            shape = inp.type.tensor_type.shape
            if shape and shape.dim:
                return len(shape.dim)
    for out in model.graph.output:
        if out.name == tensor_name:
            shape = out.type.tensor_type.shape
            if shape and shape.dim:
                return len(shape.dim)
    return None


def create_calibration_dataset(calib_dir: str, n_samples: int = 500) -> list:
    """Generate calibration dataset from training images."""
    from PIL import Image

    samples = []
    calib_path = Path(calib_dir) if calib_dir else None

    if calib_path and calib_path.exists():
        image_files = sorted(calib_path.glob("*.jpg")) + sorted(calib_path.glob("*.png"))
        image_files = image_files[:n_samples]
        print(f"  Usando {len(image_files)} imágenes de calibración")

        for i, img_path in enumerate(image_files):
            img = Image.open(img_path).convert("RGB").resize((IMGSZ, IMGSZ))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC → CHW
            tensor = torch.from_numpy(np.expand_dims(arr, 0))
            samples.append(tensor)
            if (i + 1) % 100 == 0:
                print(f"    ... {i + 1}/{len(image_files)} imágenes")
    else:
        print(f"  ⚠️ Sin calibración real — usando datos aleatorios ({n_samples} muestras)")
        for _ in range(n_samples):
            samples.append(torch.rand(1, 3, IMGSZ, IMGSZ))

    return samples


def convert_mixed_precision(calib_dir: str, n_samples: int, target: str = "esp32s3"):
    """Quantize FCOS T3 with InstanceNorm ops in FP32."""
    print("=" * 60)
    print("  FCOS T3 — Cuantización INT8 con Precisión Mixta")
    print(f"  FPN + Head completo → FP32 ({len(FP32_OPS)} ops)")
    print(f"  Backbone → INT8")
    print(f"  Target: {target}")
    print("=" * 60)

    if not ONNX_PATH.exists():
        print(f"  ❌ ONNX no encontrado: {ONNX_PATH}")
        sys.exit(1)

    onnx_size = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"\n  ONNX: {ONNX_PATH.name} ({onnx_size:.2f} MB)")

    # Fix negative axes
    actual_onnx = fix_negative_axes(str(ONNX_PATH))

    # Calibration
    print(f"\n  Preparando calibración...")
    calib_data = create_calibration_dataset(calib_dir, n_samples)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configure mixed-precision quantization
    print(f"\n  Configurando precisión mixta...")
    setting = QuantizationSettingFactory.espdl_setting()

    # Dispatch InstanceNorm + neighbors to FP32
    dispatched = 0
    for op_name in FP32_OPS:
        setting.dispatching_table.append(op_name, TargetPlatform.FP32)
        dispatched += 1
    print(f"  ✓ {dispatched} operadores asignados a FP32")
    print(f"    - {len(FPN_OPS)} FPN ops")
    print(f"    - {len(HEAD_OPS)} Head ops (towers + predictions)")

    # Quantize
    print(f"\n  Cuantizando...")
    t0 = time.time()

    try:
        ppq_graph = espdl_quantize_onnx(
            onnx_import_file=actual_onnx,
            espdl_export_file=str(ESPDL_FILE),
            calib_dataloader=calib_data,
            calib_steps=min(len(calib_data), 500),
            input_shape=INPUT_SHAPE,
            target=target,
            setting=setting,
        )

        elapsed = time.time() - t0
        print(f"\n  Tiempo de cuantización: {elapsed:.1f}s")

        if ESPDL_FILE.exists():
            espdl_size = ESPDL_FILE.stat().st_size / (1024 * 1024)
            compression = onnx_size / espdl_size if espdl_size > 0 else 0

            print(f"\n  ✅ ESPDL generado: {ESPDL_FILE}")
            print(f"     ONNX:  {onnx_size:.2f} MB")
            print(f"     ESPDL: {espdl_size:.2f} MB")
            print(f"     Compresión: {compression:.1f}x")

            # Check ESP32-S3 viability
            if espdl_size < 2.0:
                print(f"     🟢 Viable para ESP32-S3 (< 2 MB)")
            elif espdl_size < 4.0:
                print(f"     🟡 Ajustado para ESP32-S3 (2-4 MB)")
            else:
                print(f"     🔴 Probablemente NO cabe en ESP32-S3 (> 4 MB)")

            # Save summary
            summary = {
                "model": "fcos_v3s_t3_mixed",
                "strategy": "mixed_precision",
                "fp32_ops": len(FP32_OPS),
                "instancenorm_fp32": len(INSTANCENORM_OPS),
                "onnx_size_mb": round(onnx_size, 3),
                "espdl_size_mb": round(espdl_size, 3),
                "compression_ratio": round(compression, 2),
                "conversion_time_s": round(elapsed, 1),
                "target": target,
                "n_calib_samples": n_samples,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            summary_path = OUTPUT_DIR / "conversion_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n  📄 Resumen: {summary_path}")
        else:
            print(f"  ❌ ESPDL no generado")
            sys.exit(1)

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ Error en cuantización ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n  Siguiente paso:")
    print(f"  python scripts/eval_fp32_vs_int8.py --models fcos_v3s_t3_mixed --skip-viz")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="FCOS T3 — Cuantización INT8 con precisión mixta (InstanceNorm → FP32)"
    )
    parser.add_argument(
        "--calib-dir", type=str,
        default=str(TRAIN_MLOPS_DIR / "../datasets/IODC/coco/train/images"),
        help="Directorio con imágenes de calibración"
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="Número de imágenes de calibración"
    )
    parser.add_argument(
        "--target", type=str, default="esp32s3",
        choices=["esp32", "esp32s3", "esp32p4"],
        help="Target chip"
    )
    args = parser.parse_args()

    os.chdir(str(TRAIN_MLOPS_DIR))
    convert_mixed_precision(args.calib_dir, args.n_samples, args.target)


if __name__ == "__main__":
    main()
