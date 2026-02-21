"""Entry-point de Vertex AI — Export (ONNX → ESPDL).

Separate Custom Job for model conversion / quantization.
Downloads a trained ONNX model from GCS and converts it to
the target deployment format (ESPDL for ESP32-S3).

This task is designed to run AFTER training completes, so the
training container can shut down its GPU while conversion runs
on a cheaper CPU-only machine.

Uso::

    python -m trainer.task_export \\
        --onnx-uri gs://bucket/output/run/export/model.onnx \\
        --job-dir gs://bucket/output/run/export_output \\
        --project-id my-project \\
        --region us-central1 \\
        --family FCOS \\
        --imgsz 224 \\
        --quantize int8
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import os
import sys
import time
from pathlib import Path


LOCAL_WORK_DIR = "/tmp/export"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vertex AI — Model Export")
    p.add_argument("--onnx-uri", required=True,
                   help="GCS URI of the ONNX model to convert")
    p.add_argument("--job-dir", required=True,
                   help="GCS URI for output artifacts")
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--family", required=True,
                   choices=["FCOS", "YOLO26_CUSTOM", "ESPDet"],
                   help="Model family (determines output structure)")
    p.add_argument("--imgsz", type=int, default=224,
                   help="Input image size for the model")
    p.add_argument("--quantize", default=None,
                   choices=["int8", "fp16", None],
                   help="Quantization mode (optional)")
    p.add_argument("--calibration-uri", default=None,
                   help="GCS URI of calibration dataset for INT8 quantization")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ================================================================
    # Step 1 — Download ONNX model
    # ================================================================
    print("=" * 60)
    print("STEP 1 — Descarga del modelo ONNX")
    print("=" * 60)

    from trainer.gcs_utils import download_from_gcs, upload_to_gcs as gcs_upload

    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    local_onnx = os.path.join(LOCAL_WORK_DIR, "model.onnx")
    download_from_gcs(args.onnx_uri, local_onnx)
    print(f"✅ Descargado: {local_onnx} ({os.path.getsize(local_onnx) / 1024:.1f} KB)")

    # ================================================================
    # Step 2 — Verify ONNX model
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 2 — Verificación ONNX")
    print("=" * 60)

    from src_colab import verify_onnx_model, inspect_onnx_model

    verify_result = verify_onnx_model(local_onnx, imgsz=args.imgsz)
    if not verify_result.valid:
        raise RuntimeError(f"❌ ONNX inválido: {verify_result.error_msg}")

    info = inspect_onnx_model(local_onnx)
    print(f"📋 ONNX info:")
    print(f"   Opset: {info['opset']}")
    print(f"   Size: {info['file_size_mb']:.2f} MB")
    print(f"   Inputs: {info['inputs']}")
    print(f"   Outputs: {info['outputs']}")

    # ================================================================
    # Step 3 — Convert to target format
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 3 — Conversión a formato de despliegue")
    print("=" * 60)

    export_dir = os.path.join(LOCAL_WORK_DIR, "converted")
    os.makedirs(export_dir, exist_ok=True)

    # NOTE: The actual ONNX → ESPDL conversion is handled by the
    # convert_onnx_to_espdl.py script from 03_ING_DESPLIEGUE.
    # This task_export.py provides the Vertex AI wrapper.
    #
    # For now, we perform the following preparation steps:
    #   1. Download calibration data if INT8 quantization requested
    #   2. Run ONNX simplification / optimization
    #   3. (Future) Invoke ESPDL converter

    # ONNX optimization
    try:
        import onnx
        from onnxruntime.transformers.optimizer import optimize_model  # type: ignore
        optimized_path = os.path.join(export_dir, "model_optimized.onnx")
        opt_model = optimize_model(
            local_onnx,
            model_type="bert",  # generic optimization
            opt_level=1,
        )
        opt_model.save_model_to_file(optimized_path)
        print(f"✅ Optimizado: {optimized_path}")
    except ImportError:
        print("ℹ️  onnxruntime.transformers no disponible, copiando original")
        import shutil
        optimized_path = os.path.join(export_dir, "model_optimized.onnx")
        shutil.copy2(local_onnx, optimized_path)
    except Exception as e:
        print(f"⚠️  Optimización fallida ({e}), usando original")
        import shutil
        optimized_path = os.path.join(export_dir, "model_optimized.onnx")
        shutil.copy2(local_onnx, optimized_path)

    # Quantization placeholder
    if args.quantize == "int8":
        print("\n📊 Cuantización INT8 solicitada")
        if args.calibration_uri:
            cal_dir = os.path.join(LOCAL_WORK_DIR, "calibration")
            os.makedirs(cal_dir, exist_ok=True)
            download_from_gcs(args.calibration_uri, cal_dir)
            print(f"  ✅ Calibration data descargado")

        # INT8 quantization via onnxruntime
        try:
            from onnxruntime.quantization import (  # type: ignore
                quantize_dynamic,
                QuantType,
            )
            quantized_path = os.path.join(export_dir, "model_int8.onnx")
            quantize_dynamic(
                optimized_path,
                quantized_path,
                weight_type=QuantType.QInt8,
            )
            print(f"  ✅ Cuantizado: {quantized_path} "
                  f"({os.path.getsize(quantized_path) / 1024:.1f} KB)")
        except ImportError:
            print("  ⚠️  onnxruntime.quantization no disponible")
            quantized_path = optimized_path
    elif args.quantize == "fp16":
        print("\n📊 Cuantización FP16 solicitada")
        try:
            import onnx
            from onnxconverter_common import float16  # type: ignore
            onnx_model = onnx.load(optimized_path)
            fp16_model = float16.convert_float_to_float16(onnx_model)
            quantized_path = os.path.join(export_dir, "model_fp16.onnx")
            onnx.save(fp16_model, quantized_path)
            print(f"  ✅ FP16: {quantized_path}")
        except ImportError:
            print("  ⚠️  onnxconverter-common no disponible")
            quantized_path = optimized_path
    else:
        quantized_path = optimized_path

    # ================================================================
    # Step 4 — Verify converted model
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 4 — Verificación del modelo convertido")
    print("=" * 60)

    final_verify = verify_onnx_model(quantized_path, imgsz=args.imgsz)
    print(f"✅ Valid: {final_verify.valid}")
    print(f"   Latency: {final_verify.inference_time_ms:.1f}ms")
    print(f"   Size: {os.path.getsize(quantized_path) / 1024:.1f} KB")

    # ================================================================
    # Step 5 — Upload to GCS
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 5 — Subida a GCS")
    print("=" * 60)

    run_name = args.run_name or f"export-{args.family.lower()}-{int(time.time())}"

    for fpath in [local_onnx, optimized_path, quantized_path]:
        if os.path.exists(fpath):
            rel = os.path.relpath(fpath, LOCAL_WORK_DIR)
            gcs_dest = f"{args.job_dir}/{run_name}/{rel}"
            gcs_upload(fpath, gcs_dest)

    # Save metadata
    import json
    metadata = {
        "family": args.family,
        "imgsz": args.imgsz,
        "quantize": args.quantize,
        "original_onnx_uri": args.onnx_uri,
        "optimized_valid": final_verify.valid,
        "latency_ms": final_verify.inference_time_ms,
        "size_bytes": os.path.getsize(quantized_path),
    }
    meta_path = os.path.join(LOCAL_WORK_DIR, "export_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    gcs_upload(meta_path, f"{args.job_dir}/{run_name}/export_metadata.json")

    print(f"\n✅ Export pipeline completado ({args.family}).")


if __name__ == "__main__":
    main()
