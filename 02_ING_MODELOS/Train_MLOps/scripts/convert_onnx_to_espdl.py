#!/usr/bin/env python3
"""
Script de conversión ONNX → ESPDL para ESP32-S3
Adaptado de 03_ING_DESPLIEGUE/convert_onnx_to_espdl.py para Train_MLOps.

Convierte los 3 mejores modelos entrenados en Vertex AI:
  - FCOS T3  (fcos_v3s_t3)       — 4.74 MB ONNX, ~1.2 MB INT8
  - YOLO26 T2 (yolo26n_t2)       — 9.97 MB ONNX, ~2.5 MB INT8
  - ESPDet T4 (espdet_pico_t4)   — 1.41 MB ONNX, ~0.4 MB INT8

Requisitos:
    pip install esp-ppq onnx numpy pillow

Uso:
    cd 02_ING_MODELOS/Train_MLOps
    python scripts/convert_onnx_to_espdl.py \\
        --calib-dir ../datasets/IODC/coco/train/images \\
        --target esp32s3 \\
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
    from esp_ppq import *
    from esp_ppq.api import espdl_quantize_onnx
except ImportError:
    print("ERROR: esp-ppq no instalado. Ejecutar: pip install esp-ppq")
    sys.exit(1)


# ─── Configuración de modelos (Train_MLOps best runs) ────────────────────
MODELS = {
    "fcos_v3s_t3": {
        "onnx": "outputs/fcos_v3s_v1-1771690809/export/fcos_v3s.onnx",
        "input_shape": [1, 3, 224, 224],  # NCHW (PyTorch convention)
        "channel_format": "nchw",
        "family": "FCOS",
        "run_id": "fcos_v3s_v1-1771690809",
        "description": "FCOS MobileNetV3-Small + SimpleFPN — Train 3 (best F1)",
    },
    "yolo26n_t2": {
        "onnx": "outputs/yolo26n_custom_v2-run1/export/best.onnx",
        "input_shape": [1, 3, 224, 224],  # NCHW
        "channel_format": "nchw",
        "family": "YOLO26_CUSTOM",
        "run_id": "yolo26n_custom_v2-run1",
        "description": "YOLO26n MuSGD — Train 2 (best mAP@50)",
    },
    "espdet_pico_t4": {
        "onnx": "outputs/espdet-pico-v4-t4/export/espdet_pico.onnx",
        "input_shape": [1, 3, 224, 224],  # NCHW
        "channel_format": "nchw",
        "family": "ESPDet",
        "run_id": "espdet-pico-v4-t4",
        "description": "ESPDet-Pico BCE+NMS tuning — Train 4 (primary ESP32-S3 candidate)",
    },
}

# Salida relativa a Train_MLOps/
OUTPUT_DIR = Path("outputs/espdl")


def fix_negative_axes(onnx_path: str) -> str:
    """
    Preprocesa un modelo ONNX para reemplazar ejes negativos (e.g. axis=-1)
    por equivalentes positivos. Necesario para esp-ppq cuyo ESPDL exporter
    no soporta valores negativos en atributos de axis.

    Retorna la ruta al modelo corregido (mismo directorio, sufijo _fixed).
    Si no hay cambios necesarios, retorna la ruta original.
    """
    model = onnx.load(onnx_path)
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=False)

    fixed = False
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "axis" and attr.i < 0:
                # Determinar el rango del tensor de entrada
                # para convertir axis negativo a positivo
                rank = _get_tensor_rank(model, node.input[0])
                if rank is not None and rank > 0:
                    new_axis = attr.i + rank
                    print(f"    fix_negative_axes: {node.op_type} '{node.name}' "
                          f"axis={attr.i} → axis={new_axis} (rank={rank})")
                    attr.i = new_axis
                    fixed = True
                else:
                    print(f"    ⚠️ No se pudo resolver rango para {node.name} "
                          f"(axis={attr.i})")

    if fixed:
        fixed_path = onnx_path.replace(".onnx", "_fixed.onnx")
        onnx.save(model, fixed_path)
        print(f"    ✅ Modelo con ejes corregidos: {fixed_path}")
        return fixed_path
    return onnx_path


def _get_tensor_rank(model, tensor_name: str) -> int | None:
    """Obtiene el rango (número de dimensiones) de un tensor del grafo ONNX."""
    # Buscar en value_info (inferido por shape_inference)
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            shape = vi.type.tensor_type.shape
            if shape and shape.dim:
                return len(shape.dim)
    # Buscar en inputs del grafo
    for inp in model.graph.input:
        if inp.name == tensor_name:
            shape = inp.type.tensor_type.shape
            if shape and shape.dim:
                return len(shape.dim)
    # Buscar en outputs del grafo
    for out in model.graph.output:
        if out.name == tensor_name:
            shape = out.type.tensor_type.shape
            if shape and shape.dim:
                return len(shape.dim)
    return None


def create_calibration_dataset(
    calib_dir: str,
    input_shape: list,
    channel_format: str,
    n_samples: int = 500,
) -> list:
    """
    Genera un dataset de calibración a partir de imágenes reales del train set.

    Si calib_dir está vacío o no existe, genera datos aleatorios (no ideal
    pero funcional para verificar la pipeline).

    Args:
        calib_dir: Directorio con imágenes jpg/png.
        input_shape: Shape del input del modelo [N, C, H, W] o [N, H, W, C].
        channel_format: 'nchw' o 'nhwc'.
        n_samples: Número máximo de imágenes de calibración a usar.

    Returns:
        Lista de arrays numpy con shape = input_shape.
    """
    from PIL import Image

    h, w = 224, 224
    samples = []

    calib_path = Path(calib_dir) if calib_dir else None

    if calib_path and calib_path.exists():
        image_files = sorted(calib_path.glob("*.jpg")) + sorted(calib_path.glob("*.png"))
        image_files = image_files[:n_samples]
        print(f"  Usando {len(image_files)} imágenes de calibración de {calib_path}")

        for i, img_path in enumerate(image_files):
            img = Image.open(img_path).convert("RGB").resize((w, h))
            arr = np.array(img, dtype=np.float32) / 255.0

            if channel_format == "nchw":
                arr = arr.transpose(2, 0, 1)  # HWC → CHW

            # esp-ppq collate_fn expects torch.Tensor
            tensor = torch.from_numpy(np.expand_dims(arr, 0))
            samples.append(tensor)

            # Progreso cada 100 imágenes
            if (i + 1) % 100 == 0:
                print(f"    ... {i + 1}/{len(image_files)} imágenes procesadas")

        print(f"  ✅ Dataset de calibración: {len(samples)} muestras listas")
    else:
        print(f"  ⚠️ Sin directorio de calibración — usando datos aleatorios ({n_samples} muestras)")
        for _ in range(n_samples):
            if channel_format == "nchw":
                samples.append(torch.rand(1, 3, h, w))
            else:
                samples.append(torch.rand(1, h, w, 3))

    return samples


def convert_model(
    name: str,
    config: dict,
    calib_dir: str,
    n_samples: int,
    target: str = "esp32s3",
) -> dict:
    """
    Convierte un modelo ONNX a formato ESPDL (INT8 simétrico).

    Returns:
        dict con metadatos del resultado (éxito, tamaños, tiempos).
    """
    onnx_path = Path(config["onnx"])
    result = {
        "name": name,
        "family": config["family"],
        "run_id": config["run_id"],
        "description": config["description"],
        "success": False,
        "onnx_path": str(onnx_path),
        "onnx_size_mb": 0.0,
        "espdl_path": "",
        "espdl_size_mb": 0.0,
        "compression_ratio": 0.0,
        "conversion_time_s": 0.0,
        "target": target,
        "n_calib_samples": n_samples,
        "error": "",
    }

    if not onnx_path.exists():
        msg = f"ONNX no encontrado: {onnx_path}"
        print(f"  ❌ {msg}")
        result["error"] = msg
        return result

    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    result["onnx_size_mb"] = round(onnx_size_mb, 3)

    print(f"\n{'='*60}")
    print(f"Convirtiendo: {name}")
    print(f"  {config['description']}")
    print(f"  ONNX: {onnx_path} ({onnx_size_mb:.2f} MB)")
    print(f"  Input shape: {config['input_shape']}")
    print(f"  Target: {target}")

    # Generar dataset de calibración
    calib_data = create_calibration_dataset(
        calib_dir, config["input_shape"], config["channel_format"], n_samples
    )

    # Crear directorio de salida
    output_path = OUTPUT_DIR / name
    output_path.mkdir(parents=True, exist_ok=True)

    # Preprocesar: fix ejes negativos (necesario para YOLO26 y similares)
    actual_onnx = fix_negative_axes(str(onnx_path))

    espdl_file = output_path / f"{name}.espdl"
    result["espdl_path"] = str(espdl_file)

    t0 = time.time()

    try:
        # Cuantización y exportación con esp-ppq
        quant_setting = QuantizationSettingFactory.espdl_setting()

        # Cuantización INT8 simétrica (estándar para ESP32-S3)
        ppq_graph = espdl_quantize_onnx(
            onnx_import_file=actual_onnx,
            espdl_export_file=str(espdl_file),
            calib_dataloader=calib_data,
            calib_steps=min(len(calib_data), 500),
            input_shape=config["input_shape"],
            target=target,
            setting=quant_setting,
            do_quantize=True,
        )

        elapsed = time.time() - t0
        result["conversion_time_s"] = round(elapsed, 1)

        if espdl_file.exists():
            espdl_size_mb = espdl_file.stat().st_size / (1024 * 1024)
            compression = onnx_size_mb / espdl_size_mb if espdl_size_mb > 0 else 0
            result["espdl_size_mb"] = round(espdl_size_mb, 3)
            result["compression_ratio"] = round(compression, 2)
            result["success"] = True

            print(f"  ✅ ESPDL generado: {espdl_file}")
            print(f"     Tamaño ONNX:  {onnx_size_mb:.2f} MB")
            print(f"     Tamaño ESPDL: {espdl_size_mb:.2f} MB")
            print(f"     Compresión:   {compression:.1f}x")
            print(f"     Tiempo:       {elapsed:.1f}s")

            # Evaluar viabilidad para ESP32-S3 (Flash ≈ 8 MB, firmware ~3-4 MB)
            if espdl_size_mb < 2.0:
                print(f"     🟢 Viable para ESP32-S3 (< 2 MB)")
            elif espdl_size_mb < 4.0:
                print(f"     🟡 Ajustado para ESP32-S3 (2-4 MB)")
            else:
                print(f"     🔴 Probablemente NO cabe en ESP32-S3 (> 4 MB)")
        else:
            result["error"] = "Archivo ESPDL no generado"
            print(f"  ❌ Error: archivo ESPDL no generado")

    except Exception as e:
        elapsed = time.time() - t0
        result["conversion_time_s"] = round(elapsed, 1)
        result["error"] = str(e)
        print(f"  ❌ Error en conversión: {e}")
        import traceback
        traceback.print_exc()

    return result


def save_export_summary(results: list, output_dir: Path) -> None:
    """Guarda un resumen JSON de todas las conversiones."""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "convert_onnx_to_espdl.py (Train_MLOps)",
        "quantization": "INT8 symmetric (esp-ppq)",
        "models": results,
    }
    summary_path = output_dir / "export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Resumen guardado: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convertir ONNX → ESPDL (INT8) para ESP32-S3 — Train_MLOps"
    )
    parser.add_argument(
        "--calib-dir", type=str, default="",
        help="Directorio con imágenes de calibración (jpg/png). "
             "Recomendado: ../datasets/IODC/coco/train/images"
    )
    parser.add_argument(
        "--target", type=str, default="esp32s3",
        choices=["esp32", "esp32s3", "esp32p4"],
        help="Target de Espressif (default: esp32s3)"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Modelos a convertir (default: todos). "
             "Opciones: fcos_v3s_t3, yolo26n_t2, espdet_pico_t4"
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="Número de imágenes de calibración a usar (default: 500)"
    )
    args = parser.parse_args()

    # Establecer directorio de trabajo en Train_MLOps/
    script_dir = Path(__file__).resolve().parent
    train_mlops_root = script_dir.parent  # scripts/ → Train_MLOps/
    os.chdir(train_mlops_root)

    print(f"{'='*60}")
    print(f"ONNX → ESPDL Conversion Pipeline (Train_MLOps)")
    print(f"{'='*60}")
    print(f"Directorio de trabajo: {os.getcwd()}")
    print(f"Target: {args.target}")
    print(f"Calibración: {args.calib_dir or '(datos aleatorios)'}")
    print(f"Muestras calibración: {args.n_samples}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models_to_convert = args.models or list(MODELS.keys())
    results = []

    for name in models_to_convert:
        if name not in MODELS:
            print(f"⚠️ Modelo '{name}' no reconocido. Disponibles: {list(MODELS.keys())}")
            continue
        result = convert_model(
            name, MODELS[name], args.calib_dir, args.n_samples, args.target
        )
        results.append(result)

    # Resumen en consola
    print(f"\n{'='*60}")
    print("RESUMEN DE CONVERSIÓN ONNX → ESPDL")
    print(f"{'='*60}")
    print(f"{'Modelo':<22} {'ONNX (MB)':>10} {'ESPDL (MB)':>11} {'Ratio':>7} {'Estado':>8}")
    print(f"{'-'*60}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        espdl_str = f"{r['espdl_size_mb']:.2f}" if r["success"] else "—"
        ratio_str = f"{r['compression_ratio']:.1f}x" if r["success"] else "—"
        print(f"  {r['name']:<20} {r['onnx_size_mb']:>10.2f} {espdl_str:>11} {ratio_str:>7} {status:>8}")

    # Guardar resumen JSON
    save_export_summary(results, OUTPUT_DIR)

    # Exit code basado en resultados
    n_ok = sum(1 for r in results if r["success"])
    n_total = len(results)
    print(f"\n{'✅' if n_ok == n_total else '⚠️'} {n_ok}/{n_total} modelos convertidos exitosamente")

    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
