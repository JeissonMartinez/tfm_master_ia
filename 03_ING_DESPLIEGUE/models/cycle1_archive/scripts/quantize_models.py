"""
quantize_models.py
==================
Cuantiza modelos ONNX a formato ESPDL (INT8) usando esp-ppq,
con los datasets de calibración generados por create_calib_set.py.

Modelos soportados:
  - MBNTv3S_ssdlite_v1_p2_best.onnx  (NHWC, input: 1×224×224×3)
  - yolo11n_v1_best.onnx              (NCHW, input: 1×3×224×224)
  - yolo26n_v1_best.onnx              (NCHW, input: 1×3×224×224)

Uso:
  python models/quantize_models.py
"""

import os
import sys
import pickle
import time

import numpy as np
import torch

# esp-ppq imports
try:
    from esp_ppq import *
    from esp_ppq.api import espdl_quantize_onnx
except ImportError:
    print("[ERROR] No se encontró el paquete 'esp-ppq'.")
    print("  Instálalo con: pip install esp-ppq==1.2.4")
    sys.exit(1)

# ============================================================================
# Configuración
# ============================================================================

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_CHIP = "esp32s3"

# Configuración de cada modelo
MODELS_CONFIG = [
    {
        "name": "MBNTv3S_ssdlite_v1_p2_best",
        "onnx_file": "MBNTv3S_ssdlite_v1_p2_best.onnx",
        "espdl_file": "MBNTv3S_ssdlite_v1_p2_best.espdl",
        "input_shape": [1, 224, 224, 3],   # NHWC
        "calib_file": "calib_set_nhwc.pkl",
    },
    {
        "name": "yolo11n_v1_best",
        "onnx_file": "yolo11n_v1_best_fixed.onnx",       # Fixed: -1 en Reshape resueltos
        "espdl_file": "yolo11n_v1_best.espdl",
        "input_shape": [1, 3, 224, 224],   # NCHW
        "calib_file": "calib_set_nchw.pkl",
    },
    {
        "name": "yolo26n_v1_best",
        "onnx_file": "yolo26n_v1_best_fixed.onnx",       # Fixed: -1 resueltos + NMS truncado
        "espdl_file": "yolo26n_v1_best.espdl",
        "input_shape": [1, 3, 224, 224],   # NCHW
        "calib_file": "calib_set_nchw.pkl",
    },
]


# ============================================================================
# Funciones
# ============================================================================

def load_calibration_data(pkl_path: str) -> list:
    """Carga el dataset de calibración desde un archivo pickle y convierte a torch.Tensor."""
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"Archivo de calibración no encontrado: {pkl_path}\n"
            f"  Ejecuta primero: python models/create_calib_set.py"
        )
    with open(pkl_path, "rb") as f:
        np_data = pickle.load(f)

    # Convertir np.ndarray → torch.Tensor (requerido por esp-ppq)
    tensor_data = [torch.from_numpy(arr).float() for arr in np_data]
    print(f"  Calibración cargada: {len(tensor_data)} muestras, shape={tensor_data[0].shape}")
    return tensor_data


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    """Función de collate para el dataloader de calibración."""
    return batch.float()


def quantize_single_model(config: dict, calib_cache: dict) -> bool:
    """
    Cuantiza un modelo ONNX a ESPDL.

    Args:
        config: Diccionario con la configuración del modelo.
        calib_cache: Cache de datasets ya cargados {filename: data}.

    Returns:
        True si la cuantización fue exitosa, False en caso de error.
    """
    name = config["name"]
    onnx_path = os.path.join(MODELS_DIR, config["onnx_file"])
    espdl_path = os.path.join(MODELS_DIR, config["espdl_file"])
    calib_pkl = os.path.join(MODELS_DIR, config["calib_file"])
    input_shape = config["input_shape"]

    print(f"\n  Modelo ONNX:   {config['onnx_file']}")
    print(f"  Salida ESPDL:  {config['espdl_file']}")
    print(f"  Input shape:   {input_shape}")
    print(f"  Target:        {TARGET_CHIP}")

    # Verificar que existe el ONNX
    if not os.path.isfile(onnx_path):
        print(f"  [ERROR] Modelo ONNX no encontrado: {onnx_path}")
        return False

    # Cargar datos de calibración (con cache para no recargar)
    calib_key = config["calib_file"]
    if calib_key not in calib_cache:
        print(f"  Cargando dataset de calibración: {calib_key}")
        calib_cache[calib_key] = load_calibration_data(calib_pkl)
    else:
        print(f"  Reutilizando dataset de calibración: {calib_key} (en cache)")
    calib_data = calib_cache[calib_key]

    # Configurar cuantización
    quant_setting = QuantizationSettingFactory.espdl_setting()

    # Ejecutar cuantización
    print(f"  Iniciando cuantización INT8...")
    t0 = time.time()

    espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_path,
        calib_dataloader=calib_data,
        calib_steps=min(len(calib_data), 256),
        input_shape=input_shape,
        target=TARGET_CHIP,
        setting=quant_setting,
        collate_fn=collate_fn,
    )

    elapsed = time.time() - t0
    espdl_size = os.path.getsize(espdl_path) / 1024  # KB
    print(f"  Cuantización completada en {elapsed:.1f}s")
    print(f"  Archivo generado: {espdl_path} ({espdl_size:.1f} KB)")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  CUANTIZACIÓN DE MODELOS ONNX → ESPDL (INT8)")
    print(f"  Target: {TARGET_CHIP}")
    print("=" * 60)

    results = []
    calib_cache = {}  # Cache para no recargar el mismo pkl varias veces

    for i, config in enumerate(MODELS_CONFIG, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(MODELS_CONFIG)}] {config['name']}")
        print(f"{'─' * 60}")

        try:
            success = quantize_single_model(config, calib_cache)
            results.append((config["name"], "OK" if success else "FAIL"))
        except Exception as e:
            import traceback
            print(f"  [ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((config["name"], f"ERROR: {e}"))

    # --- Resumen final ---
    print(f"\n{'=' * 60}")
    print(f"  RESUMEN DE CUANTIZACIÓN")
    print(f"{'=' * 60}")
    for name, status in results:
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {name}: {status}")

    # Listar archivos .espdl generados
    espdl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".espdl")]
    if espdl_files:
        print(f"\n  Archivos .espdl en {MODELS_DIR}:")
        for f in sorted(espdl_files):
            size = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
            print(f"    → {f} ({size:.1f} KB)")

    print(f"{'=' * 60}")

    # Retornar código de salida
    all_ok = all(s == "OK" for _, s in results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
