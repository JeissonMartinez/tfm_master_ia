"""
quantize_models_esp.py — Cuantiza los ONNX ESP-format (6 salidas) a ESPDL
============================================================================
Usa los modelos exportados por export_onnx_esp.py que NO incluyen el detection
head (DFL/sigmoid/concat), permitiendo cuantización por tensor independiente
para cada salida.
"""

import os
import sys
import pickle
import time
import numpy as np
import torch

try:
    from esp_ppq import *
    from esp_ppq.api import espdl_quantize_onnx
except ImportError:
    print("[ERROR] esp-ppq no encontrado. pip install esp-ppq==1.2.4")
    sys.exit(1)

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_CHIP = "esp32s3"

MODELS_CONFIG = [
    {
        "name": "yolo11n_v1_best",
        "onnx_file": "yolo11n_v1_best_esp.onnx",
        "espdl_file": "yolo11n_v1_best.espdl",
        "input_shape": [1, 3, 224, 224],   # NCHW
        "calib_file": "calib_set_nchw.pkl",
    },
    {
        "name": "yolo26n_v1_best",
        "onnx_file": "yolo26n_v1_best_esp.onnx",
        "espdl_file": "yolo26n_v1_best.espdl",
        "input_shape": [1, 3, 224, 224],   # NCHW
        "calib_file": "calib_set_nchw.pkl",
    },
]


def load_calibration_data(pkl_path):
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Archivo de calibración no encontrado: {pkl_path}")
    with open(pkl_path, "rb") as f:
        np_data = pickle.load(f)
    tensor_data = [torch.from_numpy(arr).float() for arr in np_data]
    print(f"  Calibración: {len(tensor_data)} muestras, shape={tensor_data[0].shape}")
    return tensor_data


def collate_fn(batch):
    return batch.float()


def quantize_single_model(config, calib_cache):
    name = config["name"]
    onnx_path = os.path.join(MODELS_DIR, config["onnx_file"])
    espdl_path = os.path.join(MODELS_DIR, config["espdl_file"])
    calib_pkl = os.path.join(MODELS_DIR, config["calib_file"])
    input_shape = config["input_shape"]

    print(f"\n  ONNX:   {config['onnx_file']}")
    print(f"  ESPDL:  {config['espdl_file']}")
    print(f"  Shape:  {input_shape}")

    if not os.path.isfile(onnx_path):
        print(f"  [ERROR] ONNX no encontrado: {onnx_path}")
        return False

    # Cargar calibración
    calib_key = config["calib_file"]
    if calib_key not in calib_cache:
        calib_cache[calib_key] = load_calibration_data(
            os.path.join(MODELS_DIR, calib_key))
    calib_data = calib_cache[calib_key]

    # Cuantizar
    quant_setting = QuantizationSettingFactory.espdl_setting()

    # --- Ciclo 2: Activar equalization (referencia: espressif/esp-detection) ---
    # Redistribuye rangos de pesos entre capas adyacentes para reducir
    # degradación de cuantización en tensores de scores de pocas clases.
    quant_setting.equalization = True
    quant_setting.equalization_setting.iterations = 3
    quant_setting.equalization_setting.value_threshold = 2.0

    print(f"  Cuantizando INT8 (equalization=True, iter=3)...")
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
    espdl_size = os.path.getsize(espdl_path) / 1024
    print(f"  OK en {elapsed:.1f}s — {espdl_size:.1f} KB")
    return True


def main():
    print("=" * 60)
    print("  CUANTIZACIÓN ONNX ESP-FORMAT → ESPDL")
    print(f"  Target: {TARGET_CHIP}")
    print("=" * 60)

    results = []
    calib_cache = {}

    for i, config in enumerate(MODELS_CONFIG, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(MODELS_CONFIG)}] {config['name']}")
        print(f"{'─' * 60}")
        try:
            ok = quantize_single_model(config, calib_cache)
            results.append((config["name"], "OK" if ok else "FAIL"))
        except Exception as e:
            import traceback
            print(f"  [ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((config["name"], f"ERROR: {e}"))

    print(f"\n{'=' * 60}")
    print(f"  RESUMEN")
    print(f"{'=' * 60}")
    for name, status in results:
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {name}: {status}")

    espdl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".espdl")]
    if espdl_files:
        print(f"\n  Archivos .espdl:")
        for f in sorted(espdl_files):
            size = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
            print(f"    → {f} ({size:.1f} KB)")

    all_ok = all(s == "OK" for _, s in results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
