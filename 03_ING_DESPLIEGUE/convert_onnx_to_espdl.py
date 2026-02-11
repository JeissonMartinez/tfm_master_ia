#!/usr/bin/env python3
"""
Script de conversión ONNX → ESPDL para ESP32-S3
Generado automáticamente por Conversion_ModelosTFLite.ipynb

Requisitos:
    pip install esp-ppq onnx numpy pillow

Uso:
    python convert_onnx_to_espdl.py [--calib-dir <path>] [--target esp32s3]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx

try:
    from esp_ppq import *
    from esp_ppq.api import espdl_quantize_onnx
except ImportError:
    print("ERROR: esp-ppq no instalado. Ejecutar: pip install esp-ppq")
    sys.exit(1)


# ─── Configuración de modelos ────────────────────────────────────────────
MODELS = {
    "mobilenetv2_ssdlite_v1": {
        "onnx": "02_ING_MODELOS/GoogleCloudAI/outputs/MBNTv2_ssdlite_v1/MBNTv2_ssdlite_v1.onnx",
        "input_shape": [1, 224, 224, 3],  # NHWC (TF convention)
        "channel_format": "nhwc",
    },
    "yolo11n_v1": {
        "onnx": "02_ING_MODELOS/GoogleCloudAI/outputs/yolo11n_v1/train/weights/best.onnx",
        "input_shape": [1, 3, 224, 224],  # NCHW (PyTorch/ONNX convention)
        "channel_format": "nchw",
    },
    "yolo26n_v1": {
        "onnx": "02_ING_MODELOS/GoogleCloudAI/outputs/yolo26n_v1/train/weights/best.onnx",
        "input_shape": [1, 3, 224, 224],  # NCHW
        "channel_format": "nchw",
    },
}

OUTPUT_DIR = Path("03_ING_DESPLIEGUE/main/models/espdl")


def create_calibration_dataset(calib_dir: str, input_shape: list, 
                                channel_format: str, n_samples: int = 64):
    """
    Genera un dataset de calibración a partir de imágenes.
    
    Si calib_dir está vacío o no existe, genera datos aleatorios (no ideal
    pero funcional para verificar la pipeline).
    """
    from PIL import Image
    
    h, w = 224, 224
    samples = []
    
    calib_path = Path(calib_dir) if calib_dir else None
    
    if calib_path and calib_path.exists():
        image_files = sorted(calib_path.glob("*.jpg")) + sorted(calib_path.glob("*.png"))
        image_files = image_files[:n_samples]
        print(f"  Usando {len(image_files)} imágenes de calibración de {calib_path}")
        
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB").resize((w, h))
            arr = np.array(img, dtype=np.float32) / 255.0
            
            if channel_format == "nchw":
                arr = arr.transpose(2, 0, 1)  # HWC → CHW
            
            samples.append(np.expand_dims(arr, 0))
    else:
        print(f"  ⚠️ Sin directorio de calibración — usando datos aleatorios ({n_samples} muestras)")
        for _ in range(n_samples):
            if channel_format == "nchw":
                samples.append(np.random.rand(1, 3, h, w).astype(np.float32))
            else:
                samples.append(np.random.rand(1, h, w, 3).astype(np.float32))
    
    return samples


def convert_model(name: str, config: dict, calib_dir: str, target: str = "esp32s3"):
    """Convierte un modelo ONNX a formato ESPDL."""
    onnx_path = Path(config["onnx"])
    
    if not onnx_path.exists():
        print(f"  ❌ ONNX no encontrado: {onnx_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Convirtiendo: {name}")
    print(f"  ONNX: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Input shape: {config['input_shape']}")
    print(f"  Target: {target}")
    
    # Generar dataset de calibración
    calib_data = create_calibration_dataset(
        calib_dir, config["input_shape"], config["channel_format"]
    )
    
    # Crear directorio de salida
    output_path = OUTPUT_DIR / name
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Cuantización y exportación con esp-ppq
        quant_setting = QuantizationSettingFactory.espdl_setting()
        
        # Cuantización INT8 simétrica (estándar para ESP32-S3)
        ppq_graph = espdl_quantize_onnx(
            onnx_import_file=str(onnx_path),
            espdl_export_file=str(output_path / f"{name}.espdl"),
            calib_dataloader=calib_data,
            calib_steps=min(len(calib_data), 32),
            input_shape=config["input_shape"],
            target=target,
            setting=quant_setting,
            do_quantize=True,
        )
        
        espdl_file = output_path / f"{name}.espdl"
        if espdl_file.exists():
            print(f"  ✅ ESPDL generado: {espdl_file}")
            print(f"     Tamaño: {espdl_file.stat().st_size / (1024*1024):.2f} MB")
            return True
        else:
            print(f"  ❌ Error: archivo ESPDL no generado")
            return False
            
    except Exception as e:
        print(f"  ❌ Error en conversión: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Convertir ONNX → ESPDL para ESP32-S3")
    parser.add_argument("--calib-dir", type=str, default="",
                        help="Directorio con imágenes de calibración (jpg/png)")
    parser.add_argument("--target", type=str, default="esp32s3",
                        choices=["esp32", "esp32s3", "esp32p4"],
                        help="Target de Espressif")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Modelos a convertir (default: todos)")
    args = parser.parse_args()
    
    # Cambiar al directorio raíz del proyecto
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent if script_dir.name == "03_ING_DESPLIEGUE" else script_dir
    os.chdir(root)
    
    print(f"Directorio de trabajo: {os.getcwd()}")
    print(f"Target: {args.target}")
    print(f"Calibración: {args.calib_dir or '(datos aleatorios)'}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    models_to_convert = args.models or list(MODELS.keys())
    results = {}
    
    for name in models_to_convert:
        if name not in MODELS:
            print(f"⚠️ Modelo '{name}' no reconocido. Disponibles: {list(MODELS.keys())}")
            continue
        results[name] = convert_model(name, MODELS[name], args.calib_dir, args.target)
    
    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE CONVERSIÓN")
    print(f"{'='*60}")
    for name, success in results.items():
        print(f"  {'✅' if success else '❌'} {name}")


if __name__ == "__main__":
    main()
