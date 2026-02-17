"""
requantize_yolo_advanced.py — Re-cuantización INT8 mejorada para YOLO
=====================================================================
ESP-DL no soporta Conv en FP32, así que todo debe ser INT8.
Estrategias para mejorar la calidad de cuantización:

1. Equalization: balancea rangos de pesos entre capas adyacentes
2. Bias Correction: corrige sesgo en activaciones post-cuantización
3. Calibration con 'minmax' en vez de 'kl': preserva mejor los extremos
4. Más pasos de calibración

Uso:
    python models/requantize_yolo_advanced.py
"""

import os
import sys
import pickle
import time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from esp_ppq import *
    from esp_ppq.api import espdl_quantize_onnx
except ImportError:
    print("[ERROR] No se encontró 'esp-ppq'")
    sys.exit(1)


def load_calib_data():
    pkl_path = os.path.join(BASE_DIR, "calib_set_nchw.pkl")
    with open(pkl_path, "rb") as f:
        np_data = pickle.load(f)
    tensor_data = [torch.from_numpy(arr).float() for arr in np_data]
    print(f"  Calibración: {len(tensor_data)} muestras, shape={tensor_data[0].shape}")
    return tensor_data


def collate_fn(batch):
    return batch.float()


def validate_espdl_info(info_path):
    """Lee el .info y muestra exponents de outputs."""
    if not os.path.isfile(info_path):
        return
    print(f"\n  Output tensors from {os.path.basename(info_path)}:")
    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('%score') or line.startswith('%box'):
                print(f"    {line}")


def quantize_variant(onnx_path, espdl_path, calib_data, label,
                     equalization=False, bias_correct=False,
                     calib_algorithm='kl'):
    """Cuantiza con opciones configurables."""
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  equalization={equalization}, bias_correct={bias_correct}, calib={calib_algorithm}")
    
    setting = QuantizationSettingFactory.espdl_setting()
    setting.equalization = equalization
    setting.bias_correct = bias_correct
    
    if calib_algorithm != 'kl':
        setting.quantize_activation_setting.calib_algorithm = calib_algorithm
    
    t0 = time.time()
    try:
        espdl_quantize_onnx(
            onnx_import_file=onnx_path,
            espdl_export_file=espdl_path,
            calib_dataloader=calib_data,
            calib_steps=min(len(calib_data), 256),
            input_shape=[1, 3, 224, 224],
            target="esp32s3",
            setting=setting,
            collate_fn=collate_fn,
        )
        elapsed = time.time() - t0
        size = os.path.getsize(espdl_path)
        print(f"  OK en {elapsed:.1f}s, size={size:,} bytes")
        
        # Check .info
        info_path = espdl_path.replace('.espdl', '.info')
        validate_espdl_info(info_path)
        return size
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return 0


def validate_quantized_outputs(espdl_info_path, onnx_path, calib_data):
    """Compara outputs cuantizados vs float."""
    import onnxruntime as ort
    
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    
    # Test con imagen 3 (la que tenía mejores scores)
    img = calib_data[3]
    if isinstance(img, np.ndarray):
        img = img.astype(np.float32)
    else:
        img = np.array(img.numpy(), dtype=np.float32)
    if img.ndim == 3:
        img = img[np.newaxis, ...]
    
    outputs = sess.run(None, {input_name: img})
    output_names = [o.name for o in sess.get_outputs()]
    
    print(f"\n  Float reference (imagen 3):")
    for name, data in zip(output_names, outputs):
        arr = np.array(data)
        if "score" in name:
            sigmoid_scores = 1.0 / (1.0 + np.exp(-arr))
            max_sig = sigmoid_scores.max()
            above_03 = (sigmoid_scores > 0.3).sum()
            above_01 = (sigmoid_scores > 0.1).sum()
            print(f"    {name}: logit_max={arr.max():.2f} sigmoid_max={max_sig:.4f} "
                  f"above_0.3={above_03} above_0.1={above_01}")


def main():
    onnx_path = os.path.join(BASE_DIR, "yolo11n_v1_best_esp.onnx")
    
    if not os.path.isfile(onnx_path):
        print(f"[ERROR] No encontrado: {onnx_path}")
        sys.exit(1)
    
    calib_data = load_calib_data()
    
    # Referencia float
    validate_quantized_outputs(None, onnx_path, calib_data)
    
    results = {}
    
    # ====================================================================
    # Variante A: Equalization + Bias Correction (INT8, calib KL)
    # ====================================================================
    espdl_a = os.path.join(BASE_DIR, "yolo11n_v1_best_eqbc.espdl")
    results['EQ+BC (kl)'] = quantize_variant(
        onnx_path, espdl_a, calib_data,
        label="Variante A: Equalization + BiasCorrect + KL",
        equalization=True, bias_correct=True, calib_algorithm='kl'
    )
    
    # ====================================================================
    # Variante B: Equalization + Bias Correction + minmax calib
    # ====================================================================
    espdl_b = os.path.join(BASE_DIR, "yolo11n_v1_best_eqbc_minmax.espdl")
    results['EQ+BC (minmax)'] = quantize_variant(
        onnx_path, espdl_b, calib_data,
        label="Variante B: Equalization + BiasCorrect + minmax",
        equalization=True, bias_correct=True, calib_algorithm='minmax'
    )
    
    # ====================================================================
    # Variante C: Solo minmax calib (sin equalization)
    # ====================================================================
    espdl_c = os.path.join(BASE_DIR, "yolo11n_v1_best_minmax.espdl")
    results['minmax'] = quantize_variant(
        onnx_path, espdl_c, calib_data,
        label="Variante C: Solo minmax calibration",
        equalization=False, bias_correct=False, calib_algorithm='minmax'
    )
    
    # ====================================================================
    # Variante D: Solo percentile calib
    # ====================================================================
    espdl_d = os.path.join(BASE_DIR, "yolo11n_v1_best_percentile.espdl")
    results['percentile'] = quantize_variant(
        onnx_path, espdl_d, calib_data,
        label="Variante D: Percentile calibration",
        equalization=False, bias_correct=False, calib_algorithm='percentile'
    )
    
    # ====================================================================
    # Resumen
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"  RESUMEN")
    print(f"{'='*60}")
    
    orig_path = os.path.join(BASE_DIR, "yolo11n_v1_best_int8orig.espdl")
    if os.path.isfile(orig_path):
        print(f"  Original (kl, no opts):  {os.path.getsize(orig_path):>10,} bytes")
    
    for label, size in results.items():
        status = f"{size:>10,} bytes" if size else "FAILED"
        print(f"  {label:25s}: {status}")
    
    # Recomendar la mejor variante
    best_label = None
    best_size = 0
    for label, size in results.items():
        if size > 0:
            if best_label is None:
                best_label = label
                best_size = size
    
    if best_label:
        print(f"\n  Versiones generadas — probar cada una en el device.")
        print(f"  Copiar como yolo11n_v1_best.espdl y re-flashear.")


if __name__ == "__main__":
    main()
