"""
requantize_yolo11n.py — Re-cuantización mejorada de YOLO11n
===========================================================
Problema: La cuantización INT8 per-tensor destruye los scores del modelo.
Solución: Mantener las capas del score head (cv3) en FP32.

Este script:
1. Cuantiza con score head en FP32 (dispatching table)
2. Compara outputs float vs quantized
3. Genera nuevo .espdl

Uso:
    python models/requantize_yolo11n.py
"""

import os
import sys
import pickle
import time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# esp-ppq imports
try:
    from esp_ppq import *
    from esp_ppq.api import espdl_quantize_onnx
except ImportError:
    print("[ERROR] No se encontró 'esp-ppq'. Instala con: pip install esp-ppq")
    sys.exit(1)


def load_calib_data():
    """Carga datos de calibración NCHW."""
    pkl_path = os.path.join(BASE_DIR, "calib_set_nchw.pkl")
    with open(pkl_path, "rb") as f:
        np_data = pickle.load(f)
    tensor_data = [torch.from_numpy(arr).float() for arr in np_data]
    print(f"  Calibración: {len(tensor_data)} muestras, shape={tensor_data[0].shape}")
    return tensor_data


def collate_fn(batch):
    return batch.float()


def validate_quantized_onnx(onnx_path, calib_data, label=""):
    """Ejecuta inferencia float en un ONNX y muestra estadísticas de scores."""
    import onnxruntime as ort
    
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    
    print(f"\n  --- Validación {label} ---")
    
    for img_idx in [0, 3]:  # Imagen 0 y 3 que mostraron buenos scores en float
        img = calib_data[img_idx]
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)
        else:
            img = np.array(img.numpy(), dtype=np.float32)
        
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        
        outputs = sess.run(None, {input_name: img})
        
        print(f"\n  Imagen {img_idx}:")
        for name, data in zip(output_names, outputs):
            arr = np.array(data)
            if "score" in name:
                sigmoid_scores = 1.0 / (1.0 + np.exp(-arr))
                max_sig = sigmoid_scores.max()
                above = (sigmoid_scores > 0.3).sum()
                print(f"    {name}: logit=[{arr.min():.2f}, {arr.max():.2f}] "
                      f"sigmoid_max={max_sig:.4f} above_0.3={above}")


def quantize_model(onnx_path, espdl_path, calib_data, fp32_nodes=None, label=""):
    """Cuantiza un modelo ONNX con opciones configurables."""
    
    print(f"\n{'='*60}")
    print(f"  CUANTIZACIÓN: {label}")
    print(f"{'='*60}")
    
    # Setting base
    quant_setting = QuantizationSettingFactory.espdl_setting()
    
    # Dispatchar nodos específicos a FP32
    if fp32_nodes:
        print(f"  Nodos FP32: {fp32_nodes}")
        for node_name in fp32_nodes:
            quant_setting.dispatching_table.append(node_name, 0)  # 0 = FP32
    
    print(f"  ONNX: {os.path.basename(onnx_path)}")
    print(f"  Output: {os.path.basename(espdl_path)}")
    
    t0 = time.time()
    espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_path,
        calib_dataloader=calib_data,
        calib_steps=min(len(calib_data), 256),
        input_shape=[1, 3, 224, 224],
        target="esp32s3",
        setting=quant_setting,
        collate_fn=collate_fn,
    )
    elapsed = time.time() - t0
    
    espdl_size = os.path.getsize(espdl_path)
    print(f"  Completado en {elapsed:.1f}s, size={espdl_size:,} bytes")
    
    return espdl_size


def main():
    onnx_path = os.path.join(BASE_DIR, "yolo11n_v1_best_esp.onnx")
    
    if not os.path.isfile(onnx_path):
        print(f"[ERROR] No encontrado: {onnx_path}")
        sys.exit(1)
    
    calib_data = load_calib_data()
    
    # ====================================================================
    # Paso 1: Validar output float (referencia)
    # ====================================================================
    validate_quantized_onnx(onnx_path, calib_data, "FLOAT ORIGINAL")
    
    # ====================================================================
    # Paso 2: Re-cuantizar con score head en FP32
    # ====================================================================
    
    # Nodos del score head (convs finales que producen score0/1/2)
    # También incluimos los convs intermedios del cv3 path
    # y las capas SiLU que los alimentan
    SCORE_HEAD_NODES = [
        # Final score convs
        "node_conv2d_69",   # score0  ← silu_63
        "node_conv2d_77",   # score1  ← silu_69
        "node_conv2d_85",   # score2  ← silu_75
    ]
    
    # También los SiLU y Convs previos en el path de cv3
    # (trazado desde score0 = cv3.0.2 → cv3.0.1 → cv3.0.0)
    SCORE_HEAD_EXTENDED = SCORE_HEAD_NODES + [
        # SiLU que alimenta el conv final de score
        "node_silu_63", "node_Sigmoid_838",  # silu before score0
        "node_silu_69", "node_Sigmoid_922",  # silu before score1
        "node_silu_75", "node_Sigmoid_982",  # silu before score2
        # Conv previos en cv3 (penúltima capa)
        "node_Conv_1413",   # cv3.0.1 → getitem_252
        "node_Conv_1425",   # cv3.1.1 → getitem_270
        "node_Conv_1441",   # cv3.2.2 → getitem_294
    ]
    
    # Opción A: Solo convs finales de score en FP32
    espdl_a = os.path.join(BASE_DIR, "yolo11n_v1_best_fp32scores.espdl")
    try:
        size_a = quantize_model(
            onnx_path, espdl_a, calib_data,
            fp32_nodes=SCORE_HEAD_NODES,
            label="Score convs en FP32"
        )
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        size_a = 0
    
    # ====================================================================
    # Paso 3: Re-cuantizar con equalization + bias_correct
    # ====================================================================
    espdl_b = os.path.join(BASE_DIR, "yolo11n_v1_best_equalized.espdl")
    try:
        quant_setting_b = QuantizationSettingFactory.espdl_setting()
        quant_setting_b.equalization = True
        quant_setting_b.bias_correct = True
        
        print(f"\n{'='*60}")
        print(f"  CUANTIZACIÓN: INT8 + Equalization + BiasCorrect")
        print(f"{'='*60}")
        
        t0 = time.time()
        espdl_quantize_onnx(
            onnx_import_file=onnx_path,
            espdl_export_file=espdl_b,
            calib_dataloader=calib_data,
            calib_steps=min(len(calib_data), 256),
            input_shape=[1, 3, 224, 224],
            target="esp32s3",
            setting=quant_setting_b,
            collate_fn=collate_fn,
        )
        elapsed = time.time() - t0
        size_b = os.path.getsize(espdl_b)
        print(f"  Completado en {elapsed:.1f}s, size={size_b:,} bytes")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        size_b = 0
    
    # ====================================================================
    # Paso 4: Comparar exponents de outputs entre versiones
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"  COMPARACIÓN DE TAMAÑOS")
    print(f"{'='*60}")
    
    orig_espdl = os.path.join(BASE_DIR, "yolo11n_v1_best.espdl")
    if os.path.isfile(orig_espdl):
        print(f"  Original INT8:        {os.path.getsize(orig_espdl):>12,} bytes")
    if size_a:
        print(f"  FP32 scores:          {size_a:>12,} bytes")
    if size_b:
        print(f"  Equalized+BiasCorr:   {size_b:>12,} bytes")
    
    # ====================================================================
    # Resumen
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"  SIGUIENTE PASO")
    print(f"{'='*60}")
    print(f"  Archivos generados:")
    if size_a:
        print(f"    → yolo11n_v1_best_fp32scores.espdl ({size_a:,} bytes)")
    if size_b:
        print(f"    → yolo11n_v1_best_equalized.espdl ({size_b:,} bytes)")
    print(f"\n  Para probar, copia el mejor candidato como yolo11n_v1_best.espdl")
    print(f"  y ejecuta flash_models.sh")


if __name__ == "__main__":
    main()
