"""
validate_onnx.py — Validación del modelo ONNX re-exportado (6 salidas)
======================================================================
Verifica que el modelo ONNX float produce scores razonables.
Si los scores son todos muy negativos, el re-export rompió el modelo.
Si los scores son razonables, el problema es la cuantización.

También compara con el modelo original .pt de Ultralytics.
"""

import os
import sys
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 1. Validar ONNX de 6 salidas con onnxruntime
# ============================================================================

def validate_onnx_6output():
    """Ejecuta inferencia float en el ONNX de 6 salidas."""
    import onnxruntime as ort
    
    onnx_path = os.path.join(BASE_DIR, "yolo11n_v1_best_esp.onnx")
    if not os.path.isfile(onnx_path):
        print(f"[ERROR] No encontrado: {onnx_path}")
        return
    
    print("=" * 60)
    print("  VALIDACIÓN ONNX 6-SALIDAS (float32)")
    print("=" * 60)
    
    # Cargar modelo
    sess = ort.InferenceSession(onnx_path)
    
    # Mostrar inputs/outputs
    print("\n  Inputs:")
    for inp in sess.get_inputs():
        print(f"    {inp.name}: {inp.shape} {inp.type}")
    print("\n  Outputs:")
    for out in sess.get_outputs():
        print(f"    {out.name}: {out.shape} {out.type}")
    
    # Cargar una imagen de calibración
    calib_path = os.path.join(BASE_DIR, "calib_set_nchw.pkl")
    if os.path.isfile(calib_path):
        with open(calib_path, "rb") as f:
            calib_data = pickle.load(f)
        print(f"\n  Calibración cargada: {len(calib_data)} imágenes")
        
        # Usar primeras 5 imágenes
        num_test = min(5, len(calib_data))
    else:
        print("\n  [WARN] No hay datos de calibración, usando imagen aleatoria")
        calib_data = [np.random.rand(1, 3, 224, 224).astype(np.float32)]
        num_test = 1
    
    for img_idx in range(num_test):
        img = calib_data[img_idx]
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)
        else:
            img = np.array(img, dtype=np.float32)
        
        if img.ndim == 3:
            img = img[np.newaxis, ...]  # Agregar batch dim
        
        print(f"\n  --- Imagen {img_idx} ---")
        print(f"  Input shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Inferencia
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: img})
        
        # Analizar cada salida
        output_names = [o.name for o in sess.get_outputs()]
        for name, data in zip(output_names, outputs):
            arr = np.array(data)
            print(f"    {name:8s} shape={str(arr.shape):20s} "
                  f"min={arr.min():8.3f}  max={arr.max():8.3f}  "
                  f"mean={arr.mean():8.3f}  std={arr.std():6.3f}")
            
            # Para score outputs, aplicar sigmoid y mostrar max
            if "score" in name:
                sigmoid_scores = 1.0 / (1.0 + np.exp(-arr))
                max_sigmoid = sigmoid_scores.max()
                # Encontrar la posición del max score
                max_pos = np.unravel_index(sigmoid_scores.argmax(), sigmoid_scores.shape)
                print(f"             sigmoid: max={max_sigmoid:.4f} at {max_pos}")
                
                # Contar cuántos superan 0.3
                above_thr = (sigmoid_scores > 0.3).sum()
                print(f"             scores > 0.3: {above_thr}")
                
                # Top-5 scores
                flat = sigmoid_scores.flatten()
                top5_idx = np.argsort(flat)[-5:][::-1]
                top5_vals = flat[top5_idx]
                print(f"             top-5: {[f'{v:.4f}' for v in top5_vals]}")


# ============================================================================
# 2. Validar modelo original .pt con Ultralytics
# ============================================================================

def validate_original_pt():
    """Ejecuta inferencia con el modelo .pt original usando Ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n  [SKIP] Ultralytics no disponible")
        return
    
    pt_path = os.path.join(BASE_DIR, "yolo11n_v1_best.pt")
    if not os.path.isfile(pt_path):
        print(f"\n  [SKIP] No encontrado: {pt_path}")
        return
    
    print("\n" + "=" * 60)
    print("  VALIDACIÓN MODELO ORIGINAL .pt (Ultralytics)")
    print("=" * 60)
    
    model = YOLO(pt_path)
    
    # Cargar imagen de calibración
    calib_path = os.path.join(BASE_DIR, "calib_set_nchw.pkl")
    if os.path.isfile(calib_path):
        with open(calib_path, "rb") as f:
            calib_data = pickle.load(f)
        # Convertir NCHW [0,1] → HWC [0,255] uint8
        img = calib_data[0]
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)
        else:
            img = np.array(img, dtype=np.float32)
        if img.ndim == 4:
            img = img[0]
        # NCHW → HWC
        img_hwc = np.transpose(img, (1, 2, 0))  # [224, 224, 3]
        img_uint8 = (img_hwc * 255).clip(0, 255).astype(np.uint8)
    else:
        img_uint8 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    print(f"  Input: {img_uint8.shape}, range: [{img_uint8.min()}, {img_uint8.max()}]")
    
    # Inferencia
    results = model(img_uint8, imgsz=224, conf=0.01, verbose=False)
    
    for r in results:
        boxes = r.boxes
        print(f"  Detecciones (conf>0.01): {len(boxes)}")
        if len(boxes) > 0:
            for i, box in enumerate(boxes[:10]):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                cls_name = r.names.get(cls, f"cls{cls}")
                print(f"    [{i}] {cls_name} conf={conf:.3f} box={[f'{v:.1f}' for v in xyxy]}")
        else:
            print("    (ninguna)")


# ============================================================================
# 3. Validar ONNX original (1 salida, con detect head)
# ============================================================================

def validate_onnx_original():
    """Ejecuta inferencia con el ONNX original (single output)."""
    import onnxruntime as ort
    
    onnx_path = os.path.join(BASE_DIR, "yolo11n_v1_best.onnx")
    if not os.path.isfile(onnx_path):
        print(f"\n  [SKIP] No encontrado: {onnx_path}")
        return
    
    print("\n" + "=" * 60)
    print("  VALIDACIÓN ONNX ORIGINAL (1 salida, con detect head)")
    print("=" * 60)
    
    sess = ort.InferenceSession(onnx_path)
    
    print("\n  Outputs:")
    for out in sess.get_outputs():
        print(f"    {out.name}: {out.shape} {out.type}")
    
    # Cargar imagen
    calib_path = os.path.join(BASE_DIR, "calib_set_nchw.pkl")
    with open(calib_path, "rb") as f:
        calib_data = pickle.load(f)
    img = np.array(calib_data[0], dtype=np.float32)
    if img.ndim == 3:
        img = img[np.newaxis, ...]
    
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: img})
    
    for name, data in zip([o.name for o in sess.get_outputs()], outputs):
        arr = np.array(data)
        print(f"\n  {name}: shape={arr.shape}")
        # Esperable: [1, 9, 1029] donde 9 = 4 bbox + 5 classes
        if arr.ndim == 3 and arr.shape[1] > 4:
            # Class scores = arr[0, 4:, :] (ya post-sigmoid)
            scores = arr[0, 4:, :]
            print(f"  Class scores shape: {scores.shape}")
            print(f"  Class scores range: [{scores.min():.4f}, {scores.max():.4f}]")
            max_conf = scores.max()
            print(f"  Max confidence: {max_conf:.4f}")
            above = (scores > 0.3).sum()
            print(f"  Scores > 0.3: {above}")
            
            # Top detecciones
            max_per_box = scores.max(axis=0)
            top10_idx = np.argsort(max_per_box)[-10:][::-1]
            print(f"  Top-10 max confidence per box:")
            for idx in top10_idx:
                cls = scores[:, idx].argmax()
                conf = max_per_box[idx]
                print(f"    box {idx}: cls={cls} conf={conf:.4f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VALIDACIÓN DE MODELOS YOLO11n")
    print("=" * 60)
    
    # Test 1: ONNX re-exportado con 6 salidas
    validate_onnx_6output()
    
    # Test 2: Modelo original .pt
    validate_original_pt()
    
    # Test 3: ONNX original (si existe)
    validate_onnx_original()
    
    print("\n" + "=" * 60)
    print("  FIN VALIDACIÓN")
    print("=" * 60)
