"""
quantize_and_validate.py — Cuantiza YOLO11n con varias estrategias y
VALIDA los outputs cuantizados simulados (sin necesidad de flashear).

Usa espdl_quantize_onnx (que funciona en CPU) para cuantizar+exportar,
luego simula inferencia INT8 con TorchExecutor para comparar scores.
"""

import os, sys, pickle, time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from esp_ppq import QuantizationSettingFactory, TargetPlatform
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.executor import TorchExecutor


def load_calib_data():
    pkl_path = os.path.join(BASE_DIR, "calib_set_nchw.pkl")
    with open(pkl_path, "rb") as f:
        np_data = pickle.load(f)
    tensor_data = [torch.from_numpy(arr).float() for arr in np_data]
    print(f"  Calibración: {len(tensor_data)} muestras, shape={tensor_data[0].shape}")
    return tensor_data


def collate_fn(batch):
    return batch.float()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def evaluate_scores(outputs, output_names, label=""):
    """Evalúa calidad de scores."""
    results = {}
    for name, data in zip(output_names, outputs):
        if "score" not in name:
            continue
        arr = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.array(data)
        sig = sigmoid(arr)
        results[name] = {
            'logit_max': float(arr.max()),
            'logit_min': float(arr.min()),
            'sigmoid_max': float(sig.max()),
            'above_0.3': int((sig > 0.3).sum()),
            'above_0.1': int((sig > 0.1).sum()),
            'above_0.05': int((sig > 0.05).sum()),
        }
        print(f"    {name}: logit[{arr.min():.2f}, {arr.max():.2f}] "
              f"sig_max={sig.max():.4f} >0.3={results[name]['above_0.3']} "
              f">0.1={results[name]['above_0.1']} >0.05={results[name]['above_0.05']}")
    return results


def quantize_variant(onnx_path, espdl_path, calib_data, test_images, label,
                     equalization=False, bias_correct=False,
                     calib_algorithm='kl'):
    """Cuantiza, exporta ESPDL, y evalúa outputs simulados."""
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  eq={equalization}, bc={bias_correct}, calib={calib_algorithm}")
    
    setting = QuantizationSettingFactory.espdl_setting()
    setting.equalization = equalization
    setting.bias_correct = bias_correct
    if calib_algorithm != 'kl':
        setting.quantize_activation_setting.calib_algorithm = calib_algorithm
    
    t0 = time.time()
    try:
        ppq_graph = espdl_quantize_onnx(
            onnx_import_file=onnx_path,
            espdl_export_file=espdl_path,
            calib_dataloader=calib_data,
            calib_steps=min(len(calib_data), 256),
            input_shape=[1, 3, 224, 224],
            target="esp32s3",
            setting=setting,
            collate_fn=collate_fn,
            error_report=False,
        )
        elapsed = time.time() - t0
        size = os.path.getsize(espdl_path)
        print(f"  Cuantización+Export: {elapsed:.1f}s, size={size:,} bytes")
        
        # Mostrar exponents del .info
        info_path = espdl_path.replace('.espdl', '.info')
        if os.path.isfile(info_path):
            with open(info_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if 'exponents' in line:
                        print(f"    {line}")
        
        # Simular inferencia cuantizada
        print(f"\n  --- Simulación INT8 ---")
        executor = TorchExecutor(ppq_graph, device='cpu')
        output_names = [name for name in ppq_graph.outputs]
        
        all_results = {}
        for idx in test_images:
            img = test_images[idx]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            if img.ndim == 3:
                img = img.unsqueeze(0)
            
            print(f"\n  Imagen {idx} ({label}):")
            outputs = executor.forward(img)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            
            all_results[idx] = evaluate_scores(outputs, output_names)
        
        return size, all_results, elapsed
        
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return 0, {}, elapsed


def main():
    onnx_path = os.path.join(BASE_DIR, "yolo11n_v1_best_esp.onnx")
    
    if not os.path.isfile(onnx_path):
        print(f"[ERROR] No encontrado: {onnx_path}")
        sys.exit(1)
    
    calib_data = load_calib_data()
    test_images = {3: calib_data[3], 4: calib_data[4], 10: calib_data[10]}
    
    # ---------------------------------------------------------------
    # 1. Referencia float
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("  REFERENCIA FLOAT (ONNX)")
    print("="*60)
    
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_names_float = [o.name for o in sess.get_outputs()]
    
    float_results = {}
    for idx in test_images:
        img = test_images[idx]
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = img
        if img_np.ndim == 3:
            img_np = img_np[np.newaxis, ...]
        print(f"\n  Float imagen {idx}:")
        outputs = sess.run(None, {input_name: img_np.astype(np.float32)})
        float_results[idx] = evaluate_scores(outputs, output_names_float)
    
    # ---------------------------------------------------------------
    # 2. Variantes de cuantización
    # ---------------------------------------------------------------
    variants = {}
    
    # A: Default KL (como el original)
    espdl_a = os.path.join(BASE_DIR, "yolo11n_v1_best_A_kl.espdl")
    size_a, results_a, t_a = quantize_variant(
        onnx_path, espdl_a, calib_data, test_images,
        label="A: Default KL")
    variants['A_kl'] = (size_a, results_a, t_a)
    
    # B: minmax
    espdl_b = os.path.join(BASE_DIR, "yolo11n_v1_best_B_minmax.espdl")
    size_b, results_b, t_b = quantize_variant(
        onnx_path, espdl_b, calib_data, test_images,
        label="B: minmax calibration",
        calib_algorithm='minmax')
    variants['B_minmax'] = (size_b, results_b, t_b)
    
    # C: percentile
    espdl_c = os.path.join(BASE_DIR, "yolo11n_v1_best_C_percentile.espdl")
    size_c, results_c, t_c = quantize_variant(
        onnx_path, espdl_c, calib_data, test_images,
        label="C: percentile calibration",
        calib_algorithm='percentile')
    variants['C_percentile'] = (size_c, results_c, t_c)
    
    # ---------------------------------------------------------------
    # 3. Comparativa
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("  COMPARATIVA")
    print("="*60)
    
    header = f"  {'Variante':<22s}"
    for idx in [3, 4, 10]:
        header += f" {'img'+str(idx)+'_max':>10s}"
    header += f" {'total>0.1':>10s} {'time':>8s}"
    print(header)
    print(f"  {'-'*22}" + f" {'-'*10}" * 3 + f" {'-'*10} {'-'*8}")
    
    # Float
    line = f"  {'Float ref':<22s}"
    total_above = 0
    for idx in [3, 4, 10]:
        fr = float_results.get(idx, {})
        max_sig = max(fr.get(f'score{i}', {}).get('sigmoid_max', 0) for i in range(3))
        total_above += sum(fr.get(f'score{i}', {}).get('above_0.1', 0) for i in range(3))
        line += f" {max_sig:10.4f}"
    line += f" {total_above:10d} {'---':>8s}"
    print(line)
    
    best_name = None
    best_total_sig = -1
    
    for vname, (size, results, t) in variants.items():
        line = f"  {vname:<22s}"
        if not results:
            line += f" {'FAILED':>10s}"
            print(line)
            continue
        total_above = 0
        total_sig = 0
        for idx in [3, 4, 10]:
            r = results.get(idx, {})
            max_sig = max(r.get(f'score{i}', {}).get('sigmoid_max', 0) for i in range(3))
            total_above += sum(r.get(f'score{i}', {}).get('above_0.1', 0) for i in range(3))
            total_sig += max_sig
            line += f" {max_sig:10.4f}"
        line += f" {total_above:10d} {t:7.1f}s"
        print(line)
        if total_sig > best_total_sig:
            best_total_sig = total_sig
            best_name = vname
    
    # ---------------------------------------------------------------
    # 4. Variante D: EQ+BC+KL (solo si las rápidas son malas)
    # ---------------------------------------------------------------
    run_slow = True
    if best_name and variants[best_name][1]:
        best_results = variants[best_name][1]
        best_sig = max(
            max(best_results.get(idx, {}).get(f'score{i}', {}).get('sigmoid_max', 0)
                for i in range(3))
            for idx in [3, 4, 10]
        )
        if best_sig >= 0.1:
            run_slow = False
            print(f"\n  Mejor variante rápida ({best_name}) tiene sig_max={best_sig:.4f}")
            print(f"  No es necesario ejecutar variantes lentas.")
    
    if run_slow:
        print(f"\n  Ejecutando variante lenta (EQ+BC+KL)...")
        espdl_d = os.path.join(BASE_DIR, "yolo11n_v1_best_D_eqbc_kl.espdl")
        size_d, results_d, t_d = quantize_variant(
            onnx_path, espdl_d, calib_data, test_images,
            label="D: EQ + BiasCorrect + KL",
            equalization=True, bias_correct=True, calib_algorithm='kl')
        variants['D_eqbc_kl'] = (size_d, results_d, t_d)
    
    # ---------------------------------------------------------------
    # 5. Resumen
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("  RESUMEN FINAL")
    print("="*60)
    
    # Recalculate best
    best_name = None
    best_total_sig = -1
    for vname, (size, results, t) in variants.items():
        if not results:
            continue
        total_sig = 0
        for idx in [3, 4, 10]:
            r = results.get(idx, {})
            total_sig += max(r.get(f'score{i}', {}).get('sigmoid_max', 0) for i in range(3))
        if total_sig > best_total_sig:
            best_total_sig = total_sig
            best_name = vname
    
    for vname, (size, results, t) in variants.items():
        if not results:
            print(f"  {vname}: FAILED")
            continue
        total_sig = 0
        for idx in [3, 4, 10]:
            r = results.get(idx, {})
            total_sig += max(r.get(f'score{i}', {}).get('sigmoid_max', 0) for i in range(3))
        marker = " <<<BEST" if vname == best_name else ""
        print(f"  {vname}: size={size:,}B total_sig={total_sig:.4f} time={t:.0f}s{marker}")
    
    if best_name:
        print(f"\n  >>> MEJOR: {best_name} (total_sigmoid_max = {best_total_sig:.4f})")
        best_espdl = os.path.join(BASE_DIR, f"yolo11n_v1_best_{best_name}.espdl")
        print(f"  ESPDL: {best_espdl}")
        print(f"  Tamaño: {os.path.getsize(best_espdl):,} bytes")
        print(f"\n  Para usar:")
        print(f"    cp models/yolo11n_v1_best_{best_name}.espdl models/yolo11n_v1_best.espdl")
    else:
        print("\n  >>> NINGUNA VARIANTE FUNCIONO")


if __name__ == "__main__":
    main()
