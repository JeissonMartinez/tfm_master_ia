#!/usr/bin/env python3
"""Inspect TFLite models to determine quantization type (full INT8 vs hybrid)."""
import os
from ai_edge_litert.interpreter import Interpreter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

MODELS = {
    'yolo11n_v1_fullint8': '02_ING_MODELOS/GoogleCloudAI/outputs/yolo11n_v1/train/weights/best_saved_model/best_full_integer_quant.tflite',
    'yolo26n_v1_fullint8': '02_ING_MODELOS/GoogleCloudAI/outputs/yolo26n_v1/train/weights/best_saved_model/best_full_integer_quant.tflite',
    'yolo11n_v1_int8':     '02_ING_MODELOS/GoogleCloudAI/outputs/yolo11n_v1/tflite/best_int8.tflite',
    'yolo26n_v1_int8':     '02_ING_MODELOS/GoogleCloudAI/outputs/yolo26n_v1/tflite/best_int8.tflite',
    'mobilenetv2_int8':    '02_ING_MODELOS/GoogleCloudAI/outputs/MBNTv2_ssdlite_v1/tflite/MBNTv2_ssdlite_v1_int8.tflite',
}


def inspect_model(name, path):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    print()
    print('=' * 60)
    print(f'{name}: {size_mb:.2f} MB')

    all_int8 = True

    print(f'  Inputs ({len(inp)}):')
    for d in inp:
        dn = d['dtype'].__name__
        qp = d.get('quantization_parameters', {})
        sc = qp.get('scales', [])
        zp = qp.get('zero_points', [])
        print(f"    {d['name']}: shape={d['shape'].tolist()}  dtype={dn}")
        if len(sc) > 0:
            print(f'      scale={sc[:3]}  zp={zp[:3]}')
        if dn != 'int8':
            all_int8 = False

    print(f'  Outputs ({len(out)}):')
    for d in out:
        dn = d['dtype'].__name__
        qp = d.get('quantization_parameters', {})
        sc = qp.get('scales', [])
        zp = qp.get('zero_points', [])
        print(f"    {d['name']}: shape={d['shape'].tolist()}  dtype={dn}")
        if len(sc) > 0:
            print(f'      scale={sc[:3]}  zp={zp[:3]}')
        if dn != 'int8':
            all_int8 = False

    verdict = 'FULL INT8' if all_int8 else 'HYBRID (has non-int8 I/O)'
    print(f'  >>> {verdict}')
    return all_int8


if __name__ == '__main__':
    print(f'Project root: {ROOT}')
    results = {}
    for name, rel_path in MODELS.items():
        path = os.path.join(ROOT, rel_path)
        if os.path.exists(path):
            results[name] = inspect_model(name, path)
        else:
            print()
            print('=' * 60)
            print(f'{name}: FILE NOT FOUND - {rel_path}')
            results[name] = None

    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for name, is_full in results.items():
        if is_full is None:
            s = 'NOT FOUND'
        elif is_full:
            s = 'FULL INT8  <<<< compatible ESP32'
        else:
            s = 'HYBRID     ---- will fail on ESP32'
        print(f'  {name}: {s}')
