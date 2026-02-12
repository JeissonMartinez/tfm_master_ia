#!/usr/bin/env python3
"""List all TFLite opcodes used in fullint8 models."""
import os

ROOT = '/Users/admin/Documents/TFM_UNIR'

MODELS = {
    'yolo11n_fullint8': '02_ING_MODELOS/GoogleCloudAI/outputs/yolo11n_v1/train/weights/best_saved_model/best_full_integer_quant.tflite',
    'yolo26n_fullint8': '02_ING_MODELOS/GoogleCloudAI/outputs/yolo26n_v1/train/weights/best_saved_model/best_full_integer_quant.tflite',
}

# TFLite builtin opcode names (partial, covers the ones we care about)
OP_NAMES = {
    0: 'ADD', 1: 'AVERAGE_POOL_2D', 2: 'CONCATENATION', 3: 'CONV_2D',
    4: 'DEPTHWISE_CONV_2D', 6: 'DEQUANTIZE', 9: 'FULLY_CONNECTED',
    14: 'LOGISTIC', 17: 'MAX_POOL_2D', 18: 'MUL', 22: 'PAD',
    25: 'RESHAPE', 27: 'SOFTMAX', 34: 'SUB', 39: 'STRIDED_SLICE',
    41: 'TRANSPOSE', 42: 'MEAN', 44: 'SQUEEZE', 45: 'SPLIT_V',
    48: 'TOPK_V2', 49: 'SPLIT', 51: 'TILE', 52: 'EXPAND_DIMS',
    56: 'GATHER', 64: 'BATCH_MATMUL', 69: 'SHAPE', 73: 'GATHER_ND',
    83: 'RESIZE_NEAREST_NEIGHBOR', 86: 'LESS', 92: 'FLOOR_MOD',
    94: 'REDUCE_MAX', 96: 'PACK', 97: 'CAST', 114: 'QUANTIZE',
}

for name, rel_path in MODELS.items():
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        print(f'{name}: NOT FOUND')
        continue

    with open(path, 'rb') as f:
        data = f.read()

    try:
        from ai_edge_litert import schema_py_generated as schema
        buf = bytearray(data)
        model = schema.ModelT.InitFromPackedBuf(buf, 0)
        opcodes = model.operatorCodes
        print(f'{name}: {len(opcodes)} opcodes')
        for i, oc in enumerate(opcodes):
            code = oc.builtinCode
            dep = oc.deprecatedBuiltinCode
            opname = OP_NAMES.get(code, f'UNKNOWN_{code}')
            print(f'  [{i:2d}] {opname} (code={code}, deprecated={dep})')
        print()
    except Exception as e:
        print(f'{name}: schema parse failed ({e}), trying flatbuffers raw...')
        # Fallback: just search for known op bytes
        if b'TOPK_V2' in data:
            print(f'  WARNING: contains TOPK_V2 string')
        print()
