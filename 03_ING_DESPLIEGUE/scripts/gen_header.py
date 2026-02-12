#!/usr/bin/env python3
"""Generate C header from TFLite model file."""
import os
import sys

def tflite_to_header(src_path, dst_path, var_name, guard_name):
    data = open(src_path, 'rb').read()
    size = len(data)
    size_mb = size / (1024*1024)

    with open(dst_path, 'w') as f:
        f.write(f'// Auto-generated from {os.path.basename(src_path)}\n')
        f.write(f'// Model size: {size:,} bytes ({size_mb:.2f} MB)\n')
        f.write('// FULL INT8 quantization - compatible with TFLite Micro on ESP32\n')
        f.write('// DO NOT EDIT\n\n')
        f.write(f'#ifndef {guard_name}\n')
        f.write(f'#define {guard_name}\n\n')
        f.write('#include <cstdint>\n\n')
        f.write(f'alignas(16) const unsigned char {var_name}[] = {{\n')

        for i in range(0, size, 12):
            chunk = data[i:i+12]
            hex_vals = ', '.join(f'0x{b:02x}' for b in chunk)
            if i + 12 < size:
                f.write(f'    {hex_vals},\n')
            else:
                f.write(f'    {hex_vals}\n')

        f.write('};\n\n')
        f.write(f'const size_t {var_name}_len = sizeof({var_name});\n\n')
        f.write(f'#endif // {guard_name}\n')

    print(f'OK: {dst_path} ({size:,} bytes, {size_mb:.2f} MB)')


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    # YOLO26n fullint8
    tflite_to_header(
        os.path.join(root, '..', '02_ING_MODELOS/GoogleCloudAI/outputs/yolo26n_v1/train/weights/best_saved_model/best_full_integer_quant.tflite'),
        os.path.join(root, 'main/models/tflite/yolo26n_v1_fullint8.h'),
        'yolo26n_v1_fullint8_data',
        'YOLO26N_V1_FULLINT8_H'
    )

    # YOLO11n fullint8
    tflite_to_header(
        os.path.join(root, '..', '02_ING_MODELOS/GoogleCloudAI/outputs/yolo11n_v1/train/weights/best_saved_model/best_full_integer_quant.tflite'),
        os.path.join(root, 'main/models/tflite/yolo11n_v1_fullint8.h'),
        'yolo11n_v1_fullint8_data',
        'YOLO11N_V1_FULLINT8_H'
    )
