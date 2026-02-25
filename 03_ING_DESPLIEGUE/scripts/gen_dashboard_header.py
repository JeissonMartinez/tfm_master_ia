#!/usr/bin/env python3
"""Generate dashboard.h from gzipped dashboard.html"""
import gzip
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.join(script_dir, '..', 'main')
html_path = os.path.join(main_dir, 'frontend', 'dashboard.html')
out_path = os.path.join(main_dir, 'dashboard.h')

with open(html_path, 'rb') as f:
    data = f.read()

compressed = gzip.compress(data, compresslevel=9)
length = len(compressed)

lines = []
lines.append('// Auto-generated from frontend/dashboard.html — do not edit.')
lines.append('// Regenerate: python3 scripts/gen_dashboard_header.py')
lines.append('#pragma once')
lines.append('#include <cstdint>')
lines.append(f'static const size_t dashboard_html_gz_len = {length};')
lines.append('static const uint8_t dashboard_html_gz[] = {')

for i in range(0, length, 16):
    chunk = compressed[i:i+16]
    hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
    if i + 16 < length:
        lines.append(f'    {hex_str},')
    else:
        lines.append(f'    {hex_str}')

lines.append('};')
lines.append('')

with open(out_path, 'w') as f:
    f.write('\n'.join(lines))

print(f'Generated {out_path}: {length} bytes compressed from {len(data)} bytes HTML')
