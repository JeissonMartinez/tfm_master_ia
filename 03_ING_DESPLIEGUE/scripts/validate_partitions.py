#!/usr/bin/env python3
"""Diagnostic: validate partition layout for TFM TinyML Detector."""

partitions = [
    ('nvs',          'data', 'nvs',     0x9000,   0x6000),
    ('phy_init',     'data', 'phy',     0xF000,   0x1000),
    ('factory',      'app',  'factory', 0x10000,  0xA00000),
    ('model_espdet', 'data', '0x40',    0xA10000, 0x100000),
    ('model_yolo26', 'data', '0x40',    0xB10000, 0x300000),
]

flash_size = 16 * 1024 * 1024  # 16 MB
print('=== Partition Validation ===')
print(f'Flash total: {flash_size / (1024*1024):.0f} MB ({flash_size:#x})')
print()

for name, ptype, subtype, offset, size in partitions:
    end = offset + size
    align_ok = (offset % 0x1000 == 0) and (size % 0x1000 == 0)
    in_flash = end <= flash_size
    tag = 'OK' if (align_ok and in_flash) else 'FAIL'
    print(f'  [{tag:4s}] {name:14s}  offset={offset:#010x}  size={size:#010x} ({size/1024:7.0f} KB)  end={end:#010x}')

# Check overlaps
print()
print('=== Overlap Check ===')
ok = True
for i in range(len(partitions)):
    for j in range(i+1, len(partitions)):
        n1, _, _, o1, s1 = partitions[i]
        n2, _, _, o2, s2 = partitions[j]
        e1 = o1 + s1
        e2 = o2 + s2
        if o1 < e2 and o2 < e1:
            print(f'  OVERLAP: {n1} [{o1:#x}-{e1:#x}] vs {n2} [{o2:#x}-{e2:#x}]')
            ok = False
if ok:
    print('  No overlaps detected')

# Check model sizes vs partition
import os
proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
espdet_path = os.path.join(proj, 'models', 'espdl', 'espdet_pico_t4.espdl')
yolo26_path = os.path.join(proj, 'models', 'espdl', 'yolo26n_t2_esp.espdl')

print()
print('=== Model vs Partition Size ===')
for path, pname, psize in [(espdet_path, 'model_espdet', 0x100000),
                            (yolo26_path, 'model_yolo26', 0x300000)]:
    if os.path.isfile(path):
        fsize = os.path.getsize(path)
        fits = 'OK' if fsize <= psize else 'TOO BIG'
        print(f'  [{fits:7s}] {os.path.basename(path)}: {fsize:,} bytes / {psize:,} bytes ({fsize/psize*100:.1f}%)')
    else:
        print(f'  [MISSING] {path}')

# Total used
last_part = partitions[-1]
total_used = last_part[3] + last_part[4]
free = flash_size - total_used
print()
print(f'=== Flash Usage ===')
print(f'  Used up to: {total_used / (1024*1024):.2f} MB ({total_used:#x})')
print(f'  Free:       {free / (1024*1024):.2f} MB ({free:#x})')
