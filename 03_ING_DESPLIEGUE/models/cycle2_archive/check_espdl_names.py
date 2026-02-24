"""Quick script to check output tensor names inside an ESPDL binary."""
import sys

fpath = "/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.espdl"
with open(fpath, "rb") as f:
    data = f.read()

print(f"File size: {len(data)} bytes\n")

# Search for expected output names
names = [b"box0", b"score0", b"box1", b"score1", b"box2", b"score2", b"images"]
print("=== Expected tensor names ===")
for name in names:
    positions = []
    start = 0
    while True:
        pos = data.find(name, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    status = f"FOUND at {[hex(p) for p in positions[:5]]}" if positions else "NOT FOUND"
    print(f"  {name.decode():10s}: {status}")

# Search for alternative output patterns
print("\n=== Alternative patterns ===")
alt_names = [b"output", b"/cv2", b"/cv3", b"model.", b"getitem"]
for name in alt_names:
    count = data.count(name)
    if count > 0:
        first = data.find(name)
        print(f"  {name.decode():10s}: {count} occurrences, first at {hex(first)}")
    else:
        print(f"  {name.decode():10s}: not found")

# Also check the cycle1 archive version
print("\n=== Checking cycle1 C_percentile version ===")
fpath2 = "/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/cycle1_archive/yolo11n_v1_best_C_percentile.espdl"
try:
    with open(fpath2, "rb") as f:
        data2 = f.read()
    print(f"File size: {len(data2)} bytes")
    for name in [b"box0", b"score0"]:
        found = data2.find(name) >= 0
        print(f"  {name.decode():10s}: {'FOUND' if found else 'NOT FOUND'}")
except FileNotFoundError:
    print("  File not found in archive")
