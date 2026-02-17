"""Inspect quantization config for score vs box conv layers."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
cfg_path = os.path.join(BASE, "yolo11n_v1_best_C_percentile.json")

with open(cfg_path) as f:
    cfg = json.load(f)

configs = cfg.get("configs", {})

# List all node names containing cv2 or cv3
print("=== Score conv nodes (cv3) ===")
for key in sorted(configs.keys()):
    if "cv3" in key:
        params = configs[key]
        print(f"\n{key}:")
        for tn, tc in params.items():
            if isinstance(tc, dict) and "bit_width" in tc:
                print(f"  {tn}: bw={tc['bit_width']} state={tc.get('state','?')}")

print("\n=== Box conv nodes (cv2) ===")
for key in sorted(configs.keys()):
    if "cv2" in key and "cv2." in key:
        params = configs[key]
        print(f"\n{key}:")
        for tn, tc in params.items():
            if isinstance(tc, dict) and "bit_width" in tc:
                print(f"  {tn}: bw={tc['bit_width']} state={tc.get('state','?')}")

# Check final score convs specifically (the ones that output score0/1/2)
print("\n=== Looking for final score/box conv nodes ===")
for key in sorted(configs.keys()):
    if "23" in key and ("cv3" in key or "cv2" in key):
        found_weight = False
        params = configs[key]
        for tn in params:
            if "2.weight" in tn:  # final conv in the 3-conv sequence
                found_weight = True
        if found_weight:
            print(f"\n{key} (FINAL CONV):")
            for tn, tc in params.items():
                if isinstance(tc, dict) and "bit_width" in tc:
                    print(f"  {tn}: bw={tc['bit_width']} state={tc.get('state','?')}")
