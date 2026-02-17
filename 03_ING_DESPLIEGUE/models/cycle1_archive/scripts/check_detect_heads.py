"""Check YOLO detect head architecture for both models."""
from ultralytics import YOLO

for pt_path in ["models/yolo11n_v1_best.pt", "models/yolo26n_v1_best.pt"]:
    print(f"\n{'='*60}")
    print(f"  {pt_path}")
    print(f"{'='*60}")
    
    model = YOLO(pt_path)
    
    for name, mod in model.model.named_modules():
        cname = type(mod).__name__
        if 'Detect' in cname or name == '':
            if name == '' and 'Model' not in cname:
                continue
            print(f"  Module: '{name}' -> {cname}")
            for attr in ['nc', 'reg_max', 'nl']:
                if hasattr(mod, attr):
                    print(f"    {attr}: {getattr(mod, attr)}")
            if hasattr(mod, 'cv2'):
                print(f"    cv2: {len(mod.cv2)} heads, type={type(mod.cv2[0]).__name__}")
                for i, head in enumerate(mod.cv2):
                    print(f"      cv2[{i}]: {head}")
            if hasattr(mod, 'cv3'):
                print(f"    cv3: {len(mod.cv3)} heads")

    # Check Attention modules
    attn_count = sum(1 for _, m in model.model.named_modules() if type(m).__name__ == 'Attention')
    if attn_count:
        print(f"  Attention modules: {attn_count}")

print("\nDone.")
