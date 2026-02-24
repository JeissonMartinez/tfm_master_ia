"""Diagnostic: compare FP32 vs INT8 raw output statistics for FCOS T3 and YOLO26 T2."""
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from PIL import Image
import warnings, tempfile, os
warnings.filterwarnings("ignore")

IMGSZ = 224
BASE = Path(__file__).resolve().parent.parent
TEST_IMG = BASE / ".." / "datasets" / "IODC" / "coco" / "test" / "images"
CALIB_IMG = BASE / ".." / "datasets" / "IODC" / "coco" / "train" / "images"

img_path = sorted(TEST_IMG.glob("*.jpg"))[0]
img = Image.open(str(img_path)).convert("RGB").resize((IMGSZ, IMGSZ))
arr = np.array(img, dtype=np.float32) / 255.0
arr = arr.transpose(2, 0, 1)
img_np = np.expand_dims(arr, 0)

# Quick calibration
calib_files = sorted(CALIB_IMG.glob("*.jpg"))[:50]
calib_data = []
for f in calib_files:
    im = Image.open(str(f)).convert("RGB").resize((IMGSZ, IMGSZ))
    a = np.array(im, dtype=np.float32) / 255.0
    a = a.transpose(2, 0, 1)
    calib_data.append(torch.from_numpy(np.expand_dims(a, 0)))

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.executor import TorchExecutor


def diagnose_model(name, onnx_path, input_name):
    print(f"\n{'='*60}")
    print(f"  {name} — FP32 vs INT8 raw output diagnostic")
    print(f"{'='*60}")

    if not os.path.exists(onnx_path):
        print(f"  ONNX not found: {onnx_path}")
        return

    # FP32
    sess = ort.InferenceSession(onnx_path)
    fp32_outs = sess.run(None, {input_name: img_np})
    fp32_names = [o.name for o in sess.get_outputs()]

    print(f"\n  FP32 outputs:")
    for n, out in zip(fp32_names, fp32_outs):
        print(f"    {n:22s} shape={str(out.shape):20s} min={out.min():10.4f} max={out.max():10.4f} mean={out.mean():10.4f} std={out.std():10.4f}")

    # INT8
    tmp = tempfile.mktemp(suffix=".espdl")
    setting = QuantizationSettingFactory.espdl_setting()

    # Fix negative axes for YOLO26
    actual_onnx = onnx_path
    if "yolo26" in name.lower():
        import onnx
        model = onnx.load(onnx_path)
        onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
        fixed = False
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.name == "axis" and attr.i < 0:
                    for vi in list(model.graph.value_info) + list(model.graph.input):
                        if vi.name == node.input[0]:
                            rank = len(vi.type.tensor_type.shape.dim)
                            if rank > 0:
                                attr.i = attr.i + rank
                                fixed = True
                            break
        if fixed:
            actual_onnx = onnx_path.replace(".onnx", "_fixed_diag.onnx")
            onnx.save(model, actual_onnx)
            print(f"  Fixed negative axes -> {actual_onnx}")

    print(f"\n  Quantizing ({len(calib_data)} calib)...")
    ppq_graph = espdl_quantize_onnx(
        onnx_import_file=actual_onnx,
        espdl_export_file=tmp,
        calib_dataloader=calib_data,
        calib_steps=50,
        input_shape=[1, 3, IMGSZ, IMGSZ],
        target="esp32s3",
        setting=setting,
        do_quantize=True,
    )

    executor = TorchExecutor(ppq_graph, device="cpu")
    int8_outs = executor.forward(torch.from_numpy(img_np).float())
    if not isinstance(int8_outs, (list, tuple)):
        int8_outs = [int8_outs]
    int8_names = list(ppq_graph.outputs)

    print(f"\n  INT8 outputs:")
    for n, out in zip(int8_names, int8_outs):
        o = out.detach().numpy()
        print(f"    {n:22s} shape={str(o.shape):20s} min={o.min():10.4f} max={o.max():10.4f} mean={o.mean():10.4f} std={o.std():10.4f}")

    print(f"\n  FP32 vs INT8 per-output comparison:")
    for i, (fn, fp) in enumerate(zip(fp32_names, fp32_outs)):
        if i < len(int8_outs):
            io = int8_outs[i].detach().numpy()
            if fp.shape == io.shape:
                diff = np.abs(fp - io)
                fp_flat = fp.flatten()
                io_flat = io.flatten()
                if fp_flat.std() > 1e-8 and io_flat.std() > 1e-8:
                    corr = np.corrcoef(fp_flat, io_flat)[0, 1]
                else:
                    corr = 0.0
                # Cosine similarity
                norm_fp = np.linalg.norm(fp_flat)
                norm_io = np.linalg.norm(io_flat)
                if norm_fp > 0 and norm_io > 0:
                    cosim = np.dot(fp_flat, io_flat) / (norm_fp * norm_io)
                else:
                    cosim = 0.0
                print(f"    {fn:22s} MAE={diff.mean():.4f} MaxErr={diff.max():.4f} Corr={corr:.4f} CosSim={cosim:.4f}")
            else:
                print(f"    {fn:22s} SHAPE MISMATCH fp={fp.shape} int8={io.shape}")

    # Quantization config analysis
    print(f"\n  Quantization config analysis:")
    n_fp32 = 0
    n_int8 = 0
    for op in ppq_graph.operations.values():
        for cfg in op.config.input_quantization_config + op.config.output_quantization_config:
            if hasattr(cfg, 'state'):
                state_name = str(cfg.state)
                if 'FP32' in state_name:
                    n_fp32 += 1
                elif 'ACTIVATED' in state_name:
                    n_int8 += 1
    print(f"    FP32 configs: {n_fp32}")
    print(f"    INT8 (ACTIVATED) configs: {n_int8}")

    # Check for exponent info if available
    print(f"\n  Output quantization details:")
    for out_name in int8_names:
        for op in ppq_graph.operations.values():
            for cfg in op.config.output_quantization_config:
                if hasattr(cfg, 'scale') and cfg.scale is not None:
                    # Check if this output connects to graph output
                    pass
    
    # Cleanup
    for f_path in [tmp]:
        if os.path.exists(f_path):
            os.remove(f_path)
    for ext in [".info", ".json"]:
        t = tmp.replace(".espdl", ext)
        if os.path.exists(t):
            os.remove(t)
    if actual_onnx != onnx_path and os.path.exists(actual_onnx):
        os.remove(actual_onnx)


# Run diagnostics
diagnose_model(
    "FCOS T3",
    str(BASE / "outputs/fcos_v3s_v1-1771690809/export/fcos_v3s.onnx"),
    "input",
)

diagnose_model(
    "YOLO26 T2",
    str(BASE / "outputs/yolo26n_custom_v2-run1/export/best.onnx"),
    "images",
)

# Also test with ESPDet as baseline (known good)
diagnose_model(
    "ESPDet T4 (baseline)",
    str(BASE / "outputs/espdet-pico-v4-t4/export/espdet_pico.onnx"),
    "input",
)
