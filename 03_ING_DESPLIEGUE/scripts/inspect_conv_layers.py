#!/usr/bin/env python3
"""Inspect Conv2D layer details in the YOLO11n fullint8 TFLite model.

Checks dimension compatibility and parameters that could trigger
TFLITE_DCHECK assertions in esp-tflite-micro Conv2D kernel.
"""

import sys
sys.path.insert(0, ".")

from ai_edge_litert import interpreter as tfl

MODEL = "../02_ING_MODELOS/GoogleCloudAI/outputs/yolo11n_v1/train/weights/best_saved_model/best_full_integer_quant.tflite"


def main():
    interp = tfl.Interpreter(model_path=MODEL)
    interp.allocate_tensors()

    # Get model details using the internal _get_ops_details
    try:
        ops = interp._get_ops_details()
    except AttributeError:
        print("Cannot get ops details from this interpreter version")
        return

    tensors = interp.get_tensor_details()
    tensor_map = {t["index"]: t for t in tensors}

    print(f"Total operators: {len(ops)}")
    print(f"Total tensors:   {len(tensors)}")
    print()

    conv_count = 0
    for i, op in enumerate(ops):
        op_name = op.get("op_name", "unknown")
        if "CONV_2D" not in op_name:
            continue

        conv_count += 1
        inputs = op["inputs"]
        outputs = op["outputs"]

        inp = tensor_map.get(inputs[0], {})
        filt = tensor_map.get(inputs[1], {})
        bias = tensor_map.get(inputs[2], {}) if len(inputs) > 2 else {}
        out = tensor_map.get(outputs[0], {})

        inp_shape = inp.get("shape", [])
        filt_shape = filt.get("shape", [])
        bias_shape = bias.get("shape", [])
        out_shape = out.get("shape", [])

        inp_dtype = inp.get("dtype", None)
        filt_dtype = filt.get("dtype", None)
        out_dtype = out.get("dtype", None)

        # Check dimension issues
        issues = []
        if len(inp_shape) != 4:
            issues.append(f"INPUT ndim={len(inp_shape)} (expected 4)")
        if len(filt_shape) != 4:
            issues.append(f"FILTER ndim={len(filt_shape)} (expected 4)")
        if len(out_shape) != 4:
            issues.append(f"OUTPUT ndim={len(out_shape)} (expected 4)")

        # For standard Conv2D: input_channels = filter_input_channels
        if len(inp_shape) == 4 and len(filt_shape) == 4:
            in_ch = inp_shape[3]
            filt_in_ch = filt_shape[3]
            if in_ch != filt_in_ch:
                # Could be group conv
                if in_ch % filt_in_ch != 0:
                    issues.append(f"input_ch({in_ch}) % filter_in_ch({filt_in_ch}) != 0")
                else:
                    issues.append(f"GROUPED: input_ch={in_ch}, filter_in_ch={filt_in_ch}, groups={in_ch // filt_in_ch}")

            # MatchingDim checks in EvalQuantizedPerChannel
            if inp_shape[0] != out_shape[0]:
                issues.append(f"BATCH MISMATCH: in[0]={inp_shape[0]} vs out[0]={out_shape[0]}")
            if len(out_shape) == 4 and filt_shape[0] != out_shape[3]:
                issues.append(f"OUTPUT_DEPTH MISMATCH: filter[0]={filt_shape[0]} vs out[3]={out_shape[3]}")

        # Type issues
        if inp_dtype != out_dtype:
            issues.append(f"TYPE MISMATCH: in={inp_dtype} out={out_dtype}")

        status = "❌ " + "; ".join(issues) if issues else "✅"

        print(f"Op[{i:3d}] CONV_2D #{conv_count}")
        print(f"  Input:  {inp_shape} dtype={inp_dtype}")
        print(f"  Filter: {filt_shape} dtype={filt_dtype}")
        print(f"  Bias:   {bias_shape}")
        print(f"  Output: {out_shape} dtype={out_dtype}")
        print(f"  Status: {status}")

        # Get builtin options if available
        opts = op.get("builtin_options", {})
        if opts:
            print(f"  Options: {opts}")
        print()

    print(f"\nTotal CONV_2D layers: {conv_count}")


if __name__ == "__main__":
    main()
