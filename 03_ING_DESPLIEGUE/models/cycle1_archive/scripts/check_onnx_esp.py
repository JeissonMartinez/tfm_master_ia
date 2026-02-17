"""Check exported ONNX files for issues before quantization."""
import onnx
import numpy as np

for name in ['yolo11n_v1_best_esp.onnx', 'yolo26n_v1_best_esp.onnx']:
    path = f'models/{name}'
    m = onnx.load(path)
    print(f'\n=== {name} ===')
    print(f'Opset: {[o.version for o in m.opset_import]}')
    
    # Check for -1 in Reshape
    init_map = {i.name: i for i in m.graph.initializer}
    has_neg1 = False
    for n in m.graph.node:
        if n.op_type == 'Reshape' and len(n.input) > 1 and n.input[1] in init_map:
            vals = np.frombuffer(init_map[n.input[1]].raw_data, dtype=np.int64)
            if -1 in vals:
                has_neg1 = True
                print(f'  WARNING: Reshape {n.name}: shape={vals.tolist()}')
    if not has_neg1:
        print(f'  OK: No Reshape with -1')
    
    ops = sorted(set(n.op_type for n in m.graph.node))
    print(f'  Ops ({len(ops)}): {ops}')
    
    for out in m.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f'  Output: {out.name} {dims}')
