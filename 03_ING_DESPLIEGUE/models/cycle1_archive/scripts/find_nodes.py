"""Find ONNX node names for score head convolutions."""
import onnx

model = onnx.load('/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_esp.onnx')

# Nodes producing score/box outputs
print('=== Nodes producing score/box outputs ===')
for node in model.graph.node:
    for out in node.output:
        if 'score' in out or 'box' in out:
            print(f'  Node: "{node.name}"  Op: {node.op_type}')
            print(f'    Outputs: {list(node.output)}')
            print(f'    Inputs: {list(node.input)}')

# All cv3 related nodes (score head)
print('\n=== cv3 nodes (score head) ===')
for node in model.graph.node:
    if 'cv3' in node.name:
        print(f'  "{node.name}" Op={node.op_type} Out={list(node.output)}')

# All cv2 related nodes (box head)
print('\n=== cv2 nodes (box head) ===')
for node in model.graph.node:
    if 'cv2' in node.name:
        print(f'  "{node.name}" Op={node.op_type} Out={list(node.output)}')

# Last 30 nodes
print('\n=== Last 30 nodes ===')
for node in model.graph.node[-30:]:
    print(f'  "{node.name}" {node.op_type} -> {list(node.output)}')
