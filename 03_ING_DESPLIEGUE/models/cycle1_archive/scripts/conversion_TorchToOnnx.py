"""
Conversión de modelos YOLO (.pt Ultralytics) a ONNX.

Los archivos .pt de Ultralytics contienen el modelo completo (arquitectura + pesos),
por lo que se deben cargar con la clase YOLO de ultralytics.

Modelos:
  - YOLO11n_v1: input 224x224 RGB, output [1, 9, 1029]
  - YOLO26n_v1: input 224x224 RGB, output [1, 300, 6] (NMS integrado)
"""

from ultralytics import YOLO

BASE = "/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models"

models = {
    "yolo11n_v1": f"{BASE}/yolo11n_v1_best.pt",
    "yolo26n_v1": f"{BASE}/yolo26n_v1_best.pt",
}

for name, path_model in models.items():
    print(f"\n{'='*60}")
    print(f"Convirtiendo {name}: {path_model}")
    print(f"{'='*60}")

    # 1. Cargar el modelo con Ultralytics (incluye arquitectura + pesos)
    model = YOLO(path_model)

    # 2. Exportar a ONNX (imgsz=224 según el entrenamiento, opset 13)
    output_path = model.export(
        format="onnx",
        imgsz=224,        # 224x224 RGB (3 canales, NCHW automático)
        opset=13,
        simplify=True,    # Simplificar grafo ONNX con onnxsim
    )

    print(f"✅ {name} exportado a: {output_path}")

print("\n✅ Conversión completada para todos los modelos.") 