#!/usr/bin/env python3
"""
export_yolo26_esp.py — Re-exportar YOLO26 T2 con 6 salidas ESP-compatible
==========================================================================
Exporta el modelo YOLO26 T2 (.pt) en formato ONNX con 6 salidas separadas
(box0/score0, box1/score1, box2/score2) SIN detection head post-processing.

Esto resuelve el problema de cuantización INT8 donde el output unificado
[1, 9, 1029] comparte el mismo exponente para boxes (0-224) y scores (0-1),
destruyendo la resolución de los scores.

Basado en el enfoque probado de:
    03_ING_DESPLIEGUE/models/export_onnx_esp.py

Salidas por nivel de detección (P3, P4, P5):
  - box{i}: [1, reg_max*4, H, W]  → [1, 64, H, W] (reg_max=16)
  - score{i}: [1, nc, H, W]       → [1, 5, H, W]

    P3 (stride 8):  H=W=28 (224/8)  → 784 candidates
    P4 (stride 16): H=W=14 (224/16) → 196 candidates
    P5 (stride 32): H=W=7  (224/32) → 49 candidates
    Total: 1029 candidates

Uso:
    cd 02_ING_MODELOS/Train_MLOps
    python scripts/export_yolo26_esp.py
"""

import os
import sys
import types
from pathlib import Path

import torch
import onnx

try:
    from onnxsim import simplify
except ImportError:
    print("[ERROR] Necesitas onnxsim: pip install onnxsim")
    sys.exit(1)

try:
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect
except ImportError:
    print("[ERROR] Necesitas ultralytics: pip install ultralytics")
    sys.exit(1)

# Intentar importar Attention (YOLO26 usa A2C2f con Attention)
try:
    from ultralytics.nn.modules.block import Attention
    HAS_ATTENTION = True
except ImportError:
    try:
        from ultralytics.nn.modules import Attention
        HAS_ATTENTION = True
    except ImportError:
        HAS_ATTENTION = False
        print("[WARN] No se pudo importar Attention, se omitirá el parche")


# ============================================================================
# Configuración
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_MLOPS_DIR = SCRIPT_DIR.parent  # scripts/ → Train_MLOps/

PT_FILE = TRAIN_MLOPS_DIR / "outputs/yolo26n_custom_v2-run1/yolo_project/phase2/weights/best.pt"
OUTPUT_DIR = TRAIN_MLOPS_DIR / "outputs/yolo26n_custom_v2-run1/export"
OUTPUT_FILE = OUTPUT_DIR / "best_esp.onnx"

IMGSZ = 224


# ============================================================================
# Replacement forwards — eliminar detection head post-processing
# ============================================================================

class ESP_Detect_Forward:
    """Devuelve las salidas crudas de cv2/cv3 por cada nivel,
    sin DFL, sigmoid ni concat."""

    @staticmethod
    def forward(self, x):
        """Devuelve box0, score0, box1, score1, box2, score2."""
        results = []
        for i in range(self.nl):
            results.append(self.cv2[i](x[i]))   # box: [1, reg_max*4, H, W]
            results.append(self.cv3[i](x[i]))   # score: [1, nc, H, W]
        return tuple(results)


class ESP_Attention_Forward:
    """Replacement forward para Attention que evita operaciones
    no soportadas en ESP-DL (como einsum)."""

    @staticmethod
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(
            B, self.num_heads, self.key_dim * 2 + self.head_dim, N
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        # Scaled dot-product attention (sin einsum)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        x = self.proj(x)
        return x


# ============================================================================
# Funciones principales
# ============================================================================

def patch_model(yolo_model):
    """Monkey-patch el modelo para export ESP-compatible."""
    patched_detect = False
    patched_attention = 0

    for name, module in yolo_model.model.named_modules():
        # Patch Detect head
        if isinstance(module, Detect):
            module.forward = types.MethodType(ESP_Detect_Forward.forward, module)
            patched_detect = True
            print(f"  ✓ Patched Detect at '{name}' "
                  f"(nc={module.nc}, reg_max={module.reg_max}, nl={module.nl})")

        # Patch Attention (YOLO26 usa A2C2f con Attention) para evitar einsum
        if HAS_ATTENTION and isinstance(module, Attention):
            module.forward = types.MethodType(ESP_Attention_Forward.forward, module)
            patched_attention += 1

    if patched_attention:
        print(f"  ✓ Patched {patched_attention} Attention module(s)")

    if not patched_detect:
        print("  [ERROR] No se encontró módulo Detect para patchear")
        return False
    return True


def export():
    """Exporta YOLO26 T2 .pt → ONNX con 6 salidas ESP-compatible."""
    print("=" * 60)
    print("  EXPORTACIÓN YOLO26 T2 → ONNX ESP-DL (6 salidas)")
    print(f"  Input size: {IMGSZ}×{IMGSZ}")
    print(f"  PT: {PT_FILE}")
    print("=" * 60)

    if not PT_FILE.exists():
        print(f"  [ERROR] No se encontró: {PT_FILE}")
        sys.exit(1)

    # 1. Cargar modelo
    print(f"\n  Cargando modelo...")
    yolo = YOLO(str(PT_FILE))
    model = yolo.model

    # Obtener info del Detect head
    detect_module = None
    for _, mod in model.named_modules():
        if isinstance(mod, Detect):
            detect_module = mod
            break

    if not detect_module:
        print("  [ERROR] No se encontró módulo Detect")
        sys.exit(1)

    nc = detect_module.nc
    reg_max = detect_module.reg_max
    nl = detect_module.nl

    print(f"  nc={nc}, reg_max={reg_max}, nl={nl}")
    print(f"  Box output channels: {reg_max * 4} (reg_max × 4)")
    print(f"  Score output channels: {nc}")

    # 2. Patch modelo
    print(f"\n  Parcheando modelo...")
    if not patch_model(yolo):
        sys.exit(1)

    # 3. Preparar para exportación
    model.eval()
    model.float()

    dummy_input = torch.randn(1, 3, IMGSZ, IMGSZ)

    # Nombres de salida
    output_names = []
    for i in range(nl):
        output_names.extend([f"box{i}", f"score{i}"])

    # 4. Exportar a ONNX
    print(f"\n  Exportando ONNX con salidas: {output_names}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_FILE),
        opset_version=13,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,  # shapes fijos
    )

    # 5. Simplificar con onnxsim
    print(f"  Simplificando con onnxsim...")
    onnx_model = onnx.load(str(OUTPUT_FILE))
    onnx_model_sim, check = simplify(
        onnx_model,
        overwrite_input_shapes={"images": [1, 3, IMGSZ, IMGSZ]},
    )
    if check:
        onnx.save(onnx_model_sim, str(OUTPUT_FILE))
        print(f"  ✓ Simplificación verificada")
    else:
        print(f"  [WARN] Simplificación no verificada, guardando original")

    # 6. Verificar salidas
    onnx_final = onnx.load(str(OUTPUT_FILE))
    size_kb = os.path.getsize(str(OUTPUT_FILE)) / 1024

    print(f"\n  {'─' * 50}")
    print(f"  Modelo exportado: {OUTPUT_FILE.name}")
    print(f"  Tamaño: {size_kb:.1f} KB")

    for inp in onnx_final.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name} {dims}")

    for out in onnx_final.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} {dims}")

    # Verificar que no hay ops problemáticas
    ops = set(n.op_type for n in onnx_final.graph.node)
    problematic = ops & {"TopK", "GatherElements", "NonMaxSuppression",
                         "ScatterND", "Einsum"}
    if problematic:
        print(f"  ⚠️ Ops potencialmente problemáticas: {problematic}")
    else:
        print(f"  ✓ Sin operaciones problemáticas para ESP-DL")

    # Verificar shapes esperados
    expected_shapes = {
        "box0": [1, reg_max * 4, 28, 28],
        "score0": [1, nc, 28, 28],
        "box1": [1, reg_max * 4, 14, 14],
        "score1": [1, nc, 14, 14],
        "box2": [1, reg_max * 4, 7, 7],
        "score2": [1, nc, 7, 7],
    }
    all_ok = True
    for out in onnx_final.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        expected = expected_shapes.get(out.name)
        if expected and dims != expected:
            print(f"  ⚠️ Shape inesperado para {out.name}: {dims} (esperado {expected})")
            all_ok = False

    if all_ok:
        print(f"\n  ✅ Exportación exitosa — 6 salidas separadas")
    else:
        print(f"\n  ⚠️ Exportación completada con advertencias")

    print(f"\n  Siguiente paso:")
    print(f"  python scripts/convert_onnx_to_espdl.py "
          f"--calib-dir ../datasets/IODC/coco/train/images "
          f"--models yolo26n_t2_esp")
    print("=" * 60)


if __name__ == "__main__":
    export()
