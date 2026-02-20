"""
export_onnx_esp.py — Exportación ONNX compatible con ESP-DL
============================================================
Exporta modelos YOLO (.pt) en formato ONNX con 6 salidas separadas
(box0/score0, box1/score1, box2/score2) SIN detection head.

Esto permite a esp-ppq cuantizar cada tensor de salida con su propio exponent,
evitando el problema de cuantización por tensor donde class scores (0-1) y
bbox coords (0-224) comparten el mismo exponent.

Basado en el enfoque de Espressif en:
    esp-dl/models/coco_detect/models/export_onnx.py

Salidas por nivel de detección (P3, P4, P5):
  - box{i}: [1, reg_max*4, H, W]  (YOLO11n: [1,64,H,W], YOLO26n: [1,4,H,W])
  - score{i}: [1, nc, H, W]       (siempre [1,5,H,W] para 5 clases)

    P3 (stride 8):  H=W=28 (224/8)  → 784 candidates
    P4 (stride 16): H=W=14 (224/16) → 196 candidates
    P5 (stride 32): H=W=7  (224/32) → 49 candidates
    Total: 1029 candidates

Uso:
    python models/export_onnx_esp.py
"""

import os
import sys
import torch
import onnx
from onnxsim import simplify

# Importar Ultralytics
try:
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect
except ImportError:
    print("[ERROR] Necesitas ultralytics: pip install ultralytics")
    sys.exit(1)

# Intentar importar Attention (puede estar en diferentes ubicaciones)
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
# Detect head "sin cabeza" — solo convs crudas por nivel
# ============================================================================

class ESP_Detect_Forward:
    """Replacement forward para Detect que devuelve las salidas crudas de cv2/cv3
    por cada nivel, sin DFL, sigmoid ni concat."""

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
        q, k, v = qkv.reshape(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        # Scaled dot-product attention (sin einsum)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        x = self.proj(x)
        return x


# ============================================================================
# Configuración
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMGSZ = 224

MODELS = [
    {
        "name": "yolo11n_v1_best",
        "pt_file": "yolo11n_v1_best.pt",
        "output_file": "yolo11n_v1_best_esp.onnx",
    },
    {
        "name": "yolo26n_v1_best",
        "pt_file": "yolo26n_v1_best.pt",
        "output_file": "yolo26n_v1_best_esp.onnx",
    },
]


# ============================================================================
# Exportación
# ============================================================================

def patch_model(model):
    """Monkey-patch el modelo para export ESP-compatible."""
    patched_detect = False
    patched_attention = 0

    for name, module in model.model.named_modules():
        # Patch Detect head
        if isinstance(module, Detect):
            # Guardar referencia original
            module._original_forward = module.forward
            # Bind el nuevo forward
            import types
            module.forward = types.MethodType(ESP_Detect_Forward.forward, module)
            patched_detect = True
            print(f"  ✓ Patched Detect at '{name}' (nc={module.nc}, "
                  f"reg_max={module.reg_max}, nl={module.nl})")

        # Patch Attention (si existe) para evitar einsum
        if HAS_ATTENTION and isinstance(module, Attention):
            import types
            module.forward = types.MethodType(ESP_Attention_Forward.forward, module)
            patched_attention += 1

    if patched_attention:
        print(f"  ✓ Patched {patched_attention} Attention module(s)")

    if not patched_detect:
        print("  [ERROR] No se encontró módulo Detect para patchear")
        return False
    return True


def export_model(config):
    """Exporta un modelo .pt a ONNX con 6 salidas ESP-compatible."""
    pt_path = os.path.join(BASE_DIR, config["pt_file"])
    onnx_path = os.path.join(BASE_DIR, config["output_file"])

    if not os.path.isfile(pt_path):
        print(f"  [ERROR] No se encontró: {pt_path}")
        return False

    # 1. Cargar modelo
    print(f"\n  Cargando {config['pt_file']}...")
    yolo = YOLO(pt_path)
    model = yolo.model

    # Obtener info del Detect head
    detect_module = None
    for _, mod in model.named_modules():
        if isinstance(mod, Detect):
            detect_module = mod
            break

    if not detect_module:
        print("  [ERROR] No se encontró módulo Detect")
        return False

    nc = detect_module.nc
    reg_max = detect_module.reg_max
    nl = detect_module.nl

    # 2. Patchear modelo
    print(f"  Parcheando modelo...")
    if not patch_model(yolo):
        return False

    # 3. Preparar para exportación
    model.eval()
    model.float()

    # Input dummy
    dummy_input = torch.randn(1, 3, IMGSZ, IMGSZ)

    # Nombres de salida
    output_names = []
    for i in range(nl):
        output_names.extend([f"box{i}", f"score{i}"])

    # 4. Exportar a ONNX
    print(f"  Exportando ONNX con salidas: {output_names}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,  # shapes fijos
    )

    # 5. Simplificar con onnxsim
    print(f"  Simplificando con onnxsim...")
    onnx_model = onnx.load(onnx_path)
    onnx_model_sim, check = simplify(
        onnx_model,
        overwrite_input_shapes={"images": [1, 3, IMGSZ, IMGSZ]},
    )
    if check:
        onnx.save(onnx_model_sim, onnx_path)
        print(f"  ✓ Simplificación verificada")
    else:
        print(f"  [WARN] Simplificación no verificada, guardando original")

    # 6. Verificar salidas
    onnx_final = onnx.load(onnx_path)
    print(f"\n  Modelo exportado: {os.path.basename(onnx_path)}")
    print(f"  Tamaño: {os.path.getsize(onnx_path) / 1024:.1f} KB")

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
        print(f"  ⚠ Ops potencialmente problemáticas: {problematic}")
    else:
        print(f"  ✓ Sin operaciones problemáticas")

    return True


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  EXPORTACIÓN ONNX ESP-DL COMPATIBLE (6 salidas)")
    print(f"  Input size: {IMGSZ}×{IMGSZ}")
    print("=" * 60)

    results = []
    for config in MODELS:
        print(f"\n{'─' * 60}")
        print(f"  {config['name']}")
        print(f"{'─' * 60}")

        try:
            success = export_model(config)
            results.append((config["name"], "OK" if success else "FAIL"))
        except Exception as e:
            import traceback
            print(f"  [ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((config["name"], f"ERROR: {e}"))

    # Resumen
    print(f"\n{'=' * 60}")
    print(f"  RESUMEN")
    print(f"{'=' * 60}")
    for name, status in results:
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {name}: {status}")

    print(f"\n  Siguiente paso:")
    print(f"  1. Verificar los ONNX generados")
    print(f"  2. Ejecutar fix_onnx_for_espdl.py si es necesario")
    print(f"  3. Ejecutar quantize_models.py con los nuevos ONNX")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
