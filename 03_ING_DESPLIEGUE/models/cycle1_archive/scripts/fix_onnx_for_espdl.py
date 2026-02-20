"""
fix_onnx_for_espdl.py
=====================
Pre-procesa modelos ONNX de YOLO para hacerlos compatibles con el exportador
ESPDL de esp-ppq.

Problemas que resuelve:
  1. Reshape con dimensiones dinámicas (-1) → las resuelve a valores concretos
     usando onnxsim con input shape fijo.
  2. Operaciones no soportadas (TopK, GatherElements, etc.) en modelos con NMS
     integrado → trunca el grafo antes del post-procesamiento NMS.

Uso:
  python models/fix_onnx_for_espdl.py
"""

import os
import numpy as np
import onnx
from onnx import shape_inference
from onnxsim import simplify

# ============================================================================
# Configuración
# ============================================================================

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_SHAPE = [1, 3, 224, 224]  # NCHW
INPUT_NAME = "images"


def get_onnx_info(model_path: str):
    """Imprime información básica del modelo ONNX."""
    m = onnx.load(model_path)
    print(f"  Opset: {[o.version for o in m.opset_import]}")
    for inp in m.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param
                for d in inp.type.tensor_type.shape.dim]
        print(f"  Input: {inp.name} {dims}")
    for out in m.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param
                for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} {dims}")
    ops = sorted(set(n.op_type for n in m.graph.node))
    print(f"  Ops ({len(ops)}): {ops}")

    # Check for -1 in Reshape initializers
    init_map = {init.name: init for init in m.graph.initializer}
    has_neg1 = False
    for n in m.graph.node:
        if n.op_type == "Reshape" and len(n.input) > 1 and n.input[1] in init_map:
            vals = np.frombuffer(init_map[n.input[1]].raw_data, dtype=np.int64)
            if -1 in vals:
                has_neg1 = True
                print(f"  ⚠ Reshape {n.name}: shape={vals.tolist()} (contiene -1)")
    if not has_neg1:
        print(f"  ✓ No hay Reshape con -1")

    return m


def resolve_dynamic_shapes(model_path: str, output_path: str):
    """
    Resuelve dimensiones dinámicas (-1) en Reshape:
      1. onnxsim con input shape fijo
      2. Shape inference para obtener shapes concretos
      3. Reemplazo manual de -1 con valores concretos (cada Reshape
         recibe su propia constante de shape)
    """
    print(f"\n  Simplificando con input shape fijo {INPUT_SHAPE}...")
    model = onnx.load(model_path)

    # Paso 1: Shape inference
    model = shape_inference.infer_shapes(model)

    # Paso 2: Simplificar con onnxsim y shapes fijos
    model_sim, check = simplify(
        model,
        overwrite_input_shapes={INPUT_NAME: INPUT_SHAPE},
    )

    if not check:
        print(f"  [WARN] onnxsim reportó que el modelo simplificado no es equivalente")
    else:
        print(f"  ✓ Simplificación verificada")

    # Paso 3: Resolver -1 manualmente en Reshape ops
    model_fixed = _replace_neg1_in_reshapes(model_sim)

    onnx.save(model_fixed, output_path)
    print(f"  Guardado en: {os.path.basename(output_path)}")
    return model_fixed


def _replace_neg1_in_reshapes(model):
    """
    Para cada Reshape que usa -1, calcula el shape concreto usando la info
    inferida y crea una constante dedicada con el valor correcto.
    """
    from onnx import numpy_helper, TensorProto

    # Shape inference para tener todos los shapes intermedios
    model = shape_inference.infer_shapes(model)

    # Crear mapa de shapes inferidos: nombre_tensor → shape
    inferred_shapes = {}
    for vi in model.graph.value_info:
        if vi.type.tensor_type.HasField("shape"):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if all(d > 0 for d in dims):
                inferred_shapes[vi.name] = dims
    for vi in model.graph.output:
        if vi.type.tensor_type.HasField("shape"):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if all(d > 0 for d in dims):
                inferred_shapes[vi.name] = dims
    for vi in model.graph.input:
        if vi.type.tensor_type.HasField("shape"):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if all(d > 0 for d in dims):
                inferred_shapes[vi.name] = dims

    # Mapa de inicializadores
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = init

    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue

        shape_input_name = node.input[1]
        if shape_input_name not in init_map:
            continue

        shape_data = np.frombuffer(init_map[shape_input_name].raw_data, dtype=np.int64).copy()
        if -1 not in shape_data:
            continue

        # Obtener el shape de salida inferido del Reshape
        output_name = node.output[0]
        if output_name not in inferred_shapes:
            print(f"  [WARN] No se pudo inferir shape para {output_name}, "
                  f"se mantiene -1")
            continue

        concrete_shape = np.array(inferred_shapes[output_name], dtype=np.int64)

        # Crear un nuevo inicializador único para este Reshape
        new_name = f"{shape_input_name}_fixed_{replaced}"
        new_init = numpy_helper.from_array(concrete_shape, name=new_name)
        model.graph.initializer.append(new_init)

        # Actualizar el input del Reshape
        node.input[1] = new_name

        print(f"  ✓ {node.name}: {shape_data.tolist()} → {concrete_shape.tolist()}")
        replaced += 1

    if replaced > 0:
        print(f"  Total: {replaced} Reshape ops corregidos")
    else:
        print(f"  No se encontraron Reshape con -1 que resolver")

    return model


def check_unsupported_ops(model):
    """Verifica si hay operaciones que esp-ppq/ESPDL no soporta bien."""
    # Ops problemáticas para ESPDL export
    problematic_ops = {"TopK", "GatherElements", "Gather", "Mod", "Tile",
                       "NonMaxSuppression", "ScatterND", "ScatterElements"}
    found = set()
    for n in model.graph.node:
        if n.op_type in problematic_ops:
            found.add(n.op_type)
    return found


def truncate_before_nms(model_path: str, output_path: str, raw_output_names: list):
    """
    Trunca el modelo ONNX para que termine en las salidas raw (antes del NMS),
    eliminando el post-procesamiento que contiene ops no soportadas.

    Args:
        model_path: Ruta al modelo ONNX.
        output_path: Ruta de salida.
        raw_output_names: Nombres de los tensores intermedios que serán las nuevas salidas.
    """
    print(f"\n  Truncando modelo antes del NMS...")
    model = onnx.load(model_path)

    # Encontrar los value_info de las salidas deseadas
    # Primero hacer shape inference para tener toda la info
    model = shape_inference.infer_shapes(model)

    # Crear mapa de value_info
    vi_map = {}
    for vi in model.graph.value_info:
        vi_map[vi.name] = vi
    for vi in model.graph.output:
        vi_map[vi.name] = vi

    # Crear nuevas salidas
    new_outputs = []
    for name in raw_output_names:
        if name in vi_map:
            new_outputs.append(vi_map[name])
        else:
            print(f"  [ERROR] No se encontró el tensor '{name}' en el grafo")
            return None

    # Reemplazar outputs
    while len(model.graph.output) > 0:
        model.graph.output.pop()
    for out in new_outputs:
        model.graph.output.append(out)

    # Eliminar nodos no alcanzables desde las nuevas salidas
    # Usamos onnxsim para limpiar
    model_clean, check = simplify(
        model,
        overwrite_input_shapes={INPUT_NAME: INPUT_SHAPE},
    )

    if not check:
        print(f"  [WARN] onnxsim reportó que el modelo truncado no es equivalente")

    onnx.save(model_clean, output_path)
    print(f"  Guardado en: {os.path.basename(output_path)}")
    return model_clean


def find_concat_before_nms(model):
    """
    Para YOLO models con NMS integrado, encuentra los Concat que combinan
    las salidas de las cabezas de detección (antes de NMS/postprocess).
    """
    # Buscar los Concat del detection head (fusionan bbox y cls de las 3 escalas)
    concat_nodes = [n for n in model.graph.node if n.op_type == "Concat"]

    # En YOLO, los dos Concat principales son:
    # - Concat de bbox branches (cv2) de las 3 escalas
    # - Concat de cls branches (cv3) de las 3 escalas
    # Después viene otro Concat que los junta, y luego el postprocess

    # Buscar el Concat final que alimenta a los ops de NMS
    nms_related = {"TopK", "GatherElements", "Gather", "Mod", "Tile"}
    nms_inputs = set()
    for n in model.graph.node:
        if n.op_type in nms_related:
            nms_inputs.update(n.input)

    # Encontrar el último Concat antes de NMS
    pre_nms_concats = []
    for n in concat_nodes:
        for out in n.output:
            if out in nms_inputs:
                pre_nms_concats.append((n.name, n.output[0]))

    return pre_nms_concats


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  FIX ONNX MODELS FOR ESPDL EXPORT")
    print("=" * 60)

    # ---- yolo11n ----
    print(f"\n{'─' * 60}")
    print(f"  yolo11n_v1_best.onnx")
    print(f"{'─' * 60}")
    yolo11n_path = os.path.join(MODELS_DIR, "yolo11n_v1_best.onnx")
    yolo11n_fixed = os.path.join(MODELS_DIR, "yolo11n_v1_best_fixed.onnx")

    print("\n  [ANTES]")
    get_onnx_info(yolo11n_path)

    resolve_dynamic_shapes(yolo11n_path, yolo11n_fixed)

    print("\n  [DESPUÉS]")
    m11 = get_onnx_info(yolo11n_fixed)

    unsupported = check_unsupported_ops(m11)
    if unsupported:
        print(f"  ⚠ Ops potencialmente no soportadas: {unsupported}")
    else:
        print(f"  ✓ Todas las ops deberían ser compatibles con ESPDL")

    # ---- yolo26n ----
    print(f"\n{'─' * 60}")
    print(f"  yolo26n_v1_best.onnx")
    print(f"{'─' * 60}")
    yolo26n_path = os.path.join(MODELS_DIR, "yolo26n_v1_best.onnx")
    yolo26n_fixed = os.path.join(MODELS_DIR, "yolo26n_v1_best_fixed.onnx")

    print("\n  [ANTES]")
    m26_orig = get_onnx_info(yolo26n_path)

    unsupported = check_unsupported_ops(m26_orig)
    if unsupported:
        print(f"\n  ⚠ Ops no soportadas para ESPDL: {unsupported}")
        print(f"  → Se truncará el modelo antes del post-procesamiento NMS")

        # Primero simplificar
        yolo26n_sim_path = os.path.join(MODELS_DIR, "yolo26n_v1_best_sim.onnx")
        m26_sim = resolve_dynamic_shapes(yolo26n_path, yolo26n_sim_path)

        # Buscar las salidas raw del detection head
        # En YOLO26 (one2one detection), las ramas son:
        # - one2one_cv2 (bbox): 3 escalas → Reshape → Concat
        # - one2one_cv3 (cls): 3 escalas → Reshape → Concat
        # El Concat final de ambas ramas es la salida pre-NMS

        # Buscar el Concat que fusiona bbox+cls (último Concat grande antes de NMS)
        concat_nodes = [n for n in m26_sim.graph.node if n.op_type == "Concat"]
        print(f"\n  Buscando punto de corte (Concat nodes):")
        for cn in concat_nodes:
            print(f"    {cn.name}: output={cn.output[0]}, "
                  f"axis={cn.attribute[0].i if cn.attribute else '?'}, "
                  f"num_inputs={len(cn.input)}")

        # El Concat final que combina bbox (dim 64→4) y cls (dim 5) debería tener
        # axis=1 y 2 inputs (bbox_concat + cls_concat)
        # Buscamos Concat con 2 inputs y axis=1 que están justo antes de Transpose
        main_concat = None
        for cn in concat_nodes:
            if len(cn.input) == 2:
                axis = cn.attribute[0].i if cn.attribute else -1
                if axis == 1:
                    main_concat = cn
                    print(f"\n  → Concat principal encontrado: {cn.name} "
                          f"(output: {cn.output[0]})")

        if main_concat:
            # Buscar el Transpose que sigue al Concat (en YOLO, box+cls → Transpose → NMS)
            for n in m26_sim.graph.node:
                if n.op_type == "Transpose" and main_concat.output[0] in n.input:
                    raw_output = n.output[0]
                    print(f"  → Transpose encontrado: {n.name} (output: {raw_output})")
                    break
            else:
                raw_output = main_concat.output[0]

            # Truncar
            truncated = truncate_before_nms(yolo26n_sim_path, yolo26n_fixed, [raw_output])
            if truncated:
                print("\n  [DESPUÉS - TRUNCADO]")
                get_onnx_info(yolo26n_fixed)
        else:
            # Fallback: intentar simplemente resolver shapes
            print(f"\n  [WARN] No se encontró punto de corte automático.")
            print(f"  Intentando simplificación directa...")
            resolve_dynamic_shapes(yolo26n_path, yolo26n_fixed)
            print("\n  [DESPUÉS]")
            get_onnx_info(yolo26n_fixed)
    else:
        # Sin ops problemáticas, solo resolver -1
        resolve_dynamic_shapes(yolo26n_path, yolo26n_fixed)
        print("\n  [DESPUÉS]")
        get_onnx_info(yolo26n_fixed)

    # ---- Resumen ----
    print(f"\n{'=' * 60}")
    print(f"  RESUMEN")
    print(f"{'=' * 60}")
    for name, path in [("yolo11n_v1_best_fixed.onnx", yolo11n_fixed),
                       ("yolo26n_v1_best_fixed.onnx", yolo26n_fixed)]:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {name} ({size:.2f} MB)")
        else:
            print(f"  ✗ {name} — no generado")

    print(f"\n  Siguiente paso:")
    print(f"  Actualiza quantize_models.py para usar los modelos *_fixed.onnx")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
