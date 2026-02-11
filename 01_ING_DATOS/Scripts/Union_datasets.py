"""
Union_datasets.py
=================
Script para unificar múltiples datasets en formato YOLO en un único
Dataset Maestro. Las imágenes se serializan como img_000001, img_000002, ...
y los class_id de los labels se remapean a un mapeo unificado.

Flujo:
  1. Lee data.yaml de cada dataset → extrae clases locales.
  2. Normaliza nombres de clases (lowercase + singular + excepciones).
  3. Construye mapeo global unificado de clases.
  4. Mueve imágenes y reescribe labels con el nuevo class_id.
  5. Genera data.yaml unificado + log de trazabilidad + análisis de distribución.

Autor: TFM UNIR - Ingeniería de Datos
"""

import csv
import shutil
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

# ╔══════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURACIÓN EDITABLE                       ║
# ╚══════════════════════════════════════════════════════════════════╝

# Directorio base (donde están los datasets descomprimidos)
BASE_DIR = Path(__file__).parent

# Directorio de salida para el dataset unificado
OUTPUT_DIR = BASE_DIR / "unified"

# Lista de datasets a unificar (rutas relativas desde BASE_DIR).
# El usuario debe editar esta lista para incluir/excluir datasets.
DATASETS = [
    "Roboflow_dataset10-yolo26",
    "Roboflow_Dataset11-yolo26",
    "Roboflow_Dataset12-yolo26",
    "Roboflow_Dataset13-stair-yolo26",
    "Roboflow_Dataset14-stair-yolo26",
    "Roboflow_Dataset15-MyHome-yolo26",
    "Roboflow_Dataset8_IndoorObstacleDetection-yolo26",
    "Roboflow_Door-Detection-yolo26",
    "Roboflow_DoorDetection-yolo26",
    "Roboflow_DoorsOD-yolo26",
    "Roboflow_ObstacleUnique-yolo26",
    "Roboflow_obstacle-avoidnace-yolo26",
    "Roboflow_obstacle-door-person-yolo26",
    "Roboflow_obstacle-person2-yolo26",
    "Roboflow_obstacle-yolo26",
]

# Diccionario de EXCEPCIONES de normalización.
# Se aplica DESPUÉS de lowercase + singular automático.
# Formato: { "nombre_tal_como_viene_en_data.yaml" : "nombre_destino" }
# Añade aquí cualquier mapeo especial que la normalización automática no resuelva.
CLASS_OVERRIDE = {
    "escalator": "stair",
}

# Extensiones de imagen válidas
VALID_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       FUNCIONES AUXILIARES                      ║
# ╚══════════════════════════════════════════════════════════════════╝


def normalize_class_name(name: str) -> str:
    """
    Normaliza un nombre de clase:
      1. Aplica excepciones del diccionario CLASS_OVERRIDE (case-insensitive).
      2. Convierte a minúsculas.
      3. Elimina 's' final para pasar a singular (regla simple).

    Ejemplos:
      'Door'       → 'door'
      'obstacles'  → 'obstacle'
      'Obstacles'  → 'obstacle'
      'escalator'  → 'stair'  (por override)
      'person'     → 'person' (no termina en 's')
      'stair'      → 'stair'  (no termina en 's')
      'dog'        → 'dog'
    """
    # 1. Override exacto (case-insensitive)
    lower = name.lower().strip()
    if lower in CLASS_OVERRIDE:
        return CLASS_OVERRIDE[lower]

    # 2. Lowercase ya aplicado
    # 3. Singular: quitar 's' final solo si tiene más de 3 caracteres
    #    (evita romper palabras como 'bus', 'gas', etc.)
    if lower.endswith("s") and len(lower) > 3:
        lower = lower[:-1]

    return lower


def parse_data_yaml(yaml_path: Path) -> list[str]:
    """
    Lee data.yaml y retorna la lista de nombres de clases en orden (index = class_id).
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", [])
    if not names:
        raise ValueError(f"No se encontró 'names' en {yaml_path}")
    return names


def discover_splits(dataset_path: Path) -> list[Path]:
    """
    Descubre qué splits (train, valid, test) existen en un dataset.
    Retorna lista de paths a las carpetas de splits que existen.
    """
    splits = []
    for split_name in ["train", "valid", "test"]:
        split_dir = dataset_path / split_name
        if split_dir.is_dir():
            splits.append(split_dir)
    return splits


def build_global_class_map(
    datasets: list[str], base_dir: Path
) -> tuple[dict[str, int], dict[str, dict[int, int]]]:
    """
    Fase de descubrimiento: lee los data.yaml de todos los datasets,
    normaliza nombres y construye:
      - global_map: { nombre_normalizado: global_class_id }
      - translation_tables: { dataset_name: { local_id: global_id } }
    """
    # Recolectar todas las clases normalizadas (preservar orden de aparición)
    all_normalized: OrderedDict[str, None] = OrderedDict()
    dataset_local_classes: dict[str, list[str]] = {}

    for ds_name in datasets:
        yaml_path = base_dir / ds_name / "data.yaml"
        if not yaml_path.exists():
            print(f"  ⚠ ADVERTENCIA: No se encontró {yaml_path}. Saltando dataset.")
            continue

        local_names = parse_data_yaml(yaml_path)
        normalized = [normalize_class_name(n) for n in local_names]
        dataset_local_classes[ds_name] = normalized

        for n in normalized:
            all_normalized[n] = None  # OrderedDict para preservar orden

    # Asignar IDs globales
    global_map = {name: idx for idx, name in enumerate(all_normalized)}

    # Construir tablas de traducción
    translation_tables = {}
    for ds_name, normalized_names in dataset_local_classes.items():
        table = {}
        for local_id, norm_name in enumerate(normalized_names):
            table[local_id] = global_map[norm_name]
        translation_tables[ds_name] = table

    return global_map, translation_tables


def remap_label_line(line: str, translation: dict[int, int]) -> str | None:
    """
    Toma una línea de un archivo label YOLO y remapea el class_id.
    Retorna la línea con el nuevo class_id, o None si es inválida.

    Formato de entrada/salida: 'class_id x_center y_center width height'
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        local_id = int(parts[0])
    except ValueError:
        return None

    if local_id not in translation:
        return None

    parts[0] = str(translation[local_id])
    return " ".join(parts)


def print_class_map_summary(
    global_map: dict[str, int],
    translation_tables: dict[str, dict[int, int]],
    datasets: list[str],
    base_dir: Path,
):
    """Imprime un resumen del mapeo de clases para verificación del usuario."""
    print("\n" + "=" * 60)
    print("RESUMEN DE MAPEO DE CLASES")
    print("=" * 60)

    print("\n📋 Mapeo global unificado:")
    for name, idx in global_map.items():
        print(f"   {idx}: {name}")

    print(f"\n   Total de clases únicas: {len(global_map)}")

    print("\n📊 Traducción por dataset:")
    for ds_name in datasets:
        if ds_name not in translation_tables:
            continue
        yaml_path = base_dir / ds_name / "data.yaml"
        if not yaml_path.exists():
            continue
        original_names = parse_data_yaml(yaml_path)
        translation = translation_tables[ds_name]
        print(f"\n   {ds_name}:")
        for local_id, name in enumerate(original_names):
            global_id = translation[local_id]
            norm = normalize_class_name(name)
            if name != norm:
                print(f"      {local_id}:'{name}' → {global_id}:'{norm}'")
            else:
                print(f"      {local_id}:'{name}' → {global_id}:'{norm}'")

    print("\n" + "=" * 60)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      FUNCIÓN PRINCIPAL                          ║
# ╚══════════════════════════════════════════════════════════════════╝


def unify_datasets():
    """Ejecuta la unificación completa de datasets."""

    print("🔧 Unificación de Datasets YOLO")
    print("=" * 60)

    # ── Validar datasets ──────────────────────────────────────────
    valid_datasets = []
    for ds_name in DATASETS:
        ds_path = BASE_DIR / ds_name
        if not ds_path.is_dir():
            print(f"  ⚠ '{ds_name}' no existe. Saltando.")
        else:
            valid_datasets.append(ds_name)

    if not valid_datasets:
        print("❌ No se encontraron datasets válidos. Revisa la lista DATASETS.")
        sys.exit(1)

    print(f"\n✅ Datasets a procesar: {len(valid_datasets)}")
    for ds in valid_datasets:
        print(f"   • {ds}")

    # ── Fase 1: Descubrimiento y mapeo de clases ──────────────────
    print("\n📂 Fase 1: Descubrimiento de clases...")
    global_map, translation_tables = build_global_class_map(valid_datasets, BASE_DIR)
    print_class_map_summary(global_map, translation_tables, valid_datasets, BASE_DIR)

    # ── Fase 2: Crear estructura de destino ───────────────────────
    print("\n📁 Fase 2: Creando estructura de destino...")
    images_dir = OUTPUT_DIR / "data" / "images"
    labels_dir = OUTPUT_DIR / "data" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Destino: {OUTPUT_DIR.relative_to(BASE_DIR)}/")

    # ── Fase 3: Mover imágenes y reescribir labels ────────────────
    print("\n🚚 Fase 3: Procesando imágenes y labels...")
    counter = 0  # Contador global para serialización
    log_entries = []  # Para merge_log.csv
    class_counter = Counter()  # Para análisis de distribución
    warnings = []
    errors = []

    for ds_name in valid_datasets:
        ds_path = BASE_DIR / ds_name
        translation = translation_tables.get(ds_name)
        if translation is None:
            continue

        splits = discover_splits(ds_path)
        if not splits:
            warnings.append(f"{ds_name}: No se encontraron splits (train/valid/test)")
            continue

        ds_img_count = 0

        for split_dir in splits:
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"

            if not img_dir.is_dir():
                warnings.append(f"{ds_name}/{split_dir.name}: No existe carpeta 'images/'")
                continue

            # Obtener lista de imágenes ordenada (para reproducibilidad)
            img_files = sorted(
                [f for f in img_dir.iterdir() if f.suffix.lower() in VALID_IMG_EXTENSIONS]
            )

            for img_file in img_files:
                counter += 1
                new_basename = f"img_{counter:06d}"
                new_img_ext = img_file.suffix.lower()  # Normalizar extensión a minúsculas
                new_img_name = f"{new_basename}{new_img_ext}"
                new_lbl_name = f"{new_basename}.txt"

                # ── Mover imagen ──
                dst_img = images_dir / new_img_name
                try:
                    shutil.move(str(img_file), str(dst_img))
                except Exception as e:
                    errors.append(f"Error moviendo {img_file.name}: {e}")
                    counter -= 1  # Revertir contador
                    continue

                # ── Procesar label ──
                has_label = False
                original_lbl = lbl_dir / (img_file.stem + ".txt") if lbl_dir.is_dir() else None

                if original_lbl and original_lbl.exists():
                    try:
                        with open(original_lbl, "r", encoding="utf-8") as f:
                            lines = f.readlines()

                        new_lines = []
                        for line in lines:
                            if not line.strip():
                                continue
                            remapped = remap_label_line(line, translation)
                            if remapped is not None:
                                new_lines.append(remapped)
                                # Contar instancia de clase
                                cls_id = int(remapped.split()[0])
                                class_counter[cls_id] += 1
                            else:
                                errors.append(
                                    f"Línea inválida en {original_lbl.name}: '{line.strip()}'"
                                )

                        # Escribir label remapeado
                        dst_lbl = labels_dir / new_lbl_name
                        with open(dst_lbl, "w", encoding="utf-8") as f:
                            f.write("\n".join(new_lines))
                            if new_lines:
                                f.write("\n")
                        has_label = True

                        # Eliminar label original
                        original_lbl.unlink()

                    except Exception as e:
                        errors.append(f"Error procesando label {original_lbl.name}: {e}")
                else:
                    # Imagen sin label → crear label vacío
                    dst_lbl = labels_dir / new_lbl_name
                    dst_lbl.touch()
                    warnings.append(
                        f"{ds_name}/{split_dir.name}/{img_file.name}: Sin label → creado vacío"
                    )

                # ── Registrar en log ──
                log_entries.append(
                    {
                        "new_name": new_img_name,
                        "original_name": img_file.name,
                        "dataset": ds_name,
                        "split": split_dir.name,
                        "has_label": has_label,
                    }
                )
                ds_img_count += 1

        print(f"   ✓ {ds_name}: {ds_img_count} imágenes procesadas")

    print(f"\n   Total de imágenes unificadas: {counter}")

    # ── Fase 4: Generar data.yaml unificado ───────────────────────
    print("\n📝 Fase 4: Generando data.yaml unificado...")
    names_list = [name for name, _ in sorted(global_map.items(), key=lambda x: x[1])]
    data_yaml = {
        "path": "./data",
        "train": "images",
        "val": "images",
        "nc": len(names_list),
        "names": names_list,
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"   ✓ {yaml_path.relative_to(BASE_DIR)}")
    print(f"   Clases ({len(names_list)}): {names_list}")

    # ── Fase 5: Generar log de trazabilidad ───────────────────────
    print("\n📋 Fase 5: Generando log de trazabilidad...")
    log_path = OUTPUT_DIR / "merge_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["new_name", "original_name", "dataset", "split", "has_label"]
        )
        writer.writeheader()
        writer.writerows(log_entries)
    print(f"   ✓ {log_path.relative_to(BASE_DIR)} ({len(log_entries)} registros)")

    # ── Fase 6: Análisis de distribución de clases ────────────────
    print("\n📊 Fase 6: Análisis de distribución de clases...")
    total_instances = sum(class_counter.values())

    # Construir tabla
    print(f"\n{'Clase':<20} {'ID':>4} {'Instancias':>12} {'Porcentaje':>10}")
    print("-" * 50)
    for name in names_list:
        cls_id = global_map[name]
        count = class_counter.get(cls_id, 0)
        pct = (count / total_instances * 100) if total_instances > 0 else 0
        print(f"{name:<20} {cls_id:>4} {count:>12,} {pct:>9.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<20} {'':>4} {total_instances:>12,} {'100.0':>9}%")

    # Generar gráfica
    fig, ax = plt.subplots(figsize=(10, 6))
    class_names = names_list
    class_counts = [class_counter.get(global_map[n], 0) for n in class_names]
    colors = plt.cm.Set2.colors[: len(class_names)]

    bars = ax.bar(class_names, class_counts, color=colors, edgecolor="black", linewidth=0.5)

    # Añadir etiquetas de valor sobre las barras
    for bar, count in zip(bars, class_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(class_counts) * 0.01,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Clase", fontsize=12)
    ax.set_ylabel("Número de Instancias (Bounding Boxes)", fontsize=12)
    ax.set_title("Distribución de Clases - Dataset Maestro Unificado", fontsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n   ✓ Gráfica guardada: {chart_path.relative_to(BASE_DIR)}")

    # ── Resumen final ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ UNIFICACIÓN COMPLETADA")
    print("=" * 60)
    print(f"   📂 Directorio: {OUTPUT_DIR.relative_to(BASE_DIR)}/")
    print(f"   🖼  Imágenes:  {counter:,}")
    print(f"   🏷  Clases:    {len(names_list)} → {names_list}")
    print(f"   📦 BB totales: {total_instances:,}")

    if warnings:
        print(f"\n   ⚠ Advertencias: {len(warnings)}")
        for w in warnings[:10]:
            print(f"      - {w}")
        if len(warnings) > 10:
            print(f"      ... y {len(warnings) - 10} más")

    if errors:
        print(f"\n   ❌ Errores: {len(errors)}")
        for e in errors[:10]:
            print(f"      - {e}")
        if len(errors) > 10:
            print(f"      ... y {len(errors) - 10} más")

    print()


# ╔══════════════════════════════════════════════════════════════════╗
# ║                         EJECUCIÓN                               ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    unify_datasets()
