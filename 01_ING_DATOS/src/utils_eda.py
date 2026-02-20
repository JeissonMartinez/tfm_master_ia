"""
Utilidades para Análisis Exploratorio de Datos (EDA) de datasets COCO.

Este módulo contiene funciones para analizar datasets en formato COCO,
incluyendo visualización de distribuciones, análisis geométrico de bounding boxes,
y detección de anomalías en las anotaciones.

Todas las funciones de visualización y generación de reportes guardan automáticamente
los artefactos generados (PNG para figuras, CSV para tablas) en el directorio especificado.
"""

import os
import json
import csv
import shutil
import hashlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import random
import yaml
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import Counter, OrderedDict
from pycocotools.coco import COCO
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from pathlib import Path
import zipfile


# ============================================================================
# FASE 0: DESCOMPRIMIT ARCHIVO ZIP
# ============================================================================

def unzip_datasets(source_dir=None):
    """
    Descomprime todos los archivos .zip en el directorio especificado.
    
    Args:
        source_dir: Directorio donde se encuentran los archivos .zip.
                   Si es None, usa el directorio actual del script.
    """
    # Si no se especifica directorio, usar el directorio del script
    if source_dir is None:
        source_dir = Path(__file__).parent
    else:
        source_dir = Path(source_dir)
    
    # Buscar todos los archivos .zip
    zip_files = list(source_dir.glob("*.zip"))
    
    if not zip_files:
        print(f"No se encontraron archivos .zip en {source_dir}")
        return
    
    print(f"Se encontraron {len(zip_files)} archivos .zip")
    print("-" * 50)
    
    # Descomprimir cada archivo
    for zip_path in zip_files:
        # Crear nombre de carpeta de destino (sin la extensión .zip)
        output_dir = source_dir / zip_path.stem
        
        print(f"\nDescomprimiendo: {zip_path.name}")
        print(f"Destino: {output_dir.name}/")
        
        try:
            # Crear carpeta si no existe
            output_dir.mkdir(exist_ok=True)
            
            # Descomprimir
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Obtener número de archivos extraídos
            num_files = sum(1 for _ in output_dir.rglob('*') if _.is_file())
            print(f"✓ Completado - {num_files} archivos extraídos")
            
        except zipfile.BadZipFile:
            print(f"✗ Error: {zip_path.name} no es un archivo .zip válido")
        except PermissionError:
            print(f"✗ Error: Sin permisos para escribir en {output_dir}")
        except Exception as e:
            print(f"✗ Error inesperado: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Proceso completado")


# ============================================================================
# FASE 0B: PROCESAMIENTO DE DATASET YOLO (NORMALIZACIÓN + REESTRUCTURACIÓN)
# ============================================================================

# Extensiones de imagen válidas para procesamiento YOLO
_VALID_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_class_name(name: str, overrides: Optional[Dict[str, str]] = None) -> str:
    """
    Normaliza un nombre de clase YOLO:
      1. Convierte a minúsculas.
      2. Aplica excepciones del diccionario de overrides (case-insensitive).
      3. Elimina 's' final para pasar a singular (regla simple, solo si len > 3).

    Args:
        name: Nombre de clase original (e.g. 'Obstacles', 'escalator').
        overrides: Diccionario de excepciones de normalización.
                   Formato: {"nombre_lowercase": "nombre_destino"}.
                   Ejemplo: {"escalator": "stair"}.

    Returns:
        Nombre de clase normalizado.

    Examples:
        >>> normalize_class_name('Door')
        'door'
        >>> normalize_class_name('obstacles')
        'obstacle'
        >>> normalize_class_name('escalator', overrides={'escalator': 'stair'})
        'stair'
    """
    lower = name.lower().strip()

    # 1. Override exacto (case-insensitive)
    if overrides and lower in overrides:
        return overrides[lower]

    # 2. Singular: quitar 's' final solo si tiene más de 3 caracteres
    #    (evita romper palabras como 'bus', 'gas', etc.)
    if lower.endswith("s") and len(lower) > 3:
        lower = lower[:-1]

    return lower


def parse_data_yaml(yaml_path) -> list:
    """
    Lee un archivo data.yaml de un dataset YOLO y retorna la lista de
    nombres de clases en orden (index = class_id local).

    Args:
        yaml_path: Ruta al archivo data.yaml (str o Path).

    Returns:
        Lista de nombres de clases ['dog', 'door', 'obstacle', ...].

    Raises:
        ValueError: Si el archivo no contiene la clave 'names'.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", [])
    if not names:
        raise ValueError(f"No se encontró 'names' en {yaml_path}")
    return names


def discover_splits(dataset_path) -> list:
    """
    Descubre qué splits (train, valid, test) existen en un dataset YOLO.

    Args:
        dataset_path: Ruta raíz del dataset (str o Path).

    Returns:
        Lista de Paths a las carpetas de splits que existen.
    """
    dataset_path = Path(dataset_path)
    splits = []
    for split_name in ["train", "valid", "test"]:
        split_dir = dataset_path / split_name
        if split_dir.is_dir():
            splits.append(split_dir)
    return splits


def remap_label_line(line: str, translation: Dict[int, int]) -> Optional[str]:
    """
    Toma una línea de un archivo label YOLO y remapea el class_id
    según una tabla de traducción.

    Args:
        line: Línea del archivo .txt en formato 'class_id x_center y_center width height'.
        translation: Diccionario {local_class_id: global_class_id}.

    Returns:
        Línea con el class_id remapeado, o None si la línea es inválida.
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


def build_class_map(
    dataset_path,
    class_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, int], Dict[int, int]]:
    """
    Lee el data.yaml de un dataset YOLO, normaliza los nombres de clases
    y construye el mapeo global unificado.

    Args:
        dataset_path: Ruta raíz del dataset que contiene data.yaml.
        class_overrides: Diccionario de excepciones de normalización.
                         Ejemplo: {"escalator": "stair"}.

    Returns:
        Tupla (global_map, translation):
          - global_map: Dict[str, int]  →  {nombre_normalizado: global_class_id}
          - translation: Dict[int, int] →  {local_class_id: global_class_id}
    """
    dataset_path = Path(dataset_path)
    yaml_path = dataset_path / "data.yaml"

    local_names = parse_data_yaml(yaml_path)
    normalized = [normalize_class_name(n, overrides=class_overrides) for n in local_names]

    # Construir mapeo global preservando orden de aparición
    all_normalized: OrderedDict = OrderedDict()
    for n in normalized:
        all_normalized[n] = None

    global_map = {name: idx for idx, name in enumerate(all_normalized)}

    # Tabla de traducción local_id → global_id
    translation = {}
    for local_id, norm_name in enumerate(normalized):
        translation[local_id] = global_map[norm_name]

    return global_map, translation


def process_yolo_dataset(
    dataset_path,
    output_dir,
    class_overrides: Optional[Dict[str, str]] = None,
    output_dir_reports: Optional[str] = None,
) -> Dict:
    """
    Procesa un dataset YOLO: normaliza clases, reestructura archivos y genera
    la estructura unificada data/images + data/labels + data.yaml compatible
    con load_yolo_as_coco().

    Flujo:
      1. Lee data.yaml → extrae clases locales.
      2. Normaliza nombres de clases (lowercase + singular + excepciones).
      3. Construye mapeo global unificado de clases.
      4. Mueve imágenes y reescribe labels con el nuevo class_id.
      5. Genera data.yaml unificado + log de trazabilidad + distribución.

    Args:
        dataset_path: Ruta al dataset YOLO descomprimido (con data.yaml,
                      train/images, train/labels, etc.).
        output_dir:   Ruta de destino donde se creará la estructura
                      data/images + data/labels + data.yaml.
        class_overrides: Diccionario de excepciones de normalización.
                         Ejemplo: {"escalator": "stair"}.
        output_dir_reports: Directorio para guardar gráficas de distribución.
                            Si es None, se guarda en output_dir.

    Returns:
        Diccionario con estadísticas del procesamiento:
          - total_images: int
          - total_annotations: int
          - class_distribution: Counter
          - global_map: Dict[str, int]
          - warnings: list[str]
          - errors: list[str]
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    report_dir = Path(output_dir_reports) if output_dir_reports else output_dir

    print("🔧 Procesamiento de Dataset YOLO")
    print("=" * 60)
    print(f"   📂 Origen:  {dataset_path}")
    print(f"   📂 Destino: {output_dir}")

    # ── Fase 1: Descubrimiento y mapeo de clases ──────────────────
    print("\n📂 Fase 1: Descubrimiento de clases...")
    global_map, translation = build_class_map(dataset_path, class_overrides)

    # Imprimir resumen del mapeo
    print("\n   📋 Mapeo global unificado:")
    for name, idx in global_map.items():
        print(f"      {idx}: {name}")
    print(f"\n   Total de clases únicas: {len(global_map)}")

    # Mostrar traducción detallada
    original_names = parse_data_yaml(dataset_path / "data.yaml")
    print("\n   📊 Traducción de clases:")
    for local_id, name in enumerate(original_names):
        global_id = translation[local_id]
        norm = normalize_class_name(name, overrides=class_overrides)
        arrow = " (sin cambio)" if name.lower() == norm and local_id == global_id else ""
        print(f"      {local_id}:'{name}' → {global_id}:'{norm}'{arrow}")

    # ── Fase 2: Crear estructura de destino ───────────────────────
    print("\n📁 Fase 2: Creando estructura de destino...")
    images_dir = output_dir / "data" / "images"
    labels_dir = output_dir / "data" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ {images_dir}")
    print(f"   ✅ {labels_dir}")

    # ── Fase 3: Mover imágenes y reescribir labels ────────────────
    print("\n🚚 Fase 3: Procesando imágenes y labels...")
    counter = 0
    log_entries = []
    class_counter = Counter()
    warnings = []
    errors = []

    splits = discover_splits(dataset_path)
    if not splits:
        msg = f"No se encontraron splits (train/valid/test) en {dataset_path}"
        print(f"   ⚠️ {msg}")
        warnings.append(msg)
    else:
        print(f"   Splits encontrados: {[s.name for s in splits]}")

    for split_dir in splits:
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.is_dir():
            msg = f"{split_dir.name}: No existe carpeta 'images/'"
            warnings.append(msg)
            continue

        # Obtener lista de imágenes ordenada (para reproducibilidad)
        img_files = sorted(
            [f for f in img_dir.iterdir() if f.suffix.lower() in _VALID_IMG_EXTENSIONS]
        )

        ds_img_count = 0
        for img_file in img_files:
            counter += 1
            new_basename = f"img_{counter:06d}"
            new_img_ext = img_file.suffix.lower()
            new_img_name = f"{new_basename}{new_img_ext}"
            new_lbl_name = f"{new_basename}.txt"

            # ── Mover imagen ──
            dst_img = images_dir / new_img_name
            try:
                shutil.move(str(img_file), str(dst_img))
            except Exception as e:
                errors.append(f"Error moviendo {img_file.name}: {e}")
                counter -= 1
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
                    f"{split_dir.name}/{img_file.name}: Sin label → creado vacío"
                )

            # ── Registrar en log ──
            log_entries.append({
                "new_name": new_img_name,
                "original_name": img_file.name,
                "split": split_dir.name,
                "has_label": has_label,
            })
            ds_img_count += 1

        print(f"   ✅ {split_dir.name}: {ds_img_count} imágenes procesadas")

    print(f"\n   📊 Total de imágenes procesadas: {counter}")

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
    yaml_out = output_dir / "data.yaml"
    with open(yaml_out, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"   ✅ {yaml_out}")
    print(f"   Clases ({len(names_list)}): {names_list}")

    # ── Fase 5: Generar log de trazabilidad ───────────────────────
    print("\n📋 Fase 5: Generando log de trazabilidad...")
    log_path = output_dir / "process_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["new_name", "original_name", "split", "has_label"]
        )
        writer.writeheader()
        writer.writerows(log_entries)
    print(f"   ✅ {log_path} ({len(log_entries)} registros)")

    # ── Fase 6: Análisis de distribución de clases ────────────────
    print("\n📊 Fase 6: Distribución de clases...")
    total_instances = sum(class_counter.values())

    print(f"\n   {'Clase':<20} {'ID':>4} {'Instancias':>12} {'Porcentaje':>10}")
    print("   " + "-" * 50)
    for name in names_list:
        cls_id = global_map[name]
        count = class_counter.get(cls_id, 0)
        pct = (count / total_instances * 100) if total_instances > 0 else 0
        print(f"   {name:<20} {cls_id:>4} {count:>12,} {pct:>9.1f}%")
    print("   " + "-" * 50)
    print(f"   {'TOTAL':<20} {'':>4} {total_instances:>12,} {'100.0':>9}%")

    # Generar gráfica de distribución
    fig, ax = plt.subplots(figsize=(10, 6))
    class_counts = [class_counter.get(global_map[n], 0) for n in names_list]
    colors = sns.color_palette("magma", n_colors=len(names_list))

    bars = ax.bar(names_list, class_counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, class_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(class_counts) * 0.01,
            f"{count:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Clase", fontsize=12)
    ax.set_ylabel("Número de Instancias (Bounding Boxes)", fontsize=12)
    ax.set_title("Distribución de Clases - Dataset Procesado", fontsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    os.makedirs(str(report_dir), exist_ok=True)
    chart_path = report_dir / "class_distribution.png"
    plt.savefig(str(chart_path), dpi=150)
    plt.show()
    plt.close()
    print(f"\n   💾 Gráfica guardada: {chart_path}")

    # ── Resumen final ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"   📂 Directorio: {output_dir}")
    print(f"   🖼  Imágenes:  {counter:,}")
    print(f"   🏷  Clases:    {len(names_list)} → {names_list}")
    print(f"   📦 BB totales: {total_instances:,}")

    if warnings:
        print(f"\n   ⚠️ Advertencias: {len(warnings)}")
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

    return {
        "total_images": counter,
        "total_annotations": total_instances,
        "class_distribution": class_counter,
        "global_map": global_map,
        "warnings": warnings,
        "errors": errors,
    }


def ensure_dir(directory: str) -> None:
    """
    Crea un directorio si no existe.
    
    Args:
        directory: Ruta del directorio a crear
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Directorio creado: {directory}")


# ============================================================================
# FASE 1: ANÁLISIS DE DISTRIBUCIÓN DE CLASES
# ============================================================================

def get_category_distribution(coco) -> pd.DataFrame:
    """
    Obtiene la distribución de categorías y sus conteos.
    
    Args:
        coco: Objeto COCO con las anotaciones cargadas
        
    Returns:
        DataFrame con columnas: Category, Count, Category_ID
    """
    cats = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in cats]
    cat_ids = [cat['id'] for cat in cats]
    
    cat_counts = []
    for cat_id in cat_ids:
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        count = len(ann_ids)
        cat_counts.append(count)
    
    df_counts = pd.DataFrame({
        'Category': cat_names,
        'Count': cat_counts,
        'Category_ID': cat_ids
    }).sort_values('Count', ascending=False)
    
    return df_counts


def plot_category_distribution(df_counts: pd.DataFrame,
                               scale_type: str = 'linear',
                               palette: str = 'magma',
                               figsize: Tuple[int, int] = (12, 8),
                               output_dir: Optional[str] = None,
                               filename: str = 'category_distribution',
                               title: str = 'Frecuencia de Clases') -> None:
    """
    Grafica la distribución de categorías en un gráfico de barras horizontal.
    
    Args:
        df_counts: DataFrame con la distribución de categorías
        scale_type: 'linear' o 'log' para la escala del eje X
        palette: Paleta de colores de seaborn
        figsize: Tamaño de la figura
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título del gráfico
    """
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df_counts, y='Category', x='Count', 
                     palette=palette, hue='Category', legend=False)
    plt.title(title)
    plt.xlabel('Número de Instancias (Bounding Boxes)')
    plt.ylabel('Clase (Obstáculo)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Añadir valores al lado de cada barra
    for i, v in enumerate(df_counts['Count']):
        ax.text(v, i, f' {v}', va='center', fontsize=10)
    
    if scale_type == 'log':
        plt.xscale('log')
        plt.xlabel('Número de Instancias (Bounding Boxes) - Escala Logarítmica')
    
    plt.tight_layout()
    
    # Guardar figura si se especifica directorio
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")
    
    plt.show()
    
    # Imprimir y guardar tabla resumen
    print("\n--- Tabla de Distribución ---")
    print(df_counts[['Category', 'Count']].to_markdown(index=False))
    
    if output_dir:
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df_counts[['Category', 'Count']].to_csv(csv_path, index=False)
        print(f"💾 Tabla guardada: {csv_path}")


def calculate_image_object_density(coco, cat_ids: List[int]) -> Tuple[pd.DataFrame, List[int]]:
    """
    Calcula la densidad de imágenes vs objetos por categoría.
    
    Args:
        coco: Objeto COCO con las anotaciones cargadas
        cat_ids: Lista de IDs de categorías a analizar
        
    Returns:
        Tupla con (DataFrame con columnas: Category, Images, Objects; lista de IDs activos)
    """
    img_counts = []
    obj_counts = []
    cat_labels = []
    active_cat_ids = []
    
    for cat_id in cat_ids:
        img_ids = coco.getImgIds(catIds=[cat_id])
        n_imgs = len(img_ids)
        
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        n_objs = len(ann_ids)
        
        if n_objs > 0:
            cat_info = coco.loadCats([cat_id])[0]
            cat_labels.append(cat_info['name'])
            img_counts.append(n_imgs)
            obj_counts.append(n_objs)
            active_cat_ids.append(cat_id)
    
    df_density = pd.DataFrame({
        'Category': cat_labels,
        'Images': img_counts,
        'Objects': obj_counts
    })
    
    return df_density, active_cat_ids


def plot_image_object_density(df_density: pd.DataFrame,
                              palette: str = 'magma',
                              figsize: Tuple[int, int] = (12, 6),
                              output_dir: Optional[str] = None,
                              filename: str = 'image_object_density',
                              title: str = 'Densidad: Cantidad de Imágenes vs. Cantidad de Objetos') -> None:
    """
    Grafica la densidad de imágenes vs objetos.
    
    Args:
        df_density: DataFrame con la densidad calculada
        palette: Paleta de colores
        figsize: Tamaño de la figura
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título del gráfico
    """
    df_melted = df_density.melt(id_vars='Category', var_name='Metric', value_name='Count')
    
    plt.figure(figsize=figsize)
    sns.barplot(data=df_melted, x='Category', y='Count', hue='Metric', palette=palette)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")    
    plt.show()


def calculate_and_plot_density_ratio(df_density: pd.DataFrame,
                                     palette: str = 'magma',
                                     figsize: Tuple[int, int] = (10, 6),
                                     output_dir: Optional[str] = None,
                                     filename: str = 'density_ratio',
                                     title: str = 'Ratio de Densidad: Promedio de Objetos por Imagen') -> pd.DataFrame:
    """
    Calcula el ratio Objetos/Imagen por categoría y genera gráfico de barras horizontal.

    El ratio indica el promedio de objetos por imagen donde aparece cada clase.
    Un ratio > 1 significa que esa clase tiende a aparecer agrupada (varios objetos
    por imagen), mientras que un ratio ≈ 1 indica que normalmente aparece un solo
    objeto por imagen.

    Args:
        df_density: DataFrame con columnas Category, Images, Objects
                    (salida de calculate_image_object_density)
        palette: Paleta de colores de seaborn
        figsize: Tamaño de la figura
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)

    Returns:
        DataFrame con columnas Category, Images, Objects, Ratio ordenado
        descendentemente por Ratio
    """
    df = df_density.copy()
    df['Ratio'] = (df['Objects'] / df['Images']).round(4)
    df_sorted = df.sort_values('Ratio', ascending=False).reset_index(drop=True)

    # ── Tabla resumen ──────────────────────────────────────────────
    print("\n--- Ratio de Densidad (Promedio de objetos por imagen donde aparece la clase) ---")
    print(df_sorted[['Category', 'Images', 'Objects', 'Ratio']].to_markdown(index=False))

    # ── Gráfico de barras horizontal ───────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette(palette, n_colors=len(df_sorted))
    bars = ax.barh(df_sorted['Category'], df_sorted['Ratio'], color=colors,
                   edgecolor='black', linewidth=0.5)

    # Línea de referencia ratio = 1
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.2,
               label='Ratio = 1 (1 obj/img)')

    # Etiquetas de valor en cada barra
    for bar, ratio in zip(bars, df_sorted['Ratio']):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{ratio:.2f}', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Ratio (Objetos / Imágenes)', fontsize=12)
    ax.set_ylabel('Categoría', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Categoría con mayor ratio arriba
    plt.tight_layout()

    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")

        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df_sorted[['Category', 'Images', 'Objects', 'Ratio']].to_csv(csv_path, index=False)
        print(f"💾 Tabla guardada: {csv_path}")

    plt.show()

    return df_sorted


def calculate_cooccurrence_matrix(coco, active_cat_ids: List[int]) -> Tuple[np.ndarray, List[str]]:
    """
    Calcula la matriz de co-ocurrencia entre categorías.
    
    Args:
        coco: Objeto COCO con las anotaciones cargadas
        active_cat_ids: Lista de IDs de categorías activas
        
    Returns:
        Tupla con (matriz de co-ocurrencia, lista de nombres de categorías)
    """
    id_to_idx = {cat_id: i for i, cat_id in enumerate(active_cat_ids)}
    num_classes = len(active_cat_ids)
    co_occurrence_matrix = np.zeros((num_classes, num_classes))
    
    cat_labels = [coco.loadCats([cat_id])[0]['name'] for cat_id in active_cat_ids]
    
    all_img_ids = coco.getImgIds()
    
    for img_id in all_img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        present_cat_ids = set([ann['category_id'] for ann in anns 
                               if ann['category_id'] in id_to_idx])
        
        for id1 in present_cat_ids:
            for id2 in present_cat_ids:
                idx1 = id_to_idx[id1]
                idx2 = id_to_idx[id2]
                co_occurrence_matrix[idx1, idx2] += 1
    
    return co_occurrence_matrix, cat_labels


def plot_cooccurrence_matrix(co_occurrence_matrix: np.ndarray,
                             cat_labels: List[str],
                             cmap: str = 'magma',
                             figsize: Tuple[int, int] = (10, 8),
                             output_dir: Optional[str] = None,
                             filename: str = 'cooccurrence_matrix',
                             title: str = 'Matriz de Co-ocurrencia (Frecuencia conjunta en imágenes)') -> None:
    """
    Grafica la matriz de co-ocurrencia como heatmap.
    
    Args:
        co_occurrence_matrix: Matriz de co-ocurrencia
        cat_labels: Lista de nombres de categorías
        cmap: Colormap para el heatmap
        figsize: Tamaño de la figura
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título del gráfico
    """
    plt.figure(figsize=figsize)
    sns.heatmap(co_occurrence_matrix, annot=True, fmt='g', cmap=cmap,
                xticklabels=cat_labels, yticklabels=cat_labels)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")
        
        # Guardar matriz como CSV
        df_matrix = pd.DataFrame(co_occurrence_matrix, 
                                index=cat_labels, 
                                columns=cat_labels)
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df_matrix.to_csv(csv_path)
        print(f"💾 Matriz guardada: {csv_path}")
    
    plt.show()


# ============================================================================
# FASE 2: ANÁLISIS GEOMÉTRICO DE BOUNDING BOXES
# ============================================================================

def extract_bbox_geometry(coco, target_cat_ids: List[int],
                          output_dir: Optional[str] = None,
                          filename: str = 'bbox_geometry') -> pd.DataFrame:
    """
    Extrae información geométrica de los bounding boxes.
    
    Args:
        coco: Objeto COCO con las anotaciones cargadas
        target_cat_ids: Lista de IDs de categorías objetivo
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        
    Returns:
        DataFrame con geometría de cada bbox (área, ratio, centro, etc.)
    """
    id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(target_cat_ids)}
    
    areas = []
    ratios = []
    centers_x = []
    centers_y = []
    categories = []
    sizes_coco = []
    
    for cat_id in target_cat_ids:
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            
            if w <= 0 or h <= 0:
                continue
            
            # Área y clasificación COCO
            area = w * h
            if area < 32**2:
                size_label = 'Small (<32²)'
            elif area < 96**2:
                size_label = 'Medium'
            else:
                size_label = 'Large (>96²)'
            
            # Aspect Ratio
            ratio = w / h
            
            # Posición normalizada
            img_info = coco.loadImgs(ann['image_id'])[0]
            img_w, img_h = img_info['width'], img_info['height']
            cx = (x + w/2) / img_w
            cy = (y + h/2) / img_h
            
            areas.append(area)
            ratios.append(ratio)
            centers_x.append(cx)
            centers_y.append(cy)
            categories.append(id_to_name[cat_id])
            sizes_coco.append(size_label)
    
    df_geo = pd.DataFrame({
        'Category': categories,
        'Area': areas,
        'Ratio': ratios,
        'Size_COCO': sizes_coco,
        'Center_X': centers_x,
        'Center_Y': centers_y
    })

    # Imprimir resumen
    print(f"\n📐 Geometría de Bounding Boxes extraída: {len(df_geo)} anotaciones")
    summary = df_geo.groupby('Category').agg(
        Count=('Area', 'size'),
        Area_mean=('Area', 'mean'),
        Area_median=('Area', 'median'),
        Ratio_mean=('Ratio', 'mean'),
        Ratio_std=('Ratio', 'std')
    ).round(2)
    print(summary.to_markdown())

    if output_dir:
        ensure_dir(output_dir)
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df_geo.to_csv(csv_path, index=False)
        print(f"💾 Geometría guardada: {csv_path}")

        csv_summary_path = os.path.join(output_dir, f'{filename}_summary.csv')
        summary.to_csv(csv_summary_path)
        print(f"💾 Resumen guardado: {csv_summary_path}")
    
    return df_geo


def plot_size_distribution(df_geo: pd.DataFrame,
                           colormap: str = 'viridis',
                           figsize: Tuple[int, int] = (10, 6),
                           output_dir: Optional[str] = None,
                           filename: str = 'size_distribution',
                           title: str = 'Distribución de Tamaños (Estándar COCO)') -> None:
    """
    Grafica la distribución de tamaños COCO por categoría.
    
    Args:
        df_geo: DataFrame con geometría de bboxes
        colormap: Colormap para el gráfico
        figsize: Tamaño de la figura
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título del gráfico
    """
    size_counts = df_geo.groupby(['Category', 'Size_COCO']).size().unstack(fill_value=0)
    
    plt.figure(figsize=figsize)
    size_counts.plot(kind='bar', stacked=True, colormap=colormap)
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Cantidad de Objetos')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend(title='Size_COCO')
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")
    
    plt.show()
    
    print("\n--- Porcentaje de Tamaños por Clase ---")
    size_pct = size_counts.div(size_counts.sum(axis=1), axis=0) * 100
    print(size_pct.round(2).to_markdown())
    
    if output_dir:
        csv_path = os.path.join(output_dir, f'{filename}_percentages.csv')
        size_pct.round(2).to_csv(csv_path)
        print(f"💾 Tabla guardada: {csv_path}")


def plot_aspect_ratio(df_geo: pd.DataFrame,
                      palette: str = 'magma',
                      figsize: Tuple[int, int] = (10, 8),
                      output_dir: Optional[str] = None,
                      filename: str = 'aspect_ratio',
                      title: str = 'Relación de Aspecto (Ancho / Alto)') -> None:
    """
    Grafica la relación de aspecto (ancho/alto) por categoría.
    
    Args:
        df_geo: DataFrame con geometría de bboxes
        palette: Paleta de colores
        figsize: Tamaño de la figura
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título del gráfico
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=df_geo, x='Category', y='Ratio', palette=palette)
    plt.title(title)
    plt.axhline(1, color='r', linestyle='--', label='Cuadrado (1:1)')
    plt.ylim(0, 5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")
    
    plt.show()


def plot_spatial_heatmap(df_geo: pd.DataFrame,
                        cmap: str = 'magma',
                        figsize: Tuple[int, int] = (8, 8),
                        output_dir: Optional[str] = None,
                        filename: str = 'spatial_heatmap',
                        title: str = 'Mapa de Calor (Posición de Objetos)') -> None:
    """
    Grafica un mapa de calor de la posición de los objetos.
    
    Args:
        df_geo: DataFrame con geometría de bboxes
        cmap: Colormap para el heatmap
        figsize: Tamaño de la figura
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título del gráfico
    """
    plt.figure(figsize=figsize)
    h = plt.hist2d(df_geo['Center_X'], 1 - df_geo['Center_Y'], bins=30, cmap=cmap)
    plt.title(title)
    plt.xlabel('Posición X (Normalizada)')
    plt.ylabel('Posición Y (Normalizada)')
    plt.gca().set_aspect('equal')
    plt.colorbar(h[3], label='Frecuencia')
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")
    
    plt.show()


# ============================================================================
# FASE 3: COMPLEJIDAD DE LA ESCENA
# ============================================================================

def calculate_density_and_iou(coco, target_cat_ids: List[int]) -> Tuple[List[int], List[float]]:
    """
    Calcula densidad de objetos por imagen e IoU entre bboxes vecinos.
    
    Args:
        coco: Objeto COCO con las anotaciones cargadas
        target_cat_ids: Lista de IDs de categorías objetivo
        
    Returns:
        Tupla con (lista de densidades, lista de IoUs)
    """
    img_ids = coco.getImgIds()
    density_counts = []
    ious = []
    
    print("Procesando imágenes para calcular IoU y Densidad...")
    
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=target_cat_ids)
        anns = coco.loadAnns(ann_ids)
        
        count = len(anns)
        density_counts.append(count)
        
        if count > 1:
            boxes = [ann['bbox'] for ann in anns]
            boxes_xyxy = []
            for b in boxes:
                boxes_xyxy.append([b[0], b[1], b[0]+b[2], b[1]+b[3]])
            
            boxes_np = np.array(boxes_xyxy)
            areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
            
            for i in range(count):
                boxA = boxes_np[i]
                max_iou_for_this_box = 0
                
                for j in range(count):
                    if i == j:
                        continue
                    
                    boxB = boxes_np[j]
                    
                    xA = max(boxA[0], boxB[0])
                    yA = max(boxA[1], boxB[1])
                    xB = min(boxA[2], boxB[2])
                    yB = min(boxA[3], boxB[3])
                    
                    interWidth = max(0, xB - xA)
                    interHeight = max(0, yB - yA)
                    interArea = interWidth * interHeight
                    
                    unionArea = areas[i] + areas[j] - interArea
                    iou = interArea / unionArea if unionArea > 0 else 0
                    
                    if iou > max_iou_for_this_box:
                        max_iou_for_this_box = iou
                
                ious.append(max_iou_for_this_box)
    
    return density_counts, ious


def plot_density_histogram(density_counts: List[int],
                           color: str = 'orange',
                           yscale: str = 'linear',
                           figsize: Tuple[int, int] = (10, 6),
                           output_dir: Optional[str] = None,
                           filename: str = 'density_histogram',
                           title: str = 'Densidad de Escena (Objetos por Imagen)') -> None:
    """
    Grafica histograma de densidad de escena.
    
    Args:
        density_counts: Lista con conteo de objetos por imagen
        color: Color de las barras
        yscale: Escala del eje Y ('linear' o 'log')
        figsize: Tamaño de la figura
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título del gráfico
    """
    plt.figure(figsize=figsize)
    sns.histplot(density_counts, bins=range(0, max(density_counts)+2), color=color, kde=False)
    plt.title(title)
    plt.xlabel('Cantidad de Objetos')
    plt.ylabel('Cantidad de Imágenes')
    plt.axvline(float(np.mean(density_counts)), color='red', linestyle='--', 
                label=f'Promedio: {np.mean(density_counts):.1f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.yscale(yscale)
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")
    
    plt.show()


def plot_iou_histogram(ious: List[float],
                       color: str = 'purple',
                       yscale: str = 'linear',
                       figsize: Tuple[int, int] = (10, 6),
                       output_dir: Optional[str] = None,
                       filename: str = 'iou_histogram',
                       title: str = 'Oclusión (IoU entre cajas vecinas)') -> None:
    """
    Grafica histograma de oclusión (IoU).
    
    Args:
        ious: Lista de valores de IoU
        color: Color de las barras
        yscale: Escala del eje Y ('linear' o 'log')
        figsize: Tamaño de la figura
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título del gráfico
    """
    plt.figure(figsize=figsize)
    sns.histplot(ious, bins=30, color=color, kde=True)
    plt.title(title)
    plt.xlabel('Intersection over Union (IoU)')
    plt.ylabel('Frecuencia')
    plt.axvline(0.5, color='red', linestyle='--', label='Umbral Crítico (0.5)')
    plt.text(0.55, plt.gca().get_ylim()[1]*0.8, 'Alta Oclusión ->', color='red', fontsize=10)
    plt.xlim(0, max(1.0, max(ious) * 1.1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.yscale(yscale)
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")
    
    plt.show()


def print_density_stats(img_ids: List[int], density_counts: List[int],
                       output_dir: Optional[str] = None,
                       filename: str = 'density_stats') -> None:
    """
    Imprime estadísticas de densidad.
    
    Args:
        img_ids: Lista de IDs de imágenes
        density_counts: Lista con conteo de objetos por imagen
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
    """
    stats = {
        'Total Imágenes Analizadas': len(img_ids),
        'Imágenes Vacías (Background)': density_counts.count(0),
        'Promedio de objetos por imagen': round(np.mean(density_counts), 2),
        'Máximo de objetos en una imagen': max(density_counts)
    }
    
    print(f"Total Imágenes Analizadas: {stats['Total Imágenes Analizadas']}")
    print(f"Imágenes Vacías (Background): {stats['Imágenes Vacías (Background)']}")
    print(f"Promedio de objetos por imagen: {stats['Promedio de objetos por imagen']:.2f}")
    print(f"Máximo de objetos en una imagen: {stats['Máximo de objetos en una imagen']}")
    
    if output_dir:
        ensure_dir(output_dir)
        df_stats = pd.DataFrame([stats]).T
        df_stats.columns = ['Valor']
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df_stats.to_csv(csv_path)
        print(f"💾 Estadísticas guardadas: {csv_path}")


# ============================================================================
# FASE 4: INSPECCIÓN VISUAL Y DETECCIÓN DE ANOMALÍAS
# ============================================================================

def visualize_sample(coco, target_cat_ids: List[int], id_to_name: Dict[int, str],
                    class_colors: Dict[str, Tuple[int, int, int]], img_dir: str,
                    num_samples: int = 3, img_id: Optional[int] = None,
                    img_ids: Optional[List[int]] = None,
                    output_dir: Optional[str] = None,
                    filename: str = 'sample_visualization',
                    title: Optional[str] = None) -> None:
    """
    Visualiza imágenes con sus anotaciones en una grilla de 4 columnas.
    
    Args:
        coco: Objeto COCO con las anotaciones cargadas
        target_cat_ids: Lista de IDs de categorías objetivo
        id_to_name: Diccionario de mapeo ID -> nombre de categoría
        class_colors: Diccionario de colores BGR por clase
        img_dir: Directorio donde están las imágenes
        num_samples: Número de imágenes aleatorias a mostrar
        img_id: ID específico de imagen (opcional, para compatibilidad)
        img_ids: Lista de IDs específicos de imágenes (opcional)
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título general de la figura (None = sin título)
    """
    # Filtrar imágenes
    target_img_ids = set()
    for cat_id in target_cat_ids:
        target_img_ids.update(coco.getImgIds(catIds=[cat_id]))
    target_img_ids = list(target_img_ids)
    
    # Determinar qué imágenes mostrar (prioridad: img_ids > img_id > random)
    if img_ids is not None:
        random_ids = img_ids
        num_samples = len(img_ids)
    elif img_id is not None:
        random_ids = [img_id]
        num_samples = 1
    else:
        n_classes = len(id_to_name)
        if num_samples >= n_classes:
            # Garantizar al menos 1 imagen por clase
            random_ids = []
            used_ids = set()
            for cat_id in target_cat_ids:
                cat_imgs = coco.getImgIds(catIds=[cat_id])
                available = [i for i in cat_imgs if i in target_img_ids and i not in used_ids]
                if available:
                    chosen = random.choice(available)
                    random_ids.append(chosen)
                    used_ids.add(chosen)
            # Rellenar slots restantes al azar (sin repetir)
            remaining_pool = [i for i in target_img_ids if i not in used_ids]
            extra = num_samples - len(random_ids)
            if extra > 0 and remaining_pool:
                random_ids.extend(random.sample(remaining_pool, min(extra, len(remaining_pool))))
            random.shuffle(random_ids)
        else:
            random_ids = random.sample(target_img_ids, num_samples)
    
    # Configuración de la grilla: 4 columnas
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    
    # Asegurar que axes sea siempre un array 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    for i, img_id in enumerate(random_ids):
        ax = axes_flat[i]
        
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo cargar: {img_path}")
            ax.axis('off')
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=target_cat_ids)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            x, y, w, h = [int(v) for v in ann['bbox']]
            cat_name = id_to_name[ann['category_id']]
            color = class_colors.get(cat_name, (255, 255, 255))
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            text_size, _ = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - 20), (x + text_size[0], y), color, -1)
            cv2.putText(img, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"ID: {img_id} - {img_info['file_name']}", fontsize=8)
    
    # Ocultar ejes sobrantes
    for i in range(num_samples, len(axes_flat)):
        axes_flat[i].axis('off')
    
    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")
    
    plt.show()


def detect_anomalies(coco, target_cat_ids: List[int], id_to_name: Dict[int, str],
                     min_area: float = 225, max_ratio: float = 0.95,
                     output_dir: Optional[str] = None,
                     filename: str = 'anomalies') -> Tuple[List[Dict], List[Dict]]:
    """
    Detecta anomalías en las anotaciones (cajas minúsculas y gigantes).
    
    Args:
        coco: Objeto COCO con las anotaciones cargadas
        target_cat_ids: Lista de IDs de categorías objetivo
        id_to_name: Diccionario de mapeo ID -> nombre de categoría
        min_area: Área mínima para considerar una caja como válida
        max_ratio: Ratio máximo del área de la imagen que puede ocupar una caja
        output_dir: Directorio donde guardar los archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        
    Returns:
        Tupla con (lista de cajas minúsculas, lista de cajas gigantes)
    """
    print("--- Iniciando búsqueda de anomalías ---")
    
    suspect_tiny = []
    suspect_huge = []
    
    for cat_id in target_cat_ids:
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            area = w * h
            
            img_info = coco.loadImgs(ann['image_id'])[0]
            img_area = img_info['width'] * img_info['height']
            
            if area < min_area:
                suspect_tiny.append({
                    'img_id': ann['image_id'],
                    'category': id_to_name[cat_id],
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'area': area
                })
            
            if area > (img_area * max_ratio):
                suspect_huge.append({
                    'img_id': ann['image_id'],
                    'category': id_to_name[cat_id],
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'ratio': area/img_area
                })
    
    print(f"\nResultados del escaneo:")
    print(f"⚠️ Cajas Minúsculas (<15x15px): {len(suspect_tiny)}")
    if len(suspect_tiny) > 0:
        print(pd.DataFrame(suspect_tiny).head(5).to_markdown())
    
    print(f"\n⚠️ Cajas Gigantes (>{max_ratio*100}% imagen): {len(suspect_huge)}")
    if len(suspect_huge) > 0:
        print(pd.DataFrame(suspect_huge).head(5).to_markdown())
    
    # Guardar resultados
    if output_dir:
        ensure_dir(output_dir)
        if len(suspect_tiny) > 0:
            df_tiny = pd.DataFrame(suspect_tiny)
            csv_path = os.path.join(output_dir, f'{filename}_tiny.csv')
            df_tiny.to_csv(csv_path, index=False)
            print(f"💾 Cajas minúsculas guardadas: {csv_path}")
        
        if len(suspect_huge) > 0:
            df_huge = pd.DataFrame(suspect_huge)
            csv_path = os.path.join(output_dir, f'{filename}_huge.csv')
            df_huge.to_csv(csv_path, index=False)
            print(f"💾 Cajas gigantes guardadas: {csv_path}")
    
    return suspect_tiny, suspect_huge


# ═══════════════════════════════════════════════════════════════════
# FASE 1: ANÁLISIS TÉCNICO DEL DATASET
# ═══════════════════════════════════════════════════════════════════

def load_yolo_as_coco(dataset_path: str,
                      output_json: Optional[str] = None) -> Tuple['COCO', str]:
    """
    Carga un dataset YOLO (data.yaml + data/images + data/labels) y lo convierte
    a un objeto COCO de pycocotools para reutilizar todas las funciones de análisis.

    Args:
        dataset_path: Ruta al directorio raíz del dataset YOLO
                      (debe contener data.yaml y data/images, data/labels).
        output_json:  Ruta donde guardar el JSON COCO generado.
                      Si es None se guarda como _annotations.coco.json
                      dentro del propio dataset_path.

    Returns:
        Tupla (coco_obj, img_dir) donde img_dir es la ruta absoluta al
        directorio de imágenes.
    """
    # ── 1. Leer data.yaml ──────────────────────────────────────────
    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml['names']

    # ── 2. Directorios de imágenes y labels ────────────────────────
    img_dir = os.path.join(dataset_path, "data", "images")
    label_dir = os.path.join(dataset_path, "data", "labels")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    img_files = sorted([f for f in os.listdir(img_dir)
                        if os.path.splitext(f)[1].lower() in valid_extensions])

    # ── 3. Construir estructura COCO ───────────────────────────────
    coco_dict = {"images": [], "annotations": [], "categories": []}

    for i, name in enumerate(class_names):
        coco_dict["categories"].append({
            "id": i, "name": name, "supercategory": "none"
        })

    ann_id = 1
    for img_idx, img_file in enumerate(img_files, start=1):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        coco_dict["images"].append({
            "id": img_idx,
            "file_name": img_file,
            "width": img_w,
            "height": img_h
        })

        # Buscar label correspondiente
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))

                # YOLO normalizado → COCO absoluto
                bbox_w = w * img_w
                bbox_h = h * img_h
                bbox_x = (cx - w / 2) * img_w
                bbox_y = (cy - h / 2) * img_h
                area = bbox_w * bbox_h

                coco_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": class_id,
                    "bbox": [round(bbox_x, 2), round(bbox_y, 2),
                             round(bbox_w, 2), round(bbox_h, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0
                })
                ann_id += 1

    # ── 4. Guardar JSON y cargar con pycocotools ───────────────────
    if output_json is None:
        output_json = os.path.join(dataset_path, "_annotations.coco.json")

    with open(output_json, 'w') as f:
        json.dump(coco_dict, f)

    coco = COCO(output_json)

    print(f"✅ Dataset YOLO cargado como COCO:")
    print(f"   📂 Imágenes: {len(coco_dict['images'])}")
    print(f"   📝 Anotaciones: {len(coco_dict['annotations'])}")
    print(f"   🏷️  Categorías: {[c['name'] for c in coco_dict['categories']]}")

    return coco, img_dir


def load_all_splits(train_path: str, valid_path: str, test_path: str) -> Dict[str, COCO]:
    """
    Carga los 3 splits del dataset (train/valid/test) como objetos COCO.
    
    Args:
        train_path: Ruta al archivo _annotations.coco.json de train
        valid_path: Ruta al archivo _annotations.coco.json de valid
        test_path: Ruta al archivo _annotations.coco.json de test
        
    Returns:
        Diccionario con formato {'train': coco_obj, 'valid': coco_obj, 'test': coco_obj}
    """
    splits = {}
    for split_name, path in [('train', train_path), ('valid', valid_path), ('test', test_path)]:
        try:
            splits[split_name] = COCO(path)
            print(f"✅ {split_name.upper()}: {len(splits[split_name].getImgIds())} imágenes cargadas")
        except Exception as e:
            print(f"❌ Error al cargar {split_name}: {e}")
            splits[split_name] = None
    
    return splits


def analyze_split_distribution(splits: Dict[str, COCO],
                               output_dir: Optional[str] = None,
                               filename: str = 'split_distribution') -> pd.DataFrame:
    """
    Analiza la distribución de imágenes y anotaciones por split.
    
    Args:
        splits: Diccionario con objetos COCO por split
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        
    Returns:
        DataFrame con columnas: Split, Images, Annotations, Annotations/Image
    """
    data = []
    for split_name, coco_obj in splits.items():
        if coco_obj is None:
            continue
        
        n_images = len(coco_obj.getImgIds())
        n_annotations = len(coco_obj.getAnnIds())
        ratio = n_annotations / n_images if n_images > 0 else 0
        
        data.append({
            'Split': split_name,
            'Images': n_images,
            'Annotations': n_annotations,
            'Annotations/Image': round(ratio, 2)
        })
    
    df = pd.DataFrame(data)
    
    print("\n📊 Distribución del Dataset por Split:")
    print(df.to_markdown(index=False))
    
    if output_dir:
        ensure_dir(output_dir)
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Tabla guardada: {csv_path}")
    
    return df


def get_image_dimensions(splits: Dict[str, COCO]) -> pd.DataFrame:
    """
    Extrae las dimensiones (ancho x alto) de todas las imágenes en todos los splits.
    
    Args:
        splits: Diccionario con objetos COCO por split
        
    Returns:
        DataFrame con columnas: image_id, width, height, split
    """
    data = []
    for split_name, coco_obj in splits.items():
        if coco_obj is None:
            continue
        
        for img_id in coco_obj.getImgIds():
            img_info = coco_obj.loadImgs(img_id)[0]
            data.append({
                'image_id': img_id,
                'width': img_info['width'],
                'height': img_info['height'],
                'split': split_name
            })
    
    return pd.DataFrame(data)


def plot_resolution_scatter(df_dims: pd.DataFrame,
                            output_dir: Optional[str] = None,
                            filename: str = 'resolution_scatter',
                            title: str = 'Distribución de Resoluciones por Split') -> None:
    """
    Grafica un scatter plot de ancho vs alto, coloreado por split.
    
    Args:
        df_dims: DataFrame con columnas width, height, split
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título del gráfico
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for split in df_dims['split'].unique():
        subset = df_dims[df_dims['split'] == split]
        ax.scatter(subset['width'], subset['height'], label=split, alpha=0.6, s=50)
    
    ax.set_xlabel('Ancho (píxeles)', fontsize=12)
    ax.set_ylabel('Alto (píxeles)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Split')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Agregar estadísticas
    unique_resolutions = df_dims.groupby('split').apply(
        lambda x: len(x.groupby(['width', 'height']))
    )
    stats_text = "\n".join([f"{split}: {count} resoluciones únicas" 
                            for split, count in unique_resolutions.items()])
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)
    
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figura guardada: {fig_path}")
        
        # Guardar tabla de resoluciones únicas
        resolution_table = df_dims.groupby(['split', 'width', 'height']).size().reset_index(name='count')
        csv_path = os.path.join(output_dir, f'{filename}_table.csv')
        resolution_table.to_csv(csv_path, index=False)
        print(f"💾 Tabla de resoluciones guardada: {csv_path}")
    
    plt.show()


def check_file_integrity(splits: Dict[str, COCO], base_dir: str,
                         img_dir_map: Optional[Dict[str, str]] = None,
                         output_dir: Optional[str] = None,
                         filename: str = 'integrity_check') -> pd.DataFrame:
    """
    Verifica la integridad de archivos de imagen (corruptos, 0 bytes, formatos).
    
    Args:
        splits: Diccionario con objetos COCO por split
        base_dir: Directorio base donde están los datasets
        img_dir_map: Diccionario {split_name: ruta_img_dir} para rutas personalizadas.
                     Si es None usa la ruta por defecto del dataset COCO original.
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        
    Returns:
        DataFrame con archivos problemáticos
    """
    issues = []
    
    for split_name, coco_obj in splits.items():
        if coco_obj is None:
            continue
        
        # Construir path del directorio de imágenes según la estructura
        if img_dir_map and split_name in img_dir_map:
            img_dir = img_dir_map[split_name]
        else:
            img_dir = os.path.join(base_dir, "Datasets", "TFM_Dataset.v1-v1_2026-02-06_5-48pm.coco", 
                                   split_name)
        
        for img_id in coco_obj.getImgIds():
            img_info = coco_obj.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            # Verificar existencia
            if not os.path.exists(img_path):
                issues.append({
                    'split': split_name,
                    'image_id': img_id,
                    'filename': img_info['file_name'],
                    'issue': 'FILE_NOT_FOUND'
                })
                continue
            
            # Verificar tamaño
            file_size = os.path.getsize(img_path)
            if file_size == 0:
                issues.append({
                    'split': split_name,
                    'image_id': img_id,
                    'filename': img_info['file_name'],
                    'issue': 'ZERO_BYTES'
                })
                continue
            
            # Verificar si se puede abrir
            img = cv2.imread(img_path)
            if img is None:
                issues.append({
                    'split': split_name,
                    'image_id': img_id,
                    'filename': img_info['file_name'],
                    'issue': 'CORRUPTED_FILE'
                })
                continue
            
            # Verificar formato de color
            if len(img.shape) == 2:
                issues.append({
                    'split': split_name,
                    'image_id': img_id,
                    'filename': img_info['file_name'],
                    'issue': 'GRAYSCALE_IMAGE'
                })
            elif img.shape[2] == 4:
                issues.append({
                    'split': split_name,
                    'image_id': img_id,
                    'filename': img_info['file_name'],
                    'issue': 'RGBA_IMAGE'
                })
    
    df = pd.DataFrame(issues)
    
    if len(df) == 0:
        print("\n✅ Verificación de Integridad: SIN PROBLEMAS")
        print(f"   Todas las imágenes son válidas y legibles (RGB)")
    else:
        print(f"\n⚠️  Verificación de Integridad: {len(df)} problema(s) encontrado(s)")
        print(df.to_markdown(index=False))
    
    if output_dir and len(df) > 0:
        ensure_dir(output_dir)
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Reporte de problemas guardado: {csv_path}")
    
    return df


def detect_duplicates_across_splits(splits: Dict[str, COCO], base_dir: str,
                                     img_dir_map: Optional[Dict[str, str]] = None,
                                     output_dir: Optional[str] = None,
                                     filename: str = 'duplicates') -> pd.DataFrame:
    """
    Detecta imágenes duplicadas entre splits usando hash MD5.
    
    Args:
        splits: Diccionario con objetos COCO por split
        base_dir: Directorio base donde están los datasets
        img_dir_map: Diccionario {split_name: ruta_img_dir} para rutas personalizadas.
                     Si es None usa la ruta por defecto del dataset COCO original.
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        
    Returns:
        DataFrame con duplicados encontrados
    """
    hash_dict = {}  # {md5_hash: [(split, img_id, filename), ...]}
    
    print("\n🔍 Calculando hashes MD5 de todas las imágenes...")
    
    for split_name, coco_obj in splits.items():
        if coco_obj is None:
            continue
        
        if img_dir_map and split_name in img_dir_map:
            img_dir = img_dir_map[split_name]
        else:
            img_dir = os.path.join(base_dir, "Datasets", "TFM_Dataset.v1-v1_2026-02-06_5-48pm.coco", 
                                   split_name)
        
        for img_id in coco_obj.getImgIds():
            img_info = coco_obj.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            if not os.path.exists(img_path):
                continue
            
            # Calcular MD5 hash
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash not in hash_dict:
                hash_dict[file_hash] = []
            hash_dict[file_hash].append((split_name, img_id, img_info['file_name']))
    
    # Encontrar duplicados (hashes con más de 1 archivo)
    duplicates = []
    for file_hash, occurrences in hash_dict.items():
        if len(occurrences) > 1:
            for split, img_id, filename in occurrences:
                duplicates.append({
                    'hash': file_hash,
                    'split': split,
                    'image_id': img_id,
                    'filename': filename,
                    'duplicate_count': len(occurrences)
                })
    
    df = pd.DataFrame(duplicates)
    
    if len(df) == 0:
        print("✅ No se encontraron duplicados entre splits")
    else:
        print(f"⚠️  Se encontraron {len(df)} imágenes duplicadas:")
        print(df.groupby('hash')['split'].apply(list).to_markdown())
    
    if output_dir and len(df) > 0:
        ensure_dir(output_dir)
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Reporte de duplicados guardado: {csv_path}")
    
    return df


def calculate_pixel_statistics(splits: Dict[str, COCO], base_dir: str,
                                sample_size: int = 100,
                                img_dir_map: Optional[Dict[str, str]] = None,
                                output_dir: Optional[str] = None,
                                filename: str = 'channel_stats') -> pd.DataFrame:
    """
    Calcula estadísticas de píxeles (mean/std) por canal RGB para normalización.
    
    Args:
        splits: Diccionario con objetos COCO por split
        base_dir: Directorio base donde están los datasets
        sample_size: Número de imágenes a muestrear por split (None = todas)
        img_dir_map: Diccionario {split_name: ruta_img_dir} para rutas personalizadas.
                     Si es None usa la ruta por defecto del dataset COCO original.
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        
    Returns:
        DataFrame con mean y std por canal y split
    """
    stats_data = []
    
    print("\n📊 Calculando estadísticas de canales RGB...")
    
    for split_name, coco_obj in splits.items():
        if coco_obj is None:
            continue
        
        if img_dir_map and split_name in img_dir_map:
            img_dir = img_dir_map[split_name]
        else:
            img_dir = os.path.join(base_dir, "Datasets", "TFM_Dataset.v1-v1_2026-02-06_5-48pm.coco", 
                                   split_name)
        
        img_ids = coco_obj.getImgIds()
        if sample_size and len(img_ids) > sample_size:
            img_ids = random.sample(img_ids, sample_size)
        
        r_pixels, g_pixels, b_pixels = [], [], []
        
        for img_id in img_ids:
            img_info = coco_obj.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # cv2 lee en BGR, convertir a RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalizar a [0, 1]
            img_normalized = img_rgb / 255.0
            
            r_pixels.extend(img_normalized[:, :, 0].flatten())
            g_pixels.extend(img_normalized[:, :, 1].flatten())
            b_pixels.extend(img_normalized[:, :, 2].flatten())
        
        # Calcular estadísticas
        r_mean, r_std = np.mean(r_pixels), np.std(r_pixels)
        g_mean, g_std = np.mean(g_pixels), np.std(g_pixels)
        b_mean, b_std = np.mean(b_pixels), np.std(b_pixels)
        
        stats_data.append({
            'split': split_name,
            'R_mean': round(r_mean, 4),
            'R_std': round(r_std, 4),
            'G_mean': round(g_mean, 4),
            'G_std': round(g_std, 4),
            'B_mean': round(b_mean, 4),
            'B_std': round(b_std, 4),
            'sample_size': len(img_ids)
        })
        
        print(f"   {split_name.upper()}: RGB mean=[{r_mean:.4f}, {g_mean:.4f}, {b_mean:.4f}], "
              f"std=[{r_std:.4f}, {g_std:.4f}, {b_std:.4f}]")
    
    df = pd.DataFrame(stats_data)
    
    if output_dir:
        ensure_dir(output_dir)
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Estadísticas guardadas: {csv_path}")
    
    return df


def plot_channel_histograms(splits: Dict[str, COCO], base_dir: str,
                            sample_size: int = 100,
                            img_dir_map: Optional[Dict[str, str]] = None,
                            output_dir: Optional[str] = None,
                            filename: str = 'rgb_histograms',
                            title: str = 'Distribución de Intensidad por Canal RGB') -> None:
    """
    Grafica histogramas de intensidad RGB superpuestos por split.
    
    Args:
        splits: Diccionario con objetos COCO por split
        base_dir: Directorio base donde están los datasets
        sample_size: Número de imágenes a muestrear por split
        img_dir_map: Diccionario {split_name: ruta_img_dir} para rutas personalizadas.
                     Si es None usa la ruta por defecto del dataset COCO original.
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título general de la figura
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    channel_names = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    
    split_pixels = {split: {0: [], 1: [], 2: []} for split in splits.keys() if splits[split] is not None}
    
    print("\n🎨 Generando histogramas de distribución de píxeles...")
    
    for split_name, coco_obj in splits.items():
        if coco_obj is None:
            continue
        
        if img_dir_map and split_name in img_dir_map:
            img_dir = img_dir_map[split_name]
        else:
            img_dir = os.path.join(base_dir, "Datasets", "TFM_Dataset.v1-v1_2026-02-06_5-48pm.coco", 
                                   split_name)
        
        img_ids = coco_obj.getImgIds()
        if sample_size and len(img_ids) > sample_size:
            img_ids = random.sample(img_ids, sample_size)
        
        for img_id in img_ids:
            img_info = coco_obj.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for channel in range(3):
                split_pixels[split_name][channel].extend(img_rgb[:, :, channel].flatten())
    
    # Graficar histogramas
    for channel, ax in enumerate(axes):
        for split_name in split_pixels.keys():
            if len(split_pixels[split_name][channel]) > 0:
                ax.hist(split_pixels[split_name][channel], bins=50, alpha=0.5, 
                       label=split_name, density=True)
        
        ax.set_xlabel('Intensidad del Píxel (0-255)', fontsize=11)
        ax.set_ylabel('Densidad', fontsize=11)
        ax.set_title(f'Canal {channel_names[channel]}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Histogramas guardados: {fig_path}")
    
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# FASE 2: DISTRIBUCIÓN DE CLASES (MULTI-SPLIT)
# ═══════════════════════════════════════════════════════════════════

def compare_class_distribution_splits(splits: Dict[str, COCO],
                                      output_dir: Optional[str] = None,
                                      filename: str = 'class_comparison_splits') -> pd.DataFrame:
    """
    Compara la distribución de clases entre train/valid/test con gráfico de barras agrupadas.
    
    Args:
        splits: Diccionario con objetos COCO por split
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        
    Returns:
        DataFrame con conteo de instancias por clase y split
    """
    # Construir DataFrame con todas las clases y splits
    all_data = []
    
    for split_name, coco_obj in splits.items():
        if coco_obj is None:
            continue
        
        categories = coco_obj.loadCats(coco_obj.getCatIds())
        
        for cat in categories:
            ann_ids = coco_obj.getAnnIds(catIds=[cat['id']])
            count = len(ann_ids)
            
            all_data.append({
                'Category': cat['name'],
                'Split': split_name,
                'Count': count
            })
    
    df = pd.DataFrame(all_data)
    
    # Pivot para visualización
    df_pivot = df.pivot(index='Category', columns='Split', values='Count').fillna(0)
    
    # Graficar
    fig, ax = plt.subplots(figsize=(14, 8))
    df_pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Categoría', fontsize=12)
    ax.set_ylabel('Número de Instancias', fontsize=12)
    ax.set_title('Comparación de Distribución de Clases por Split', fontsize=14, fontweight='bold')
    ax.legend(title='Split', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico de comparación guardado: {fig_path}")
        
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Tabla de distribución guardada: {csv_path}")
    
    plt.show()
    
    print("\n📊 Distribución de Clases por Split:")
    print(df_pivot.to_markdown())
    
    return df


def chi_square_split_test(df_distribution: pd.DataFrame,
                          output_dir: Optional[str] = None,
                          filename: str = 'chi_square_test') -> Optional[Dict]:
    """
    Realiza test Chi-cuadrado para verificar homogeneidad de distribución entre splits.
    
    Args:
        df_distribution: DataFrame con columnas Category, Split, Count
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        
    Returns:
        Diccionario con resultados del test o None si no hay suficientes datos válidos
    """
    # Crear tabla de contingencia
    contingency_table = df_distribution.pivot(index='Category', columns='Split', values='Count').fillna(0)
    
    # Filtrar categorías que tengan 0 en algún split (no válidas para Chi²)
    categories_with_zeros = (contingency_table == 0).any(axis=1)
    if categories_with_zeros.any():
        excluded_categories = contingency_table[categories_with_zeros].index.tolist()
        print(f"\n⚠️  Categorías excluidas del test (tienen 0 en algún split): {excluded_categories}")
        contingency_table = contingency_table[~categories_with_zeros]
    
    # Verificar que queden suficientes categorías
    if len(contingency_table) < 2:
        print("\n❌ ERROR: No hay suficientes categorías válidas para el test Chi-cuadrado")
        return None
    
    # Realizar test Chi-cuadrado
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    result = {
        'chi2_statistic': round(float(chi2), 4),
        'p_value': round(float(p_value), 6),
        'degrees_of_freedom': int(dof),
        'is_homogeneous': float(p_value) > 0.05  # Nivel de significancia alfa=0.05
    }
    
    print("\n🧪 Test Chi-Cuadrado de Homogeneidad entre Splits:")
    print(f"   Chi² estatístico: {result['chi2_statistic']}")
    print(f"   p-value: {result['p_value']}")
    print(f"   Grados de libertad: {result['degrees_of_freedom']}")
    
    if result['is_homogeneous']:
        print("   ✅ RESULTADO: Las distribuciones son homogéneas (p > 0.05)")
        print("      Los splits tienen distribuciones estadísticamente similares.")
    else:
        print("   ⚠️  RESULTADO: Las distribuciones NO son homogéneas (p ≤ 0.05)")
        print("      Existe sesgo significativo entre los splits.")
    
    if output_dir:
        ensure_dir(output_dir)
        result_df = pd.DataFrame([result])
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        result_df.to_csv(csv_path, index=False)
        print(f"💾 Resultados del test guardados: {csv_path}")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# FASE 3: OPTIMIZACIÓN DE ANCHORS
# ═══════════════════════════════════════════════════════════════════

def extract_bbox_dimensions(coco, target_cat_ids: Optional[List[int]] = None) -> np.ndarray:
    """
    Extrae dimensiones (ancho, alto) normalizadas de todos los bboxes.
    
    Args:
        coco: Objeto COCO con las anotaciones cargadas
        target_cat_ids: Lista de IDs de categorías (None = todas)
        
    Returns:
        Array numpy de forma (N, 2) con [width_norm, height_norm] en rango [0, 1]
    """
    dimensions = []
    
    if target_cat_ids is None:
        target_cat_ids = coco.getCatIds()
    
    for cat_id in target_cat_ids:
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            img_info = coco.loadImgs(ann['image_id'])[0]
            img_width = img_info['width']
            img_height = img_info['height']
            
            x, y, w, h = ann['bbox']
            
            # Normalizar a [0, 1]
            w_norm = w / img_width
            h_norm = h / img_height
            
            dimensions.append([w_norm, h_norm])
    
    return np.array(dimensions)


def kmeans_anchor_optimization(bbox_dims: np.ndarray, n_anchors: int = 9,
                               random_state: int = 42) -> np.ndarray:
    """
    Calcula anchors óptimos usando K-Means clustering sobre dimensiones de bboxes.
    
    Args:
        bbox_dims: Array (N, 2) con dimensiones normalizadas [width, height]
        n_anchors: Número de anchors a generar (típicamente 6-9 para YOLO)
        random_state: Semilla para reproducibilidad
        
    Returns:
        Array (n_anchors, 2) con centroides ordenados por área
    """
    print(f"\n🔬 Ejecutando K-Means con {n_anchors} clusters sobre {len(bbox_dims)} bboxes...")
    
    kmeans = KMeans(n_clusters=n_anchors, random_state=random_state, n_init=10)
    kmeans.fit(bbox_dims)
    
    # Obtener centroides y ordenar por área (ancho * alto)
    centroids = kmeans.cluster_centers_
    areas = centroids[:, 0] * centroids[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = centroids[sorted_indices]
    
    print("✅ Anchors optimizados calculados")
    
    return anchors


def plot_anchor_clusters(bbox_dims: np.ndarray, anchors: np.ndarray,
                         output_dir: Optional[str] = None,
                         filename: str = 'anchor_clusters',
                         title: Optional[str] = None) -> None:
    """
    Visualiza bboxes y anchors optimizados en un scatter plot.
    
    Args:
        bbox_dims: Array (N, 2) con dimensiones normalizadas
        anchors: Array (n_anchors, 2) con centroides
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título del gráfico (None = título por defecto con nro. de anchors)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter de todos los bboxes
    ax.scatter(bbox_dims[:, 0], bbox_dims[:, 1], alpha=0.3, s=10, c='blue', label='Bounding Boxes')
    
    # Marcar anchors
    ax.scatter(anchors[:, 0], anchors[:, 1], s=200, c='red', marker='X', 
              edgecolors='black', linewidths=2, label='Anchors Optimizados', zorder=5)
    
    # Anotar cada anchor
    for i, (w, h) in enumerate(anchors):
        ax.annotate(f'A{i+1}\n({w:.3f},{h:.3f})', 
                   xy=(w, h), xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Ancho Normalizado', fontsize=12)
    ax.set_ylabel('Alto Normalizado', fontsize=12)
    ax.set_title(title or f'Clustering de Bounding Boxes ({len(anchors)} Anchors)', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualización de anchors guardada: {fig_path}")
    
    plt.show()


def export_yolo_anchors(anchors: np.ndarray, img_size: int = 640,
                       output_dir: Optional[str] = None,
                       filename: str = 'yolo_anchors') -> None:
    """
    Exporta anchors en formato YOLO (píxeles absolutos) como archivo YAML.
    
    Args:
        anchors: Array (n_anchors, 2) con dimensiones normalizadas
        img_size: Tamaño de imagen objetivo para YOLO (típicamente 640)
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
    """
    # Convertir a píxeles absolutos
    anchors_pixels = (anchors * img_size).astype(int)
    
    # Organizar en 3 escalas (típico de YOLOv5/v8)
    n_anchors = len(anchors_pixels)
    anchors_per_scale = n_anchors // 3
    
    anchor_config = {
        'anchors': [
            anchors_pixels[:anchors_per_scale].tolist(),      # P3/8 (pequeños)
            anchors_pixels[anchors_per_scale:2*anchors_per_scale].tolist(),  # P4/16 (medianos)
            anchors_pixels[2*anchors_per_scale:].tolist()     # P5/32 (grandes)
        ]
    }
    
    print("\n⚙️  Configuración de Anchors para YOLO:")
    print(f"   Imagen: {img_size}x{img_size}")
    print(f"   Anchors P3/8 (pequeños): {anchor_config['anchors'][0]}")
    print(f"   Anchors P4/16 (medianos): {anchor_config['anchors'][1]}")
    print(f"   Anchors P5/32 (grandes): {anchor_config['anchors'][2]}")
    
    if output_dir:
        ensure_dir(output_dir)
        yaml_path = os.path.join(output_dir, f'{filename}.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(anchor_config, f, default_flow_style=False)
        print(f"💾 Configuración YAML guardada: {yaml_path}")
        print(f"   Usar en modelo: copiar anchors a tu archivo de configuración YOLO")


# ═══════════════════════════════════════════════════════════════════
# FASE 6: DATA AUGMENTATION (ANÁLISIS Y PLANIFICACIÓN)
# ═══════════════════════════════════════════════════════════════════

def analyze_augmentation_needs(coco, target_instances: int = 400,
                               output_dir: Optional[str] = None,
                               filename: str = 'augmentation_analysis') -> pd.DataFrame:
    """
    Identifica clases minoritarias que necesitan Data Augmentation.
    
    Args:
        coco: Objeto COCO con las anotaciones cargadas
        target_instances: Número objetivo de instancias por clase
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        
    Returns:
        DataFrame con análisis de déficit por clase
    """
    categories = coco.loadCats(coco.getCatIds())
    analysis = []
    
    for cat in categories:
        ann_ids = coco.getAnnIds(catIds=[cat['id']])
        current_count = len(ann_ids)
        deficit = max(0, target_instances - current_count)
        
        # Contar imágenes únicas que contienen esta clase
        img_ids_with_class = set(coco.getImgIds(catIds=[cat['id']]))
        n_images = len(img_ids_with_class)
        
        status = "✅ Balanceado" if deficit == 0 else "⚠️  Minoritaria"
        
        analysis.append({
            'Category': cat['name'],
            'Current_Instances': current_count,
            'Target_Instances': target_instances,
            'Deficit': deficit,
            'Images_with_Class': n_images,
            'Avg_Instances_per_Image': round(current_count / n_images, 2) if n_images > 0 else 0,
            'Status': status
        })
    
    df = pd.DataFrame(analysis).sort_values('Deficit', ascending=False)
    
    print("\n📊 Análisis de Necesidades de Augmentation:")
    print(df.to_markdown(index=False))
    
    minority_classes = df[df['Deficit'] > 0]
    if len(minority_classes) > 0:
        print(f"\n⚠️  {len(minority_classes)} clase(s) minoritaria(s) requieren augmentation:")
        for _, row in minority_classes.iterrows():
            print(f"   • {row['Category']}: {row['Deficit']} instancias faltantes")
    else:
        print("\n✅ Todas las clases están balanceadas")
    
    if output_dir:
        ensure_dir(output_dir)
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Análisis guardado: {csv_path}")
    
    return df


def compute_augmentation_strategy(df_analysis: pd.DataFrame,
                                  output_dir: Optional[str] = None,
                                  filename: str = 'augmentation_plan') -> pd.DataFrame:
    """
    Calcula estrategia de augmentation (copias necesarias por imagen).
    
    Args:
        df_analysis: DataFrame de analyze_augmentation_needs
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        
    Returns:
        DataFrame con plan de augmentation
    """
    minority_classes = df_analysis[df_analysis['Deficit'] > 0].copy()
    
    if len(minority_classes) == 0:
        print("\n✅ No se requiere augmentation")
        return pd.DataFrame()
    
    strategy = []
    
    for _, row in minority_classes.iterrows():
        deficit = row['Deficit']
        n_images = row['Images_with_Class']
        avg_inst = row['Avg_Instances_per_Image']
        
        # Calcular copias necesarias
        if avg_inst > 0:
            total_new_images = int(np.ceil(deficit / avg_inst))
            copies_per_image = int(np.ceil(total_new_images / n_images)) if n_images > 0 else 0
        else:
            copies_per_image = 0
        
        expected_new_instances = copies_per_image * n_images * avg_inst
        
        strategy.append({
            'Category': row['Category'],
            'Deficit': deficit,
            'Candidate_Images': n_images,
            'Copies_per_Image': copies_per_image,
            'Expected_New_Instances': int(expected_new_instances),
            'Strategy': 'Mild' if deficit < 100 else 'Aggressive'
        })
    
    df_strategy = pd.DataFrame(strategy)
    
    print("\n📋 Plan de Augmentation:")
    print(df_strategy.to_markdown(index=False))
    
    total_new_images = (df_strategy['Copies_per_Image'] * df_strategy['Candidate_Images']).sum()
    print(f"\n📦 Total de imágenes a generar: ~{int(total_new_images)}")
    
    if output_dir:
        ensure_dir(output_dir)
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        df_strategy.to_csv(csv_path, index=False)
        print(f"💾 Plan de augmentation guardado: {csv_path}")
    
    return df_strategy


def visualize_augmentation_impact(df_analysis: pd.DataFrame, df_strategy: pd.DataFrame,
                                  output_dir: Optional[str] = None,
                                  filename: str = 'augmentation_impact',
                                  title: str = 'Impacto Proyectado del Data Augmentation') -> None:
    """
    Visualiza el impacto esperado del augmentation (ANTES vs DESPUÉS proyectado).
    
    Args:
        df_analysis: DataFrame de analyze_augmentation_needs
        df_strategy: DataFrame de compute_augmentation_strategy
        output_dir: Directorio donde guardar archivos (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        title: Título general de la figura
    """
    # Preparar datos
    categories = df_analysis['Category'].tolist()
    before = df_analysis['Current_Instances'].tolist()
    
    # Calcular proyección DESPUÉS
    after = []
    for cat in categories:
        current = df_analysis[df_analysis['Category'] == cat]['Current_Instances'].values[0]
        
        if cat in df_strategy['Category'].values:
            boost = df_strategy[df_strategy['Category'] == cat]['Expected_New_Instances'].values[0]
            after.append(current + boost)
        else:
            after.append(current)
    
    # Graficar
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico ANTES
    ax1 = axes[0]
    colors_before = ['#e74c3c' if df_analysis.iloc[i]['Deficit'] > 0 else '#3498db' 
                     for i in range(len(categories))]
    bars1 = ax1.bar(categories, before, color=colors_before, edgecolor='black', linewidth=0.5)
    
    for bar, count in zip(bars1, before):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(before) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    target = df_analysis['Target_Instances'].iloc[0]
    ax1.axhline(y=target, color='green', linestyle='--', alpha=0.7, label=f'Objetivo ({target:,})')
    ax1.set_title('ANTES de Augmentation', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Instancias (Bounding Boxes)', fontsize=11)
    ax1.set_xlabel('Categoría', fontsize=11)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Gráfico DESPUÉS
    ax2 = axes[1]
    colors_after = ['#2ecc71' if df_analysis.iloc[i]['Deficit'] > 0 else '#3498db' 
                    for i in range(len(categories))]
    bars2 = ax2.bar(categories, after, color=colors_after, edgecolor='black', linewidth=0.5)
    
    for bar, count in zip(bars2, after):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(after) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.axhline(y=target, color='green', linestyle='--', alpha=0.7, label=f'Objetivo ({target:,})')
    ax2.set_title('DESPUÉS de Augmentation (Proyectado)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Instancias (Bounding Boxes)', fontsize=11)
    ax2.set_xlabel('Categoría', fontsize=11)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualización de impacto guardada: {fig_path}")
    
    plt.show()


# ============================================================================
# Funciones de análisis para datasets en formato YOLO
# ============================================================================

def load_yolo_class_names(yaml_path: str) -> List[str]:
    """
    Lee los nombres de clase desde un archivo data.yaml de YOLO.

    Args:
        yaml_path: Ruta al archivo data.yaml

    Returns:
        Lista de nombres de clase en orden de índice
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]


def count_yolo_class_distribution(labels_dir: str,
                                  class_names: List[str]) -> pd.DataFrame:
    """
    Cuenta las instancias (bounding boxes) de cada clase en un directorio
    de etiquetas YOLO.

    Cada archivo .txt contiene líneas con formato:
        <class_id> <x_center> <y_center> <width> <height>

    Args:
        labels_dir: Directorio que contiene los archivos .txt de etiquetas
        class_names: Lista de nombres de clase (índice = class_id)

    Returns:
        DataFrame con columnas: Class, Count, Class_ID ordenado
        descendentemente por Count
    """
    counter: Counter = Counter()

    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(labels_dir, fname)
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                counter[class_id] += 1

    # Asegurar que todas las clases aparezcan aunque tengan 0 instancias
    rows = []
    for idx, name in enumerate(class_names):
        rows.append({
            "Class": name,
            "Count": counter.get(idx, 0),
            "Class_ID": idx,
        })

    df = pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)
    return df


def plot_yolo_class_distribution(
    dataset_path: str,
    scale_type: str = "linear",
    palette: str = "magma",
    figsize: Tuple[int, int] = (12, 8),
    output_dir: Optional[str] = None,
    filename: str = "class_distribution",
    title: str = "Distribución de Clases — Dataset YOLO",
) -> pd.DataFrame:
    """
    Grafica la distribución de clases de un dataset YOLO como barras
    horizontales, siguiendo el estilo visual del proyecto (paleta 'magma').

    Espera la estructura estándar de YOLO:
        dataset_path/
            data.yaml          ← nombres de clase y configuración
            data/labels/*.txt  ← archivos de etiquetas

    Args:
        dataset_path: Ruta raíz del dataset YOLO (contiene data.yaml y data/)
        scale_type:   'linear' o 'log' para la escala del eje X
        palette:      Paleta de colores de seaborn (por defecto 'magma')
        figsize:      Tamaño de la figura
        output_dir:   Directorio donde guardar PNG/CSV (None = no guardar)
        filename:     Nombre base del archivo (sin extensión)
        title:        Título del gráfico

    Returns:
        DataFrame con la distribución (Class, Count, Class_ID)
    """
    # --- 1. Cargar datos ------------------------------------------------
    yaml_path = os.path.join(dataset_path, "data.yaml")
    labels_dir = os.path.join(dataset_path, "data", "labels")

    class_names = load_yolo_class_names(yaml_path)
    df_counts = count_yolo_class_distribution(labels_dir, class_names)
    total = df_counts["Count"].sum()

    # --- 2. Gráfico de barras horizontales ------------------------------
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_counts,
        y="Class",
        x="Count",
        hue="Class",
        palette=palette,
        legend=False,
        edgecolor="black",
        alpha=0.85,
    )

    # Anotaciones con conteo y porcentaje al lado de cada barra
    for i, row in df_counts.iterrows():
        pct = (row["Count"] / total * 100) if total > 0 else 0
        ax.text(
            row["Count"],
            i,
            f'  {row["Count"]:,}  ({pct:.1f}%)',
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(
        "Número de Instancias (Bounding Boxes)"
        + (" — Escala Logarítmica" if scale_type == "log" else ""),
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Clase", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Ampliar eje X un 30 % respecto al valor máximo
    max_count = df_counts["Count"].max()
    ax.set_xlim(0, max_count * 1.30)

    if scale_type == "log":
        ax.set_xscale("log")

    plt.tight_layout()

    # --- 3. Guardar artefactos ------------------------------------------
    if output_dir:
        ensure_dir(output_dir)
        fig_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"💾 Figura guardada: {fig_path}")

    plt.show()

    # --- 4. Tabla resumen -----------------------------------------------
    print("\n--- Distribución de Clases (YOLO) ---")
    print(df_counts[["Class", "Count"]].to_markdown(index=False))

    if output_dir:
        csv_path = os.path.join(output_dir, f"{filename}.csv")
        df_counts[["Class", "Count"]].to_csv(csv_path, index=False)
        print(f"💾 Tabla guardada: {csv_path}")

    return df_counts

