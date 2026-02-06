"""
augmentation_dataset.py
=======================
Augmentación focalizada de datos para balancear clases minoritarias
en un dataset YOLO. Genera un nuevo dataset completo (copia del original
+ imágenes augmentadas) sin modificar el original.

Flujo:
  1. Analiza distribución de clases en el dataset fuente.
  2. Identifica imágenes candidatas (contienen clases a augmentar).
  3. Copia el dataset original completo al directorio de salida.
  4. Genera imágenes augmentadas con albumentations (bbox-aware).
  5. Produce tabla comparativa, gráfica y log de trazabilidad.

Prerequisitos:
  pip install albumentations opencv-python-headless pyyaml matplotlib tqdm

Autor: TFM UNIR - Ingeniería de Datos
"""

import csv
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

# ╔══════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURACIÓN EDITABLE                       ║
# ╚══════════════════════════════════════════════════════════════════╝

# Directorio base
BASE_DIR = Path(__file__).parent

# Dataset fuente (NO se modifica)
SOURCE_DIR = BASE_DIR / "dataset_maestro"

# Dataset de salida (copia completa + augmentaciones)
OUTPUT_DIR = BASE_DIR / "dataset_maestro_aug"

# Clases a augmentar: { class_id: "nombre" }
AUGMENT_CLASSES = {
    1: "dog",
    3: "stair",
}

# Número objetivo de instancias por clase (media aproximada)
TARGET_INSTANCES = 4111

# Semilla para reproducibilidad
RANDOM_SEED = 42

# ╔══════════════════════════════════════════════════════════════════╗
# ║                PIPELINES DE TRANSFORMACIÓN                      ║
# ╚══════════════════════════════════════════════════════════════════╝

# Parámetros comunes para bboxes YOLO
BBOX_PARAMS = A.BboxParams(
    format="yolo",
    min_visibility=0.2,
    label_fields=["class_labels"],
)


def get_mild_transform() -> A.Compose:
    """Transformaciones suaves: flip, brillo, ruido."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.05), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.3,
            ),
        ],
        bbox_params=BBOX_PARAMS,
    )


def get_aggressive_transform() -> A.Compose:
    """Transformaciones agresivas: geométricas + color + ruido."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.6
            ),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.3,
            ),
        ],
        bbox_params=BBOX_PARAMS,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    FUNCIONES AUXILIARES                          ║
# ╚══════════════════════════════════════════════════════════════════╝


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Lee un archivo label YOLO y retorna lista de (class_id, cx, cy, w, h).
    """
    bboxes = []
    if not label_path.exists():
        return bboxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls = int(float(parts[0]))
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    bboxes.append((cls, cx, cy, w, h))
                except ValueError:
                    continue
    return bboxes


def write_label_file(
    label_path: Path,
    bboxes: list[list[float]],
    class_labels: list[int],
):
    """
    Escribe un archivo label YOLO.
    bboxes: [[cx, cy, w, h], ...] (formato yolo de albumentations)
    class_labels: [class_id, ...]
    """
    with open(label_path, "w", encoding="utf-8") as f:
        for bbox, cls in zip(bboxes, class_labels):
            cx, cy, w, h = bbox
            f.write(f"{int(cls)} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n")


def count_instances(labels_dir: Path) -> Counter:
    """
    Cuenta el total de instancias (bounding boxes) por class_id
    en todos los labels de un directorio.
    """
    counter = Counter()
    for lbl_file in labels_dir.glob("*.txt"):
        for cls, *_ in parse_label_file(lbl_file):
            counter[cls] += 1
    return counter


def analyze_candidates(
    labels_dir: Path,
    augment_classes: dict[int, str],
) -> dict[int, list[tuple[str, int]]]:
    """
    Identifica imágenes candidatas para augmentación.

    Retorna: { class_id: [ (basename_sin_ext, num_instancias_de_esa_clase), ... ] }
    """
    candidates: dict[int, list[tuple[str, int]]] = defaultdict(list)

    for lbl_file in sorted(labels_dir.glob("*.txt")):
        bboxes = parse_label_file(lbl_file)
        if not bboxes:
            continue

        # Contar instancias por clase en esta imagen
        cls_counts: Counter = Counter()
        for cls, *_ in bboxes:
            cls_counts[cls] += 1

        # Si tiene al menos 1 bbox de una clase objetivo → es candidata
        for cls_id in augment_classes:
            if cls_counts[cls_id] > 0:
                candidates[cls_id].append((lbl_file.stem, cls_counts[cls_id]))

    return candidates


def compute_augmentation_plan(
    class_counter: Counter,
    candidates: dict[int, list[tuple[str, int]]],
    augment_classes: dict[int, str],
    target: int,
) -> dict[int, tuple[int, list[tuple[str, int]]]]:
    """
    Calcula cuántas copias augmentadas generar por imagen-candidata de cada clase.

    Retorna: { class_id: (num_copies_per_image, [(basename, instances), ...]) }

    Estrategia: distribuir uniformemente las copias entre todas las candidatas.
    Cada copia de una imagen aporta en promedio (total_instances / num_images) de esa clase.
    """
    plan = {}
    for cls_id, cls_name in augment_classes.items():
        current = class_counter.get(cls_id, 0)
        deficit = target - current

        if deficit <= 0:
            print(f"   ℹ {cls_name}(id={cls_id}): ya tiene {current} >= {target}. No se augmenta.")
            continue

        cands = candidates.get(cls_id, [])
        if not cands:
            print(f"   ⚠ {cls_name}(id={cls_id}): sin imágenes candidatas. No se puede augmentar.")
            continue

        # Total instances de esta clase en todas las candidatas
        total_inst_in_candidates = sum(inst for _, inst in cands)
        # Instancias promedio por imagen
        avg_inst_per_img = total_inst_in_candidates / len(cands)
        # Imágenes augmentadas necesarias para cubrir el déficit
        imgs_needed = math.ceil(deficit / avg_inst_per_img)
        # Copias por imagen (distribuir uniformemente)
        copies_per_img = math.ceil(imgs_needed / len(cands))

        plan[cls_id] = (copies_per_img, cands)

        print(f"   📊 {cls_name}(id={cls_id}):")
        print(f"      Actual: {current} | Objetivo: {target} | Déficit: {deficit}")
        print(f"      Candidatas: {len(cands)} imágenes | Avg inst/img: {avg_inst_per_img:.2f}")
        print(f"      Copias por imagen: {copies_per_img} | Total imgs a generar: ~{copies_per_img * len(cands)}")

    return plan


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      FUNCIÓN PRINCIPAL                          ║
# ╚══════════════════════════════════════════════════════════════════╝


def augment_dataset():
    """Ejecuta la augmentación focalizada para balancear clases minoritarias."""

    random.seed(RANDOM_SEED)

    print("🔧 Augmentación Focalizada de Dataset YOLO")
    print("=" * 60)

    # ── Validar directorios ───────────────────────────────────────
    src_images = SOURCE_DIR / "data" / "images"
    src_labels = SOURCE_DIR / "data" / "labels"

    if not src_images.is_dir() or not src_labels.is_dir():
        print(f"❌ No se encontró dataset fuente en {SOURCE_DIR}/data/")
        sys.exit(1)

    # Leer data.yaml
    src_yaml = SOURCE_DIR / "data.yaml"
    with open(src_yaml, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml["names"]

    print(f"   Fuente:  {SOURCE_DIR.relative_to(BASE_DIR)}/")
    print(f"   Destino: {OUTPUT_DIR.relative_to(BASE_DIR)}/")
    print(f"   Clases:  {class_names}")
    print(f"   Objetivo: {TARGET_INSTANCES} instancias por clase minoritaria")

    # ── Fase 1: Analizar distribución actual ──────────────────────
    print("\n📂 Fase 1: Analizando distribución actual...")
    class_counter = count_instances(src_labels)
    total_before = sum(class_counter.values())

    print(f"\n   {'Clase':<15} {'ID':>3} {'Instancias':>12} {'%':>7}")
    print("   " + "-" * 42)
    for idx, name in enumerate(class_names):
        cnt = class_counter.get(idx, 0)
        pct = cnt / total_before * 100 if total_before > 0 else 0
        marker = " ◄ augmentar" if idx in AUGMENT_CLASSES else ""
        print(f"   {name:<15} {idx:>3} {cnt:>12,} {pct:>6.1f}%{marker}")
    print("   " + "-" * 42)
    print(f"   {'TOTAL':<15} {'':>3} {total_before:>12,}")

    # ── Fase 2: Identificar candidatas y calcular plan ────────────
    print("\n📊 Fase 2: Calculando plan de augmentación...")
    candidates = analyze_candidates(src_labels, AUGMENT_CLASSES)
    plan = compute_augmentation_plan(class_counter, candidates, AUGMENT_CLASSES, TARGET_INSTANCES)

    if not plan:
        print("\n   ℹ No hay clases que requieran augmentación. Finalizando.")
        return

    # ── Fase 3: Copiar dataset original ───────────────────────────
    print("\n📁 Fase 3: Copiando dataset original...")
    if OUTPUT_DIR.exists():
        print(f"   ⚠ {OUTPUT_DIR.relative_to(BASE_DIR)}/ ya existe. Eliminando...")
        shutil.rmtree(OUTPUT_DIR)

    shutil.copytree(SOURCE_DIR, OUTPUT_DIR)
    print(f"   ✓ Copiado a {OUTPUT_DIR.relative_to(BASE_DIR)}/")

    # ── Fase 4: Generar imágenes augmentadas ──────────────────────
    print("\n🚀 Fase 4: Generando imágenes augmentadas...")
    out_images = OUTPUT_DIR / "data" / "images"
    out_labels = OUTPUT_DIR / "data" / "labels"

    # Preparar transformaciones
    mild_transform = get_mild_transform()
    aggressive_transform = get_aggressive_transform()

    aug_counter = 0
    aug_log = []  # Para augmentation_log.csv
    aug_class_instances = Counter()  # Instancias adicionales generadas

    # Recopilar todas las imágenes candidatas únicas (evitar duplicar si
    # una imagen es candidata para múltiples clases)
    @dataclass
    class AugEntry:
        copies: int = 0
        target_cls: set[int] = field(default_factory=set)

    img_aug_schedule: dict[str, AugEntry] = {}

    for cls_id, (copies_per_img, cands) in plan.items():
        for basename, _inst_count in cands:
            if basename not in img_aug_schedule:
                img_aug_schedule[basename] = AugEntry()
            entry = img_aug_schedule[basename]
            # Si ya está programada para más copias por otra clase, tomar el máximo
            entry.copies = max(entry.copies, copies_per_img)
            entry.target_cls.add(cls_id)

    # Generar augmentaciones
    total_to_generate = sum(e.copies for e in img_aug_schedule.values())
    print(f"   Imágenes únicas a augmentar: {len(img_aug_schedule)}")
    print(f"   Total de copias a generar:   ~{total_to_generate}")
    print()

    # Iterar con barra de progreso
    for basename, entry in tqdm(
        sorted(img_aug_schedule.items()),
        desc="   Augmentando",
        unit="img",
        ncols=80,
    ):
        copies = entry.copies

        # Buscar imagen original (probar extensiones comunes)
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate_path = src_images / f"{basename}{ext}"
            if candidate_path.exists():
                img_path = candidate_path
                break

        if img_path is None:
            continue

        # Leer imagen
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Leer label completo (todas las clases)
        lbl_path = src_labels / f"{basename}.txt"
        annotations = parse_label_file(lbl_path)
        if not annotations:
            continue

        class_labels = [a[0] for a in annotations]
        bboxes = [[a[1], a[2], a[3], a[4]] for a in annotations]  # cx, cy, w, h

        # Generar N copias
        for copy_idx in range(copies):
            aug_counter += 1
            new_basename = f"aug_{aug_counter:06d}"
            new_img_name = f"{new_basename}.jpg"
            new_lbl_name = f"{new_basename}.txt"

            # Alternar mild/aggressive: primeras copias mild, resto aggressive
            if copy_idx < max(1, copies // 2):
                transform = mild_transform
                transform_type = "mild"
            else:
                transform = aggressive_transform
                transform_type = "aggressive"

            # Aplicar transformación (con reintentos si se pierden todos los bboxes)
            success = False
            for _attempt in range(5):
                try:
                    result = transform(
                        image=img,
                        bboxes=bboxes,
                        class_labels=class_labels,
                    )
                except Exception:
                    continue

                if len(result["bboxes"]) > 0:
                    success = True
                    break

            if not success:
                aug_counter -= 1
                continue

            # Guardar imagen augmentada
            aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_images / new_img_name), aug_img)

            # Guardar label augmentado
            write_label_file(
                out_labels / new_lbl_name,
                result["bboxes"],
                result["class_labels"],
            )

            # Contabilizar instancias augmentadas
            for cls in result["class_labels"]:
                aug_class_instances[cls] += 1

            # Determinar para qué clase(s) se hizo esta augmentación
            target_cls_names = [
                AUGMENT_CLASSES[cid]
                for cid in AUGMENT_CLASSES
                if cid in entry.target_cls and cid in class_labels
            ]

            aug_log.append(
                {
                    "new_name": new_img_name,
                    "source_image": img_path.name,
                    "transform_type": transform_type,
                    "target_classes": "|".join(target_cls_names),
                    "num_bboxes": len(result["bboxes"]),
                }
            )

    print(f"\n   ✓ Generadas {aug_counter} imágenes augmentadas")

    # ── Fase 5: Guardar log de augmentación ───────────────────────
    print("\n📋 Fase 5: Guardando log de augmentación...")
    aug_log_path = OUTPUT_DIR / "augmentation_log.csv"
    with open(aug_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["new_name", "source_image", "transform_type", "target_classes", "num_bboxes"],
        )
        writer.writeheader()
        writer.writerows(aug_log)
    print(f"   ✓ {aug_log_path.name} ({len(aug_log)} registros)")

    # ── Fase 6: Distribución final y comparativa ──────────────────
    print("\n📊 Fase 6: Análisis de distribución final...")
    final_counter = count_instances(out_labels)
    total_after = sum(final_counter.values())

    # Tabla comparativa
    print(f"\n   {'Clase':<12} {'Antes':>10} {'Augment':>10} {'Después':>10} {'Δ%':>8}")
    print("   " + "-" * 54)
    for idx, name in enumerate(class_names):
        before = class_counter.get(idx, 0)
        after = final_counter.get(idx, 0)
        augmented = after - before
        delta = ((after - before) / before * 100) if before > 0 else 0
        marker = " ✓" if idx in AUGMENT_CLASSES else ""
        print(f"   {name:<12} {before:>10,} {augmented:>+10,} {after:>10,} {delta:>+7.1f}%{marker}")
    print("   " + "-" * 54)
    print(f"   {'TOTAL':<12} {total_before:>10,} {total_after - total_before:>+10,} {total_after:>10,}")

    # ── Fase 7: Generar gráfica comparativa ───────────────────────
    print("\n📈 Fase 7: Generando gráfica de distribución...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfica ANTES
    ax1 = axes[0]
    before_counts = [class_counter.get(i, 0) for i in range(len(class_names))]
    colors_before = [
        "#e74c3c" if i in AUGMENT_CLASSES else "#3498db" for i in range(len(class_names))
    ]
    bars1 = ax1.bar(class_names, before_counts, color=colors_before, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars1, before_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(before_counts) * 0.01,
            f"{count:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax1.axhline(y=TARGET_INSTANCES, color="green", linestyle="--", alpha=0.7, label=f"Objetivo ({TARGET_INSTANCES:,})")
    ax1.set_title("ANTES de Augmentación", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Instancias (Bounding Boxes)", fontsize=11)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Gráfica DESPUÉS
    ax2 = axes[1]
    after_counts = [final_counter.get(i, 0) for i in range(len(class_names))]
    colors_after = [
        "#2ecc71" if i in AUGMENT_CLASSES else "#3498db" for i in range(len(class_names))
    ]
    bars2 = ax2.bar(class_names, after_counts, color=colors_after, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars2, after_counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(after_counts) * 0.01,
            f"{count:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax2.axhline(y=TARGET_INSTANCES, color="green", linestyle="--", alpha=0.7, label=f"Objetivo ({TARGET_INSTANCES:,})")
    ax2.set_title("DESPUÉS de Augmentación", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Instancias (Bounding Boxes)", fontsize=11)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Distribución de Clases - Balanceo por Augmentación", fontsize=15, fontweight="bold")
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"   ✓ Gráfica guardada: {chart_path.name}")

    # ── Actualizar data.yaml (sin cambios de clases, mismo contenido) ──
    with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # ── Resumen final ─────────────────────────────────────────────
    total_images = len(list(out_images.glob("*")))
    total_labels = len(list(out_labels.glob("*.txt")))

    print("\n" + "=" * 60)
    print("✅ AUGMENTACIÓN COMPLETADA")
    print("=" * 60)
    print(f"   📂 Directorio:        {OUTPUT_DIR.relative_to(BASE_DIR)}/")
    print(f"   🖼  Imágenes totales:  {total_images:,} (originales + augmentadas)")
    print(f"   🏷  Labels totales:    {total_labels:,}")
    print(f"   📦 BB totales:         {total_after:,} (antes: {total_before:,})")
    print(f"   ➕ Imgs augmentadas:   {aug_counter:,}")

    # Verificación de integridad
    if total_images == total_labels:
        print(f"   ✓ Integridad OK: imágenes == labels")
    else:
        print(f"   ⚠ ATENCIÓN: imágenes ({total_images}) ≠ labels ({total_labels})")

    print()


# ╔══════════════════════════════════════════════════════════════════╗
# ║                         EJECUCIÓN                               ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    augment_dataset()
