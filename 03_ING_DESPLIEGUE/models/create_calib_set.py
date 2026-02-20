"""
create_calib_set.py
===================
Genera un dataset de calibración a partir del split 'valid' del dataset YOLO,
para usar en la cuantización de modelos ONNX → ESPDL con esp-ppq.

Produce dos archivos .pkl:
  - calib_set_nchw.pkl  → List[np.ndarray] shape (1,3,224,224) — para YOLO11n y YOLO26n
  - calib_set_nhwc.pkl  → List[np.ndarray] shape (1,224,224,3) — para MBNTv3S_ssdlite

Uso:
  python models/create_calib_set.py
"""

import os
import random
import pickle
import glob
import time

import cv2
import numpy as np

# ============================================================================
# Configuración
# ============================================================================

# Ruta al split 'valid' del dataset YOLO
DATASET_DIR = os.path.expanduser(
    "~/Documents/TFM_UNIR/02_ING_MODELOS/datasets/yolo26/valid/images"
)

# Directorio de salida (junto a los modelos ONNX)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parámetros
N_SAMPLES = 256       # Número de imágenes para calibración
IMG_SIZE = 224         # Tamaño de entrada de los modelos (224x224)
RANDOM_SEED = 42       # Semilla para reproducibilidad
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Archivos de salida
OUTPUT_NCHW = os.path.join(OUTPUT_DIR, "calib_set_nchw.pkl")
OUTPUT_NHWC = os.path.join(OUTPUT_DIR, "calib_set_nhwc.pkl")


# ============================================================================
# Funciones
# ============================================================================

def find_images(directory: str, extensions: tuple) -> list:
    """Busca recursivamente imágenes con las extensiones dadas."""
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    # Eliminar duplicados y ordenar para reproducibilidad
    image_paths = sorted(set(image_paths))
    return image_paths


def load_and_preprocess(image_path: str, img_size: int) -> np.ndarray:
    """
    Carga una imagen y la preprocesa para calibración.

    Pasos:
      1. Leer con OpenCV (BGR)
      2. Convertir BGR → RGB
      3. Redimensionar a (img_size, img_size)
      4. Normalizar a float32 [0.0, 1.0]

    Returns:
        np.ndarray con shape (H, W, C) = (224, 224, 3), dtype float32
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionar a (IMG_SIZE, IMG_SIZE)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Normalizar a float32 [0, 1]
    img = img.astype(np.float32) / 255.0

    return img


def build_calibration_datasets(image_paths: list, img_size: int):
    """
    Construye dos listas de calibración a partir de las rutas de imágenes:
      - nchw_list: List[np.ndarray] cada uno con shape (1, 3, H, W)
      - nhwc_list: List[np.ndarray] cada uno con shape (1, H, W, 3)
    """
    nchw_list = []
    nhwc_list = []
    failed = 0

    for i, path in enumerate(image_paths):
        try:
            img_hwc = load_and_preprocess(path, img_size)  # (H, W, C)

            # NHWC: agregar dimensión batch → (1, H, W, C)
            img_nhwc = np.expand_dims(img_hwc, axis=0)
            nhwc_list.append(img_nhwc)

            # NCHW: transponer (H,W,C) → (C,H,W) y agregar batch → (1, C, H, W)
            img_chw = img_hwc.transpose(2, 0, 1)  # (C, H, W)
            img_nchw = np.expand_dims(img_chw, axis=0)
            nchw_list.append(img_nchw)

            if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
                print(f"  Procesadas {i + 1}/{len(image_paths)} imágenes...")

        except Exception as e:
            print(f"  [WARN] Error procesando {os.path.basename(path)}: {e}")
            failed += 1

    if failed > 0:
        print(f"  {failed} imágenes fallaron y se omitieron.")

    return nchw_list, nhwc_list


def save_pickle(data, filepath: str):
    """Guarda datos con pickle y reporta tamaño."""
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Guardado: {os.path.basename(filepath)} ({size_mb:.1f} MB)")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  GENERACIÓN DE DATASET DE CALIBRACIÓN")
    print("=" * 60)

    # --- 1. Buscar imágenes ---
    print(f"\n[1/4] Buscando imágenes en:\n  {DATASET_DIR}")
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"No se encontró el directorio: {DATASET_DIR}")

    all_images = find_images(DATASET_DIR, EXTENSIONS)
    print(f"  Encontradas: {len(all_images)} imágenes")

    if len(all_images) < N_SAMPLES:
        print(f"  [WARN] Solo hay {len(all_images)} imágenes, se usarán todas.")
        selected = all_images
    else:
        random.seed(RANDOM_SEED)
        selected = random.sample(all_images, N_SAMPLES)
        selected.sort()  # Ordenar para consistencia
    print(f"  Seleccionadas: {len(selected)} imágenes (seed={RANDOM_SEED})")

    # --- 2. Cargar y preprocesar ---
    print(f"\n[2/4] Cargando y preprocesando imágenes ({IMG_SIZE}x{IMG_SIZE}, float32 [0,1])...")
    t0 = time.time()
    nchw_list, nhwc_list = build_calibration_datasets(selected, IMG_SIZE)
    elapsed = time.time() - t0
    print(f"  Completado en {elapsed:.1f}s")

    # --- 3. Verificar shapes ---
    print(f"\n[3/4] Verificación de shapes:")
    print(f"  NCHW: {len(nchw_list)} arrays, shape={nchw_list[0].shape}, dtype={nchw_list[0].dtype}")
    print(f"  NHWC: {len(nhwc_list)} arrays, shape={nhwc_list[0].shape}, dtype={nhwc_list[0].dtype}")

    # Sanity check
    assert nchw_list[0].shape == (1, 3, IMG_SIZE, IMG_SIZE), \
        f"Shape NCHW inesperado: {nchw_list[0].shape}"
    assert nhwc_list[0].shape == (1, IMG_SIZE, IMG_SIZE, 3), \
        f"Shape NHWC inesperado: {nhwc_list[0].shape}"

    # Verificar rango de valores
    sample = nchw_list[0]
    assert 0.0 <= sample.min() and sample.max() <= 1.0, \
        f"Valores fuera de rango [0,1]: min={sample.min()}, max={sample.max()}"
    print(f"  Rango de valores: [{sample.min():.4f}, {sample.max():.4f}] ✓")

    # --- 4. Guardar ---
    print(f"\n[4/4] Guardando datasets de calibración:")
    save_pickle(nchw_list, OUTPUT_NCHW)
    save_pickle(nhwc_list, OUTPUT_NHWC)

    # --- Resumen ---
    print(f"\n{'=' * 60}")
    print(f"  RESUMEN")
    print(f"{'=' * 60}")
    print(f"  Fuente:     {DATASET_DIR}")
    print(f"  Split:      valid")
    print(f"  Muestras:   {len(nchw_list)}")
    print(f"  Resolución: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Norm:       float32 / 255.0 → [0.0, 1.0]")
    print(f"")
    print(f"  Archivos generados:")
    print(f"    → {OUTPUT_NCHW}")
    print(f"      Shape: (1, 3, {IMG_SIZE}, {IMG_SIZE}) — para YOLO11n, YOLO26n")
    print(f"    → {OUTPUT_NHWC}")
    print(f"      Shape: (1, {IMG_SIZE}, {IMG_SIZE}, 3) — para MBNTv3S_ssdlite")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
