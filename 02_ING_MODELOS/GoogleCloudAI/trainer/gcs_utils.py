"""Utilidades de Google Cloud Storage para Vertex AI Custom Jobs.

Gestiona la descarga del dataset y la subida de artefactos
de entrenamiento al bucket GCS del proyecto.

Ejemplo de uso::

    from trainer.gcs_utils import prepare_dataset, upload_directory_to_gcs
    local_path = prepare_dataset("gs://bucket/datasets/yolo26.zip", "/tmp/data", "YOLO26")
    upload_directory_to_gcs("/tmp/training/models/yolo26n_v1", "gs://bucket/output/yolo26n_v1")
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional

from google.cloud import storage


# ── Descarga ─────────────────────────────────────────────────────────

def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    """Descarga un archivo individual desde GCS.

    Args:
        gcs_uri: URI completa ``gs://bucket/path/to/file``.
        local_path: Ruta local de destino.

    Returns:
        Ruta local al archivo descargado.
    """
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  ✅ Descargado: {gcs_uri} → {local_path} ({size_mb:.1f} MB)")
    return local_path


def download_directory_from_gcs(gcs_prefix: str, local_dir: str) -> str:
    """Descarga todos los blobs bajo un prefijo GCS a un directorio local.

    Útil para datasets TFRecord que consisten en varios archivos
    (train.tfrecord, val.tfrecord, test.tfrecord, metadata.json).

    Args:
        gcs_prefix: URI de directorio ``gs://bucket/path/to/dir/``
                     (con o sin trailing /).
        local_dir: Directorio local de destino.

    Returns:
        Ruta al directorio local.
    """
    bucket_name, prefix = _parse_gcs_uri(gcs_prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        raise FileNotFoundError(
            f"No se encontraron archivos en {gcs_prefix}"
        )

    os.makedirs(local_dir, exist_ok=True)
    count = 0
    for blob in blobs:
        # Ignorar "carpetas" vacías
        if blob.name.endswith("/"):
            continue
        relative_path = blob.name[len(prefix):]
        local_file = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        blob.download_to_filename(local_file)
        count += 1

    print(f"  ✅ Descargados {count} archivos desde {gcs_prefix} → {local_dir}")
    return local_dir


def extract_dataset(zip_path: str, dest_dir: str) -> str:
    """Descomprime un archivo .zip de dataset.

    Args:
        zip_path: Ruta local al archivo .zip.
        dest_dir: Directorio local donde extraer.

    Returns:
        Ruta al directorio extraído.
    """
    print(f"  📦 Descomprimiendo {zip_path} → {dest_dir} ...")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  ✅ Descomprimido correctamente")
    return dest_dir


def prepare_dataset(
    gcs_dataset_uri: str,
    local_base: str,
    family: str,
) -> str:
    """Orquesta la descarga y preparación del dataset.

    - Si la URI termina en ``.zip`` → descarga + extrae.
    - Si no → descarga todo el directorio GCS recursivamente.

    Args:
        gcs_dataset_uri: URI GCS del dataset.
        local_base: Directorio base local (e.g. ``/tmp/training/datasets``).
        family: Familia del modelo (``YOLO26``, ``MobileNetV3``, etc.).

    Returns:
        Ruta local al directorio del dataset listo para usar.
    """
    print(f"\n📥 Preparando dataset desde {gcs_dataset_uri}")

    if gcs_dataset_uri.endswith(".zip"):
        # Descargar .zip y extraer
        dataset_name = Path(gcs_dataset_uri).stem  # e.g. "yolo26"
        zip_local = os.path.join(local_base, f"{dataset_name}.zip")
        dataset_dir = os.path.join(local_base, dataset_name)

        download_from_gcs(gcs_dataset_uri, zip_local)
        extract_dataset(zip_local, dataset_dir)

        # Limpiar zip para liberar espacio
        os.remove(zip_local)
        print(f"  🗑️  Eliminado zip temporal: {zip_local}")
    else:
        # Descargar directorio completo (TFRecord)
        dataset_name = gcs_dataset_uri.rstrip("/").split("/")[-1]
        dataset_dir = os.path.join(local_base, dataset_name)
        download_directory_from_gcs(gcs_dataset_uri, dataset_dir)

    print(f"  📂 Dataset listo: {dataset_dir}")
    return dataset_dir


# ── Subida ───────────────────────────────────────────────────────────

def upload_file_to_gcs(local_path: str, gcs_uri: str) -> str:
    """Sube un archivo individual a GCS.

    Args:
        local_path: Ruta local del archivo.
        gcs_uri: URI de destino ``gs://bucket/path/to/file``.

    Returns:
        URI GCS del archivo subido.
    """
    bucket_name, blob_path = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(local_path)
    print(f"  ☁️  Subido: {local_path} → {gcs_uri}")
    return gcs_uri


def upload_directory_to_gcs(local_dir: str, gcs_prefix: str) -> int:
    """Sube recursivamente un directorio local a GCS.

    Args:
        local_dir: Directorio local a subir.
        gcs_prefix: URI de destino ``gs://bucket/path/to/dir``.

    Returns:
        Número de archivos subidos.
    """
    bucket_name, prefix = _parse_gcs_uri(gcs_prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    count = 0
    for root, _dirs, files in os.walk(local_dir):
        for filename in files:
            local_file = os.path.join(root, filename)
            relative_path = os.path.relpath(local_file, local_dir)
            blob_path = prefix + relative_path
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
            count += 1

    print(f"  ☁️  Subidos {count} archivos → {gcs_prefix}")
    return count


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Extrae bucket y blob path de una URI ``gs://bucket/path``.

    Returns:
        (bucket_name, blob_path)
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"URI GCS inválida (debe empezar con gs://): {uri}")
    parts = uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""
    return bucket_name, blob_path
