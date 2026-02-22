"""Carga de configuración desde archivos YAML para Vertex AI — Ciclo 2.

Convierte un archivo YAML de hiperparámetros en un ``ExperimentSetup``
compatible con toda la pipeline de ``src_colab``.

Soporta las familias:
    - ``FCOS`` — MobileNetV3-Small + FCOS (Tiny)
    - ``YOLO26_CUSTOM`` — YOLO26n con custom training loop
    - ``ESPDet`` — ESPDet-Pico (custom Espressif)
    - ``EXPORT`` — Job de exportación genérico

Ejemplo de uso::

    from trainer.config_loader import load_config_from_yaml
    setup = load_config_from_yaml("/tmp/config.yaml")
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from src_colab.utils_widgets import (
    ExperimentSetup,
    create_manual_setup,
    _FCOS_DEFAULTS,
    _YOLO26_CUSTOM_DEFAULTS,
    _ESPDET_DEFAULTS,
)
from src_colab.config import (
    is_fcos_family,
    is_yolo26_custom_family,
    is_espdet_family,
    is_export_family,
)


# ── Mapeo familia → (clave YAML, defaults) ──────────────────────────
_FAMILY_SECTIONS = {
    "FCOS":          ("fcos",          _FCOS_DEFAULTS),
    "YOLO26_CUSTOM": ("yolo26_custom", _YOLO26_CUSTOM_DEFAULTS),
    "ESPDet":        ("espdet",        _ESPDET_DEFAULTS),
}


def load_config_from_yaml(yaml_path: str) -> ExperimentSetup:
    """Lee un YAML de hiperparámetros y devuelve un ``ExperimentSetup``.

    El YAML debe tener las secciones ``model``, ``dataset``, ``common``,
    y opcionalmente ``fcos``, ``yolo26_custom`` o ``espdet`` según la familia.

    Args:
        yaml_path: Ruta local al archivo YAML.

    Returns:
        ExperimentSetup completamente configurado.
    """
    with open(yaml_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    common_cfg = cfg.get("common", {})

    family = model_cfg.get("family", "FCOS")

    # ── Construir kwargs específicos de la familia ──
    family_kwargs: Dict[str, Any] = {}
    section_info = _FAMILY_SECTIONS.get(family)
    if section_info:
        yaml_key, defaults = section_info
        family_section = cfg.get(yaml_key, {})
        # Pasar TODAS las claves de la sección YAML (no solo las del whitelist).
        # Fix T8: El patrón anterior iteraba solo sobre `defaults`, descartando
        # silenciosamente claves nuevas como focal_gamma, reg_warmup_epochs, etc.
        family_kwargs = dict(family_section)

    # ── Crear ExperimentSetup vía create_manual_setup ──
    setup = create_manual_setup(
        model_family=family,
        model_variant=model_cfg.get("variant", ""),
        version=model_cfg.get("version", "v1"),
        description=model_cfg.get("description", ""),
        dataset_name=dataset_cfg.get("name", "iodc_yolo"),
        class_names=dataset_cfg.get("class_names"),
        img_size=dataset_cfg.get("img_size", 224),
        batch_size=common_cfg.get("batch_size", 32),
        patience=common_cfg.get("patience", 30),
        seed=common_cfg.get("seed", 42),
        conf_threshold=common_cfg.get("conf_threshold", 0.25),
        iou_threshold=common_cfg.get("iou_threshold", 0.45),
        **family_kwargs,
    )

    return setup


def get_gcs_dataset_uri(yaml_path: str) -> str:
    """Extrae la URI GCS del dataset desde el YAML de configuración.

    Args:
        yaml_path: Ruta local al archivo YAML.

    Returns:
        URI GCS del dataset (e.g. ``gs://bucket/datasets/iodc_yolo.zip``).
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("dataset", {}).get("gcs_uri", "")


def get_gcs_model_uri(yaml_path: str) -> str:
    """Extrae la URI GCS del modelo entrenado (para jobs de export).

    Args:
        yaml_path: Ruta local al archivo YAML.

    Returns:
        URI GCS del modelo (e.g. ``gs://bucket/output/fcos_v1/best_model.pt``).
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("export", {}).get("model_gcs_uri", "")


def get_pretrained_uri(yaml_path: str) -> str:
    """Extrae la URI GCS de pesos pretrained (para YOLO26/ESPDet).

    Args:
        yaml_path: Ruta local al archivo YAML.

    Returns:
        URI GCS de pesos pretrained o cadena vacía.
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    family = cfg.get("model", {}).get("family", "")
    section_info = _FAMILY_SECTIONS.get(family)
    if section_info:
        yaml_key, _ = section_info
        return cfg.get(yaml_key, {}).get("pretrained_gcs_uri", "")
    return ""
