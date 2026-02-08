"""Carga de configuración desde archivos YAML para Vertex AI.

Convierte un archivo YAML de hiperparámetros en un ``ExperimentSetup``
compatible con toda la pipeline de ``src_colab``.

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
    _YOLO_DEFAULTS,
    _MOBILENET_DEFAULTS,
)
from src_colab.config import is_yolo_family, is_mobilenet_family


def load_config_from_yaml(yaml_path: str) -> ExperimentSetup:
    """Lee un YAML de hiperparámetros y devuelve un ``ExperimentSetup``.

    El YAML debe tener las secciones ``model``, ``dataset``, ``common``,
    y opcionalmente ``yolo`` o ``mobilenet`` según la familia.

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

    family = model_cfg.get("family", "YOLO26")

    # ── Construir kwargs específicos de la familia ──
    family_kwargs: Dict[str, Any] = {}
    if is_yolo_family(family):
        yolo_section = cfg.get("yolo", {})
        # Partir de defaults, sobrescribir con YAML
        for key, default_val in _YOLO_DEFAULTS.items():
            if key in yolo_section:
                family_kwargs[key] = yolo_section[key]
    elif is_mobilenet_family(family):
        mnet_section = cfg.get("mobilenet", {})
        for key, default_val in _MOBILENET_DEFAULTS.items():
            if key in mnet_section:
                family_kwargs[key] = mnet_section[key]

    # ── Crear ExperimentSetup vía create_manual_setup ──
    setup = create_manual_setup(
        model_family=family,
        model_variant=model_cfg.get("variant", ""),
        version=model_cfg.get("version", "v1"),
        description=model_cfg.get("description", ""),
        dataset_name=dataset_cfg.get("name", "yolo26"),
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
        URI GCS del dataset (e.g. ``gs://bucket/datasets/yolo26.zip``).
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("dataset", {}).get("gcs_uri", "")
