"""Centralized configuration for the unified Colab training pipeline.

Handles environment detection (Colab vs local), Google Drive mounting,
GPU configuration, and unified path resolution for all model families.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Model family constants ──────────────────────────────────────────
MODEL_FAMILIES = {
    "YOLO11": {
        "framework": "pytorch",
        "variants": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        "dataset_format": "yolo",
        "training_phases": 1,
    },
    "YOLO26": {
        "framework": "pytorch",
        "variants": ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"],
        "dataset_format": "yolo",
        "training_phases": 1,
    },
    "MobileNetV2": {
        "framework": "tensorflow",
        "variants": ["MobileNetV2_SSDLite"],
        "dataset_format": "tfrecord",
        "training_phases": 2,
    },
    "MobileNetV3": {
        "framework": "tensorflow",
        "variants": ["MobileNetV3S_SSDLite", "MobileNetV3L_SSDLite"],
        "dataset_format": "tfrecord",
        "training_phases": 2,
    },
}


# ── Dataset master class definitions ────────────────────────────────
# Maps dataset_name → full ordered list of class names as they appear
# in the original label files (class_id 0 = first, 1 = second, …).
# Used to correctly filter/remap labels when training a class subset.
DATASET_MASTER_CLASSES = {
    # ⚠️  Roboflow re-ordena las clases ALFABÉTICAMENTE al exportar.
    #     El orden original de anotación era [obstacle, dog, person, stair, door],
    #     pero los IDs en los .txt exportados siguen orden alfabético:
    #       0=dog, 1=door, 2=obstacle, 3=person, 4=stair
    "yolo26": ["dog", "door", "obstacle", "person", "stair"],
    "yolo_v11": ["dog", "door", "obstacle", "person", "stair"],
}


def is_yolo_family(family: str) -> bool:
    """Check if model family is YOLO-based (PyTorch/Ultralytics)."""
    return family in ("YOLO11", "YOLO26")


def is_mobilenet_family(family: str) -> bool:
    """Check if model family is MobileNet-based (TensorFlow/Keras)."""
    return family in ("MobileNetV2", "MobileNetV3")


def get_framework(family: str) -> str:
    """Return 'pytorch' or 'tensorflow' for the given model family."""
    return MODEL_FAMILIES[family]["framework"]


def get_dataset_format(family: str) -> str:
    """Return expected dataset format: 'yolo' or 'tfrecord'."""
    return MODEL_FAMILIES[family]["dataset_format"]


# ── Environment detection & setup ───────────────────────────────────
@dataclass
class ColabEnvironment:
    """Runtime environment information."""
    is_colab: bool = False
    is_local: bool = True
    is_vertex_ai: bool = False
    gpu_name: str = "CPU"
    gpu_available: bool = False
    gpu_memory_mb: int = 0
    cuda_available: bool = False
    mps_available: bool = False
    tf_gpu_available: bool = False


def detect_environment() -> ColabEnvironment:
    """Detect runtime environment (Colab / local / Vertex AI) and GPU availability."""
    env = ColabEnvironment()

    # Detect Vertex AI (AIP_MODEL_DIR is injected by Vertex AI Training)
    if os.environ.get("AIP_MODEL_DIR") or os.environ.get("CLOUD_ML_PROJECT_ID"):
        env.is_vertex_ai = True
        env.is_colab = False
        env.is_local = False
    else:
        # Detect Colab
        try:
            import google.colab  # noqa: F401
            env.is_colab = True
            env.is_local = False
        except ImportError:
            env.is_colab = False
            env.is_local = True

    # PyTorch GPU
    try:
        import torch
        env.cuda_available = torch.cuda.is_available()
        env.mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if env.cuda_available:
            env.gpu_name = torch.cuda.get_device_name(0)
            env.gpu_memory_mb = int(
                torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
            )
            env.gpu_available = True
        elif env.mps_available:
            env.gpu_name = "Apple MPS"
            env.gpu_available = True
    except ImportError:
        pass

    # TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        env.tf_gpu_available = len(gpus) > 0
        if env.tf_gpu_available and not env.gpu_available:
            env.gpu_available = True
            env.gpu_name = gpus[0].name if gpus else "TF GPU"
    except ImportError:
        pass

    return env


def setup_gpu(env: ColabEnvironment) -> None:
    """Configure GPU memory growth and device settings."""
    # TensorFlow: enable memory growth to coexist with PyTorch
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    # Report
    print(f"\n🖥️  Entorno: {'Google Colab' if env.is_colab else 'Local'}")
    print(f"🎮 GPU: {env.gpu_name}")
    if env.gpu_memory_mb:
        print(f"   VRAM: {env.gpu_memory_mb:,} MB")
    print(f"   CUDA: {'✅' if env.cuda_available else '❌'}  |  "
          f"MPS: {'✅' if env.mps_available else '❌'}  |  "
          f"TF-GPU: {'✅' if env.tf_gpu_available else '❌'}")


def get_yolo_device(env: ColabEnvironment) -> str:
    """Return the best device string for Ultralytics YOLO training."""
    if env.cuda_available:
        return "0"
    if env.mps_available:
        return "mps"
    return "cpu"


# ── Path management ─────────────────────────────────────────────────
@dataclass
class ProjectPaths:
    """Unified path container for all project directories."""
    project_root: Path = field(default_factory=Path)

    # Datasets
    datasets_dir: Path = field(default_factory=Path)

    # Model outputs
    models_dir: Path = field(default_factory=Path)
    checkpoints_dir: Path = field(default_factory=Path)
    final_export_dir: Path = field(default_factory=Path)

    # Logging
    logs_dir: Path = field(default_factory=Path)
    experiments_dir: Path = field(default_factory=Path)

    # Reports
    reports_dir: Path = field(default_factory=Path)

    # YOLO runs
    runs_dir: Path = field(default_factory=Path)


def _resolve_project_root(env: ColabEnvironment) -> Path:
    """Determine the project root depending on the runtime."""
    if env.is_colab:
        return Path("/content/drive/MyDrive/TFM_UNIR")
    # Local: walk up from this file's location
    # Google_Colab/src_colab/config.py → Google_Colab → 02_ING_MODELOS
    return Path(__file__).resolve().parent.parent.parent


def create_project_paths(
    project_root: Optional[Path] = None,
    env: Optional[ColabEnvironment] = None,
) -> ProjectPaths:
    """Build all output paths from a project root, creating dirs."""
    if project_root is None:
        if env is None:
            env = detect_environment()
        project_root = _resolve_project_root(env)

    p = ProjectPaths(
        project_root=project_root,
        datasets_dir=project_root / "datasets",
        models_dir=project_root / "models",
        checkpoints_dir=project_root / "models" / "checkpoints",
        final_export_dir=project_root / "models" / "final_export",
        logs_dir=project_root / "logs",
        experiments_dir=project_root / "logs" / "experiments",
        reports_dir=project_root / "reports",
        runs_dir=project_root / "runs",
    )

    # Create directories
    for d in [
        p.models_dir,
        p.checkpoints_dir,
        p.final_export_dir,
        p.logs_dir,
        p.experiments_dir,
        p.reports_dir,
        p.runs_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    return p


# ── High-level setup ────────────────────────────────────────────────
def setup_environment(
    project_root: Optional[Path] = None,
) -> Tuple[ColabEnvironment, ProjectPaths]:
    """One-call environment setup.

    1. Detects runtime (Colab / local).
    2. Mounts Google Drive if on Colab.
    3. Configures GPU memory growth.
    4. Resolves and creates all output directories.
    5. Adds src_colab to ``sys.path``.

    Returns:
        (ColabEnvironment, ProjectPaths)
    """
    env = detect_environment()

    # Mount Drive in Colab (skip in Vertex AI — data comes from GCS)
    if env.is_colab and not env.is_vertex_ai:
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
        except Exception as exc:
            print(f"⚠️ No se pudo montar Google Drive: {exc}")

    # GPU config
    setup_gpu(env)

    # Paths
    paths = create_project_paths(project_root=project_root, env=env)

    # Ensure src_colab is importable (handles Colab where the package
    # lives inside Drive)
    src_parent = str(paths.project_root / "src_colab")
    if src_parent not in sys.path:
        sys.path.insert(0, str(paths.project_root))

    print(f"\n📂 Project root: {paths.project_root}")
    print(f"📁 Datasets:     {paths.datasets_dir}")
    print(f"📁 Models:        {paths.models_dir}")
    print(f"📁 Logs:          {paths.logs_dir}")
    print(f"📁 Reports:       {paths.reports_dir}")
    print(f"📁 Experiments:   {paths.experiments_dir}")

    return env, paths
