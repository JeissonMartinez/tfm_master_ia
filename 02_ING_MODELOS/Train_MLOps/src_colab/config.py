"""Centralized configuration for the unified PyTorch training pipeline — Cycle 2.

Handles environment detection (Colab vs local vs Vertex AI),
GPU configuration, and unified path resolution for all model families.

Supported families: FCOS, YOLO26_CUSTOM, ESPDet, EXPORT.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Model family constants ──────────────────────────────────────────
MODEL_FAMILIES = {
    "FCOS": {
        "framework": "pytorch",
        "variants": ["MobileNetV3S_FCOS"],
        "dataset_format": "yolo",
        "training_phases": 2,
    },
    "YOLO26_CUSTOM": {
        "framework": "pytorch",
        "variants": ["yolo26n_custom", "yolo26s_custom"],
        "dataset_format": "yolo",
        "training_phases": 2,
    },
    "ESPDet": {
        "framework": "pytorch",
        "variants": ["espdet_pico"],
        "dataset_format": "yolo",
        "training_phases": 2,
    },
    "EXPORT": {
        "framework": "pytorch",
        "variants": ["export"],
        "dataset_format": "yolo",
        "training_phases": 0,
    },
}


# ── Derived convenience constants ───────────────────────────────────
MODEL_VARIANTS: Dict[str, List[str]] = {
    family: info["variants"] for family, info in MODEL_FAMILIES.items()
}

TRAINING_FRAMEWORKS: Dict[str, str] = {
    family: info["framework"] for family, info in MODEL_FAMILIES.items()
}

BASE_IMG_SIZE: int = 224  # Target resolution for ESP32-S3


# ── Dataset master class definitions ────────────────────────────────
DATASET_MASTER_CLASSES = {
    "iodc_yolo": ["dog", "door", "obstacle", "person", "stair"],
    "yolo26": ["dog", "door", "obstacle", "person", "stair"],
    "yolo_v11": ["dog", "door", "obstacle", "person", "stair"],
}


# ── Family classification helpers ───────────────────────────────────

def is_fcos_family(family: str) -> bool:
    """Check if model family is FCOS-based (MobileNetV3 + FCOS head)."""
    return family == "FCOS"


def is_yolo26_custom_family(family: str) -> bool:
    """Check if model family is YOLO26 with custom training loop."""
    return family == "YOLO26_CUSTOM"


def is_espdet_family(family: str) -> bool:
    """Check if model family is ESPDet (custom Espressif architecture)."""
    return family == "ESPDet"


def is_export_family(family: str) -> bool:
    """Check if this is an export-only job."""
    return family == "EXPORT"


def is_pytorch_family(family: str) -> bool:
    """All families in Cycle 2 are PyTorch-based."""
    return family in MODEL_FAMILIES


def get_framework(family: str) -> str:
    """Return 'pytorch' for all supported families."""
    return MODEL_FAMILIES.get(family, {}).get("framework", "pytorch")


def get_dataset_format(family: str) -> str:
    """Return expected dataset format (always 'yolo' in Cycle 2)."""
    return MODEL_FAMILIES.get(family, {}).get("dataset_format", "yolo")


def get_yolo_device(env: "ColabEnvironment") -> str:
    """Return the best device string for Ultralytics YOLO operations."""
    if env.cuda_available:
        return "0"
    if env.mps_available:
        return "mps"
    return "cpu"


def get_torch_device(env: "ColabEnvironment") -> str:
    """Return the best device string for PyTorch operations."""
    if env.cuda_available:
        return "cuda:0"
    if env.mps_available:
        return "mps"
    return "cpu"


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


def detect_environment() -> ColabEnvironment:
    """Detect runtime environment and GPU availability (PyTorch only)."""
    env = ColabEnvironment()

    # Detect Vertex AI
    if os.environ.get("AIP_MODEL_DIR") or os.environ.get("CLOUD_ML_PROJECT_ID"):
        env.is_vertex_ai = True
        env.is_colab = False
        env.is_local = False
    else:
        try:
            import google.colab  # noqa: F401
            env.is_colab = True
            env.is_local = False
        except ImportError:
            env.is_colab = False
            env.is_local = True

    # PyTorch GPU detection
    try:
        import torch
        env.cuda_available = torch.cuda.is_available()
        env.mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if env.cuda_available:
            env.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            raw = getattr(props, "total_memory", 0) or getattr(props, "total_mem", 0)
            env.gpu_memory_mb = int(raw / 1024 / 1024)
            env.gpu_available = True
        elif env.mps_available:
            env.gpu_name = "Apple MPS"
            env.gpu_available = True
    except ImportError:
        pass

    return env


def setup_gpu(env: ColabEnvironment) -> None:
    """Print GPU configuration summary."""
    runtime = "Vertex AI" if env.is_vertex_ai else (
        "Google Colab" if env.is_colab else "Local"
    )
    print(f"\n🖥️  Entorno: {runtime}")
    print(f"🎮 GPU: {env.gpu_name}")
    if env.gpu_memory_mb:
        print(f"   VRAM: {env.gpu_memory_mb:,} MB")
    print(
        f"   CUDA: {'✅' if env.cuda_available else '❌'}  |  "
        f"MPS: {'✅' if env.mps_available else '❌'}"
    )


# ── Path management ─────────────────────────────────────────────────
@dataclass
class ProjectPaths:
    """Unified path container for all project directories."""
    project_root: Path = field(default_factory=Path)
    datasets_dir: Path = field(default_factory=Path)
    models_dir: Path = field(default_factory=Path)
    checkpoints_dir: Path = field(default_factory=Path)
    final_export_dir: Path = field(default_factory=Path)
    logs_dir: Path = field(default_factory=Path)
    experiments_dir: Path = field(default_factory=Path)
    reports_dir: Path = field(default_factory=Path)
    runs_dir: Path = field(default_factory=Path)


def _resolve_project_root(env: ColabEnvironment) -> Path:
    """Determine the project root depending on the runtime."""
    if env.is_colab:
        return Path("/content/drive/MyDrive/TFM_UNIR")
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

    for d in [
        p.models_dir, p.checkpoints_dir, p.final_export_dir,
        p.logs_dir, p.experiments_dir, p.reports_dir, p.runs_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    return p


# ── High-level setup ────────────────────────────────────────────────
def setup_environment(
    project_root: Optional[Path] = None,
) -> Tuple[ColabEnvironment, ProjectPaths]:
    """One-call environment setup (PyTorch only)."""
    env = detect_environment()

    if env.is_colab and not env.is_vertex_ai:
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
        except Exception as exc:
            print(f"⚠️ No se pudo montar Google Drive: {exc}")

    setup_gpu(env)
    paths = create_project_paths(project_root=project_root, env=env)

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
