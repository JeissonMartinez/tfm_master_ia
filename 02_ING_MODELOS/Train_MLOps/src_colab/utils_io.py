"""Safe I/O utilities: logging, directory creation, JSON/text read-write,
and thin GCS wrappers that delegate to ``trainer.gcs_utils``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def log(msg: str) -> None:
    """Print a timestamped log message."""
    print(msg)


def safe_mkdir(path: str | Path) -> bool:
    """Create directory (and parents) safely."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as exc:
        log(f"⚠️ safe_mkdir failed: {path} → {exc}")
        return False


def read_json(path: str | Path) -> Optional[Dict[str, Any]]:
    """Read JSON file safely, return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log(f"⚠️ JSON no encontrado: {path}")
    except json.JSONDecodeError as exc:
        log(f"⚠️ JSON inválido: {path} → {exc}")
    except Exception as exc:
        log(f"⚠️ Error leyendo JSON: {path} → {exc}")
    return None


def write_json(
    path: str | Path,
    data: Dict[str, Any],
    indent: int = 2,
) -> bool:
    """Write JSON file safely, creating parent dirs."""
    try:
        safe_mkdir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except Exception as exc:
        log(f"⚠️ Error escribiendo JSON: {path} → {exc}")
        return False


def write_text(path: str | Path, content: str) -> bool:
    """Write text file safely."""
    try:
        safe_mkdir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as exc:
        log(f"⚠️ Error escribiendo texto: {path} → {exc}")
        return False


def safe_copy(src: str | Path, dst: str | Path) -> bool:
    """Copy a file safely."""
    try:
        if not os.path.exists(src):
            log(f"⚠️ Archivo no encontrado: {src}")
            return False
        safe_mkdir(Path(dst).parent)
        shutil.copy2(src, dst)
        return True
    except Exception as exc:
        log(f"⚠️ Error copiando {src} → {dst}: {exc}")
        return False


def file_exists(path: str | Path) -> bool:
    """Check if path exists safely."""
    try:
        return os.path.exists(path)
    except Exception:
        return False


def get_file_size_mb(path: str | Path) -> Optional[float]:
    """Return file size in MB, or None on error."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except FileNotFoundError:
        log(f"⚠️ Archivo no encontrado: {path}")
    except Exception as exc:
        log(f"⚠️ Error obteniendo tamaño: {path} → {exc}")
    return None


def write_yaml(path: str | Path, data: Dict[str, Any]) -> bool:
    """Write a YAML file (without PyYAML dependency)."""
    try:
        safe_mkdir(Path(path).parent)
        lines: list[str] = []
        for k, v in data.items():
            if isinstance(v, list):
                lines.append(f"{k}: {v}")
            else:
                lines.append(f"{k}: {v}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return True
    except Exception as exc:
        log(f"⚠️ Error escribiendo YAML: {path} → {exc}")
        return False


# ── GCS helpers (delegate to trainer.gcs_utils) ─────────────────────

def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    """Download a single file from GCS.

    Thin wrapper that delegates to ``trainer.gcs_utils.download_from_gcs``.
    """
    from trainer.gcs_utils import download_from_gcs as _download
    return _download(gcs_uri, local_path)


def upload_to_gcs(local_path: str, gcs_uri: str) -> str:
    """Upload a single file to GCS.

    Thin wrapper that delegates to ``trainer.gcs_utils.upload_file_to_gcs``.
    """
    from trainer.gcs_utils import upload_file_to_gcs
    return upload_file_to_gcs(local_path, gcs_uri)


def setup_experiment_dirs(base_dir: str, run_name: str) -> Dict[str, str]:
    """Create standard experiment sub-directories and return their paths.

    Returns a dict with keys: root, checkpoints, export, plots.
    """
    dirs: Dict[str, str] = {}
    dirs["root"] = os.path.join(base_dir, run_name)
    dirs["checkpoints"] = os.path.join(dirs["root"], "checkpoints")
    dirs["export"] = os.path.join(dirs["root"], "export")
    dirs["plots"] = os.path.join(dirs["root"], "plots")
    for d in dirs.values():
        safe_mkdir(d)
    log(f"📂 Directorios de experimento creados en {dirs['root']}")
    return dirs
