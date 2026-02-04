"""Utilities for safe IO operations.
Designed to fail softly with clear logs.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def log(msg: str) -> None:
    """Print a log message."""
    print(msg)


def safe_mkdir(path: str | Path) -> bool:
    """Create directory (and parents) safely."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as exc:
        log(f"⚠️ safe_mkdir failed: {path} -> {exc}")
        return False


def safe_read_json(path: str | Path) -> Optional[Dict[str, Any]]:
    """Read JSON file safely, return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log(f"⚠️ JSON no encontrado: {path}")
    except json.JSONDecodeError as exc:
        log(f"⚠️ JSON inválido: {path} -> {exc}")
    except Exception as exc:
        log(f"⚠️ Error leyendo JSON: {path} -> {exc}")
    return None


def safe_write_json(path: str | Path, data: Dict[str, Any], indent: int = 2) -> bool:
    """Write JSON file safely."""
    try:
        safe_mkdir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except Exception as exc:
        log(f"⚠️ Error escribiendo JSON: {path} -> {exc}")
        return False


def safe_write_text(path: str | Path, content: str) -> bool:
    """Write text file safely."""
    try:
        safe_mkdir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as exc:
        log(f"⚠️ Error escribiendo: {path} -> {exc}")
        return False


def safe_copy(src: str | Path, dst: str | Path) -> bool:
    """Copy file safely."""
    try:
        if not os.path.exists(src):
            log(f"⚠️ Archivo no encontrado: {src}")
            return False
        safe_mkdir(Path(dst).parent)
        shutil.copy2(src, dst)
        return True
    except Exception as exc:
        log(f"⚠️ Error copiando {src} -> {dst}: {exc}")
        return False


def safe_filesize_mb(path: str | Path) -> Optional[float]:
    """Get file size in MB safely."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except FileNotFoundError:
        log(f"⚠️ Archivo no encontrado: {path}")
    except Exception as exc:
        log(f"⚠️ Error tamaño archivo: {path} -> {exc}")
    return None


def safe_exists(path: str | Path) -> bool:
    """Check if path exists safely."""
    try:
        return os.path.exists(path)
    except Exception:
        return False
