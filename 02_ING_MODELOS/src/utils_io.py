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
    print(msg)


def safe_mkdir(path: str | Path) -> bool:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        log(f"⚠️ safe_mkdir failed: {path} -> {exc}")
        return False


def safe_read_json(path: str | Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log(f"⚠️ JSON no encontrado: {path}")
    except json.JSONDecodeError as exc:
        log(f"⚠️ JSON inválido: {path} -> {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        log(f"⚠️ Error leyendo JSON: {path} -> {exc}")
    return None


def safe_write_text(path: str | Path, content: str) -> bool:
    try:
        safe_mkdir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        log(f"⚠️ Error escribiendo: {path} -> {exc}")
        return False


def safe_copy(src: str | Path, dst: str | Path) -> bool:
    try:
        if not os.path.exists(src):
            log(f"⚠️ Archivo no encontrado: {src}")
            return False
        safe_mkdir(Path(dst).parent)
        shutil.copy2(src, dst)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        log(f"⚠️ Error copiando {src} -> {dst}: {exc}")
        return False


def safe_filesize_mb(path: str | Path) -> Optional[float]:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except FileNotFoundError:
        log(f"⚠️ Archivo no encontrado: {path}")
    except Exception as exc:  # pragma: no cover - defensive
        log(f"⚠️ Error tamaño archivo: {path} -> {exc}")
    return None


def safe_exists(path: str | Path) -> bool:
    try:
        return os.path.exists(path)
    except Exception:  # pragma: no cover - defensive
        return False
