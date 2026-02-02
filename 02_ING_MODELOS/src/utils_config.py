"""Project configuration utilities for SSD/YOLO workflows."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

from .utils_io import safe_mkdir, log


@dataclass
class ProjectConfig:
    base_project_dir: str = "/Users/admin/Documents/TFM_UNIR"

    dataset_root: str = field(init=False)
    train_img_dir: str = field(init=False)
    train_json: str = field(init=False)
    val_img_dir: str = field(init=False)
    val_json: str = field(init=False)
    test_img_dir: str = field(init=False)
    test_json: str = field(init=False)

    stage_dir: str = field(init=False)
    dirs: Dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.dataset_root = os.path.join(self.base_project_dir, "01_ING_DATOS", "Dataset")
        self.train_img_dir = os.path.join(self.dataset_root, "train", "augmented_images")
        self.train_json = os.path.join(self.train_img_dir, "train_final.json")
        self.val_img_dir = os.path.join(self.dataset_root, "valid")
        self.val_json = os.path.join(self.val_img_dir, "val_final.json")
        self.test_img_dir = os.path.join(self.dataset_root, "test")
        self.test_json = os.path.join(self.test_img_dir, "test_final.json")

        self.stage_dir = os.path.join(self.base_project_dir, "02_ING_MODELOS")
        self.dirs = {
            "yolo_dataset": os.path.join(self.stage_dir, "datasets", "yolo_v11"),
            "models_chk": os.path.join(self.stage_dir, "models", "checkpoints"),
            "models_final": os.path.join(self.stage_dir, "models", "final_export"),
            "logs": os.path.join(self.stage_dir, "logs"),
            "visuals": os.path.join(self.stage_dir, "reports", "figures"),
        }

    def ensure_dirs(self) -> None:
        log("🔧 Verificando estructura de carpetas...")
        for _, path in self.dirs.items():
            safe_mkdir(path)
