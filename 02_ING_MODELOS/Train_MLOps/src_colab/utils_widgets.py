"""Experiment configuration — Cycle 2 (PyTorch only).

Supports FCOS, YOLO26_CUSTOM, ESPDet families with 2-phase training
and progressive resizing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

from .config import (
    MODEL_FAMILIES,
    is_fcos_family,
    is_yolo26_custom_family,
    is_espdet_family,
)


# ── Experiment Setup dataclass ──────────────────────────────────────

@dataclass
class ExperimentSetup:
    """Container for all experiment parameters."""
    model_family: str = "FCOS"
    model_variant: str = "MobileNetV3S_FCOS"
    version: str = "v1"
    description: str = ""
    dataset_name: str = "iodc_yolo"
    class_names: List[str] = field(default_factory=lambda: ["dog", "door", "obstacle", "person", "stair"])
    num_classes: int = 5
    img_size: int = 224
    batch_size: int = 32
    patience: int = 30
    seed: int = 42
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

    # Family-specific config dicts
    fcos_config: Dict[str, Any] = field(default_factory=dict)
    yolo26_custom_config: Dict[str, Any] = field(default_factory=dict)
    espdet_config: Dict[str, Any] = field(default_factory=dict)

    experiment_name: str = ""

    def compute_experiment_name(self) -> str:
        """Generate experiment name from variant + version."""
        mapping = {
            "MobileNetV3S_FCOS": "fcos_v3s",
            "yolo26n_custom": "yolo26n_custom",
            "espdet_pico": "espdet_pico",
        }
        prefix = mapping.get(self.model_variant, self.model_variant)
        return f"{prefix}_{self.version}"

    @property
    def family_config(self) -> Dict[str, Any]:
        """Return the active family config dict."""
        if is_fcos_family(self.model_family):
            return self.fcos_config
        if is_yolo26_custom_family(self.model_family):
            return self.yolo26_custom_config
        if is_espdet_family(self.model_family):
            return self.espdet_config
        return {}


# ── Default hyperparameters per family ──────────────────────────────

_FCOS_DEFAULTS: Dict[str, Any] = {
    # Architecture
    "fpn_channels": 64,
    "backbone": "mobilenet_v3_small",
    # Phase 1 — Freeze backbone
    "phase1_epochs": 50,
    "phase1_lr": 1e-3,
    "phase1_freeze": "backbone",
    # Phase 2 — Fine-tune all
    "phase2_epochs": 100,
    "phase2_lr": 1e-4,
    "phase2_freeze": "none",
    # Optimizer
    "optimizer": "AdamW",
    "weight_decay": 0.0005,
    # Progressive Resizing {epoch: img_size}
    "resize_schedule": {0: 640, 30: 416, 90: 320, 120: 224},
    # Loss weights
    "loss_cls_weight": 1.0,
    "loss_reg_weight": 1.0,
    "loss_centerness_weight": 1.0,
    # Augmentation (Albumentations)
    "aug_brightness_limit": 0.2,
    "aug_contrast_limit": 0.2,
    "aug_hue_shift_limit": 20,
    "aug_sat_shift_limit": 30,
    "aug_val_shift_limit": 20,
    "aug_shift_limit": 0.1,
    "aug_scale_limit": 0.2,
    "aug_rotate_limit": 15,
    "aug_hflip_prob": 0.5,
    # Workers
    "workers": 4,
    "amp": True,
}

_YOLO26_CUSTOM_DEFAULTS: Dict[str, Any] = {
    # Architecture
    "pretrained_weights": "yolo26n.pt",
    "pretrained_gcs_uri": "",
    # Phase 1 — Freeze backbone
    "phase1_epochs": 50,
    "phase1_lr": 1e-3,
    "phase1_freeze_layers": 10,
    # Phase 2 — Fine-tune all
    "phase2_epochs": 100,
    "phase2_lr": 1e-4,
    "phase2_freeze": "none",
    # Optimizer
    "optimizer": "AdamW",
    "weight_decay": 0.0005,
    # Progressive Resizing
    "resize_schedule": {0: 640, 30: 416, 90: 320, 120: 224},
    # Augmentation
    "use_mosaic": True,
    "close_mosaic": 20,
    "use_mixup": True,
    "mixup_alpha": 0.15,
    "aug_hflip_prob": 0.5,
    "aug_brightness_limit": 0.2,
    "aug_hue_shift_limit": 20,
    # Loss
    "loss": "ultralytics",  # use v8DetectionLoss
    # Workers
    "workers": 4,
    "amp": True,
}

_ESPDET_DEFAULTS: Dict[str, Any] = {
    # Architecture
    "pretrained_weights": "espdet_pico_coco.pt",
    "pretrained_gcs_uri": "",
    "reg_max": 1,
    # Phase 1 — Freeze backbone
    "phase1_epochs": 50,
    "phase1_lr": 1e-3,
    "phase1_freeze": "backbone",
    # Phase 2 — Fine-tune all
    "phase2_epochs": 100,
    "phase2_lr": 1e-4,
    "phase2_freeze": "none",
    # Optimizer
    "optimizer": "AdamW",
    "weight_decay": 0.0005,
    # Progressive Resizing
    "resize_schedule": {0: 640, 30: 416, 90: 320, 120: 224},
    # Augmentation (aggressive illumination)
    "aug_brightness_limit": 0.3,
    "aug_contrast_limit": 0.3,
    "aug_hue_shift_limit": 25,
    "aug_sat_shift_limit": 35,
    "aug_val_shift_limit": 25,
    "aug_shift_limit": 0.1,
    "aug_scale_limit": 0.2,
    "aug_rotate_limit": 15,
    "aug_hflip_prob": 0.5,
    # Workers
    "workers": 4,
    "amp": True,
}


# ── Summary printer ─────────────────────────────────────────────────

def _print_setup_summary(s: ExperimentSetup) -> None:
    """Print a human-readable summary of the experiment configuration."""
    print(f"\n🧪 CONFIGURACIÓN DEL EXPERIMENTO")
    print(f"  Nombre:       {s.experiment_name}")
    print(f"  Familia:      {s.model_family}")
    print(f"  Variante:     {s.model_variant}")
    print(f"  Versión:      {s.version}")
    print(f"  Descripción:  {s.description or '(sin descripción)'}")
    print(f"  Dataset:      {s.dataset_name}")
    print(f"  Clases ({s.num_classes}):  {s.class_names}")
    print(f"  Img Size:     {s.img_size}×{s.img_size}")
    print(f"  Batch Size:   {s.batch_size}")
    print(f"  Patience:     {s.patience}")
    print(f"  Seed:         {s.seed}")
    print(f"  Conf Thresh:  {s.conf_threshold}")
    print(f"  IoU Thresh:   {s.iou_threshold}")

    cfg = s.family_config
    if cfg:
        # Common 2-phase info
        p1e = cfg.get("phase1_epochs", "?")
        p1lr = cfg.get("phase1_lr", "?")
        p2e = cfg.get("phase2_epochs", "?")
        p2lr = cfg.get("phase2_lr", "?")
        sched = cfg.get("resize_schedule", {})
        print(f"  📐 2-Phase Training:")
        print(f"     Phase 1: {p1e} epochs @ LR={p1lr}")
        print(f"     Phase 2: {p2e} epochs @ LR={p2lr}")
        print(f"     Resize Schedule: {sched}")
        print(f"     Optimizer: {cfg.get('optimizer', '?')} | "
              f"WD: {cfg.get('weight_decay', '?')}")

        if is_fcos_family(s.model_family):
            print(f"  🔷 FCOS Config:")
            print(f"     FPN Channels: {cfg.get('fpn_channels', 64)}")
            print(f"     Backbone: {cfg.get('backbone', 'mobilenet_v3_small')}")
        elif is_yolo26_custom_family(s.model_family):
            print(f"  🔶 YOLO26 Custom Config:")
            print(f"     Pretrained: {cfg.get('pretrained_weights', '?')}")
            print(f"     Loss: {cfg.get('loss', '?')}")
            print(f"     Mosaic: {cfg.get('use_mosaic')} | "
                  f"Mixup: {cfg.get('use_mixup')}")
        elif is_espdet_family(s.model_family):
            print(f"  🟢 ESPDet Config:")
            print(f"     Pretrained: {cfg.get('pretrained_weights', '?')}")
            print(f"     reg_max: {cfg.get('reg_max', 1)}")

    print(f"\n✅ Configuración aplicada correctamente")


# ── Manual setup (for Vertex AI / scripts) ──────────────────────────

def create_manual_setup(
    model_family: str = "FCOS",
    model_variant: str = "MobileNetV3S_FCOS",
    version: str = "v1",
    description: str = "",
    dataset_name: str = "iodc_yolo",
    class_names: Optional[List[str]] = None,
    img_size: int = 224,
    batch_size: int = 32,
    patience: int = 30,
    seed: int = 42,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    **kwargs: Any,
) -> ExperimentSetup:
    """Create ExperimentSetup programmatically (no widgets needed)."""
    if class_names is None:
        class_names = ["dog", "door", "obstacle", "person", "stair"]

    setup = ExperimentSetup(
        model_family=model_family,
        model_variant=model_variant,
        version=version,
        description=description,
        dataset_name=dataset_name,
        class_names=class_names,
        num_classes=len(class_names),
        img_size=img_size,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    # Assign kwargs to the correct family config
    if is_fcos_family(model_family):
        cfg = dict(_FCOS_DEFAULTS)
        cfg.update(kwargs)
        setup.fcos_config = cfg
    elif is_yolo26_custom_family(model_family):
        cfg = dict(_YOLO26_CUSTOM_DEFAULTS)
        cfg.update(kwargs)
        setup.yolo26_custom_config = cfg
    elif is_espdet_family(model_family):
        cfg = dict(_ESPDET_DEFAULTS)
        cfg.update(kwargs)
        setup.espdet_config = cfg

    setup.experiment_name = setup.compute_experiment_name()
    _print_setup_summary(setup)
    return setup


# ── Widget-based selector (for Colab/Jupyter) ──────────────────────

def create_model_selector(
    on_apply: Optional[Callable[[ExperimentSetup], None]] = None,
) -> ExperimentSetup:
    """Create interactive widget-based experiment configurator.

    Falls back to a plain ExperimentSetup if ipywidgets is unavailable.
    """
    setup = ExperimentSetup()

    if not WIDGETS_AVAILABLE:
        print("⚠️ ipywidgets no disponible. Usa create_manual_setup().")
        return setup

    # Simplified widget panel for Cycle 2 families
    w_family = widgets.Dropdown(
        options=list(MODEL_FAMILIES.keys()),
        value="FCOS",
        description="Familia:",
        style={"description_width": "120px"},
    )

    def _variant_options(fam: str) -> List[str]:
        return MODEL_FAMILIES.get(fam, {}).get("variants", [])

    w_variant = widgets.Dropdown(
        options=_variant_options("FCOS"),
        value="MobileNetV3S_FCOS",
        description="Variante:",
        style={"description_width": "120px"},
    )
    w_version = widgets.Text(
        value="v1", description="Versión:", style={"description_width": "120px"}
    )

    def _on_family_change(change: Dict) -> None:
        opts = _variant_options(change["new"])
        w_variant.options = opts
        w_variant.value = opts[0] if opts else ""

    w_family.observe(_on_family_change, names="value")

    w_dataset = widgets.Text(
        value="iodc_yolo", description="Dataset:",
        style={"description_width": "120px"},
    )
    w_classes = widgets.Text(
        value="dog, door, obstacle, person, stair",
        description="Clases:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="80%"),
    )
    w_imgsize = widgets.IntSlider(
        value=224, min=96, max=640, step=32,
        description="Img Size:", style={"description_width": "120px"},
    )
    w_batch = widgets.IntSlider(
        value=32, min=4, max=128, step=4,
        description="Batch Size:", style={"description_width": "120px"},
    )

    output = widgets.Output()
    btn_apply = widgets.Button(
        description="✅ Aplicar Configuración",
        button_style="success",
        layout=widgets.Layout(width="300px", height="40px"),
    )

    def _on_apply(_: Any) -> None:
        with output:
            clear_output()
            setup.model_family = w_family.value
            setup.model_variant = w_variant.value
            setup.version = w_version.value
            setup.dataset_name = w_dataset.value
            setup.class_names = [
                c.strip() for c in w_classes.value.split(",") if c.strip()
            ]
            setup.num_classes = len(setup.class_names)
            setup.img_size = w_imgsize.value
            setup.batch_size = w_batch.value
            setup.experiment_name = setup.compute_experiment_name()
            _print_setup_summary(setup)
            if on_apply:
                on_apply(setup)

    btn_apply.on_click(_on_apply)

    panel = widgets.VBox([
        widgets.HTML("<h2>🧪 Configuración — Ciclo 2</h2>"),
        w_family, w_variant, w_version,
        w_dataset, w_classes, w_imgsize, w_batch,
        btn_apply, output,
    ], layout=widgets.Layout(padding="10px"))

    display(panel)
    return setup
