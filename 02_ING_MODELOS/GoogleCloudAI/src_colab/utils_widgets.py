"""Interactive ipywidgets for experiment configuration in Colab/Jupyter.

Provides dynamic forms that adapt parameters based on the selected model
family (YOLO11, YOLO26, MobileNetV2, MobileNetV3).
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

from .config import MODEL_FAMILIES, is_yolo_family, is_mobilenet_family


# ── Experiment config container ─────────────────────────────────────
@dataclass
class ExperimentSetup:
    """Container for all experiment parameters selected via widgets."""

    # Model
    model_family: str = "YOLO26"
    model_variant: str = "yolo26n"
    version: str = "v1"
    description: str = ""

    # Dataset
    dataset_name: str = "yolo26"
    class_names: List[str] = field(default_factory=lambda: ["obstacle"])
    num_classes: int = 1
    img_size: int = 224

    # Common training
    batch_size: int = 32
    patience: int = 30
    seed: int = 42
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

    # YOLO-specific
    yolo_config: Dict[str, Any] = field(default_factory=dict)

    # MobileNet-specific
    mobilenet_config: Dict[str, Any] = field(default_factory=dict)

    # Derived
    experiment_name: str = ""

    def compute_experiment_name(self) -> str:
        """Build experiment name from model + version."""
        if is_yolo_family(self.model_family):
            return f"{self.model_variant}_{self.version}"
        # MobileNet family
        mapping = {
            "MobileNetV3S_SSDLite": "MBNTv3S_ssdlite",
            "MobileNetV3L_SSDLite": "MBNTv3L_ssdlite",
            "MobileNetV2_SSDLite": "MBNTv2_ssdlite",
        }
        prefix = mapping.get(self.model_variant, self.model_variant)
        return f"{prefix}_{self.version}"


# ── Default YOLO config ─────────────────────────────────────────────
_YOLO_DEFAULTS: Dict[str, Any] = {
    "epochs": 100,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": True,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "mosaic": 1.0,
    "mixup": 0.1,
    "close_mosaic": 10,
    "copy_paste": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "erasing": 0.0,
    "box": 7.5,
    "cls": 0.5,
    "freeze": None,
    "amp": True,
    "workers": 4,
    "max_det": 300,
}

# ── Default MobileNet config ────────────────────────────────────────
_MOBILENET_DEFAULTS: Dict[str, Any] = {
    "phase1_epochs": 30,
    "phase1_lr": 1e-3,
    "phase2_epochs": 50,
    "phase2_lr": 5e-5,
    "phase2_unfreeze_layers": 40,
    "augmentation_level": "medium",
    "backbone_alpha": 1.0,
    "minimalistic": True,
    "dropout_rate": 0.2,
    "l2_reg": 1e-4,
    "feature_channels": 128,
    "num_anchors_per_cell": 9,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "neg_pos_ratio": 3,
    "use_copy_paste": False,
    "use_class_weights": True,
    "class_weight_method": "effective_samples",
}


# ── Widget builders ─────────────────────────────────────────────────

def _variant_options(family: str) -> List[str]:
    """Return variant dropdown options for a model family."""
    return MODEL_FAMILIES.get(family, {}).get("variants", [])


def create_model_selector(
    on_apply: Optional[Callable[[ExperimentSetup], None]] = None,
) -> ExperimentSetup:
    """Create the full interactive widget panel.

    When the user clicks *Aplicar Configuración*, the returned
    ``ExperimentSetup`` object is populated and ``on_apply`` is called.

    If ipywidgets is not available, falls back to a manual dict.
    """
    setup = ExperimentSetup()

    if not WIDGETS_AVAILABLE:
        print("⚠️ ipywidgets no disponible. Configura manualmente ExperimentSetup.")
        return setup

    # ── Section 1: Model selection ──
    w_family = widgets.Dropdown(
        options=list(MODEL_FAMILIES.keys()),
        value="YOLO26",
        description="Familia:",
        style={"description_width": "120px"},
    )
    w_variant = widgets.Dropdown(
        options=_variant_options("YOLO26"),
        value="yolo26n",
        description="Variante:",
        style={"description_width": "120px"},
    )
    w_version = widgets.Text(
        value="v1", description="Versión:", style={"description_width": "120px"}
    )
    w_description = widgets.Textarea(
        value="",
        description="Descripción:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="80%"),
    )

    def _on_family_change(change: Dict) -> None:
        opts = _variant_options(change["new"])
        w_variant.options = opts
        w_variant.value = opts[0] if opts else ""
        # Show/hide family-specific panels
        yolo_box.layout.display = "block" if is_yolo_family(change["new"]) else "none"
        mnet_box.layout.display = (
            "block" if is_mobilenet_family(change["new"]) else "none"
        )

    w_family.observe(_on_family_change, names="value")

    model_box = widgets.VBox(
        [
            widgets.HTML("<h3>📦 Selección de Modelo</h3>"),
            w_family,
            w_variant,
            w_version,
            w_description,
        ]
    )

    # ── Section 2: Dataset ──
    w_dataset = widgets.Text(
        value="yolo26",
        description="Dataset:",
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
        description="Img Size:",
        style={"description_width": "120px"},
    )

    dataset_box = widgets.VBox(
        [
            widgets.HTML("<h3>📊 Dataset</h3>"),
            w_dataset,
            w_classes,
            w_imgsize,
        ]
    )

    # ── Section 3: Common training ──
    w_batch = widgets.IntSlider(
        value=32, min=4, max=128, step=4,
        description="Batch Size:",
        style={"description_width": "120px"},
    )
    w_patience = widgets.IntSlider(
        value=30, min=1, max=100, step=1,
        description="Patience:",
        style={"description_width": "120px"},
    )
    w_seed = widgets.IntText(
        value=42, description="Seed:",
        style={"description_width": "120px"},
    )
    w_conf = widgets.FloatSlider(
        value=0.25, min=0.01, max=0.9, step=0.05,
        description="Conf Thresh:",
        style={"description_width": "120px"},
    )
    w_iou = widgets.FloatSlider(
        value=0.45, min=0.1, max=0.9, step=0.05,
        description="IoU Thresh:",
        style={"description_width": "120px"},
    )

    common_box = widgets.VBox(
        [
            widgets.HTML("<h3>⚙️ Parámetros Comunes</h3>"),
            w_batch,
            w_patience,
            w_seed,
            w_conf,
            w_iou,
        ]
    )

    # ── Section 4: YOLO-specific ──
    w_yolo_epochs = widgets.IntSlider(
        value=50, min=1, max=500, step=1,
        description="Épocas:",
        style={"description_width": "140px"},
    )
    w_yolo_optimizer = widgets.Dropdown(
        options=["auto", "AdamW", "SGD", "Adam", "MuSGD"],
        value="auto",
        description="Optimizer:",
        style={"description_width": "140px"},
    )
    w_yolo_lr0 = widgets.FloatLogSlider(
        value=0.01, min=-4, max=-1, step=0.1,
        description="LR Inicial:",
        style={"description_width": "140px"},
        readout_format=".5f",
    )
    w_yolo_lrf = widgets.FloatSlider(
        value=0.01, min=0.001, max=0.5, step=0.005,
        description="LR Final (frac):",
        style={"description_width": "140px"},
    )
    w_yolo_cos_lr = widgets.Checkbox(
        value=True, description="Cosine LR",
        style={"description_width": "140px"},
    )
    w_yolo_mosaic = widgets.FloatSlider(
        value=1.0, min=0.0, max=1.0, step=0.1,
        description="Mosaic:",
        style={"description_width": "140px"},
    )
    w_yolo_mixup = widgets.FloatSlider(
        value=0.1, min=0.0, max=1.0, step=0.05,
        description="Mixup:",
        style={"description_width": "140px"},
    )
    w_yolo_copy_paste = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.05,
        description="Copy-Paste:",
        style={"description_width": "140px"},
    )
    w_yolo_close_mosaic = widgets.IntSlider(
        value=10, min=0, max=30, step=1,
        description="Close Mosaic:",
        style={"description_width": "140px"},
    )
    w_yolo_box = widgets.FloatSlider(
        value=7.5, min=1.0, max=15.0, step=0.5,
        description="Box Loss W:",
        style={"description_width": "140px"},
    )
    w_yolo_cls = widgets.FloatSlider(
        value=0.5, min=0.1, max=5.0, step=0.1,
        description="Cls Loss W:",
        style={"description_width": "140px"},
    )
    w_yolo_scale = widgets.FloatSlider(
        value=0.5, min=0.0, max=1.0, step=0.1,
        description="Scale:",
        style={"description_width": "140px"},
    )
    w_yolo_fliplr = widgets.FloatSlider(
        value=0.5, min=0.0, max=1.0, step=0.1,
        description="Flip LR:",
        style={"description_width": "140px"},
    )

    yolo_box = widgets.VBox(
        [
            widgets.HTML("<h3>🔷 Parámetros YOLO</h3>"),
            w_yolo_epochs,
            w_yolo_optimizer,
            widgets.HBox([w_yolo_lr0, w_yolo_lrf]),
            w_yolo_cos_lr,
            widgets.HTML("<b>Augmentación:</b>"),
            widgets.HBox([w_yolo_mosaic, w_yolo_mixup]),
            widgets.HBox([w_yolo_copy_paste, w_yolo_close_mosaic]),
            widgets.HBox([w_yolo_scale, w_yolo_fliplr]),
            widgets.HTML("<b>Loss:</b>"),
            widgets.HBox([w_yolo_box, w_yolo_cls]),
        ]
    )

    # ── Section 5: MobileNet-specific ──
    w_mnet_p1_epochs = widgets.IntSlider(
        value=30, min=1, max=100, step=1,
        description="Phase1 Épocas:",
        style={"description_width": "160px"},
    )
    w_mnet_p1_lr = widgets.FloatLogSlider(
        value=1e-3, min=-5, max=-2, step=0.1,
        description="Phase1 LR:",
        style={"description_width": "160px"},
        readout_format=".6f",
    )
    w_mnet_p2_epochs = widgets.IntSlider(
        value=50, min=1, max=200, step=1,
        description="Phase2 Épocas:",
        style={"description_width": "160px"},
    )
    w_mnet_p2_lr = widgets.FloatLogSlider(
        value=5e-5, min=-6, max=-3, step=0.1,
        description="Phase2 LR:",
        style={"description_width": "160px"},
        readout_format=".6f",
    )
    w_mnet_unfreeze = widgets.IntSlider(
        value=40, min=0, max=100, step=5,
        description="Unfreeze Layers:",
        style={"description_width": "160px"},
    )
    w_mnet_aug = widgets.Dropdown(
        options=["none", "light", "medium", "heavy"],
        value="medium",
        description="Augmentación:",
        style={"description_width": "160px"},
    )
    w_mnet_alpha = widgets.Dropdown(
        options=[0.35, 0.5, 0.75, 1.0],
        value=1.0,
        description="Backbone Alpha:",
        style={"description_width": "160px"},
    )
    w_mnet_minimalistic = widgets.Checkbox(
        value=True,
        description="Minimalistic (ReLU)",
        style={"description_width": "160px"},
    )
    w_mnet_dropout = widgets.FloatSlider(
        value=0.2, min=0.0, max=0.5, step=0.05,
        description="Dropout:",
        style={"description_width": "160px"},
    )
    w_mnet_l2 = widgets.FloatLogSlider(
        value=1e-4, min=-6, max=-2, step=0.1,
        description="L2 Reg:",
        style={"description_width": "160px"},
        readout_format=".6f",
    )
    w_mnet_focal_alpha = widgets.FloatSlider(
        value=0.25, min=0.0, max=1.0, step=0.05,
        description="Focal Alpha:",
        style={"description_width": "160px"},
    )
    w_mnet_focal_gamma = widgets.FloatSlider(
        value=2.0, min=0.5, max=5.0, step=0.5,
        description="Focal Gamma:",
        style={"description_width": "160px"},
    )
    w_mnet_neg_pos = widgets.IntSlider(
        value=3, min=1, max=10, step=1,
        description="Neg/Pos Ratio:",
        style={"description_width": "160px"},
    )
    w_mnet_n_anchors = widgets.IntSlider(
        value=9, min=3, max=18, step=3,
        description="Anchors/Cell:",
        style={"description_width": "160px"},
    )
    w_mnet_feat_ch = widgets.Dropdown(
        options=[64, 96, 128, 192, 256],
        value=128,
        description="Feature Channels:",
        style={"description_width": "160px"},
    )
    w_mnet_copy_paste = widgets.Checkbox(
        value=False,
        description="Copy-Paste Aug",
        style={"description_width": "160px"},
    )
    w_mnet_class_weights = widgets.Checkbox(
        value=True,
        description="Usar Class Weights",
        style={"description_width": "160px"},
    )
    w_mnet_cw_method = widgets.Dropdown(
        options=["inverse_freq", "sqrt_inverse", "effective_samples"],
        value="effective_samples",
        description="CW Method:",
        style={"description_width": "160px"},
    )

    mnet_box = widgets.VBox(
        [
            widgets.HTML("<h3>🟢 Parámetros MobileNet + SSD-Lite</h3>"),
            widgets.HTML("<b>Entrenamiento 2 Fases:</b>"),
            widgets.HBox([w_mnet_p1_epochs, w_mnet_p1_lr]),
            widgets.HBox([w_mnet_p2_epochs, w_mnet_p2_lr]),
            w_mnet_unfreeze,
            widgets.HTML("<b>Backbone:</b>"),
            widgets.HBox([w_mnet_alpha, w_mnet_minimalistic]),
            widgets.HBox([w_mnet_dropout, w_mnet_l2]),
            widgets.HTML("<b>Detección SSD:</b>"),
            widgets.HBox([w_mnet_n_anchors, w_mnet_feat_ch]),
            widgets.HTML("<b>Loss:</b>"),
            widgets.HBox([w_mnet_focal_alpha, w_mnet_focal_gamma]),
            w_mnet_neg_pos,
            widgets.HTML("<b>Dataset / Augmentación:</b>"),
            w_mnet_aug,
            widgets.HBox([w_mnet_copy_paste, w_mnet_class_weights]),
            w_mnet_cw_method,
        ],
        layout=widgets.Layout(display="none"),  # Hidden initially (YOLO is default)
    )

    # ── Apply button + output ──
    output = widgets.Output()
    btn_apply = widgets.Button(
        description="✅ Aplicar Configuración",
        button_style="success",
        layout=widgets.Layout(width="300px", height="40px"),
    )

    def _on_apply(_: Any) -> None:
        with output:
            clear_output()

            # Populate setup
            setup.model_family = w_family.value
            setup.model_variant = w_variant.value
            setup.version = w_version.value
            setup.description = w_description.value

            setup.dataset_name = w_dataset.value
            setup.class_names = [
                c.strip() for c in w_classes.value.split(",") if c.strip()
            ]
            setup.num_classes = len(setup.class_names)
            setup.img_size = w_imgsize.value

            setup.batch_size = w_batch.value
            setup.patience = w_patience.value
            setup.seed = w_seed.value
            setup.conf_threshold = w_conf.value
            setup.iou_threshold = w_iou.value

            if is_yolo_family(setup.model_family):
                setup.yolo_config = {
                    "epochs": w_yolo_epochs.value,
                    "optimizer": w_yolo_optimizer.value,
                    "lr0": w_yolo_lr0.value,
                    "lrf": w_yolo_lrf.value,
                    "cos_lr": w_yolo_cos_lr.value,
                    "mosaic": w_yolo_mosaic.value,
                    "mixup": w_yolo_mixup.value,
                    "copy_paste": w_yolo_copy_paste.value,
                    "close_mosaic": w_yolo_close_mosaic.value,
                    "box": w_yolo_box.value,
                    "cls": w_yolo_cls.value,
                    "scale": w_yolo_scale.value,
                    "fliplr": w_yolo_fliplr.value,
                }
            else:
                setup.mobilenet_config = {
                    "phase1_epochs": w_mnet_p1_epochs.value,
                    "phase1_lr": w_mnet_p1_lr.value,
                    "phase2_epochs": w_mnet_p2_epochs.value,
                    "phase2_lr": w_mnet_p2_lr.value,
                    "phase2_unfreeze_layers": w_mnet_unfreeze.value,
                    "augmentation_level": w_mnet_aug.value,
                    "backbone_alpha": w_mnet_alpha.value,
                    "minimalistic": w_mnet_minimalistic.value,
                    "dropout_rate": w_mnet_dropout.value,
                    "l2_reg": w_mnet_l2.value,
                    "feature_channels": w_mnet_feat_ch.value,
                    "num_anchors_per_cell": w_mnet_n_anchors.value,
                    "focal_alpha": w_mnet_focal_alpha.value,
                    "focal_gamma": w_mnet_focal_gamma.value,
                    "neg_pos_ratio": w_mnet_neg_pos.value,
                    "use_copy_paste": w_mnet_copy_paste.value,
                    "use_class_weights": w_mnet_class_weights.value,
                    "class_weight_method": w_mnet_cw_method.value,
                }

            setup.experiment_name = setup.compute_experiment_name()
            _print_setup_summary(setup)

            if on_apply:
                on_apply(setup)

    btn_apply.on_click(_on_apply)

    # ── Assemble ──
    panel = widgets.VBox(
        [
            widgets.HTML("<h2>🧪 Configuración del Experimento</h2>"),
            model_box,
            dataset_box,
            common_box,
            yolo_box,
            mnet_box,
            btn_apply,
            output,
        ],
        layout=widgets.Layout(padding="10px"),
    )

    display(panel)
    return setup


# ── Helpers ──────────────────────────────────────────────────────────

def _print_setup_summary(s: ExperimentSetup) -> None:
    """Print a formatted summary of the experiment configuration."""
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

    if is_yolo_family(s.model_family) and s.yolo_config:
        c = s.yolo_config
        print(f"  🔷 YOLO Config:")
        print(f"     Épocas: {c['epochs']}  |  Optimizer: {c['optimizer']}")
        print(f"     LR: {c['lr0']:.5f} → {c['lr0'] * c['lrf']:.6f}")
        print(f"     Cosine LR: {c['cos_lr']}")
        print(f"     Mosaic: {c['mosaic']}  |  Mixup: {c['mixup']}  |  "
              f"Copy-Paste: {c['copy_paste']}")
        print(f"     Box: {c['box']}  |  Cls: {c['cls']}")

    if is_mobilenet_family(s.model_family) and s.mobilenet_config:
        c = s.mobilenet_config
        print(f"  🟢 MobileNet Config:")
        print(f"     Phase1: {c['phase1_epochs']} ep @ LR={c['phase1_lr']:.6f}")
        print(f"     Phase2: {c['phase2_epochs']} ep @ LR={c['phase2_lr']:.6f}")
        print(f"     Unfreeze: {c['phase2_unfreeze_layers']} layers")
        print(f"     Alpha: {c['backbone_alpha']}  |  "
              f"Minimalistic: {c['minimalistic']}")
        print(f"     Dropout: {c['dropout_rate']}  |  L2: {c['l2_reg']:.6f}")
        print(f"     Anchors/cell: {c['num_anchors_per_cell']}  |  "
              f"Feat Ch: {c['feature_channels']}")
        print(f"     Focal α={c['focal_alpha']}  γ={c['focal_gamma']}  |  "
              f"Neg/Pos: {c['neg_pos_ratio']}")
        print(f"     Augmentation: {c['augmentation_level']}  |  "
              f"Copy-Paste: {c['use_copy_paste']}")

    print(f"\n✅ Configuración aplicada correctamente")


def create_manual_setup(
    model_family: str = "YOLO26",
    model_variant: str = "yolo26n",
    version: str = "v1",
    description: str = "",
    dataset_name: str = "yolo26",
    class_names: Optional[List[str]] = None,
    img_size: int = 224,
    batch_size: int = 32,
    patience: int = 30,
    seed: int = 42,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    **kwargs: Any,
) -> ExperimentSetup:
    """Create ExperimentSetup programmatically (for non-widget environments).

    Extra keyword arguments are routed to ``yolo_config`` or
    ``mobilenet_config`` depending on the family.
    """
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

    if is_yolo_family(model_family):
        yolo_cfg = dict(_YOLO_DEFAULTS)
        yolo_cfg.update(kwargs)
        setup.yolo_config = yolo_cfg
    else:
        mnet_cfg = dict(_MOBILENET_DEFAULTS)
        mnet_cfg.update(kwargs)
        setup.mobilenet_config = mnet_cfg

    setup.experiment_name = setup.compute_experiment_name()
    _print_setup_summary(setup)
    return setup
