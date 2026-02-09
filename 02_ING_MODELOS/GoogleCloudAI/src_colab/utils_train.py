"""Unified training utilities for YOLO and MobileNet models.

YOLO (11/26): single-phase training via Ultralytics API.
MobileNet (V2/V3 + SSD-Lite): two-phase training with
  Phase 1 = backbone frozen (warm-up)
  Phase 2 = partial unfreeze (fine-tuning)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .utils_io import log, safe_mkdir, file_exists


# =====================================================================
#  YOLO TRAINING
# =====================================================================

@dataclass
class YoloTrainConfig:
    """Training config that works for both YOLO11 and YOLO26.

    Pass to :func:`train_yolo` and it builds the correct Ultralytics call.
    """
    model: str = "yolo26n.pt"
    imgsz: int = 224
    epochs: int = 100
    patience: int = 50
    batch: int = 16
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    cos_lr: bool = True
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    mosaic: float = 1.0
    mixup: float = 0.1
    close_mosaic: int = 10
    copy_paste: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    erasing: float = 0.0
    box: float = 7.5
    cls: float = 0.5
    val: bool = True
    save: bool = True
    save_period: int = -1
    plots: bool = True
    device: Optional[str] = None
    amp: bool = True
    workers: int = 4
    project: str = "runs"
    name: str = "train"
    exist_ok: bool = True
    freeze: Optional[List[int]] = None
    max_det: int = 300

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for ``model.train(**d)``."""
        d: Dict[str, Any] = {
            "epochs": self.epochs,
            "patience": self.patience,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "cos_lr": self.cos_lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "close_mosaic": self.close_mosaic,
            "copy_paste": self.copy_paste,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "erasing": self.erasing,
            "box": self.box,
            "cls": self.cls,
            "val": self.val,
            "save": self.save,
            "save_period": self.save_period,
            "plots": self.plots,
            "amp": self.amp,
            "workers": self.workers,
            "project": self.project,
            "name": self.name,
            "exist_ok": self.exist_ok,
            "max_det": self.max_det,
            "verbose": True,
        }
        if self.device is not None:
            d["device"] = self.device
        if self.freeze is not None:
            d["freeze"] = self.freeze
        return d

    def summary(self) -> str:
        lines = [
            f"\n🔧 YOLO Training Config: {self.model}",
            f"  Img: {self.imgsz}×{self.imgsz}  |  Epochs: {self.epochs}  |  "
            f"Patience: {self.patience}  |  Batch: {self.batch}",
            f"  Optimizer: {self.optimizer}  |  LR: {self.lr0} → "
            f"{self.lr0 * self.lrf:.6f}  |  Cosine: {self.cos_lr}",
            f"  Mosaic: {self.mosaic}  |  Mixup: {self.mixup}  |  "
            f"Copy-Paste: {self.copy_paste}  |  Close: {self.close_mosaic}",
            f"  Box: {self.box}  |  Cls: {self.cls}  |  Device: {self.device}",
        ]
        return "\n".join(lines)


def train_yolo(
    data_yaml: str,
    cfg: YoloTrainConfig,
    resume: bool = False,
    resume_path: Optional[str] = None,
) -> Optional[Any]:
    """Train a YOLO model (YOLO11 or YOLO26) via Ultralytics.

    Returns the Ultralytics *results* object (or None on failure).
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        log("❌ Ultralytics no disponible.")
        return None

    if not file_exists(data_yaml):
        log(f"❌ data.yaml no encontrado: {data_yaml}")
        return None

    log("\n" + cfg.summary())

    try:
        if resume and resume_path and file_exists(resume_path):
            log(f"🔄 Reanudando desde: {resume_path}")
            model = YOLO(resume_path)
        else:
            log(f"🔄 Cargando modelo base: {cfg.model}")
            model = YOLO(cfg.model)

        params = cfg.to_dict()
        params["data"] = data_yaml
        if resume:
            params["resume"] = True

        is_mps = str(params.get("device", "")).lower() == "mps"
        is_yolo26 = "yolo26" in cfg.model.lower()
        title = "YOLO26" if is_yolo26 else "YOLO11"

        # ── MPS safety ──────────────────────────────────────────
        # MPS + AMP causa RuntimeError en backward pass
        if is_mps:
            params["amp"] = False
            log("⚠️  AMP desactivado automaticamente en MPS (incompatibilidad conocida)")

            # YOLO26 + MPS: mosaic/mixup crean tensores no contiguos
            # que causan "view size is not compatible" en backward pass.
            # Desactivarlos permite entrenar en MPS sin fallback a CPU.
            if is_yolo26:
                changed = []
                if params.get("mosaic", 0) > 0:
                    params["mosaic"] = 0.0
                    changed.append("mosaic")
                if params.get("mixup", 0) > 0:
                    params["mixup"] = 0.0
                    changed.append("mixup")
                if params.get("copy_paste", 0) > 0:
                    params["copy_paste"] = 0.0
                    changed.append("copy_paste")
                if changed:
                    log(f"⚠️  YOLO26 + MPS: desactivando {', '.join(changed)} "
                        "(causan tensores no contiguos en backward pass)")
                    log("   En Colab (CUDA) se mantienen activos automaticamente.")

        log(f"\n🚀 Iniciando entrenamiento {title}...")
        if is_yolo26:
            log("   💡 YOLO26: MuSGD optimizer, end-to-end, DFL-free, ProgLoss\n")
        else:
            log("   💡 YOLO11: Ultralytics standard pipeline\n")

        try:
            results = model.train(**params)
        except RuntimeError as mps_err:
            if is_mps and "view size is not compatible" in str(mps_err):
                log(f"\n⚠️  MPS incompatible con {title} (.view() en backward pass)")
                log("🔄 Reintentando en CPU automaticamente...\n")
                params["device"] = "cpu"
                params["amp"] = False
                # Recargar modelo limpio (el anterior quedo en estado parcial)
                model = YOLO(cfg.model)
                results = model.train(**params)
            else:
                raise

        log("✅ Entrenamiento completado")

        if results is not None:
            try:
                metrics = results.results_dict
                if metrics:
                    log("\n📊 Métricas finales:")
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            log(f"   {k}: {v:.4f}")
            except Exception:
                pass
        return results

    except KeyboardInterrupt:
        log("\n⚠️ Entrenamiento interrumpido por el usuario")
        return None
    except Exception as exc:
        log(f"❌ Error entrenamiento: {exc}")
        import traceback; traceback.print_exc()
        return None


def validate_yolo(
    model_path: str,
    data_yaml: str,
    split: str = "val",
    imgsz: int = 224,
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 300,
    project: Optional[str] = None,
    name: str = "val",
) -> Optional[Any]:
    """Validate YOLO model on a dataset split.

    Default ``split='val'`` — use **val** for post-training validation
    and **test** only for the final evaluation (Bloque 9).
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        log("❌ Ultralytics no disponible")
        return None

    if project is None:
        try:
            project = str(Path(model_path).parent.parent)
        except Exception:
            pass

    try:
        log(f"\n🔍 Validando modelo en split: {split}")
        model = YOLO(model_path)
        kwargs: Dict[str, Any] = dict(
            data=data_yaml, split=split, imgsz=imgsz,
            conf=conf, iou=iou, max_det=max_det, verbose=True,
        )
        if project is not None:
            kwargs["project"] = project
            kwargs["name"] = name
        return model.val(**kwargs)
    except Exception as exc:
        log(f"❌ Error validación: {exc}")
        return None


# =====================================================================
#  MOBILENET TRAINING (Two-phase)
# =====================================================================

def freeze_backbone(model) -> int:
    """Freeze all backbone layers (keep SSD head trainable).

    Returns number of frozen layers.
    """
    import tensorflow as tf
    frozen = 0
    for layer in model.layers:
        n = layer.name.lower()
        if "ssd" not in n and "objectness" not in n and "class_out" not in n and "bbox_out" not in n:
            layer.trainable = False
            frozen += 1
        else:
            layer.trainable = True
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    total = model.count_params()
    log(f"🔒 Backbone congelado: {frozen} layers")
    log(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return frozen


def unfreeze_backbone_layers(model, num_layers: int = -1) -> int:
    """Unfreeze last *num_layers* of the model (-1 = all)."""
    import tensorflow as tf
    if num_layers == -1:
        for layer in model.layers:
            layer.trainable = True
        log(f"🔓 Todas las capas desbloqueadas ({len(model.layers)} layers)")
        return len(model.layers)

    for layer in model.layers[-num_layers:]:
        layer.trainable = True
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    total = model.count_params()
    log(f"🔓 Desbloqueadas últimas {num_layers} capas")
    log(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return num_layers


# ── Losses ───────────────────────────────────────────────────────────

def _focal_loss(alpha: float = 0.25, gamma: float = 2.0, class_weights=None,
                label_smoothing: float = 0.0):
    """Focal loss for multi-class classification.

    Args:
        label_smoothing: If > 0, smooth hard labels towards uniform.
    """
    import tensorflow as tf

    def fn(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        # Label smoothing: suavizar targets duros → reduce sobreconfianza
        if label_smoothing > 0:
            n_cls = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / n_cls
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fw = tf.pow(1.0 - pt, gamma)
        aw = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss = aw * fw * ce
        if class_weights is not None:
            loss = loss * tf.reshape(class_weights, (1, 1, -1))
        has_obj = tf.reduce_max(y_true, axis=-1, keepdims=True)
        loss = loss * has_obj
        n_pos = tf.reduce_sum(has_obj) + 1e-4
        return tf.reduce_sum(loss) / n_pos
    return fn


def _binary_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """Binary focal loss for objectness."""
    import tensorflow as tf

    def fn(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return tf.reduce_mean(tf.pow(1.0 - pt, gamma) * (y_true * alpha + (1 - y_true) * (1 - alpha)) * bce)
    return fn


def _smooth_l1_loss(delta: float = 1.0):
    """Smooth L1 for bbox regression (only positive anchors)."""
    import tensorflow as tf

    def fn(y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        loss = tf.where(diff < delta, 0.5 * tf.square(diff), delta * diff - 0.5 * delta**2)
        valid = tf.cast(tf.reduce_max(tf.abs(y_true), axis=-1, keepdims=True) > 1e-3, tf.float32)
        loss = loss * valid
        n_pos = tf.reduce_sum(valid) + 1e-4
        return tf.reduce_sum(loss) / n_pos
    return fn


def create_ssd_loss(
    num_classes: int,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    neg_pos_ratio: int = 3,
    class_weights=None,
    label_smoothing: float = 0.0,
) -> Dict[str, Any]:
    """Create the SSD-Lite loss dict for ``model.compile(loss=...)``."""
    return {
        "objectness": _binary_focal_loss(focal_alpha, focal_gamma),
        "class_out": _focal_loss(focal_alpha, focal_gamma, class_weights,
                                  label_smoothing),
        "bbox_out": _smooth_l1_loss(1.0),
    }


# ── Optimizer builder ────────────────────────────────────────────────

def _build_optimizer(
    lr: float,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.0,
    lr_schedule: str = "reduce_on_plateau",
    total_steps: int = 0,
    warmup_steps: int = 0,
    lr_min: float = 1e-7,
):
    """Build Keras optimizer with optional cosine LR schedule.

    Args:
        lr: Peak learning rate.
        optimizer_name: ``"Adam"`` or ``"AdamW"``.
        weight_decay: Weight decay for AdamW (ignored for Adam).
        lr_schedule: ``"cosine"`` → CosineDecay, else constant LR
            (combined with ReduceLROnPlateau callback).
        total_steps: Total training steps (epochs × steps_per_epoch).
        warmup_steps: Linear warmup steps for cosine schedule.
        lr_min: Minimum learning rate.
    """
    import tensorflow as tf

    learning_rate = lr

    if lr_schedule == "cosine" and total_steps > 0:
        decay_steps = max(1, total_steps - warmup_steps)
        if warmup_steps > 0:
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr_min,
                decay_steps=total_steps,
                alpha=1.0,              # decae hasta initial_lr × alpha = lr_min
                warmup_target=lr,
                warmup_steps=warmup_steps,
            )
        else:
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=decay_steps,
                alpha=lr_min / lr if lr > 0 else 0.0,
            )
        log(f"📈 LR schedule: cosine (peak={lr}, min={lr_min}, "
            f"steps={total_steps}, warmup={warmup_steps})")

    opt_name = optimizer_name.lower().strip()
    if opt_name == "adamw":
        opt = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        log(f"⚙️  Optimizer: AdamW (weight_decay={weight_decay})")
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        log(f"⚙️  Optimizer: Adam")
    return opt


def create_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    model_name: str = "mobilenet_ssd",
    patience_reduce_lr: int = 5,
    patience_early_stop: int = 15,
    reduce_lr_factor: float = 0.2,
    min_lr: float = 1e-7,
    use_reduce_lr: bool = True,
) -> list:
    """Standard Keras callbacks for MobileNet training.

    Args:
        use_reduce_lr: If False, skip ReduceLROnPlateau (e.g. when using
            a cosine LR schedule that already manages decay).
    """
    import tensorflow as tf

    safe_mkdir(checkpoint_dir)
    safe_mkdir(log_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_best.keras"),
            monitor="val_loss", save_best_only=True, mode="min", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_early_stop,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, f"{model_name}_history.csv"), append=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, f"{model_name}_{ts}"),
            histogram_freq=0, update_freq="epoch",
        ),
    ]
    if use_reduce_lr:
        cbs.insert(1, tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=reduce_lr_factor,
            patience=patience_reduce_lr, min_lr=min_lr, verbose=1,
        ))
    sched_label = "ReduceLR, " if use_reduce_lr else ""
    log(f"📋 Callbacks creados ({len(cbs)}): checkpoint, {sched_label}EarlyStopping, CSV, TB")
    return cbs


def train_mobilenet_phase1(
    model,
    train_ds,
    val_ds,
    epochs: int = 30,
    lr: float = 1e-3,
    loss_dict: Optional[Dict] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    callbacks: Optional[list] = None,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.0,
    lr_schedule: str = "reduce_on_plateau",
    lr_min: float = 1e-7,
    lr_warmup_epochs: int = 0,
    steps_per_epoch: int = 0,
) -> Any:
    """Phase 1: warm-up with frozen backbone.

    New in v2: supports AdamW, cosine LR schedule, loss_weights.
    """
    import tensorflow as tf

    freeze_backbone(model)

    if loss_dict is None:
        loss_dict = create_ssd_loss(num_classes=4)

    total_steps = epochs * steps_per_epoch if steps_per_epoch > 0 else 0
    warmup_steps = lr_warmup_epochs * steps_per_epoch if steps_per_epoch > 0 else 0
    optimizer = _build_optimizer(
        lr=lr, optimizer_name=optimizer_name, weight_decay=weight_decay,
        lr_schedule=lr_schedule, total_steps=total_steps,
        warmup_steps=warmup_steps, lr_min=lr_min,
    )
    compile_kw = dict(optimizer=optimizer, loss=loss_dict)
    if loss_weights:
        compile_kw["loss_weights"] = loss_weights
    model.compile(**compile_kw)
    log(f"\n🚀 Phase 1: Warm-up ({epochs} épocas, LR={lr}, opt={optimizer_name})")

    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=callbacks, verbose=1,
    )
    elapsed = time.time() - t0
    log(f"✅ Phase 1 completada en {elapsed / 60:.1f} min")
    return history


def train_mobilenet_phase2(
    model,
    train_ds,
    val_ds,
    epochs: int = 50,
    lr: float = 5e-5,
    unfreeze_layers: int = 40,
    loss_dict: Optional[Dict] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    callbacks: Optional[list] = None,
    initial_epoch: int = 0,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.0,
    lr_schedule: str = "reduce_on_plateau",
    lr_min: float = 1e-7,
    lr_warmup_epochs: int = 0,
    steps_per_epoch: int = 0,
) -> Any:
    """Phase 2: fine-tuning with partially unfrozen backbone.

    New in v2: supports AdamW, cosine LR schedule, loss_weights.
    """
    import tensorflow as tf

    unfreeze_backbone_layers(model, unfreeze_layers)

    if loss_dict is None:
        loss_dict = create_ssd_loss(num_classes=4)

    total_steps = epochs * steps_per_epoch if steps_per_epoch > 0 else 0
    warmup_steps = lr_warmup_epochs * steps_per_epoch if steps_per_epoch > 0 else 0
    optimizer = _build_optimizer(
        lr=lr, optimizer_name=optimizer_name, weight_decay=weight_decay,
        lr_schedule=lr_schedule, total_steps=total_steps,
        warmup_steps=warmup_steps, lr_min=lr_min,
    )
    compile_kw = dict(optimizer=optimizer, loss=loss_dict)
    if loss_weights:
        compile_kw["loss_weights"] = loss_weights
    model.compile(**compile_kw)
    log(f"\n🚀 Phase 2: Fine-tuning ({epochs} épocas, LR={lr}, opt={optimizer_name})")

    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=initial_epoch + epochs,
        initial_epoch=initial_epoch, callbacks=callbacks, verbose=1,
    )
    elapsed = time.time() - t0
    log(f"✅ Phase 2 completada en {elapsed / 60:.1f} min")
    return history


def validate_mobilenet(model, val_ds) -> Dict[str, float]:
    """Quick validation on val set. Returns loss dict."""
    results = model.evaluate(val_ds, verbose=1, return_dict=True)
    log(f"📊 Validación: {results}")
    return results


def save_training_history(
    history,
    output_path: str,
    phase_label: str = "",
) -> None:
    """Save Keras training history to CSV."""
    import pandas as pd

    df = pd.DataFrame(history.history)
    if phase_label:
        df["phase"] = phase_label
    safe_mkdir(Path(output_path).parent)
    df.to_csv(output_path, index=False)
    log(f"💾 Historial guardado: {output_path}")


def combine_histories(
    h1_path: str,
    h2_path: str,
    output_path: str,
) -> None:
    """Combine Phase1 + Phase2 CSVs into a single file."""
    import pandas as pd

    dfs = []
    for p, label in [(h1_path, "phase1"), (h2_path, "phase2")]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "phase" not in df.columns:
                df["phase"] = label
            dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(output_path, index=False)
        log(f"💾 Historial combinado: {output_path}")
