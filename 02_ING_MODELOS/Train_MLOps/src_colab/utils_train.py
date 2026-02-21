"""Unified two-phase training for Cycle 2 PyTorch models.

All three families (FCOS, YOLO26_CUSTOM, ESPDet) share the same
two-phase training strategy with progressive resizing:
  Phase 1: backbone frozen, higher LR (warm-up head)
  Phase 2: all layers unfrozen, lower LR (fine-tuning)

The ``train_two_phase()`` function is the main entry-point.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils_io import log, safe_mkdir
from .utils_model import freeze_backbone, unfreeze_all


# =====================================================================
#  Training config
# =====================================================================

@dataclass
class TwoPhaseConfig:
    """Config for the two-phase progressive training loop."""
    # Phase 1 — frozen backbone
    phase1_epochs: int = 30
    phase1_lr: float = 1e-3
    phase1_weight_decay: float = 1e-4
    # Phase 2 — unfrozen
    phase2_epochs: int = 70
    phase2_lr: float = 1e-4
    phase2_weight_decay: float = 1e-5
    # Common
    batch_size: int = 16
    patience: int = 20
    optimizer_name: str = "AdamW"
    scheduler_name: str = "cosine"    # "cosine" | "reduce_on_plateau"
    warmup_epochs: int = 3
    amp: bool = True                  # automatic mixed precision
    grad_clip_max_norm: float = 10.0
    # Progressive resizing
    resize_schedule: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 640), (20, 416), (60, 320), (80, 224),
    ])
    # Device
    device: str = "cuda"

    def summary(self) -> str:
        lines = [
            f"\n🔧 Two-Phase Training Config",
            f"  Phase 1: {self.phase1_epochs} epochs | LR={self.phase1_lr} | "
            f"WD={self.phase1_weight_decay}",
            f"  Phase 2: {self.phase2_epochs} epochs | LR={self.phase2_lr} | "
            f"WD={self.phase2_weight_decay}",
            f"  Optimizer: {self.optimizer_name} | Scheduler: {self.scheduler_name}",
            f"  Batch: {self.batch_size} | AMP: {self.amp} | "
            f"Patience: {self.patience}",
            f"  Resize schedule: {self.resize_schedule}",
        ]
        return "\n".join(lines)


# =====================================================================
#  Training history
# =====================================================================

@dataclass
class PhaseHistory:
    """Per-epoch metrics for one training phase."""
    epoch: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    # --- loss components (model-agnostic names) ---
    train_cls_loss: List[float] = field(default_factory=list)
    train_reg_loss: List[float] = field(default_factory=list)
    train_ctr_loss: List[float] = field(default_factory=list)
    val_cls_loss: List[float] = field(default_factory=list)
    val_reg_loss: List[float] = field(default_factory=list)
    val_ctr_loss: List[float] = field(default_factory=list)
    # --- meta ---
    lr: List[float] = field(default_factory=list)
    img_size: List[int] = field(default_factory=list)
    phase_label: str = ""
    elapsed_min: float = 0.0


@dataclass
class TwoPhaseHistory:
    """Combined history from both phases."""
    phase1: PhaseHistory = field(default_factory=PhaseHistory)
    phase2: PhaseHistory = field(default_factory=PhaseHistory)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_epochs: int = 0

    @property
    def all_train_loss(self) -> List[float]:
        return self.phase1.train_loss + self.phase2.train_loss

    @property
    def all_val_loss(self) -> List[float]:
        return self.phase1.val_loss + self.phase2.val_loss

    @property
    def all_train_cls_loss(self) -> List[float]:
        return self.phase1.train_cls_loss + self.phase2.train_cls_loss

    @property
    def all_train_reg_loss(self) -> List[float]:
        return self.phase1.train_reg_loss + self.phase2.train_reg_loss

    @property
    def all_train_ctr_loss(self) -> List[float]:
        return self.phase1.train_ctr_loss + self.phase2.train_ctr_loss

    @property
    def all_val_cls_loss(self) -> List[float]:
        return self.phase1.val_cls_loss + self.phase2.val_cls_loss

    @property
    def all_val_reg_loss(self) -> List[float]:
        return self.phase1.val_reg_loss + self.phase2.val_reg_loss

    @property
    def all_val_ctr_loss(self) -> List[float]:
        return self.phase1.val_ctr_loss + self.phase2.val_ctr_loss

    @property
    def all_lr(self) -> List[float]:
        return self.phase1.lr + self.phase2.lr

    @property
    def all_epochs(self) -> List[int]:
        return self.phase1.epoch + self.phase2.epoch


# =====================================================================
#  Loss functions
# =====================================================================

def _giou_loss_ltrb(
    pred_ltrb: torch.Tensor,
    target_ltrb: torch.Tensor,
) -> torch.Tensor:
    """Compute GIoU loss for l,t,r,b encoded predictions.

    Both inputs are (N, 4) with columns [l, t, r, b] representing
    distances from the cell center to the box edges (stride-normalised).

    Returns:
        Scalar mean(1 - GIoU).  Range [0, 2].
    """
    # Clamp predictions to avoid degenerate boxes
    pred_ltrb = pred_ltrb.clamp(min=0)

    # Areas
    pred_area = (pred_ltrb[:, 0] + pred_ltrb[:, 2]) * (
        pred_ltrb[:, 1] + pred_ltrb[:, 3]
    )
    target_area = (target_ltrb[:, 0] + target_ltrb[:, 2]) * (
        target_ltrb[:, 1] + target_ltrb[:, 3]
    )

    # Intersection (both share same cell center)
    inter_w = (
        torch.min(pred_ltrb[:, 0], target_ltrb[:, 0])
        + torch.min(pred_ltrb[:, 2], target_ltrb[:, 2])
    )
    inter_h = (
        torch.min(pred_ltrb[:, 1], target_ltrb[:, 1])
        + torch.min(pred_ltrb[:, 3], target_ltrb[:, 3])
    )
    inter_area = inter_w * inter_h

    # Union
    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-7)

    # Enclosing box
    enclosing_w = (
        torch.max(pred_ltrb[:, 0], target_ltrb[:, 0])
        + torch.max(pred_ltrb[:, 2], target_ltrb[:, 2])
    )
    enclosing_h = (
        torch.max(pred_ltrb[:, 1], target_ltrb[:, 1])
        + torch.max(pred_ltrb[:, 3], target_ltrb[:, 3])
    )
    enclosing_area = enclosing_w * enclosing_h

    giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-7)
    return (1 - giou).mean()


def _smooth_l1_ltrb(
    pred_ltrb: torch.Tensor,
    target_ltrb: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 loss for l,t,r,b encoded predictions.

    Both inputs are (N, 4).  Returns scalar mean loss.
    """
    pred_ltrb = pred_ltrb.clamp(min=0)
    return nn.functional.smooth_l1_loss(pred_ltrb, target_ltrb, beta=beta)


def build_fcos_loss(
    cls_weight: float = 1.0,
    reg_weight: float = 1.0,
    ctr_weight: float = 1.0,
    reg_warmup_epochs: int = 0,
) -> Callable:
    """Build a combined FCOS loss function (cls + reg + centerness).

    This is a simplified version that expects pre-encoded targets.
    The actual encoding (strides, distances) is handled in the
    task_fcos entry-point.

    If ``reg_warmup_epochs > 0``, Smooth L1 is used for regression during
    the first *reg_warmup_epochs* global epochs, then switches to GIoU.
    This avoids the initial GIoU plateau observed in previous trains.
    Use ``fcos_loss.set_epoch(e)`` to update the current epoch.
    """
    cls_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    # Mutable state so the training loop can set the current epoch
    _state = {"epoch": 0}

    def fcos_loss(preds: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        total_cls = torch.tensor(0.0, device=preds["cls"][0].device)
        total_reg = torch.tensor(0.0, device=preds["cls"][0].device)
        total_ctr = torch.tensor(0.0, device=preds["cls"][0].device)
        n_pos = 0

        use_smooth_l1 = (reg_warmup_epochs > 0
                         and _state["epoch"] < reg_warmup_epochs)

        for lvl_idx, (cls_pred, reg_pred, ctr_pred) in enumerate(
            zip(preds["cls"], preds["reg"], preds["centerness"])
        ):
            cls_target = targets[f"cls_{lvl_idx}"]
            reg_target = targets[f"reg_{lvl_idx}"]
            ctr_target = targets[f"ctr_{lvl_idx}"]
            pos_mask = targets[f"pos_{lvl_idx}"]

            # Classification loss (all locations)
            cls_loss = cls_loss_fn(
                cls_pred.permute(0, 2, 3, 1).reshape(-1, cls_pred.shape[1]),
                cls_target.reshape(-1, cls_pred.shape[1]),
            ).sum()
            total_cls = total_cls + cls_loss

            # Regression + centerness (positive only)
            if pos_mask.any():
                n_pos += pos_mask.sum().item()
                pred_pos = reg_pred.permute(0, 2, 3, 1)[pos_mask]
                tgt_pos = reg_target[pos_mask]
                if use_smooth_l1:
                    reg_loss = _smooth_l1_ltrb(pred_pos, tgt_pos)
                else:
                    reg_loss = _giou_loss_ltrb(pred_pos, tgt_pos)
                ctr_loss = nn.functional.binary_cross_entropy_with_logits(
                    ctr_pred.permute(0, 2, 3, 1)[pos_mask].squeeze(-1),
                    ctr_target[pos_mask],
                )
                total_reg = total_reg + reg_loss
                total_ctr = total_ctr + ctr_loss

        n_pos = max(n_pos, 1.0)
        loss_dict = {
            "cls_loss": total_cls / n_pos * cls_weight,
            "reg_loss": total_reg * reg_weight,
            "ctr_loss": total_ctr * ctr_weight,
        }
        loss_dict["total"] = sum(loss_dict.values())
        return loss_dict

    def set_epoch(epoch: int) -> None:
        prev_sl1 = (reg_warmup_epochs > 0
                    and _state["epoch"] < reg_warmup_epochs)
        _state["epoch"] = epoch
        new_sl1 = (reg_warmup_epochs > 0
                   and _state["epoch"] < reg_warmup_epochs)
        if prev_sl1 and not new_sl1:
            log(f"🔄 reg_loss warmup complete at epoch {epoch}: "
                f"Smooth L1 → GIoU")

    fcos_loss.set_epoch = set_epoch

    return fcos_loss


def build_espdet_loss(
    cls_weight: float = 1.0,
    reg_weight: float = 2.0,
) -> Callable:
    """Build ESPDet-Pico loss (simplified anchor-free)."""
    cls_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def espdet_loss(preds: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        total_cls = torch.tensor(0.0, device=preds["cls"][0].device)
        total_reg = torch.tensor(0.0, device=preds["cls"][0].device)
        n_pos = 0

        for lvl_idx, (cls_pred, reg_pred) in enumerate(
            zip(preds["cls"], preds["reg"])
        ):
            cls_target = targets[f"cls_{lvl_idx}"]
            reg_target = targets[f"reg_{lvl_idx}"]
            pos_mask = targets[f"pos_{lvl_idx}"]

            cls_loss = cls_loss_fn(
                cls_pred.permute(0, 2, 3, 1).reshape(-1, cls_pred.shape[1]),
                cls_target.reshape(-1, cls_pred.shape[1]),
            ).sum()
            total_cls = total_cls + cls_loss

            if pos_mask.any():
                n_pos += pos_mask.sum().item()
                reg_loss = _giou_loss_ltrb(
                    reg_pred.permute(0, 2, 3, 1)[pos_mask],
                    reg_target[pos_mask],
                )
                total_reg = total_reg + reg_loss

        n_pos = max(n_pos, 1.0)
        loss_dict = {
            "cls_loss": total_cls / n_pos * cls_weight,
            "reg_loss": total_reg * reg_weight,
        }
        loss_dict["total"] = sum(loss_dict.values())
        return loss_dict

    return espdet_loss


# =====================================================================
#  Optimizer / Scheduler builders
# =====================================================================

def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 1e-4,
    name: str = "AdamW",
) -> torch.optim.Optimizer:
    """Build optimizer (AdamW or SGD)."""
    name = name.lower().strip()
    if name == "adamw":
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
    elif name == "sgd":
        opt = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, momentum=0.9, weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
    log(f"⚙️  Optimizer: {name.upper()} (lr={lr}, wd={weight_decay})")
    return opt


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine",
    total_epochs: int = 100,
    warmup_epochs: int = 3,
    min_lr: float = 1e-7,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build LR scheduler."""
    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr,
        )
        log(f"📈 Scheduler: CosineAnnealing (T_max={total_epochs - warmup_epochs})")
        return scheduler
    elif name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5, min_lr=min_lr,
        )
        log(f"📈 Scheduler: ReduceOnPlateau (factor=0.2, patience=5)")
        return scheduler
    return None


# =====================================================================
#  Core training loop
# =====================================================================

def _get_current_img_size(epoch: int, resize_schedule: List[Tuple[int, int]]) -> int:
    """Determine image size for current epoch from schedule."""
    current_size = resize_schedule[0][1] if resize_schedule else 640
    for start_epoch, size in resize_schedule:
        if epoch >= start_epoch:
            current_size = size
    return current_size


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip: float = 10.0,
    encode_targets_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """Train for one epoch.

    Returns:
        Dict with averaged loss components (always includes ``"total"``).
    """
    model.train()
    accum: Dict[str, float] = {}
    n_batches = 0

    for images, targets in dataloader:
        images = images.to(device)

        # Encode targets if a custom encoder is provided
        if encode_targets_fn is not None:
            encoded_targets = encode_targets_fn(targets, images.shape, device)
        else:
            encoded_targets = targets

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                preds = model(images)
                loss_dict = loss_fn(preds, encoded_targets)
                loss = loss_dict["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss_dict = loss_fn(preds, encoded_targets)
            loss = loss_dict["total"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k, v in loss_dict.items():
            accum[k] = accum.get(k, 0.0) + (v.item() if hasattr(v, 'item') else float(v))
        n_batches += 1

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in accum.items()}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    device: str,
    encode_targets_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """Validate for one epoch.

    Returns:
        Dict with averaged loss components (always includes ``"total"``).
    """
    model.eval()
    accum: Dict[str, float] = {}
    n_batches = 0

    for images, targets in dataloader:
        images = images.to(device)
        if encode_targets_fn is not None:
            encoded_targets = encode_targets_fn(targets, images.shape, device)
        else:
            encoded_targets = targets

        preds = model(images)
        loss_dict = loss_fn(preds, encoded_targets)
        for k, v in loss_dict.items():
            accum[k] = accum.get(k, 0.0) + (v.item() if hasattr(v, 'item') else float(v))
        n_batches += 1

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in accum.items()}


def _run_phase(
    model: nn.Module,
    train_dataset,
    val_dataset,
    loss_fn: Callable,
    config: TwoPhaseConfig,
    phase: int,
    initial_epoch: int,
    best_val_loss: float,
    save_dir: str,
    family: str,
    encode_targets_fn: Optional[Callable] = None,
) -> Tuple[PhaseHistory, float, int]:
    """Run a single training phase (1 or 2)."""
    if phase == 1:
        epochs = config.phase1_epochs
        lr = config.phase1_lr
        wd = config.phase1_weight_decay
        label = "Phase 1 (backbone frozen)"
    else:
        epochs = config.phase2_epochs
        lr = config.phase2_lr
        wd = config.phase2_weight_decay
        label = "Phase 2 (full fine-tuning)"

    log(f"\n🚀 {label} — {epochs} epochs, LR={lr}")

    optimizer = build_optimizer(model, lr, wd, config.optimizer_name)
    scheduler = build_scheduler(
        optimizer, config.scheduler_name,
        total_epochs=epochs,
        warmup_epochs=config.warmup_epochs if phase == 1 else 0,
    )

    scaler = torch.amp.GradScaler("cuda") if config.amp and "cuda" in config.device else None

    history = PhaseHistory(phase_label=f"phase{phase}")
    patience_counter = 0
    best_epoch = initial_epoch
    t0 = time.time()

    for epoch_rel in range(epochs):
        epoch_abs = initial_epoch + epoch_rel

        # Notify loss_fn of current epoch (for hybrid warmup)
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch_abs)

        # Progressive resizing
        new_size = _get_current_img_size(epoch_abs, config.resize_schedule)
        if hasattr(train_dataset, "set_image_size"):
            train_dataset.set_image_size(new_size)
        if hasattr(val_dataset, "set_image_size"):
            val_dataset.set_image_size(new_size)

        # Rebuild dataloaders after resize
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
            collate_fn=getattr(train_dataset, "_collate_fn", None)
            or _default_collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
            collate_fn=getattr(val_dataset, "_collate_fn", None)
            or _default_collate_fn,
        )

        # Warmup LR (linear ramp for first N epochs of phase 1)
        if phase == 1 and epoch_rel < config.warmup_epochs:
            warmup_lr = lr * (epoch_rel + 1) / config.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_result = train_one_epoch(
            model, train_loader, loss_fn, optimizer,
            config.device, scaler, config.grad_clip_max_norm,
            encode_targets_fn,
        )
        val_result = validate_one_epoch(
            model, val_loader, loss_fn, config.device, encode_targets_fn,
        )

        train_loss = train_result["total"]
        val_loss = val_result["total"]
        current_lr = optimizer.param_groups[0]["lr"]

        # Scheduler step
        if scheduler is not None and epoch_rel >= config.warmup_epochs:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Record totals
        history.epoch.append(epoch_abs)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.lr.append(current_lr)
        history.img_size.append(new_size)

        # Record loss components (keys depend on loss_fn)
        history.train_cls_loss.append(train_result.get("cls_loss", 0.0))
        history.train_reg_loss.append(train_result.get("reg_loss", 0.0))
        history.train_ctr_loss.append(train_result.get("ctr_loss", 0.0))
        history.val_cls_loss.append(val_result.get("cls_loss", 0.0))
        history.val_reg_loss.append(val_result.get("reg_loss", 0.0))
        history.val_ctr_loss.append(val_result.get("ctr_loss", 0.0))

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_abs
            patience_counter = 0
            ckpt_path = os.path.join(save_dir, f"best_{family.lower()}.pt")
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        # Log (with component breakdown)
        comp_parts = []
        for key in ["cls_loss", "reg_loss", "ctr_loss"]:
            tv = train_result.get(key, 0.0)
            if tv > 0:
                comp_parts.append(f"{key.replace('_loss','')}={tv:.4f}")
        comp_str = f" [{' | '.join(comp_parts)}]" if comp_parts else ""

        log(f"  Epoch {epoch_abs:>3d} | "
            f"train={train_loss:.4f}{comp_str} | "
            f"val={val_loss:.4f} | "
            f"lr={current_lr:.2e} | img={new_size} | "
            f"{'★ best' if patience_counter == 0 else ''}")

        # Early stopping
        if patience_counter >= config.patience:
            log(f"⏹️  Early stopping at epoch {epoch_abs} "
                f"(patience={config.patience})")
            break

    history.elapsed_min = (time.time() - t0) / 60.0
    log(f"✅ {label} completada en {history.elapsed_min:.1f} min")
    return history, best_val_loss, best_epoch


def _default_collate_fn(batch):
    """Fallback: stack images, keep targets as list."""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


# =====================================================================
#  Main entry-point
# =====================================================================

def train_two_phase(
    model: nn.Module,
    train_dataset,
    val_dataset,
    loss_fn: Callable,
    config: TwoPhaseConfig,
    family: str,
    save_dir: str = "checkpoints",
    encode_targets_fn: Optional[Callable] = None,
) -> TwoPhaseHistory:
    """Two-phase training with progressive resizing.

    Args:
        model: PyTorch model.
        train_dataset: Must support ``set_image_size(int)``.
        val_dataset: Same.
        loss_fn: Callable(preds, targets) → Dict with "total" key.
        config: TwoPhaseConfig.
        family: "FCOS" | "YOLO26_CUSTOM" | "ESPDet".
        save_dir: Directory for checkpoints.
        encode_targets_fn: Optional callable to encode raw targets
            into the format expected by loss_fn.

    Returns:
        TwoPhaseHistory with combined metrics.
    """
    safe_mkdir(save_dir)
    log(config.summary())

    model = model.to(config.device)
    history = TwoPhaseHistory()

    # ── Phase 1: frozen backbone ──
    freeze_backbone(model, family)
    h1, best_loss, best_ep = _run_phase(
        model, train_dataset, val_dataset, loss_fn, config,
        phase=1, initial_epoch=0,
        best_val_loss=float("inf"),
        save_dir=save_dir, family=family,
        encode_targets_fn=encode_targets_fn,
    )
    history.phase1 = h1

    # ── Phase 2: unfrozen ──
    # Reload best checkpoint from phase 1
    best_ckpt = os.path.join(save_dir, f"best_{family.lower()}.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=config.device, weights_only=True))
        log(f"🔄 Mejor checkpoint de Phase 1 recargado")

    unfreeze_all(model)
    phase2_start = len(h1.epoch)
    h2, best_loss, best_ep = _run_phase(
        model, train_dataset, val_dataset, loss_fn, config,
        phase=2, initial_epoch=phase2_start,
        best_val_loss=best_loss,
        save_dir=save_dir, family=family,
        encode_targets_fn=encode_targets_fn,
    )
    history.phase2 = h2

    history.best_val_loss = best_loss
    history.best_epoch = best_ep
    history.total_epochs = len(h1.epoch) + len(h2.epoch)

    log(f"\n📊 Entrenamiento completo: {history.total_epochs} epochs")
    log(f"   Mejor val_loss: {history.best_val_loss:.4f} (epoch {history.best_epoch})")
    log(f"   Tiempo total: {h1.elapsed_min + h2.elapsed_min:.1f} min")

    return history


# =====================================================================
#  YOLO26 Custom training (Ultralytics API, single-phase)
# =====================================================================

@dataclass
class Yolo26CustomConfig:
    """Config for YOLO26 via Ultralytics ``model.train()``."""
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
    mosaic: float = 1.0
    mixup: float = 0.1
    close_mosaic: int = 10
    box: float = 7.5
    cls: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    device: Optional[str] = None
    amp: bool = True
    workers: int = 4
    project: str = "runs"
    name: str = "train"
    exist_ok: bool = True
    freeze: Optional[List[int]] = None
    max_det: int = 300

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()
             if v is not None and k not in ("model",)}
        d["verbose"] = True
        return d

    def summary(self) -> str:
        return (
            f"\n🔧 YOLO26 Custom Config: {self.model}\n"
            f"  Img: {self.imgsz} | Epochs: {self.epochs} | "
            f"Patience: {self.patience} | Batch: {self.batch}\n"
            f"  LR0={self.lr0} → LRf={self.lr0 * self.lrf:.6f} | "
            f"Cosine: {self.cos_lr}\n"
            f"  Mosaic: {self.mosaic} | Mixup: {self.mixup}"
        )


def train_yolo26_custom(
    data_yaml: str,
    cfg: Yolo26CustomConfig,
    resume: bool = False,
    resume_path: Optional[str] = None,
) -> Optional[Any]:
    """Train YOLO26 via Ultralytics API with DDP env cleanup.

    Includes the DDP env variable cleanup discovered in Cycle 1.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        log("❌ Ultralytics no instalada.")
        return None

    # ── DDP env cleanup (Cycle 1 lesson) ──
    ddp_vars = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    ]
    cleaned = []
    for var in ddp_vars:
        if var in os.environ:
            del os.environ[var]
            cleaned.append(var)
    if cleaned:
        log(f"🧹 Limpiadas vars DDP: {cleaned}")

    log(cfg.summary())

    try:
        if resume and resume_path and os.path.exists(resume_path):
            log(f"🔄 Reanudando desde: {resume_path}")
            model = YOLO(resume_path)
        else:
            model = YOLO(cfg.model)

        params = cfg.to_dict()
        params["data"] = data_yaml
        if resume:
            params["resume"] = True

        log(f"\n🚀 Iniciando entrenamiento YOLO26 Custom...")
        results = model.train(**params)
        log("✅ Entrenamiento YOLO26 Custom completado")
        return results

    except Exception as exc:
        log(f"❌ Error entrenamiento: {exc}")
        import traceback
        traceback.print_exc()
        return None


def validate_yolo26_custom(
    model_path: str,
    data_yaml: str,
    split: str = "val",
    imgsz: int = 224,
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 300,
) -> Optional[Any]:
    """Validate a YOLO26 custom-trained model."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        log("❌ Ultralytics no disponible")
        return None

    try:
        log(f"\n🔍 Validando YOLO26 en split: {split}")
        model = YOLO(model_path)
        return model.val(
            data=data_yaml, split=split, imgsz=imgsz,
            conf=conf, iou=iou, max_det=max_det, verbose=True,
        )
    except Exception as exc:
        log(f"❌ Error validación: {exc}")
        return None


# =====================================================================
#  Save / combine histories
# =====================================================================

def save_two_phase_history(
    history: TwoPhaseHistory,
    output_path: str,
) -> None:
    """Save TwoPhaseHistory to CSV (with loss component breakdown)."""
    import pandas as pd

    rows = []
    for ph in [history.phase1, history.phase2]:
        for i in range(len(ph.epoch)):
            rows.append({
                "epoch": ph.epoch[i],
                "train_loss": ph.train_loss[i],
                "train_cls_loss": ph.train_cls_loss[i] if i < len(ph.train_cls_loss) else 0.0,
                "train_reg_loss": ph.train_reg_loss[i] if i < len(ph.train_reg_loss) else 0.0,
                "train_ctr_loss": ph.train_ctr_loss[i] if i < len(ph.train_ctr_loss) else 0.0,
                "val_loss": ph.val_loss[i],
                "val_cls_loss": ph.val_cls_loss[i] if i < len(ph.val_cls_loss) else 0.0,
                "val_reg_loss": ph.val_reg_loss[i] if i < len(ph.val_reg_loss) else 0.0,
                "val_ctr_loss": ph.val_ctr_loss[i] if i < len(ph.val_ctr_loss) else 0.0,
                "lr": ph.lr[i],
                "img_size": ph.img_size[i] if i < len(ph.img_size) else 0,
                "phase": ph.phase_label,
            })

    df = pd.DataFrame(rows)
    safe_mkdir(str(Path(output_path).parent))
    df.to_csv(output_path, index=False)
    log(f"💾 Historial guardado: {output_path}")
