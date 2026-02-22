"""Entry-point de Vertex AI — FCOS (MobileNetV3-Small + FPN).

8-block pipeline (Cycle 2):
  1. Setup        — descarga config + dataset + prepara entorno
  2. Verify       — verifica dataset YOLO + distribución de clases
  3. Build        — construye modelo FCOS y muestra resumen
  4. Train        — entrenamiento 2 fases (freeze → unfreeze)
  5. Curves       — extrae y grafica curvas de entrenamiento
  6. Val          — evaluación en split de validación (mAP@50)
  7. Test         — evaluación en split de test + predicciones
  8. Save+Upload  — guarda artefactos + sube a GCS

Uso::

    python -m trainer.task_fcos \\
        --config-uri gs://bucket/configs/fcos_v3s_v1.yaml \\
        --job-dir gs://bucket/output \\
        --project-id my-project \\
        --region us-central1 \\
        --experiment-name tfm-deteccion \\
        --run-name fcos_v3s_v1-20260301
"""
from __future__ import annotations

# ── Headless matplotlib
import matplotlib
matplotlib.use("Agg")

import argparse
import os
import sys
import time
from pathlib import Path

import math
import numpy as np
import torch


LOCAL_WORK_DIR = "/tmp/training"

FCOS_STRIDES = [8, 16, 32]


def encode_fcos_targets(
    targets: list,
    image_shape: tuple,
    device: str,
) -> dict:
    """Encode raw IODCDataset targets into per-level FCOS target tensors.

    Vectorized implementation — replaces Python loops over grid cells with
    tensor broadcasting for ~100× speedup on GPU.

    Args:
        targets: List[Dict] with 'boxes' (N,4 cx,cy,w,h normalized) and 'labels' (N,).
        image_shape: (B, C, H, W) of the image batch.
        device: torch device string.

    Returns:
        Dict with keys cls_{lvl}, reg_{lvl}, ctr_{lvl}, pos_{lvl} for each FPN level.
    """
    batch_size = image_shape[0]
    img_h, img_w = image_shape[2], image_shape[3]
    strides = FCOS_STRIDES

    # Detect num_classes from targets
    max_cls = 0
    for tgt in targets:
        labels = tgt["labels"]
        if len(labels) > 0:
            max_cls = max(max_cls, int(labels.max().item()))
    num_classes = max(max_cls + 1, 5)

    encoded = {}

    for lvl_idx, stride in enumerate(strides):
        feat_h = img_h // stride
        feat_w = img_w // stride
        HW = feat_h * feat_w

        # Initialize targets for this level
        cls_target = torch.zeros(batch_size, feat_h, feat_w, num_classes,
                                 device=device)
        reg_target = torch.zeros(batch_size, feat_h, feat_w, 4, device=device)
        ctr_target = torch.zeros(batch_size, feat_h, feat_w, device=device)
        pos_mask = torch.zeros(batch_size, feat_h, feat_w, dtype=torch.bool,
                               device=device)

        # Build grid of cell centers in pixel coords — (feat_h, feat_w)
        gy_range = torch.arange(feat_h, device=device, dtype=torch.float32)
        gx_range = torch.arange(feat_w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(gy_range, gx_range, indexing="ij")
        px_grid = (grid_x + 0.5) * stride   # (feat_h, feat_w)
        py_grid = (grid_y + 0.5) * stride   # (feat_h, feat_w)
        px_flat = px_grid.reshape(-1)        # (HW,)
        py_flat = py_grid.reshape(-1)        # (HW,)

        for b_idx, tgt in enumerate(targets):
            boxes = tgt["boxes"].to(device)   # (N, 4) cx, cy, w, h  normalised
            labels = tgt["labels"].to(device)  # (N,)

            if len(boxes) == 0:
                continue

            # Convert normalised cx,cy,w,h → pixel x1,y1,x2,y2  — (N,)
            bcx = boxes[:, 0] * img_w
            bcy = boxes[:, 1] * img_h
            bw  = boxes[:, 2] * img_w
            bh  = boxes[:, 3] * img_h
            x1 = bcx - bw / 2
            y1 = bcy - bh / 2
            x2 = bcx + bw / 2
            y2 = bcy + bh / 2

            # Containment check — broadcast (HW,1) vs (1,N) → (HW, N)
            inside = (
                (px_flat.unsqueeze(1) >= x1.unsqueeze(0)) &
                (px_flat.unsqueeze(1) <= x2.unsqueeze(0)) &
                (py_flat.unsqueeze(1) >= y1.unsqueeze(0)) &
                (py_flat.unsqueeze(1) <= y2.unsqueeze(0))
            )  # (HW, N)

            any_inside = inside.any(dim=1)  # (HW,)
            if not any_inside.any():
                continue

            # Smallest-area tiebreaker — (HW, N) with non-matching set to inf
            areas = bw * bh  # (N,)
            areas_bcast = torch.where(
                inside,
                areas.unsqueeze(0),
                torch.tensor(float("inf"), device=device),
            )  # (HW, N)
            best_idx = areas_bcast.argmin(dim=1)  # (HW,)

            # Positive cell indices (flat) and their assigned GT index
            pos_cells = any_inside.nonzero(as_tuple=True)[0]  # (P,)
            best_for_pos = best_idx[pos_cells]                 # (P,)

            # Regression: l, t, r, b
            l = px_flat[pos_cells] - x1[best_for_pos]
            t = py_flat[pos_cells] - y1[best_for_pos]
            r = x2[best_for_pos]   - px_flat[pos_cells]
            b = y2[best_for_pos]   - py_flat[pos_cells]

            # Safety filter (should be redundant given containment)
            valid = (l > 0) & (t > 0) & (r > 0) & (b > 0)
            if not valid.all():
                pos_cells     = pos_cells[valid]
                best_for_pos  = best_for_pos[valid]
                l, t, r, b    = l[valid], t[valid], r[valid], b[valid]

            if len(pos_cells) == 0:
                continue

            # Centerness — sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
            lr_min = torch.minimum(l, r)
            lr_max = torch.maximum(l, r)
            tb_min = torch.minimum(t, b)
            tb_max = torch.maximum(t, b)
            centerness = torch.sqrt(
                (lr_min / lr_max.clamp(min=1e-6)) *
                (tb_min / tb_max.clamp(min=1e-6))
            )

            # Flat → (gy, gx) indices
            gy_idx = pos_cells // feat_w
            gx_idx = pos_cells % feat_w

            # Write into target tensors
            cls_ids = labels[best_for_pos].long()
            cls_target[b_idx, gy_idx, gx_idx, cls_ids] = 1.0
            # Normalize l,t,r,b by stride so targets are in [0, feat_size] range
            reg_target[b_idx, gy_idx, gx_idx] = torch.stack([l, t, r, b], dim=1) / stride
            ctr_target[b_idx, gy_idx, gx_idx] = centerness
            pos_mask[b_idx, gy_idx, gx_idx] = True

        encoded[f"cls_{lvl_idx}"] = cls_target
        encoded[f"reg_{lvl_idx}"] = reg_target
        encoded[f"ctr_{lvl_idx}"] = ctr_target
        encoded[f"pos_{lvl_idx}"] = pos_mask

    return encoded


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vertex AI — FCOS Training")
    p.add_argument("--config-uri", required=True)
    p.add_argument("--job-dir", required=True)
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--experiment-name", default="tfm-deteccion-objetos")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ================================================================
    # Bloque 1 — Setup
    # ================================================================
    print("=" * 60)
    print("BLOQUE 1 — Setup y descarga de datos")
    print("=" * 60)

    from trainer.gcs_utils import download_from_gcs, prepare_dataset
    from trainer.config_loader import load_config_from_yaml, get_gcs_dataset_uri
    from trainer.vertex_logging import VertexExperimentLogger

    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    local_config = os.path.join(LOCAL_WORK_DIR, "config.yaml")
    download_from_gcs(args.config_uri, local_config)

    setup = load_config_from_yaml(local_config)
    family = setup.model_family  # "FCOS"
    fc = setup.family_config     # dict from YAML fcos: section
    gcs_dataset_uri = get_gcs_dataset_uri(local_config)

    datasets_dir = os.path.join(LOCAL_WORK_DIR, "datasets")
    dataset_path = prepare_dataset(gcs_dataset_uri, datasets_dir, family)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    run_name = args.run_name or f"fcos-{int(time.time())}"
    logger = VertexExperimentLogger(
        project_id=args.project_id,
        region=args.region,
        experiment_name=args.experiment_name,
        run_name=run_name,
        staging_bucket=args.job_dir,
    )
    logger.log_config(setup)

    # ================================================================
    # Bloque 2 — Verificación del Dataset
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 2 — Verificación del Dataset")
    print("=" * 60)

    from src_colab import (
        verify_yolo_dataset, get_class_distribution_yolo,
        plot_class_distribution, calculate_class_weights,
        generate_data_yaml, delete_yolo_cache,
        create_yolo_working_copy, DATASET_MASTER_CLASSES,
        visualize_gt_samples_per_class,
    )

    verification = verify_yolo_dataset(dataset_path)
    if not verification.get("valid", False):
        raise RuntimeError(f"❌ Dataset no válido: {verification}")

    master = DATASET_MASTER_CLASSES.get(setup.dataset_name)
    if master and (setup.class_names != master):
        dataset_path, data_yaml_path, _ = create_yolo_working_copy(
            original_dir=dataset_path,
            master_classes=master,
            selected_classes=setup.class_names,
        )
    else:
        data_yaml_path = generate_data_yaml(dataset_path, setup.class_names)
        delete_yolo_cache(dataset_path)

    dist = get_class_distribution_yolo(dataset_path, setup.class_names)
    dist_path = os.path.join(LOCAL_WORK_DIR, "class_distribution.png")
    plot_class_distribution(dist, save_path=dist_path,
                            title=f"Distribución — {setup.experiment_name}")
    logger.log_figure(dist_path, "class_distribution")

    class_weights = calculate_class_weights(dist, method="inverse_freq")
    print(f"⚖️  Class weights: {class_weights}")

    gt_path = os.path.join(LOCAL_WORK_DIR, "gt_samples.png")
    visualize_gt_samples_per_class(
        dataset_dir=dataset_path, class_names=setup.class_names,
        split="train", samples_per_class=2, save_path=gt_path,
    )
    logger.log_figure(gt_path, "gt_samples")

    # ================================================================
    # Bloque 3 — Build Model
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 3 — Construcción del Modelo FCOS")
    print("=" * 60)

    from src_colab import (
        build_fcos_model, freeze_backbone,
        print_model_summary, estimate_model_size,
        FCOS_SPECS,
    )

    num_classes = len(setup.class_names)
    model = build_fcos_model(
        num_classes=num_classes,
        fpn_channels=fc.get("fpn_channels", 64),
        pretrained_backbone=fc.get("pretrained_backbone", True),
    ).to(device)

    freeze_backbone(model, "FCOS")
    print_model_summary(model, "FCOS")
    size_info = estimate_model_size(model)
    print(f"📐 Tamaño estimado: {size_info['float32_mb']:.2f} MB (FP32), {size_info['int8_mb']:.2f} MB (INT8)")
    logger.log_params({"model_size_mb": size_info['float32_mb'], **FCOS_SPECS})

    # ================================================================
    # Bloque 4 — Entrenamiento 2 Fases
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 4 — Entrenamiento (2 fases)")
    print("=" * 60)

    from torch.utils.data import DataLoader
    from src_colab import (
        TwoPhaseConfig, train_two_phase, save_two_phase_history,
        IODCDataset, iodc_collate_fn,
        build_fcos_loss,
    )

    # Build augmentation config from YAML fcos section
    aug_config = {
        k: v for k, v in fc.items() if k.startswith("aug_")
    }

    # Build dataloaders
    train_ds = IODCDataset(
        dataset_dir=dataset_path,
        split="train",
        class_names=setup.class_names,
        img_size=fc.get("resize_schedule", {0: 640}).get(0, 640),
        augment=True,
        aug_config=aug_config,
    )
    val_ds = IODCDataset(
        dataset_dir=dataset_path,
        split="valid",
        class_names=setup.class_names,
        img_size=fc.get("resize_schedule", {0: 640}).get(0, 640),
        augment=False,
    )

    batch_size = fc.get("batch_size", setup.batch_size)
    num_workers = fc.get("workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=iodc_collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=iodc_collate_fn)

    # Training config
    resize_raw = fc.get("resize_schedule", {0: 640, 10: 416, 20: 320, 30: 224})
    # Convert dict → sorted list of (epoch, size) tuples expected by TwoPhaseConfig
    if isinstance(resize_raw, dict):
        resize_schedule = sorted((int(k), v) for k, v in resize_raw.items())
    else:
        resize_schedule = [(int(e), s) for e, s in resize_raw]

    train_cfg = TwoPhaseConfig(
        phase1_epochs=fc.get("phase1_epochs", 30),
        phase2_epochs=fc.get("phase2_epochs", 60),
        phase1_lr=fc.get("phase1_lr", 1e-3),
        phase2_lr=fc.get("phase2_lr", 1e-4),
        phase1_weight_decay=fc.get("phase1_wd", 1e-4),
        phase2_weight_decay=fc.get("phase2_wd", 1e-5),
        resize_schedule=resize_schedule,
        amp=fc.get("amp", True),
        grad_clip_max_norm=fc.get("grad_clip", 10.0),
        patience=setup.patience,
        optimizer_name=fc.get("phase1_optimizer", "adamw"),
        scheduler_name=fc.get("phase1_scheduler", "cosine"),
        batch_size=batch_size,
        device=str(device),
    )

    loss_fn = build_fcos_loss(
        cls_weight=fc.get("cls_weight", 1.0),
        reg_weight=fc.get("reg_weight", 1.5),
        ctr_weight=fc.get("centerness_weight", 1.0),
        reg_warmup_epochs=fc.get("reg_warmup_epochs", 0),
        focal_gamma=fc.get("focal_gamma", 0.0),
        focal_alpha=fc.get("focal_alpha", 0.25),
    )

    # ── Startup verification (T7+): confirm code-level features deployed ──
    _focal_g = fc.get("focal_gamma", 0.0)
    _warmup_e = fc.get("reg_warmup_epochs", 0)
    _hflip_p = fc.get("aug_hflip_prob", 0.5)
    print("═" * 60)
    print("🔍 DEPLOY VERIFICATION — package v2.2.0")
    print(f"   focal_gamma  = {_focal_g}  {'✅ Focal Loss ACTIVE' if _focal_g > 0 else '⚠️ BCE (standard)'}")
    print(f"   reg_warmup   = {_warmup_e}  {'✅ SL1→GIoU warmup ACTIVE' if _warmup_e > 0 else '⚠️ GIoU from epoch 0'}")
    print(f"   aug_hflip    = {_hflip_p}  {'✅ HFlip ON' if _hflip_p > 0 else '⚠️ HFlip OFF'}")
    print(f"   phase1_epochs= {fc.get('phase1_epochs', 30)}")
    print("═" * 60)

    checkpoint_dir = os.path.join(LOCAL_WORK_DIR, "checkpoints")
    t0 = time.time()

    history = train_two_phase(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=loss_fn,
        config=train_cfg,
        family="FCOS",
        save_dir=checkpoint_dir,
        encode_targets_fn=encode_fcos_targets,
    )

    train_time = time.time() - t0
    print(f"⏱️  Entrenamiento completado en {train_time / 60:.1f} min")

    # Save history CSV
    history_csv = os.path.join(LOCAL_WORK_DIR, "training_history.csv")
    save_two_phase_history(history, history_csv)

    logger.log_metrics({
        "train_time_min": train_time / 60,
        "best_val_loss": min(history.all_val_loss) if history.all_val_loss else 0,
        "total_epochs": len(history.all_train_loss),
    })

    # ================================================================
    # Bloque 5 — Curvas de Entrenamiento
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 5 — Curvas de Entrenamiento")
    print("=" * 60)

    from src_colab import (
        extract_two_phase_history, plot_training_curves,
        print_training_summary,
    )

    th = extract_two_phase_history(history_csv)
    curves_path = os.path.join(LOCAL_WORK_DIR, "training_curves.png")
    plot_training_curves(th, save_path=curves_path,
                         title_prefix=f"FCOS — {setup.experiment_name}")
    logger.log_figure(curves_path, "training_curves")
    print_training_summary(th)

    # ================================================================
    # Bloque 6 — Evaluación en Validación
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 6 — Evaluación en Validación")
    print("=" * 60)

    from src_colab import (
        evaluate_pytorch_model, predict_fcos,
        plot_confusion_matrix, plot_per_class_metrics,
        save_evaluation,
    )

    # Load best checkpoint
    best_ckpt = os.path.join(checkpoint_dir, "best_fcos.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        print(f"✅ Cargado mejor checkpoint: {best_ckpt}")

    def fcos_predict_fn(model_ref, images_tensor, conf_threshold=None):
        return predict_fcos(model_ref, images_tensor,
                            conf_threshold=conf_threshold or setup.conf_threshold,
                            nms_threshold=setup.iou_threshold,
                            class_names=setup.class_names,
                            ctr_power=fc.get("ctr_power", 1.0),
                            iou_aware=fc.get("iou_aware_scoring", False))

    val_results = evaluate_pytorch_model(
        model=model,
        dataloader=val_loader,
        predict_fn=fcos_predict_fn,
        class_names=setup.class_names,
        device=str(device),
        model_name="fcos_v3s",
        family="FCOS",
    )

    print(f"📊 Val mAP@50: {val_results.mAP50:.4f}")
    for cn, ap in val_results.per_class_ap50.items():
        print(f"   {cn}: {ap:.4f}")

    cm_path = os.path.join(LOCAL_WORK_DIR, "val_confusion_matrix.png")
    plot_confusion_matrix(val_results, save_path=cm_path)
    logger.log_figure(cm_path, "val_confusion_matrix")

    metrics_path = os.path.join(LOCAL_WORK_DIR, "val_per_class.png")
    plot_per_class_metrics(val_results, save_path=metrics_path)
    logger.log_figure(metrics_path, "val_per_class")

    val_json = os.path.join(LOCAL_WORK_DIR, "val_evaluation.json")
    save_evaluation(val_results, val_json)

    logger.log_metrics({
        "val_map50": val_results.mAP50,
        **{f"val_ap_{cn}": ap for cn, ap in val_results.per_class_ap50.items()},
    })

    # ================================================================
    # Bloque 7 — Evaluación en Test
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 7 — Evaluación en Test")
    print("=" * 60)

    test_ds = IODCDataset(
        dataset_dir=dataset_path,
        split="test",
        class_names=setup.class_names,
        img_size=fc.get("export_imgsz", 224),
        augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=iodc_collate_fn)

    test_results = evaluate_pytorch_model(
        model=model,
        dataloader=test_loader,
        predict_fn=fcos_predict_fn,
        class_names=setup.class_names,
        device=str(device),
        model_name="fcos_v3s",
        family="FCOS",
        split="test",
    )

    print(f"📊 Test mAP@50: {test_results.mAP50:.4f}")

    test_cm_path = os.path.join(LOCAL_WORK_DIR, "test_confusion_matrix.png")
    plot_confusion_matrix(test_results, save_path=test_cm_path)
    logger.log_figure(test_cm_path, "test_confusion_matrix")

    test_json = os.path.join(LOCAL_WORK_DIR, "test_evaluation.json")
    save_evaluation(test_results, test_json)

    logger.log_metrics({
        "test_map50": test_results.mAP50,
        **{f"test_ap_{cn}": ap for cn, ap in test_results.per_class_ap50.items()},
    })

    # ================================================================
    # Bloque 8 — Save + Upload
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 8 — Guardado y subida a GCS")
    print("=" * 60)

    from src_colab import upload_to_gcs, ExportResult
    from src_colab import (
        export_pytorch_to_onnx, verify_onnx_model,
    )

    # ONNX export (non-fatal — pipeline continues if it fails)
    export_dir = os.path.join(LOCAL_WORK_DIR, "export")
    export_result = None
    onnx_verify = None
    try:
        export_result = export_pytorch_to_onnx(
            model=model,
            export_dir=export_dir,
            model_name="fcos_v3s",
            family="FCOS",
            imgsz=fc.get("export_imgsz", 224),
            opset=fc.get("export_opset", 13),
        )
        onnx_verify = verify_onnx_model(export_result.export_path,
                                         imgsz=fc.get("export_imgsz", 224))
        logger.log_metrics({
            "onnx_size_mb": export_result.file_size_mb,
            "onnx_valid": onnx_verify.valid,
            "onnx_latency_ms": onnx_verify.inference_time_ms,
        })
    except Exception as exc:
        print(f"⚠️ ONNX export falló (no fatal): {exc}")

    # Save experiment JSON
    from src_colab import (
        create_experiment_from_setup, save_experiment,
    )
    experiment = create_experiment_from_setup(setup)
    experiment.val_map50 = val_results.mAP50
    experiment.test_map50 = test_results.mAP50
    if export_result:
        experiment.onnx_size_mb = export_result.file_size_mb
        experiment.onnx_path = export_result.export_path
    if onnx_verify:
        experiment.onnx_latency_ms = onnx_verify.inference_time_ms
    experiment.model_path = best_ckpt
    experiment.history_csv = history_csv
    experiment.mark_completed()

    exp_json = os.path.join(LOCAL_WORK_DIR, "experiment.json")
    save_experiment(experiment, exp_json)

    # Upload all artifacts to GCS
    artifacts = [
        local_config, history_csv, curves_path, dist_path, gt_path,
        cm_path, metrics_path, val_json,
        test_cm_path, test_json,
        exp_json, best_ckpt,
    ]
    if export_result and export_result.export_path:
        artifacts.append(export_result.export_path)
    for artifact in artifacts:
        if artifact and os.path.exists(artifact):
            rel = os.path.relpath(artifact, LOCAL_WORK_DIR)
            gcs_dest = f"{args.job_dir}/{run_name}/{rel}"
            upload_to_gcs(artifact, gcs_dest)

    print("\n✅ Pipeline FCOS completado exitosamente.")
    logger.end_run()


if __name__ == "__main__":
    main()
