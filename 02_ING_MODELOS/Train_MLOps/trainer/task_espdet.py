"""Entry-point de Vertex AI — ESPDet-Pico (anchor-free micro-detector).

8-block pipeline (Cycle 2):
  1. Setup        — descarga config + dataset + prepara entorno
  2. Verify       — verifica dataset YOLO + distribución
  3. Build        — construye ESPDet-Pico y muestra resumen
  4. Train        — entrenamiento 2 fases (freeze → unfreeze)
  5. Curves       — extrae y grafica curvas de entrenamiento
  6. Val          — evaluación en validación (mAP@50)
  7. Test         — evaluación en test + predicciones
  8. Save+Upload  — guarda artefactos + sube a GCS

Uso::

    python -m trainer.task_espdet \\
        --config-uri gs://bucket/configs/espdet_pico_v1.yaml \\
        --job-dir gs://bucket/output \\
        --project-id my-project \\
        --region us-central1 \\
        --run-name espdet_pico_v1-20260301
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

import numpy as np
import torch


LOCAL_WORK_DIR = "/tmp/training"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vertex AI — ESPDet-Pico Training")
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
    family = setup.model_family  # "ESPDet"
    fc = setup.family_config     # dict from espdet: section
    gcs_dataset_uri = get_gcs_dataset_uri(local_config)

    datasets_dir = os.path.join(LOCAL_WORK_DIR, "datasets")
    dataset_path = prepare_dataset(gcs_dataset_uri, datasets_dir, family)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    run_name = args.run_name or f"espdet-{int(time.time())}"
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
    print("BLOQUE 3 — Construcción del Modelo ESPDet-Pico")
    print("=" * 60)

    from src_colab import (
        build_espdet_pico, freeze_backbone,
        print_model_summary, estimate_model_size,
        ESPDET_SPECS,
    )

    num_classes = len(setup.class_names)
    model = build_espdet_pico(
        num_classes=num_classes,
        width_mult=fc.get("width_mult", 0.5),
        fpn_channels=fc.get("fpn_channels", 32),
        reg_max=fc.get("reg_max", 1),
        pretrained_weights=fc.get("pretrained_weights"),
    ).to(device)

    freeze_backbone(model, "ESPDet")
    print_model_summary(model, "ESPDet-Pico")
    size_mb = estimate_model_size(model)
    print(f"📐 Tamaño estimado: {size_mb:.2f} MB")
    logger.log_params({"model_size_mb": size_mb, **ESPDET_SPECS})

    # ================================================================
    # Bloque 4 — Entrenamiento 2 Fases
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 4 — Entrenamiento (2 fases)")
    print("=" * 60)

    from src_colab import (
        TwoPhaseConfig, train_two_phase, save_two_phase_history,
        IODCDataset, create_dataloader,
        build_espdet_loss,
    )

    # Build dataloaders
    initial_size = fc.get("resize_schedule", {0: 640}).get(0, 640) if isinstance(
        fc.get("resize_schedule"), dict
    ) else 640

    train_ds = IODCDataset(
        root=os.path.join(dataset_path, "train"),
        class_names=setup.class_names,
        img_size=initial_size,
        augment=True,
    )
    val_ds = IODCDataset(
        root=os.path.join(dataset_path, "valid"),
        class_names=setup.class_names,
        img_size=initial_size,
        augment=False,
    )

    batch_size = setup.common_config.get("batch_size", fc.get("batch_size", 32))
    train_loader = create_dataloader(train_ds, batch_size=batch_size,
                                     shuffle=True, num_workers=fc.get("workers", 4))
    val_loader = create_dataloader(val_ds, batch_size=batch_size,
                                   shuffle=False, num_workers=fc.get("workers", 4))

    resize_schedule = fc.get("resize_schedule", {0: 640, 15: 416, 30: 320, 40: 224})
    resize_schedule = {int(k): v for k, v in resize_schedule.items()}

    train_cfg = TwoPhaseConfig(
        phase1_epochs=fc.get("phase1_epochs", 40),
        phase2_epochs=fc.get("phase2_epochs", 80),
        phase1_lr=fc.get("phase1_lr", 1e-3),
        phase2_lr=fc.get("phase2_lr", 5e-5),
        phase1_wd=fc.get("phase1_wd", 1e-4),
        phase2_wd=fc.get("phase2_wd", 1e-5),
        resize_schedule=resize_schedule,
        amp=fc.get("amp", True),
        grad_clip=fc.get("grad_clip", 5.0),
        patience=setup.common_config.get("patience", 20),
        optimizer_name=fc.get("phase1_optimizer", "adamw"),
        scheduler_name=fc.get("phase1_scheduler", "cosine"),
    )

    loss_fn = build_espdet_loss(
        cls_weight=fc.get("cls_weight", 1.0),
        reg_weight=fc.get("reg_weight", 2.0),
    )

    checkpoint_dir = os.path.join(LOCAL_WORK_DIR, "checkpoints")
    t0 = time.time()

    history = train_two_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_ds,
        loss_fn=loss_fn,
        config=train_cfg,
        device=device,
        family="ESPDet",
        checkpoint_dir=checkpoint_dir,
    )

    train_time = time.time() - t0
    print(f"⏱️  Entrenamiento completado en {train_time / 60:.1f} min")

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
                         title=f"ESPDet-Pico — {setup.experiment_name}")
    logger.log_figure(curves_path, "training_curves")
    print_training_summary(th)

    # ================================================================
    # Bloque 6 — Evaluación en Validación
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 6 — Evaluación en Validación")
    print("=" * 60)

    from src_colab import (
        evaluate_pytorch_model, predict_espdet,
        plot_confusion_matrix, plot_per_class_metrics,
        save_evaluation,
    )

    best_ckpt = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"✅ Cargado mejor checkpoint: {best_ckpt}")

    strides = fc.get("strides", [4, 8, 16])

    def espdet_predict_fn(images_tensor):
        return predict_espdet(model, images_tensor,
                              conf_threshold=setup.common_config.get("conf_threshold", 0.25),
                              nms_threshold=setup.common_config.get("iou_threshold", 0.45),
                              class_names=setup.class_names,
                              strides=strides)

    val_results = evaluate_pytorch_model(
        predict_fn=espdet_predict_fn,
        dataloader=val_loader,
        class_names=setup.class_names,
        device=device,
        split_name="validation",
    )

    print(f"📊 Val mAP@50: {val_results.map50:.4f}")
    for cn, ap in val_results.per_class_ap.items():
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
        "val_map50": val_results.map50,
        **{f"val_ap_{cn}": ap for cn, ap in val_results.per_class_ap.items()},
    })

    # ================================================================
    # Bloque 7 — Evaluación en Test
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 7 — Evaluación en Test")
    print("=" * 60)

    test_ds = IODCDataset(
        root=os.path.join(dataset_path, "test"),
        class_names=setup.class_names,
        img_size=fc.get("export_imgsz", 224),
        augment=False,
    )
    test_loader = create_dataloader(test_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=fc.get("workers", 4))

    test_results = evaluate_pytorch_model(
        predict_fn=espdet_predict_fn,
        dataloader=test_loader,
        class_names=setup.class_names,
        device=device,
        split_name="test",
    )

    print(f"📊 Test mAP@50: {test_results.map50:.4f}")

    test_cm_path = os.path.join(LOCAL_WORK_DIR, "test_confusion_matrix.png")
    plot_confusion_matrix(test_results, save_path=test_cm_path)
    logger.log_figure(test_cm_path, "test_confusion_matrix")

    test_json = os.path.join(LOCAL_WORK_DIR, "test_evaluation.json")
    save_evaluation(test_results, test_json)

    logger.log_metrics({
        "test_map50": test_results.map50,
        **{f"test_ap_{cn}": ap for cn, ap in test_results.per_class_ap.items()},
    })

    # ================================================================
    # Bloque 8 — Save + Upload
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 8 — Guardado y subida a GCS")
    print("=" * 60)

    from src_colab import (
        export_pytorch_to_onnx, verify_onnx_model,
        upload_to_gcs,
        create_experiment_from_setup, save_experiment,
    )

    export_dir = os.path.join(LOCAL_WORK_DIR, "export")
    export_result = export_pytorch_to_onnx(
        model=model,
        export_dir=export_dir,
        model_name="espdet_pico",
        family="ESPDet",
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

    # Save experiment
    experiment = create_experiment_from_setup(setup)
    experiment.val_map50 = val_results.map50
    experiment.test_map50 = test_results.map50
    experiment.onnx_size_mb = export_result.file_size_mb
    experiment.onnx_latency_ms = onnx_verify.inference_time_ms
    experiment.model_path = best_ckpt
    experiment.onnx_path = export_result.export_path
    experiment.history_csv = history_csv
    experiment.mark_completed()

    exp_json = os.path.join(LOCAL_WORK_DIR, "experiment.json")
    save_experiment(experiment, exp_json)

    # Upload artifacts
    artifacts = [
        local_config, history_csv, curves_path, dist_path, gt_path,
        cm_path, metrics_path, val_json,
        test_cm_path, test_json,
        export_result.export_path, exp_json, best_ckpt,
    ]
    for artifact in artifacts:
        if artifact and os.path.exists(artifact):
            rel = os.path.relpath(artifact, LOCAL_WORK_DIR)
            gcs_dest = f"{args.job_dir}/{run_name}/{rel}"
            upload_to_gcs(artifact, gcs_dest)

    print("\n✅ Pipeline ESPDet-Pico completado exitosamente.")
    logger.end_run()


if __name__ == "__main__":
    main()
