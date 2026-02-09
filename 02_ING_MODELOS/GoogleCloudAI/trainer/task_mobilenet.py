"""Entry-point de Vertex AI para entrenamiento de modelos MobileNet SSD-Lite.

Replica los 12 bloques del notebook ``07_TrainColab.ipynb`` (rama
MobileNet) adaptados para ejecución en un Custom Job de Vertex AI
con contenedor TensorFlow pre-built.

Uso desde CLI (dentro del contenedor)::

    python -m trainer.task_mobilenet \
        --config-uri gs://bucket/configs/mobilenet_v3s_ssdlite_v1.yaml \
        --job-dir gs://bucket/output \
        --project-id my-project \
        --region us-central1 \
        --experiment-name tfm-deteccion \
        --run-name MBNTv3S_ssdlite_v1-20260207
"""
from __future__ import annotations

# Headless matplotlib — DEBE ir antes de cualquier import de matplotlib
import matplotlib
matplotlib.use("Agg")

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


# ── Constantes del contenedor ────────────────────────────────────────
LOCAL_WORK_DIR = "/tmp/training"


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Vertex AI — MobileNet SSD-Lite Training"
    )
    parser.add_argument(
        "--config-uri", required=True,
        help="URI GCS del YAML de configuración (gs://bucket/configs/file.yaml)",
    )
    parser.add_argument(
        "--job-dir", required=True,
        help="URI GCS donde subir los artefactos de salida (gs://bucket/output)",
    )
    parser.add_argument(
        "--project-id", required=True,
        help="ID del proyecto de Google Cloud",
    )
    parser.add_argument(
        "--region", default="us-central1",
        help="Región de Vertex AI",
    )
    parser.add_argument(
        "--experiment-name", default="tfm-deteccion-objetos",
        help="Nombre del experimento en Vertex AI Experiments",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Nombre del run. Si no se indica, se autogenera.",
    )
    return parser.parse_args()


def main() -> None:
    """Función principal — ejecuta los 12 bloques de entrenamiento."""
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

    # 1.1 Descargar YAML de configuración
    local_config = os.path.join(LOCAL_WORK_DIR, "config.yaml")
    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    download_from_gcs(args.config_uri, local_config)

    # 1.2 Cargar configuración → ExperimentSetup
    setup = load_config_from_yaml(local_config)
    family = setup.model_family
    gcs_dataset_uri = get_gcs_dataset_uri(local_config)

    # 1.3 Descargar y preparar dataset
    datasets_dir = os.path.join(LOCAL_WORK_DIR, "datasets")
    dataset_path = prepare_dataset(gcs_dataset_uri, datasets_dir, family)

    # 1.4 Setup de entorno (GPU config, paths)
    project_root = Path(LOCAL_WORK_DIR)
    from src_colab import setup_environment
    env, paths = setup_environment(project_root=project_root)

    # 1.5 Iniciar Vertex AI Experiments
    run_name = args.run_name or f"{setup.experiment_name}-{int(time.time())}"
    logger = VertexExperimentLogger(
        project_id=args.project_id,
        region=args.region,
        experiment_name=args.experiment_name,
        run_name=run_name,
        staging_bucket=args.job_dir,
    )
    logger.log_config(setup)

    # ================================================================
    # Bloque 3 — Verificación del Dataset
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 3 — Verificación del Dataset")
    print("=" * 60)

    from src_colab import (
        verify_dataset, get_class_distribution, plot_class_distribution,
        calculate_class_weights, is_mobilenet_family,
    )

    verification = verify_dataset(dataset_path, family)
    if not verification.get("valid", False):
        raise RuntimeError(
            f"❌ Dataset no válido: {verification.get('issues', [])}"
        )

    dist = get_class_distribution(dataset_path, family, setup.class_names)

    dist_path = os.path.join(LOCAL_WORK_DIR, "class_distribution.png")
    plot_class_distribution(
        dist,
        save_path=dist_path,
        title=f"Distribución — {setup.experiment_name}",
    )
    logger.log_figure(dist_path, "class_distribution")

    mc = setup.mobilenet_config
    class_weight_method = mc.get("class_weight_method", "inverse_freq")
    use_class_weights = mc.get("use_class_weights", True)
    class_weights = (
        calculate_class_weights(dist, method=class_weight_method)
        if use_class_weights
        else None
    )
    print(f"⚖️  Class weights: {class_weights}")

    # ================================================================
    # Bloque 4 — Construcción del Modelo
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 4 — Construcción del Modelo")
    print("=" * 60)

    from src_colab import (
        build_mobilenet_ssd, print_model_summary, estimate_model_size,
        estimate_esp32_inference, generate_anchors, compute_anchor_statistics,
        create_mobilenet_pipeline,
    )

    # 4.1 Generar anclas
    use_offset_regression = mc.get("use_offset_regression", False)
    if use_offset_regression:
        print("🔧 Offset regression ACTIVADO (SSD-standard Δcx/Δcy/Δw/Δh)")
    anchor_sizes = mc.get("anchor_sizes", [0.1, 0.2, 0.37, 0.54, 0.71, 0.88])
    anchor_ratios = mc.get("anchor_ratios", [1.0, 2.0, 0.5, 3.0, 0.33])
    anchors = generate_anchors(
        imgsz=setup.img_size,
        sizes=anchor_sizes,
        ratios=anchor_ratios,
    )
    compute_anchor_statistics(anchors)
    print(f"📦 Anclas generadas: {anchors.shape}")

    # 4.2 Crear pipelines TFRecord
    train_ds = create_mobilenet_pipeline(
        dataset_path, "train", anchors, setup.class_names,
        batch_size=setup.batch_size, imgsz=setup.img_size,
        augment_level=mc.get("augmentation_level", "medium"),
        use_offset_regression=use_offset_regression,
    )
    val_ds = create_mobilenet_pipeline(
        dataset_path, "val", anchors, setup.class_names,
        batch_size=setup.batch_size, imgsz=setup.img_size,
        augment_level="none",
        use_offset_regression=use_offset_regression,
    )
    print(f"📊 Pipeline: train={train_ds}, val={val_ds}")

    # 4.3 Construir modelo
    num_anchors_per_cell = len(anchor_sizes) * len(anchor_ratios)

    mv = (setup.model_variant or "").lower()
    if "v3l" in mv or "large" in mv:
        variant = "Large"
    elif "v3s" in mv or "small" in mv:
        variant = "Small"
    else:
        variant = "Small"

    version = "v2" if "v2" in family.lower() else "v3"

    model = build_mobilenet_ssd(
        version=version.upper(),
        variant=variant,
        num_classes=len(setup.class_names),
        num_anchors_per_cell=num_anchors_per_cell,
        img_size=setup.img_size,
        alpha=mc.get("backbone_alpha", 1.0),
        minimalistic=mc.get("minimalistic", True),
        dropout_rate=mc.get("dropout_rate", 0.2),
        feature_channels=mc.get("feature_channels", 128),
        l2_reg=mc.get("l2_reg", 1e-4),
        use_offset_regression=use_offset_regression,
    )
    print_model_summary(model, family)
    estimate_model_size(model, family)
    esp32_est = estimate_esp32_inference(family, variant)
    if esp32_est:
        print(
            f"⏱️  ESP32-S3 estimado: {esp32_est['estimated_esp32_ms']:.0f} ms "
            f"({esp32_est['estimated_esp32_fps']:.1f} FPS)"
        )

    # ================================================================
    # Bloque 5 — Entrenamiento (2 fases)
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 5 — Entrenamiento")
    print("=" * 60)

    from src_colab import (
        create_ssd_loss, create_callbacks,
        train_mobilenet_phase1, train_mobilenet_phase2,
        save_training_history, combine_histories,
        safe_mkdir,
    )

    exp_dir = os.path.join(paths.models_dir, setup.experiment_name)
    safe_mkdir(exp_dir)

    loss_dict = create_ssd_loss(
        num_classes=len(setup.class_names),
        focal_alpha=mc.get("focal_alpha", 0.25),
        focal_gamma=mc.get("focal_gamma", 2.0),
        neg_pos_ratio=mc.get("neg_pos_ratio", 3),
        class_weights=class_weights,
        label_smoothing=mc.get("label_smoothing", 0.0),
    )

    # ── Loss weights (permite subir el peso de clasificación vs bbox) ──
    loss_weights = None
    obj_w = mc.get("obj_loss_weight", 1.0)
    cls_w = mc.get("cls_loss_weight", 1.0)
    box_w = mc.get("box_loss_weight", 1.0)
    if not (obj_w == 1.0 and cls_w == 1.0 and box_w == 1.0):
        loss_weights = {
            "objectness": obj_w,
            "class_out": cls_w,
            "bbox_out": box_w,
        }
        print(f"⚖️  Loss weights: obj={obj_w}, cls={cls_w}, box={box_w}")

    # ── Hiperparámetros del optimizador ──
    optimizer_name = mc.get("optimizer", "Adam")
    weight_decay = mc.get("weight_decay", 0.0)
    lr_schedule = mc.get("lr_schedule", "reduce_on_plateau")
    lr_min = mc.get("lr_min", 1e-7)
    lr_warmup_epochs = mc.get("lr_warmup_epochs", 0)
    use_cosine = lr_schedule == "cosine"

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")

    # ── Estimar steps por época ──
    import tensorflow as tf
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    if steps_per_epoch < 0:  # UNKNOWN o INFINITE
        steps_per_epoch = 0
    print(f"📐 Steps per epoch: {steps_per_epoch}")

    # ── Kwargs compartidos del optimizador ──
    opt_kwargs = dict(
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        lr_schedule=lr_schedule,
        lr_min=lr_min,
        steps_per_epoch=int(steps_per_epoch),
    )

    t_start = time.time()

    # Phase 1 — backbone congelado
    patience = setup.patience or 30
    cbs_p1 = create_callbacks(
        ckpt_dir, log_dir,
        model_name=f"{setup.experiment_name}_p1",
        patience_reduce_lr=max(3, patience // 6),
        patience_early_stop=max(10, patience // 2),
        min_lr=lr_min,
        use_reduce_lr=not use_cosine,
    )
    p1_epochs = mc.get("phase1_epochs", 20)
    h1 = train_mobilenet_phase1(
        model, train_ds, val_ds,
        epochs=p1_epochs,
        lr=mc.get("phase1_lr", 1e-3),
        loss_dict=loss_dict,
        loss_weights=loss_weights,
        callbacks=cbs_p1,
        lr_warmup_epochs=lr_warmup_epochs,
        **opt_kwargs,
    )
    p1_csv = os.path.join(log_dir, f"{setup.experiment_name}_p1_history.csv")
    save_training_history(h1, p1_csv, phase_label="phase1")

    # Phase 2 — fine-tuning
    cbs_p2 = create_callbacks(
        ckpt_dir, log_dir,
        model_name=f"{setup.experiment_name}_p2",
        patience_reduce_lr=max(5, patience // 4),
        patience_early_stop=patience,
        min_lr=lr_min,
        use_reduce_lr=not use_cosine,
    )
    h2 = train_mobilenet_phase2(
        model, train_ds, val_ds,
        epochs=mc.get("phase2_epochs", 50),
        lr=mc.get("phase2_lr", 1e-4),
        unfreeze_layers=mc.get("phase2_unfreeze_layers", 20),
        loss_dict=loss_dict,
        loss_weights=loss_weights,
        callbacks=cbs_p2,
        initial_epoch=p1_epochs,
        lr_warmup_epochs=0,  # warmup solo en phase 1
        **opt_kwargs,
    )
    p2_csv = os.path.join(log_dir, f"{setup.experiment_name}_p2_history.csv")
    save_training_history(h2, p2_csv, phase_label="phase2")

    combined_csv = os.path.join(log_dir, f"{setup.experiment_name}_history.csv")
    combine_histories(p1_csv, p2_csv, combined_csv)

    # Guardar modelo final
    model.save(os.path.join(exp_dir, f"{setup.experiment_name}_final.keras"))

    training_time_min = (time.time() - t_start) / 60
    print(f"\n⏱️  Entrenamiento total: {training_time_min:.1f} min")

    # ================================================================
    # Bloque 6 — Curvas de Entrenamiento
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 6 — Curvas de Entrenamiento")
    print("=" * 60)

    from src_colab import (
        extract_mobilenet_history,
        plot_training_curves, print_training_summary,
    )

    history = extract_mobilenet_history(combined_csv)
    history.model_name = setup.experiment_name

    curves_path = os.path.join(exp_dir, "training_curves.png")
    plot_training_curves(history, save_path=curves_path)
    print_training_summary(history)

    # Registrar en Vertex AI
    logger.log_training_metrics(history, training_time_min)
    logger.log_time_series(history)
    logger.log_figure(curves_path, "training_curves")

    # ================================================================
    # Bloque 7 — Validación (val)
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 7 — Validación (val)")
    print("=" * 60)

    from src_colab import (
        evaluate_mobilenet_model,
        plot_confusion_matrix, plot_per_class_metrics,
        save_evaluation,
    )

    val_ev = evaluate_mobilenet_model(
        model=model,
        val_ds=val_ds,
        class_names=setup.class_names,
        imgsz=setup.img_size,
        anchors=anchors,
        model_name=setup.experiment_name,
        use_offset_regression=use_offset_regression,
    )

    cm_val_path = os.path.join(exp_dir, "val_confusion_matrix.png")
    pc_val_path = os.path.join(exp_dir, "val_per_class.png")
    plot_confusion_matrix(val_ev, save_path=cm_val_path)
    plot_per_class_metrics(val_ev, save_path=pc_val_path)
    save_evaluation(val_ev, os.path.join(exp_dir, "val_evaluation.json"))

    logger.log_evaluation(val_ev, prefix="val")
    logger.log_figure(cm_val_path, "val_confusion_matrix")
    logger.log_figure(pc_val_path, "val_per_class")

    # ================================================================
    # Bloque 8 — Inferencia Visual
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 8 — Inferencia Visual")
    print("=" * 60)

    from src_colab import predict_mobilenet, visualize_predictions

    sample_batch = next(iter(val_ds))
    sample_imgs = sample_batch[0].numpy()[:8]

    dets = predict_mobilenet(
        model=model, images=sample_imgs,
        class_names=setup.class_names,
        anchors=anchors,
        use_offset_regression=use_offset_regression,
    )
    infer_path = os.path.join(exp_dir, "inference_samples.png")
    visualize_predictions(
        [sample_imgs[i] for i in range(len(sample_imgs))],
        dets,
        max_images=8, cols=4,
        save_path=infer_path,
        title=f"Inferencia — {setup.experiment_name}",
    )
    logger.log_figure(infer_path, "inference_samples")

    # ================================================================
    # Bloque 9 — Evaluación Final (test)
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 9 — Evaluación Final (test)")
    print("=" * 60)

    test_ds = create_mobilenet_pipeline(
        dataset_path, "test", anchors, setup.class_names,
        batch_size=setup.batch_size, imgsz=setup.img_size,
        augment_level="none",
        use_offset_regression=use_offset_regression,
    )
    test_ev = evaluate_mobilenet_model(
        model=model,
        val_ds=test_ds,
        class_names=setup.class_names,
        imgsz=setup.img_size,
        anchors=anchors,
        model_name=setup.experiment_name,
        use_offset_regression=use_offset_regression,
    )
    test_ev.split = "test"

    cm_test_path = os.path.join(exp_dir, "test_confusion_matrix.png")
    pc_test_path = os.path.join(exp_dir, "test_per_class.png")
    plot_confusion_matrix(test_ev, save_path=cm_test_path)
    plot_per_class_metrics(test_ev, save_path=pc_test_path)
    save_evaluation(test_ev, os.path.join(exp_dir, "test_evaluation.json"))

    logger.log_evaluation(test_ev, prefix="test")
    logger.log_figure(cm_test_path, "test_confusion_matrix")
    logger.log_figure(pc_test_path, "test_per_class")

    print(
        f"\n📊 TEST: mAP@50={test_ev.mAP50:.4f}  P={test_ev.precision:.4f}  "
        f"R={test_ev.recall:.4f}  F1={test_ev.f1:.4f}"
    )

    # ================================================================
    # Bloque 10 — Export TFLite INT8
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 10 — Export TFLite INT8")
    print("=" * 60)

    from src_colab import (
        export_tflite_int8, create_representative_dataset,
        print_export_report, save_export_result,
    )

    export_dir = os.path.join(exp_dir, "tflite")

    rep_ds = create_representative_dataset(
        val_ds, n_samples=100, imgsz=setup.img_size
    )
    export_result = export_tflite_int8(
        model=model,
        family=family,
        output_dir=export_dir,
        model_name=setup.experiment_name,
        imgsz=setup.img_size,
        representative_dataset=rep_ds,
    )

    print_export_report(export_result)
    save_export_result(export_result, os.path.join(exp_dir, "export_result.json"))

    # Determinar si la exportación TFLite fue exitosa
    tflite_ok = bool(export_result.tflite_path) and not export_result.errors

    if not tflite_ok:
        print("\n⚠️  La exportación TFLite falló o produjo errores:")
        for e in (export_result.errors or []):
            print(f"     ❌ {e}")
        print("     → Se omiten Bloques 11 (comparación) y métricas TFLite.")
        print("     → El job continuará con registro parcial y subida a GCS.\n")

    # ================================================================
    # Bloque 11 — Comparación Framework vs TFLite
    # ================================================================
    comparison = None
    tflite_test_ev = None

    if tflite_ok:
        print("\n" + "=" * 60)
        print("BLOQUE 11 — Comparación Framework vs TFLite")
        print("=" * 60)

        from src_colab import (
            compare_framework_vs_tflite, save_comparison_result,
            evaluate_tflite_model,
            plot_fw_vs_tflite_metrics, visualize_fw_vs_tflite_samples,
            predict_tflite,
        )

        # 11.1 Comparación rápida (agreement, IoU, Δconf)
        N_COMPARE = 20
        compare_batch = next(iter(val_ds))
        compare_imgs = compare_batch[0].numpy()[:N_COMPARE]

        comparison = compare_framework_vs_tflite(
            framework_model=model,
            tflite_path=export_result.tflite_path,
            images=compare_imgs,
            class_names=setup.class_names,
            family=family,
            anchors=anchors,
            imgsz=setup.img_size,
            use_offset_regression=use_offset_regression,
        )
        save_comparison_result(
            comparison, os.path.join(exp_dir, "comparison_result.json")
        )

        # 11.2 Evaluación completa TFLite sobre test
        print("🔍  Evaluación TFLite sobre test split (mAP, P, R, F1)")
        tflite_test_ev = evaluate_tflite_model(
            tflite_path=export_result.tflite_path,
            class_names=setup.class_names,
            imgsz=setup.img_size,
            conf_threshold=0.25,
            iou_threshold=0.5,
            model_name=f"{setup.experiment_name}_tflite",
            test_ds=test_ds,
            anchors=anchors,
            use_offset_regression=use_offset_regression,
        )

        # 11.3 Gráfica comparativa
        fw_tfl_path = os.path.join(exp_dir, "fw_vs_tflite_metrics.png")
        plot_fw_vs_tflite_metrics(
            fw_ev=test_ev, tfl_ev=tflite_test_ev,
            save_path=fw_tfl_path,
        )
        logger.log_figure(fw_tfl_path, "fw_vs_tflite_metrics")

        # 11.4 Visualización side-by-side
        fw_vis_dets = predict_mobilenet(
            model=model, images=compare_imgs,
            class_names=setup.class_names,
            anchors=anchors,
            use_offset_regression=use_offset_regression,
        )
        tfl_vis_dets, _ = predict_tflite(
            tflite_path=export_result.tflite_path,
            images=compare_imgs,
            class_names=setup.class_names,
            conf_threshold=0.25,
            iou_threshold=0.45,
            anchors=anchors,
            use_offset_regression=use_offset_regression,
        )

        sbs_path = os.path.join(exp_dir, "fw_vs_tflite_samples.png")
        visualize_fw_vs_tflite_samples(
            images=[compare_imgs[i] for i in range(compare_imgs.shape[0])],
            fw_dets=fw_vis_dets,
            tfl_dets=tfl_vis_dets,
            class_names=setup.class_names,
            samples_per_class=1,
            save_path=sbs_path,
        )
        logger.log_figure(sbs_path, "fw_vs_tflite_samples")

        logger.log_export_metrics(export_result, comparison)
        logger.log_tflite_test(tflite_test_ev)
    else:
        print("BLOQUE 11 — OMITIDO (exportación TFLite fallida)")

    # ================================================================
    # Bloque 12 — Registro y Comparación de Experimentos
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 12 — Registro y Comparación")
    print("=" * 60)

    from src_colab import (
        create_experiment_from_setup,
        save_experiment, load_all_experiments,
        plot_experiments_comparison, print_experiments_table,
        save_comparison_csv,
    )

    experiment = create_experiment_from_setup(setup)
    r = experiment.results

    # Training metrics
    r.training_time_min = training_time_min
    r.total_epochs_run = history.n_epochs
    if history.val_total_loss:
        r.best_val_loss = min(history.val_total_loss)
        r.best_epoch = history.best_epoch_by_val_loss
        r.final_val_loss = history.val_total_loss[-1]
    if history.train_total_loss:
        r.final_train_loss = history.train_total_loss[-1]

    # Val metrics (Framework)
    r.val_mAP50 = val_ev.mAP50
    r.val_mAP50_95 = val_ev.mAP50_95
    r.val_precision = val_ev.precision
    r.val_recall = val_ev.recall
    r.val_f1 = val_ev.f1
    r.val_per_class_ap50 = val_ev.per_class_ap50

    # Test metrics (Framework)
    r.test_mAP50 = test_ev.mAP50
    r.test_mAP50_95 = test_ev.mAP50_95
    r.test_precision = test_ev.precision
    r.test_recall = test_ev.recall
    r.test_f1 = test_ev.f1
    r.test_per_class_ap50 = test_ev.per_class_ap50

    # Export metrics
    if tflite_ok:
        r.tflite_size_mb = export_result.size_mb
        r.tflite_esp32_ok = export_result.esp32_compatible
    else:
        r.tflite_size_mb = 0.0
        r.tflite_esp32_ok = False
        r.export_errors = "; ".join(export_result.errors or ["unknown"])
    if comparison:
        r.tflite_agreement = comparison.agreement_rate
        r.tflite_avg_latency_ms = comparison.avg_inference_ms
    if tflite_test_ev:
        r.tflite_test_mAP50 = tflite_test_ev.mAP50
        r.tflite_test_mAP50_95 = tflite_test_ev.mAP50_95
        r.tflite_test_precision = tflite_test_ev.precision
        r.tflite_test_recall = tflite_test_ev.recall
        r.tflite_test_f1 = tflite_test_ev.f1
        r.tflite_test_per_class_ap50 = tflite_test_ev.per_class_ap50

    save_experiment(experiment, exp_dir)

    # ================================================================
    # Subida de artefactos a GCS
    # ================================================================
    print("\n" + "=" * 60)
    print("SUBIDA DE ARTEFACTOS A GCS")
    print("=" * 60)

    from trainer.gcs_utils import upload_directory_to_gcs

    gcs_output = f"{args.job_dir}/{setup.experiment_name}"
    upload_directory_to_gcs(exp_dir, gcs_output)

    # Cerrar Vertex AI Experiments
    logger.end_run()

    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"   Artefactos: {gcs_output}")
    print(f"   Experiment: {args.experiment_name} / {run_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
