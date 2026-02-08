"""Entry-point de Vertex AI para entrenamiento de modelos YOLO.

Replica los 12 bloques del notebook ``07_TrainColab.ipynb`` (rama
YOLO) adaptados para ejecución en un Custom Job de Vertex AI
con contenedor PyTorch pre-built.

Uso desde CLI (dentro del contenedor)::

    python -m trainer.task_yolo \
        --config-uri gs://bucket/configs/yolo26n_v1.yaml \
        --job-dir gs://bucket/output \
        --project-id my-project \
        --region us-central1 \
        --experiment-name tfm-deteccion \
        --run-name yolo26n_v1-20260207
"""
from __future__ import annotations

# Headless matplotlib — DEBE ir antes de cualquier import de matplotlib
import matplotlib
matplotlib.use("Agg")

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ── Constantes del contenedor ────────────────────────────────────────
LOCAL_WORK_DIR = "/tmp/training"


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Vertex AI — YOLO Training"
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
        calculate_class_weights, generate_data_yaml, delete_yolo_cache,
        create_yolo_working_copy, DATASET_MASTER_CLASSES,
        is_yolo_family, visualize_gt_samples_per_class,
    )

    verification = verify_dataset(dataset_path, family)
    if not verification.get("valid", False):
        raise RuntimeError(
            f"❌ Dataset no válido: {verification.get('issues', [])}"
        )

    # Working copy para subconjuntos de clases
    master = DATASET_MASTER_CLASSES.get(setup.dataset_name)
    if master and (setup.class_names != master):
        print(f"\n⚠️  Subconjunto o reorden de clases detectado:")
        print(f"   Master ({len(master)})    : {master}")
        print(f"   Seleccionadas ({len(setup.class_names)}): {setup.class_names}")
        dataset_path, data_yaml_path, filter_stats = create_yolo_working_copy(
            original_dir=dataset_path,
            master_classes=master,
            selected_classes=setup.class_names,
        )
    else:
        if master is None:
            print(
                f"\n⚠️  Dataset '{setup.dataset_name}' no tiene master_classes "
                f"definido. Se asume labels alineados con: {setup.class_names}"
            )
        data_yaml_path = generate_data_yaml(
            dataset_dir=dataset_path,
            class_names=setup.class_names,
        )
        delete_yolo_cache(dataset_path)

    print(f"\n📂 Dataset de trabajo: {dataset_path}")
    print(f"📄 data.yaml: {data_yaml_path}")

    dist = get_class_distribution(dataset_path, family, setup.class_names)

    dist_path = os.path.join(LOCAL_WORK_DIR, "class_distribution.png")
    plot_class_distribution(
        dist,
        save_path=dist_path,
        title=f"Distribución — {setup.experiment_name}",
    )
    logger.log_figure(dist_path, "class_distribution")

    class_weights = calculate_class_weights(dist, method="inverse_freq")
    print(f"⚖️  Class weights: {class_weights}")

    # GT samples
    gt_path = os.path.join(LOCAL_WORK_DIR, "gt_samples.png")
    visualize_gt_samples_per_class(
        dataset_dir=dataset_path,
        class_names=setup.class_names,
        split="train",
        samples_per_class=3,
        title=f"GT Samples (train) — {setup.experiment_name}",
        save_path=gt_path,
    )
    logger.log_figure(gt_path, "gt_samples")

    # ================================================================
    # Bloque 4 — Construcción del Modelo
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 4 — Construcción del Modelo (YOLO)")
    print("=" * 60)

    from src_colab import (
        load_yolo_model, print_model_summary,
        estimate_model_size, estimate_esp32_inference,
    )

    model = load_yolo_model(family, setup.model_variant)
    print_model_summary(model, family)
    estimate_model_size(model, family)
    esp32_est = estimate_esp32_inference(family, setup.model_variant)
    if esp32_est:
        print(
            f"⏱️  ESP32-S3 estimado: {esp32_est['estimated_esp32_ms']:.0f} ms "
            f"({esp32_est['estimated_esp32_fps']:.1f} FPS)"
        )

    # ================================================================
    # Bloque 5 — Entrenamiento
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 5 — Entrenamiento (YOLO)")
    print("=" * 60)

    from src_colab import (
        YoloTrainConfig, train_yolo,
        get_yolo_device, safe_mkdir,
    )

    exp_dir = os.path.join(paths.models_dir, setup.experiment_name)
    safe_mkdir(exp_dir)

    yc = setup.yolo_config

    t_start = time.time()

    cfg = YoloTrainConfig(
        model=f"{setup.model_variant}.pt",
        imgsz=setup.img_size,
        epochs=yc.get("epochs", 100),
        patience=setup.patience,
        batch=setup.batch_size,
        optimizer=yc.get("optimizer", "auto"),
        lr0=yc.get("lr0", 0.01),
        lrf=yc.get("lrf", 0.01),
        mosaic=yc.get("mosaic", 1.0),
        mixup=yc.get("mixup", 0.0),
        device=get_yolo_device(env),  # "0" en Vertex AI (CUDA)
        project=exp_dir,
        name="train",
    )
    results = train_yolo(data_yaml_path, cfg)

    training_time_min = (time.time() - t_start) / 60
    print(f"\n⏱️  Entrenamiento total: {training_time_min:.1f} min")

    # ================================================================
    # Bloque 6 — Curvas de Entrenamiento
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 6 — Curvas de Entrenamiento")
    print("=" * 60)

    from src_colab import (
        extract_yolo_history,
        plot_training_curves, print_training_summary,
    )

    results_csv = os.path.join(exp_dir, "train", "results.csv")
    history = extract_yolo_history(results_csv)
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
        evaluate_yolo_model,
        plot_confusion_matrix, plot_per_class_metrics,
        save_evaluation,
    )

    best_pt = os.path.join(exp_dir, "train", "weights", "best.pt")
    val_ev = evaluate_yolo_model(
        model_path=best_pt,
        data_yaml=data_yaml_path,
        split="val",
        imgsz=setup.img_size,
        class_names=setup.class_names,
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

    from src_colab import predict_yolo, visualize_predictions

    # Buscar directorio de validación
    for val_name in ["valid", "val"]:
        val_images_dir = os.path.join(dataset_path, val_name, "images")
        if os.path.isdir(val_images_dir):
            break
    else:
        val_images_dir = os.path.join(dataset_path, "images", "val")

    sample_paths = sorted(glob.glob(os.path.join(val_images_dir, "*.jpg")))[:20]
    if not sample_paths:
        sample_paths = sorted(
            glob.glob(os.path.join(val_images_dir, "*.png"))
        )[:20]

    if not sample_paths:
        print(f"⚠️  No se encontraron imágenes en: {val_images_dir}")
    else:
        vis_paths = sample_paths[:8]
        dets = predict_yolo(
            model_path=best_pt,
            image_paths=vis_paths,
            imgsz=setup.img_size,
            class_names=setup.class_names,
        )
        infer_path = os.path.join(exp_dir, "inference_samples.png")
        visualize_predictions(
            vis_paths, dets,
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

    test_ev = evaluate_yolo_model(
        model_path=best_pt,
        data_yaml=data_yaml_path,
        split="test",
        imgsz=setup.img_size,
        class_names=setup.class_names,
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
        export_tflite_int8, print_export_report, save_export_result,
    )

    export_dir = os.path.join(exp_dir, "tflite")

    export_result = export_tflite_int8(
        model=best_pt,
        family=family,
        output_dir=export_dir,
        model_name=setup.experiment_name,
        imgsz=setup.img_size,
        data_yaml=data_yaml_path,
    )

    print_export_report(export_result)
    save_export_result(export_result, os.path.join(exp_dir, "export_result.json"))

    # ================================================================
    # Bloque 11 — Comparación Framework vs TFLite
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 11 — Comparación Framework vs TFLite")
    print("=" * 60)

    from src_colab import (
        compare_framework_vs_tflite, save_comparison_result,
        evaluate_tflite_model,
        plot_fw_vs_tflite_metrics, visualize_fw_vs_tflite_samples,
        predict_tflite,
    )

    # 11.1 Comparación rápida
    N_COMPARE = 20
    if sample_paths:
        compare_paths = sample_paths[:N_COMPARE]
        compare_imgs = []
        for p in compare_paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (setup.img_size, setup.img_size))
            compare_imgs.append(img / 255.0)
        compare_imgs = np.array(compare_imgs, dtype=np.float32)

        comparison = compare_framework_vs_tflite(
            framework_model=best_pt,
            tflite_path=export_result.tflite_path,
            images=compare_imgs,
            class_names=setup.class_names,
            family=family,
            model_path=best_pt,
            imgsz=setup.img_size,
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
            dataset_dir=dataset_path,
            split="test",
        )

        # 11.3 Gráfica comparativa
        fw_tfl_path = os.path.join(exp_dir, "fw_vs_tflite_metrics.png")
        plot_fw_vs_tflite_metrics(
            fw_ev=test_ev, tfl_ev=tflite_test_ev,
            save_path=fw_tfl_path,
        )
        logger.log_figure(fw_tfl_path, "fw_vs_tflite_metrics")

        # 11.4 Visualización side-by-side
        fw_vis_dets = predict_yolo(
            model_path=best_pt,
            image_paths=[compare_imgs[i] for i in range(compare_imgs.shape[0])],
            imgsz=setup.img_size, conf=0.25, iou=0.45,
            class_names=setup.class_names,
        )
        tfl_vis_dets, _ = predict_tflite(
            tflite_path=export_result.tflite_path,
            images=compare_imgs,
            class_names=setup.class_names,
            conf_threshold=0.25,
            iou_threshold=0.45,
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
        print("⚠️  Sin imágenes de muestra — se omite comparación visual")
        comparison = None
        tflite_test_ev = None

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
    r.tflite_size_mb = export_result.size_mb
    r.tflite_esp32_ok = export_result.esp32_compatible
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
