"""Entry-point de Vertex AI — YOLO26 Custom (Ultralytics 2-phase).

8-block pipeline (Cycle 2):
  1. Setup        — descarga config + dataset + DDP cleanup
  2. Verify       — verifica dataset YOLO + distribución
  3. Build        — carga modelo YOLO26n preentrenado
  4. Train        — 2 fases via Ultralytics (freeze → unfreeze)
  5. Curves       — extrae y grafica curvas de entrenamiento
  6. Val          — evaluación en validación (mAP@50)
  7. Test         — evaluación en test + predicciones
  8. Save+Upload  — guarda artefactos + sube a GCS

Uso::

    python -m trainer.task_yolo26_custom \\
        --config-uri gs://bucket/configs/yolo26n_custom_v1.yaml \\
        --job-dir gs://bucket/output \\
        --project-id my-project \\
        --region us-central1 \\
        --run-name yolo26n_custom_v1-20260301
"""
from __future__ import annotations

# ── DDP cleanup: Vertex AI inyecta vars distribuidas que rompen
#    Ultralytics en single-GPU (lección Ciclo 1).
import os as _os
for _var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
    _os.environ.pop(_var, None)

# ── Install ultralytics (not in setup.py to avoid container conflicts)
import subprocess, sys as _sys
subprocess.check_call(
    [_sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4"],
)

# ── Monkey-patch: PyTorch 2.4 + ultralytics single-GPU
import torch.utils.data as _tud
if not hasattr(_tud.RandomSampler, "set_epoch"):
    _tud.RandomSampler.set_epoch = lambda self, epoch: None  # type: ignore

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
    p = argparse.ArgumentParser(description="Vertex AI — YOLO26 Custom Training")
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
    family = setup.model_family  # "YOLO26_CUSTOM"
    fc = setup.family_config     # dict from yolo26_custom: section
    gcs_dataset_uri = get_gcs_dataset_uri(local_config)

    datasets_dir = os.path.join(LOCAL_WORK_DIR, "datasets")
    dataset_path = prepare_dataset(gcs_dataset_uri, datasets_dir, family)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    run_name = args.run_name or f"yolo26-{int(time.time())}"
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

    # ================================================================
    # Bloque 3 — Build Model
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 3 — Carga del Modelo YOLO26 Custom")
    print("=" * 60)

    from src_colab import (
        build_yolo26_custom_model, YOLO26_CUSTOM_SPECS,
        estimate_model_size,
    )

    pretrained = fc.get("pretrained_weights", "yolo11n.pt")
    model = build_yolo26_custom_model(pretrained_weights=pretrained)
    print(f"✅ YOLO26 cargado desde: {pretrained}")
    print(f"📐 Specs: {YOLO26_CUSTOM_SPECS}")

    logger.log_params(YOLO26_CUSTOM_SPECS)

    # ================================================================
    # Bloque 4 — Entrenamiento (2 fases Ultralytics)
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 4 — Entrenamiento (2 fases)")
    print("=" * 60)

    from src_colab import Yolo26CustomConfig, train_yolo26_custom

    project_dir = os.path.join(LOCAL_WORK_DIR, "yolo_project")

    yolo_cfg = Yolo26CustomConfig(
        data_yaml=data_yaml_path,
        project_dir=project_dir,
        experiment_name=setup.experiment_name,
        imgsz=fc.get("imgsz", setup.common_config.get("img_size", 640)),
        batch=fc.get("batch", setup.common_config.get("batch_size", 16)),
        pretrained_weights=pretrained,
        # Phase 1
        phase1_epochs=fc.get("phase1_epochs", 30),
        phase1_freeze=fc.get("phase1_freeze_layers", 10),
        phase1_lr0=fc.get("phase1_lr0", 0.01),
        phase1_lrf=fc.get("phase1_lrf", 0.01),
        # Phase 2
        phase2_epochs=fc.get("phase2_epochs", 70),
        phase2_freeze=fc.get("phase2_freeze_layers", 0),
        phase2_lr0=fc.get("phase2_lr0", 0.001),
        phase2_lrf=fc.get("phase2_lrf", 0.001),
        # Aug
        mosaic=fc.get("mosaic", 1.0),
        mixup=fc.get("mixup", 0.1),
        close_mosaic=fc.get("close_mosaic", 10),
        # Other
        patience=setup.common_config.get("patience", 30),
        amp=fc.get("amp", True),
        workers=fc.get("workers", 4),
        seed=setup.common_config.get("seed", 42),
    )

    t0 = time.time()
    results, best_weights = train_yolo26_custom(model, yolo_cfg)
    train_time = time.time() - t0
    print(f"⏱️  Entrenamiento completado en {train_time / 60:.1f} min")
    print(f"📦 Best weights: {best_weights}")

    logger.log_metrics({"train_time_min": train_time / 60})

    # ================================================================
    # Bloque 5 — Curvas de Entrenamiento
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 5 — Curvas de Entrenamiento")
    print("=" * 60)

    from src_colab import (
        extract_yolo26_history, plot_training_curves,
        print_training_summary,
    )

    # Find results.csv from Ultralytics output
    import glob
    results_csvs = glob.glob(os.path.join(project_dir, "**", "results.csv"),
                             recursive=True)
    if results_csvs:
        history_csv = results_csvs[-1]  # latest
        th = extract_yolo26_history(history_csv)
        curves_path = os.path.join(LOCAL_WORK_DIR, "training_curves.png")
        plot_training_curves(th, save_path=curves_path,
                             title=f"YOLO26 Custom — {setup.experiment_name}")
        logger.log_figure(curves_path, "training_curves")
        print_training_summary(th)
    else:
        print("⚠️ No results.csv found — skipping curves")
        history_csv = ""
        curves_path = ""

    # ================================================================
    # Bloque 6 — Evaluación en Validación
    # ================================================================
    print("\n" + "=" * 60)
    print("BLOQUE 6 — Evaluación en Validación")
    print("=" * 60)

    from src_colab import (
        evaluate_yolo26_model,
        plot_confusion_matrix, plot_per_class_metrics,
        save_evaluation,
    )

    val_results = evaluate_yolo26_model(
        model_path=best_weights,
        data_yaml=data_yaml_path,
        imgsz=fc.get("export_imgsz", fc.get("imgsz", 224)),
        conf=setup.common_config.get("conf_threshold", 0.25),
        iou=setup.common_config.get("iou_threshold", 0.45),
        split="val",
        class_names=setup.class_names,
    )

    print(f"📊 Val mAP@50: {val_results.map50:.4f}")

    cm_path = os.path.join(LOCAL_WORK_DIR, "val_confusion_matrix.png")
    plot_confusion_matrix(val_results, save_path=cm_path)
    logger.log_figure(cm_path, "val_confusion_matrix")

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

    test_results = evaluate_yolo26_model(
        model_path=best_weights,
        data_yaml=data_yaml_path,
        imgsz=fc.get("export_imgsz", fc.get("imgsz", 224)),
        conf=setup.common_config.get("conf_threshold", 0.25),
        iou=setup.common_config.get("iou_threshold", 0.45),
        split="test",
        class_names=setup.class_names,
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
        export_yolo26_to_onnx, verify_onnx_model,
        upload_to_gcs,
        create_experiment_from_setup, save_experiment,
    )

    # ONNX export via Ultralytics
    export_dir = os.path.join(LOCAL_WORK_DIR, "export")
    export_result = export_yolo26_to_onnx(
        model_path=best_weights,
        export_dir=export_dir,
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
    experiment.model_path = best_weights
    experiment.onnx_path = export_result.export_path
    experiment.history_csv = history_csv
    experiment.mark_completed()

    exp_json = os.path.join(LOCAL_WORK_DIR, "experiment.json")
    save_experiment(experiment, exp_json)

    # Upload artifacts
    artifacts = [
        local_config, history_csv, curves_path, dist_path,
        cm_path, val_json, test_cm_path, test_json,
        export_result.export_path, exp_json, best_weights,
    ]
    for artifact in artifacts:
        if artifact and os.path.exists(artifact):
            rel = os.path.relpath(artifact, LOCAL_WORK_DIR)
            gcs_dest = f"{args.job_dir}/{run_name}/{rel}"
            upload_to_gcs(artifact, gcs_dest)

    print("\n✅ Pipeline YOLO26 Custom completado exitosamente.")
    logger.end_run()


if __name__ == "__main__":
    main()
