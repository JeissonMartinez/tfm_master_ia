"""Integración con Vertex AI Experiments para registro de métricas y artefactos.

Encapsula la interacción con ``google.cloud.aiplatform`` para que los
scripts ``task_mobilenet.py`` y ``task_yolo.py`` puedan registrar
hiperparámetros, métricas por época, evaluaciones y figuras PNG
en la consola de Google Cloud de forma transparente.

Ejemplo de uso::

    from trainer.vertex_logging import VertexExperimentLogger

    logger = VertexExperimentLogger(
        project_id="my-project",
        region="us-central1",
        experiment_name="tfm-deteccion",
        run_name="yolo26n_v1-20260207",
        staging_bucket="gs://my-bucket",
    )
    logger.log_config(setup)
    logger.log_training_metrics(history, training_time_min)
    logger.log_figure("/tmp/curves.png", "training_curves")
    logger.end_run()
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from google.cloud import aiplatform


class VertexExperimentLogger:
    """Wrapper para Vertex AI Experiments dentro de un Custom Job.

    Registra hiperparámetros, métricas escalares, series temporales
    y figuras PNG como artefactos del experimento.
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        experiment_name: str,
        run_name: str,
        staging_bucket: str,
    ) -> None:
        self.project_id = project_id
        self.region = region
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.staging_bucket = staging_bucket
        self._run = None

        # Inicializar SDK
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=staging_bucket,
            experiment=experiment_name,
        )

        # Iniciar run
        self._run = aiplatform.start_run(run_name)
        print(f"📊 Vertex AI Experiment: {experiment_name} / {run_name}")

    # ── Hiperparámetros ──────────────────────────────────────────────

    def log_config(self, setup) -> None:
        """Registra todos los hiperparámetros del ``ExperimentSetup``.

        Vertex AI Experiments solo acepta str / int / float como valores
        de parámetro, así que las listas se serializan como JSON strings.
        """
        params: Dict[str, Any] = {
            "model_family": setup.model_family,
            "model_variant": setup.model_variant,
            "version": setup.version,
            "experiment_name": setup.experiment_name,
            "dataset_name": setup.dataset_name,
            "num_classes": setup.num_classes,
            "img_size": setup.img_size,
            "batch_size": setup.batch_size,
            "patience": setup.patience,
            "seed": setup.seed,
            "conf_threshold": setup.conf_threshold,
            "iou_threshold": setup.iou_threshold,
            "class_names": json.dumps(setup.class_names),
        }

        # Añadir config específica de la familia
        family_cfg = setup.yolo_config or setup.mobilenet_config
        if family_cfg:
            for k, v in family_cfg.items():
                if isinstance(v, (list, dict)):
                    params[k] = json.dumps(v)
                elif isinstance(v, bool):
                    params[k] = int(v)
                else:
                    params[k] = v

        aiplatform.log_params(params)
        print(f"  📝 Registrados {len(params)} hiperparámetros")

    # ── Métricas de entrenamiento ────────────────────────────────────

    def log_training_metrics(
        self,
        history,
        training_time_min: float,
    ) -> None:
        """Registra métricas finales del entrenamiento.

        Args:
            history: ``TrainingHistory`` con las curvas de loss.
            training_time_min: Tiempo total de entrenamiento en minutos.
        """
        metrics: Dict[str, float] = {
            "training_time_min": round(training_time_min, 2),
            "total_epochs": history.n_epochs,
        }

        if history.val_total_loss:
            metrics["best_val_loss"] = round(min(history.val_total_loss), 6)
            metrics["final_val_loss"] = round(history.val_total_loss[-1], 6)
            metrics["best_epoch"] = history.best_epoch_by_val_loss

        if history.train_total_loss:
            metrics["final_train_loss"] = round(history.train_total_loss[-1], 6)

        aiplatform.log_metrics(metrics)
        print(f"  📈 Registradas métricas de entrenamiento")

    def log_time_series(self, history) -> None:
        """Registra métricas por época como series temporales.

        Permite visualizar curvas de loss directamente en la consola
        de Vertex AI Experiments.
        """
        for i in range(history.n_epochs):
            step_metrics: Dict[str, float] = {}

            if history.train_total_loss and i < len(history.train_total_loss):
                step_metrics["epoch_train_loss"] = round(
                    history.train_total_loss[i], 6
                )
            if history.val_total_loss and i < len(history.val_total_loss):
                step_metrics["epoch_val_loss"] = round(
                    history.val_total_loss[i], 6
                )
            if history.mAP50 and i < len(history.mAP50):
                step_metrics["epoch_mAP50"] = round(history.mAP50[i], 6)
            if history.lr and i < len(history.lr):
                step_metrics["epoch_lr"] = history.lr[i]

            if step_metrics:
                aiplatform.log_time_series_metrics(
                    step_metrics, step=i + 1
                )

        print(f"  📉 Registradas {history.n_epochs} épocas de series temporales")

    # ── Métricas de evaluación ───────────────────────────────────────

    def log_evaluation(self, ev, prefix: str = "val") -> None:
        """Registra métricas de evaluación (val o test).

        Args:
            ev: ``EvaluationResults`` con mAP, precision, recall, etc.
            prefix: Prefijo para las claves (``"val"`` o ``"test"``).
        """
        metrics = {
            f"{prefix}_mAP50": round(ev.mAP50, 6),
            f"{prefix}_mAP50_95": round(ev.mAP50_95, 6),
            f"{prefix}_precision": round(ev.precision, 6),
            f"{prefix}_recall": round(ev.recall, 6),
            f"{prefix}_f1": round(ev.f1, 6),
        }

        # Per-class AP50
        if ev.per_class_ap50 and ev.class_names:
            for cls_name, ap in zip(ev.class_names, ev.per_class_ap50):
                metrics[f"{prefix}_ap50_{cls_name}"] = round(ap, 6)

        aiplatform.log_metrics(metrics)
        print(f"  📊 Registradas métricas {prefix}")

    def log_export_metrics(self, export_result, comparison) -> None:
        """Registra métricas del export TFLite y comparación FW vs TFLite.

        Args:
            export_result: ``TFLiteExportResult``.
            comparison: ``TFLiteVerificationResult``.
        """
        metrics = {
            "tflite_size_mb": round(export_result.size_mb, 4),
            "tflite_esp32_ok": int(export_result.esp32_compatible),
            "tflite_agreement": round(comparison.agreement_rate, 4),
            "tflite_avg_latency_ms": round(comparison.avg_inference_ms, 2),
        }
        aiplatform.log_metrics(metrics)
        print(f"  📊 Registradas métricas de export")

    def log_tflite_test(self, tflite_ev) -> None:
        """Registra métricas del TFLite evaluado sobre test split.

        Args:
            tflite_ev: ``EvaluationResults`` del modelo TFLite.
        """
        self.log_evaluation(tflite_ev, prefix="tflite_test")

    # ── Figuras / Artefactos ─────────────────────────────────────────

    def log_figure(self, local_path: str, display_name: str) -> None:
        """Sube un archivo PNG como artefacto vinculado al run.

        El archivo se sube al staging bucket y se registra como un
        ``Artifact`` de tipo ``system.Artifact`` en Vertex AI.

        Args:
            local_path: Ruta local al PNG.
            display_name: Nombre descriptivo para el artefacto.
        """
        if not os.path.isfile(local_path):
            print(f"  ⚠️  Archivo no encontrado: {local_path}")
            return

        # Construir URI GCS en staging
        gcs_uri = (
            f"{self.staging_bucket}/artifacts/"
            f"{self.experiment_name}/{self.run_name}/{display_name}"
        )
        if not gcs_uri.endswith(".png"):
            gcs_uri += ".png"

        # Subir a GCS
        from trainer.gcs_utils import upload_file_to_gcs
        upload_file_to_gcs(local_path, gcs_uri)

        # Registrar como artefacto de Vertex AI Experiments
        try:
            with aiplatform.start_execution(
                display_name=display_name,
                resume=True,
            ) as exc:
                exc.assign_output_artifacts(
                    [
                        aiplatform.Artifact.create(
                            schema_title="system.Artifact",
                            display_name=display_name,
                            uri=gcs_uri,
                        )
                    ]
                )
        except Exception:
            # Fallback: registrar como métrica de texto con la URI
            aiplatform.log_metrics(
                {f"artifact_{display_name}": gcs_uri}
            )

        print(f"  🖼️  Artefacto registrado: {display_name}")

    # ── Cierre ───────────────────────────────────────────────────────

    def end_run(self) -> None:
        """Finaliza el run de Vertex AI Experiments."""
        try:
            aiplatform.end_run()
            print(f"  ✅ Run '{self.run_name}' finalizado")
        except Exception as e:
            print(f"  ⚠️  Error al cerrar run: {e}")
