"""Lanzador unificado de Custom Jobs en Vertex AI — Ciclo 2.

Lee un YAML de configuración, determina la familia del modelo
y lanza el entrenamiento usando el contenedor PyTorch pre-built.

Familias soportadas:
    - FCOS          → trainer.task_fcos
    - YOLO26_CUSTOM → trainer.task_yolo26_custom
    - ESPDet        → trainer.task_espdet
    - EXPORT        → trainer.task_export

Uso::

    python vertex_ai/launch_job.py \\
        --config vertex_ai/configs/fcos_v3s_v1.yaml

    python vertex_ai/launch_job.py \\
        --config vertex_ai/configs/espdet_pico_v1.yaml \\
        --run-name espdet-pico-v1-test

Requisitos previos:
    1. ``gcloud auth application-default login``
    2. Usar ``vertex_ai/build_and_launch.sh`` que empaqueta, sube y
       pasa --package-uri automáticamente.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
from google.cloud import aiplatform


# ── Configuración del proyecto ───────────────────────────────────────
PROJECT_ID = "project-18f58341-12cf-47bc-861"
REGION = "us-central1"
BUCKET_URI = "gs://project-18f58341-12cf-47bc-861-tfm-data"
EXPERIMENT_NAME = "tfm-deteccion-objetos"

# ── Contenedor pre-built de Vertex AI (solo PyTorch) ─────────────────
CONTAINER_PYTORCH = (
    "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
)

# ── Paquete en GCS (fallback; preferir --package-uri de build_and_launch.sh) ─
DEFAULT_PACKAGE_GCS_URI = f"{BUCKET_URI}/packages/tfm_trainer-2.2.0.tar.gz"

# ── Mapeo familia → módulo Python ────────────────────────────────────
FAMILY_MAP: dict[str, str] = {
    "FCOS":          "trainer.task_fcos",
    "YOLO26_CUSTOM": "trainer.task_yolo26_custom",
    "ESPDet":        "trainer.task_espdet",
    "EXPORT":        "trainer.task_export",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lanza un Custom Training Job en Vertex AI (Ciclo 2)"
    )
    parser.add_argument(
        "--config", required=True,
        help="Ruta local al YAML de configuración del experimento",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Nombre del run en Vertex AI Experiments (autogenerado si se omite)",
    )
    parser.add_argument(
        "--machine-type", default="n1-standard-8",
        help="Tipo de máquina (default: n1-standard-8 — 8 vCPU, 30 GB RAM)",
    )
    parser.add_argument(
        "--accelerator-type", default="NVIDIA_TESLA_T4",
        help="Tipo de GPU (default: NVIDIA_TESLA_T4)",
    )
    parser.add_argument(
        "--accelerator-count", type=int, default=1,
        help="Número de GPUs (default: 1)",
    )
    parser.add_argument(
        "--package-uri", default=None,
        help="URI GCS del paquete sdist (si se omite, usa DEFAULT_PACKAGE_GCS_URI)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo muestra la configuración sin lanzar el job",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Leer YAML para determinar familia ─────────────────────────
    # ── 0. Resolver URI del paquete ──────────────────────────────────
    package_gcs_uri = args.package_uri or DEFAULT_PACKAGE_GCS_URI

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    family = cfg["model"]["family"]
    variant = cfg["model"]["variant"]
    version = cfg["model"]["version"]
    experiment_name_from_config = f"{variant}_{version}"

    # ── 2. Seleccionar módulo de entrada ─────────────────────────────
    if family not in FAMILY_MAP:
        raise ValueError(
            f"Familia '{family}' no soportada. "
            f"Opciones: {list(FAMILY_MAP.keys())}"
        )

    python_module = FAMILY_MAP[family]
    container_uri = CONTAINER_PYTORCH  # Todos usan PyTorch

    # ── 3. Subir config YAML al bucket ───────────────────────────────
    run_name = (
        args.run_name
        or f"{experiment_name_from_config}-{int(time.time())}"
    )
    config_gcs_uri = f"{BUCKET_URI}/configs/{run_name}.yaml"

    # ── 4. Construir argumentos del job ──────────────────────────────
    job_dir = f"{BUCKET_URI}/output"
    job_args = [
        f"--config-uri={config_gcs_uri}",
        f"--job-dir={job_dir}",
        f"--project-id={PROJECT_ID}",
        f"--region={REGION}",
        f"--experiment-name={EXPERIMENT_NAME}",
        f"--run-name={run_name}",
    ]

    display_name = f"tfm-{experiment_name_from_config}-{int(time.time())}"

    # ── 5. Mostrar resumen ───────────────────────────────────────────
    print("=" * 60)
    print("🚀 VERTEX AI CUSTOM JOB — CONFIGURACIÓN")
    print("=" * 60)
    print(f"  Proyecto:      {PROJECT_ID}")
    print(f"  Región:        {REGION}")
    print(f"  Experimento:   {EXPERIMENT_NAME}")
    print(f"  Run:           {run_name}")
    print(f"  Familia:       {family}")
    print(f"  Módulo:        {python_module}")
    print(f"  Contenedor:    {container_uri}")
    print(f"  Máquina:       {args.machine_type}")
    print(f"  GPU:           {args.accelerator_type} x{args.accelerator_count}")
    print(f"  Paquete:       {package_gcs_uri}")
    print(f"  Config GCS:    {config_gcs_uri}")
    print(f"  Job Dir:       {job_dir}")
    print(f"  Args:          {job_args}")
    print("=" * 60)

    if args.dry_run:
        print("\n⏭️  DRY RUN — No se lanza el job.")
        return

    # ── 6. Subir config al bucket ────────────────────────────────────
    from google.cloud import storage

    bucket_name, blob_path = config_gcs_uri[5:].split("/", 1)
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(config_path))
    print(f"\n☁️  Config subido: {config_gcs_uri}")

    # ── 7. Inicializar Vertex AI SDK ─────────────────────────────────
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=BUCKET_URI,
    )

    # ── 8. Crear y lanzar Custom Job ─────────────────────────────────
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=package_gcs_uri,
        python_module_name=python_module,
        container_uri=container_uri,
    )

    print(f"\n🚀 Lanzando Custom Job: {display_name}")
    print(
        f"   Revisa el progreso en: https://console.cloud.google.com/"
        f"vertex-ai/training/custom-jobs?project={PROJECT_ID}"
    )

    job.run(
        replica_count=1,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        args=job_args,
        sync=True,
    )

    print("\n" + "=" * 60)
    print("✅ Custom Job completado exitosamente")
    print(f"   Resultados en: {job_dir}/{experiment_name_from_config}")
    print(f"   Experiments:   {EXPERIMENT_NAME} / {run_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
