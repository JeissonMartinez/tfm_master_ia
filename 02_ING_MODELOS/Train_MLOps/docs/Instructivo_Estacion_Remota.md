# Instructivo: Configuración de la Estación Remota (Vertex AI + GCS)

> **Fecha:** Ciclo 2 — Marzo 2026  
> **Alcance:** Preparación de la infraestructura remota para entrenar FCOS, YOLO26_CUSTOM y ESPDet-Pico  
> **Prerequisito:** Haber completado la configuración del workspace local `Train_MLOps/`

---

## Tabla de Contenidos

1. [Visión General](#1-visión-general)
2. [Prerequisitos](#2-prerequisitos)
3. [Paso 1 — Verificar Proyecto y Credenciales](#paso-1--verificar-proyecto-y-credenciales)
4. [Paso 2 — Subir Dataset a GCS](#paso-2--subir-dataset-a-gcs)
5. [Paso 3 — Subir Configuraciones YAML](#paso-3--subir-configuraciones-yaml)
6. [Paso 4 — Empaquetar y Subir el Paquete Python](#paso-4--empaquetar-y-subir-el-paquete-python)
7. [Paso 5 — Lanzar Custom Job](#paso-5--lanzar-custom-job)
8. [Paso 6 — Monitorizar Ejecución](#paso-6--monitorizar-ejecución)
9. [Paso 7 — Descargar Resultados](#paso-7--descargar-resultados)
10. [Paso 8 — Lanzar Export Job (Opcional)](#paso-8--lanzar-export-job-opcional)
11. [Notas y Troubleshooting](#notas-y-troubleshooting)

---

## 1. Visión General

### Arquitectura Ciclo 2

```
                    ┌─────────────────────────────┐
                    │  Dataset IODC (YOLO format) │
                    │  datasets/IODC/yolo/        │
                    └──────────┬──────────────────┘
                               │ gsutil cp → GCS
                               ▼
               ┌───────────────────────────────────┐
               │  GCS: gs://...-tfm-data/          │
               │  ├── datasets/iodc_yolo.zip       │
               │  ├── configs/*.yaml               │
               │  └── packages/tfm_trainer-2.0.0   │
               └──────────┬────────────────────────┘
                          │
        ┌─────────────────┼──────────────────────┐
        ▼                 ▼                      ▼
  ┌───────────┐   ┌──────────────┐    ┌──────────────┐
  │ YAML cfg  │   │ sdist .tar.gz│    │ iodc_yolo.zip│
  │ (hparams) │   │ (src_colab + │    │              │
  │           │   │  trainer)    │    │              │
  └─────┬─────┘   └──────┬───────┘    └───────┬──────┘
        │                │                    │
        └────────┬───────┘                    │
                 ▼                            │
  ┌──────────────────────────┐                │
  │  build_and_launch.sh     │                │
  │  ó launch_job.py         │◄───────────────┘
  └──────────┬───────────────┘
             ▼
  ┌──────────────────────────────────────┐
  │  Vertex AI Custom Job (T4 GPU)       │
  │  ┌─────────────────────────────────┐ │
  │  │ pytorch-gpu.2-4.py310:latest    │ │
  │  │ + tfm-trainer-2.0.0.tar.gz      │ │
  │  └─────────────────────────────────┘ │
  │  8 Bloques: Setup → Verify → Build   │
  │  → Train → Curves → Val → Test       │
  │  → Save+Upload                       │
  └──────────┬───────────────────────────┘
             ▼
  ┌──────────────────────────────────────┐
  │  GCS: gs://...-tfm-data/output/      │
  │  ├── best_model.pt / best.pt         │
  │  ├── export/model.onnx               │
  │  ├── experiment.json                 │
  │  ├── training_curves.png             │
  │  └── *_evaluation.json               │
  └──────────────────────────────────────┘
```

### Cambios respecto a Ciclo 1

| Aspecto | Ciclo 1 | Ciclo 2 |
|---|---|---|
| **Contenedores** | 2 (PyTorch + TF) | 1 (PyTorch único) |
| **Modelos** | YOLO11/26, MobileNet SSD | FCOS, YOLO26_CUSTOM, ESPDet |
| **Dataset format** | YOLO + TFRecord | YOLO únicamente |
| **Pipeline blocks** | 12 | 8 (sin quantize/deploy) |
| **Export** | Dentro del training job | Separate Custom Job |
| **Package version** | 1.0.0 | 2.0.0 |

---

## 2. Prerequisitos

- **gcloud CLI** instalado y autenticado
- **gsutil** disponible
- **Python 3.10+** con `google-cloud-aiplatform` instalado
- Proyecto GCP: `project-18f58341-12cf-47bc-861`
- Región: `us-central1`
- Bucket: `gs://project-18f58341-12cf-47bc-861-tfm-data`

---

## Paso 1 — Verificar Proyecto y Credenciales

```bash
# Verificar proyecto activo
gcloud config get-value project
# Debe mostrar: project-18f58341-12cf-47bc-861

# Si no está configurado:
gcloud config set project project-18f58341-12cf-47bc-861

# Verificar autenticación
gcloud auth list

# Habilitar APIs necesarias (si no están habilitadas)
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
```

---

## Paso 2 — Subir Dataset a GCS

El dataset IODC en formato YOLO debe estar en GCS como `.zip`:

```bash
# Desde el directorio raíz del proyecto
cd 02_ING_MODELOS/datasets/IODC/

# Comprimir el dataset YOLO
cd yolo
zip -r ../iodc_yolo.zip train/ valid/ test/
cd ..

# Subir a GCS
gsutil cp iodc_yolo.zip \
  gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip

# Verificar
gsutil ls -l gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
```

**Estructura esperada del zip:**
```
iodc_yolo.zip
├── train/
│   ├── images/   (1470 imágenes)
│   └── labels/   (1470 .txt YOLO)
├── valid/
│   ├── images/   (188 imágenes)
│   └── labels/   (188 .txt YOLO)
└── test/
    ├── images/   (187 imágenes)
    └── labels/   (187 .txt YOLO)
```

---

## Paso 3 — Subir Configuraciones YAML

```bash
cd 02_ING_MODELOS/Train_MLOps/

# Subir los 3 YAML de configuración
gsutil cp vertex_ai/configs/fcos_v3s_v1.yaml \
  gs://project-18f58341-12cf-47bc-861-tfm-data/configs/fcos_v3s_v1.yaml

gsutil cp vertex_ai/configs/yolo26n_custom_v1.yaml \
  gs://project-18f58341-12cf-47bc-861-tfm-data/configs/yolo26n_custom_v1.yaml

gsutil cp vertex_ai/configs/espdet_pico_v1.yaml \
  gs://project-18f58341-12cf-47bc-861-tfm-data/configs/espdet_pico_v1.yaml

# Verificar
gsutil ls gs://project-18f58341-12cf-47bc-861-tfm-data/configs/
```

---

## Paso 4 — Empaquetar y Subir el Paquete Python

```bash
cd 02_ING_MODELOS/Train_MLOps/

# Opción A: Usar el script automatizado
chmod +x vertex_ai/build_and_launch.sh
# (ver Paso 5 para el lanzamiento completo)

# Opción B: Manual
# 1. Generar sdist
python setup.py sdist

# 2. Verificar que se generó
ls -la dist/tfm_trainer-2.0.0.tar.gz

# 3. Subir a GCS
gsutil cp dist/tfm_trainer-2.0.0.tar.gz \
  gs://project-18f58341-12cf-47bc-861-tfm-data/packages/tfm_trainer-2.0.0.tar.gz
```

---

## Paso 5 — Lanzar Custom Job

### Opción A: Script automatizado (recomendado)

```bash
cd 02_ING_MODELOS/Train_MLOps/

# El script hace: sdist → upload → launch
bash vertex_ai/build_and_launch.sh \
  FCOS \
  fcos_v3s_v1.yaml \
  fcos_v3s_v1-$(date +%Y%m%d)
```

### Opción B: Python launcher

```bash
python vertex_ai/launch_job.py \
  --config vertex_ai/configs/fcos_v3s_v1.yaml \
  --run-name fcos_v3s_v1-$(date +%Y%m%d%H%M) \
  --machine-type n1-standard-8 \
  --accelerator-type NVIDIA_TESLA_T4 \
  --accelerator-count 1
```

### Lanzar los 3 modelos

```bash
# FCOS
python vertex_ai/launch_job.py \
  --config vertex_ai/configs/fcos_v3s_v1.yaml --run-name fcos_v3s_v1-run1

# YOLO26 Custom
python vertex_ai/launch_job.py \
  --config vertex_ai/configs/yolo26n_custom_v1.yaml --run-name yolo26n_custom_v1-run1

# ESPDet-Pico
python vertex_ai/launch_job.py \
  --config vertex_ai/configs/espdet_pico_v1.yaml --run-name espdet_pico_v1-run1
```

---

## Paso 6 — Monitorizar Ejecución

### Consola Web

1. Abrir [Vertex AI → Training → Custom Jobs](https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=project-18f58341-12cf-47bc-861)
2. Buscar el job por nombre del run
3. Click en el job → ver logs en tiempo real

### CLI

```bash
# Listar custom jobs recientes
gcloud ai custom-jobs list --region=us-central1 --limit=5

# Ver logs de un job específico
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

### Vertex AI Experiments

```bash
# Verificar que se registraron métricas
gcloud ai experiments describe tfm-deteccion-objetos \
  --region=us-central1
```

---

## Paso 7 — Descargar Resultados

```bash
# Descargar todos los artefactos de un run
gsutil -m cp -r \
  gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-run1/ \
  outputs/fcos_v3s_v1-run1/

# O usar el script helper
bash scripts/download_results.sh fcos_v3s_v1-run1
```

**Artefactos esperados:**
```
output/<run-name>/
├── config.yaml              # Copia del YAML usado
├── training_history.csv     # Historial de entrenamiento
├── training_curves.png      # Gráfica de curvas
├── class_distribution.png   # Distribución de clases
├── gt_samples.png           # Muestras GT por clase
├── val_confusion_matrix.png # CM validación
├── val_per_class.png        # Métricas por clase
├── val_evaluation.json      # Resultados evaluación val
├── test_confusion_matrix.png
├── test_evaluation.json
├── export/
│   └── model.onnx           # Modelo exportado
├── checkpoints/
│   └── best_model.pt        # Mejor checkpoint
└── experiment.json          # Metadatos del experimento
```

---

## Paso 8 — Lanzar Export Job (Opcional)

El job de exportación convierte ONNX → formato de despliegue (INT8, ESPDL):

```bash
python vertex_ai/launch_job.py \
  --config vertex_ai/configs/fcos_v3s_v1.yaml \
  --run-name fcos-export-$(date +%Y%m%d) \
  --accelerator-count 0
```

> **Nota:** El Export Job puede ejecutarse en CPU (sin GPU), lo que reduce costes.

---

## Notas y Troubleshooting

### Error DDP en Ultralytics (YOLO26_CUSTOM)

Vertex AI inyecta variables de entorno distribuidas (`RANK`, `WORLD_SIZE`, etc.)  
que hacen que Ultralytics asuma DDP. **Solución aplicada** en `task_yolo26_custom.py`:

```python
for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
    os.environ.pop(var, None)
```

### Contenedor único (PyTorch)

Ciclo 2 usa **un solo contenedor** para las 3 familias:
```
us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
```

### Cuotas GPU

Si el job falla con cuota insuficiente:
```bash
gcloud compute regions describe us-central1 \
  --format="value(quotas[quota_metric=NVIDIA_T4_GPUS])"
```

Solicitar incremento en: [IAM & Admin → Quotas](https://console.cloud.google.com/iam-admin/quotas)

### Costes estimados

| Recurso | Coste/hora | Job típico (2h) |
|---|---|---|
| n1-standard-8 | ~$0.38 | ~$0.76 |
| NVIDIA T4 | ~$0.35 | ~$0.70 |
| **Total** | ~$0.73 | ~**$1.46** |

Para 3 modelos × 2h ≈ **~$4.38** (estimación conservadora).
