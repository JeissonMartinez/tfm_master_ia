# Análisis de la Infraestructura de Entrenamiento en Google Vertex AI

> **Fecha:** 20 de febrero de 2026  
> **Alcance:** Revisión completa de `02_ING_MODELOS/GoogleCloudAI/`  
> **Objetivo:** Documentar metodología, recursos, aprendizajes y sugerencias antes de iniciar un nuevo ciclo de entrenamientos con dataset actualizado.

---

## Tabla de Contenidos

1. [Arquitectura General del Proyecto](#1-arquitectura-general-del-proyecto)
2. [Metodología: Paso a Paso para Nuevos Entrenamientos](#2-metodología-paso-a-paso-para-nuevos-entrenamientos)
3. [Recursos Disponibles (ya creados y reutilizables)](#3-recursos-disponibles-ya-creados-y-reutilizables)
4. [Resultados Obtenidos (Ciclos 1-3)](#4-resultados-obtenidos-ciclos-1-3)
5. [Aprendizajes y Lecciones Aprendidas](#5-aprendizajes-y-lecciones-aprendidas)
6. [Sugerencias para el Próximo Ciclo](#6-sugerencias-para-el-próximo-ciclo)

---

## 1. Arquitectura General del Proyecto

### 1.1 Estructura de Carpetas

```
GoogleCloudAI/
├── setup.py                    # Empaquetado sdist para Vertex AI
├── requirements_mobilenet.txt  # Deps extra para contenedor TF
├── requirements_yolo.txt       # Deps extra para contenedor PyTorch
├── trainer/                    # Entry-points para Vertex AI Custom Jobs
│   ├── task_mobilenet.py       # Pipeline completa MobileNet (12 bloques)
│   ├── task_yolo.py            # Pipeline completa YOLO (12 bloques)
│   ├── config_loader.py        # YAML → ExperimentSetup
│   ├── gcs_utils.py            # Download/upload GCS
│   └── vertex_logging.py       # Wrapper Vertex AI Experiments
├── src_colab/                  # Toolkit compartido (13 módulos)
│   ├── config.py               # Familias, rutas, entorno
│   ├── utils_data.py           # Dataset, TFRecord, anchors
│   ├── utils_model.py          # Build MobileNet SSD, load YOLO
│   ├── utils_train.py          # Train loops, callbacks, losses
│   ├── utils_metrics.py        # History, curvas, resúmenes
│   ├── utils_eval.py           # mAP, confusion matrix
│   ├── utils_infer.py          # Predicción FW y TFLite
│   ├── utils_export.py         # Export TFLite INT8
│   ├── utils_compare.py        # FW vs TFLite comparison
│   ├── utils_experiment.py     # UnifiedExperiment, CSV comparativo
│   ├── utils_widgets.py        # ExperimentSetup (Colab widgets)
│   └── utils_io.py             # Helpers I/O
├── vertex_ai/
│   ├── build_and_launch.sh     # Script orquestador (build → upload → launch)
│   ├── launch_job.py           # Python launcher con CLI args
│   └── configs/                # 7 YAML de configuración de experimentos
├── notebooks/                  # Conclusiones, eval local, export
├── outputs/                    # Artefactos descargados de cada experimento
├── logs/                       # Logs JSON de Cloud Logging
└── scripts/                    # Anchor analysis con k-means
```

### 1.2 Flujo de Datos General

```
                    ┌─────────────────────────────┐
                    │  Dataset local (COCO/YOLO)   │
                    │  02_ING_MODELOS/datasets/     │
                    └──────────┬──────────────────┘
                               │ subir a GCS
                               ▼
               ┌───────────────────────────────────┐
               │  GCS: gs://...tfm-data/datasets/  │
               │  - yolo26.zip (formato YOLO)       │
               │  - tfrecord/ (formato TFRecord)    │
               └──────────┬────────────────────────┘
                          │
        ┌─────────────────┼─────────────────────┐
        │                 │                     │
        ▼                 ▼                     ▼
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐
  │ YAML cfg │    │ setup.py     │    │ src_colab/   │
  │ (hparams)│    │ sdist → GCS  │    │ trainer/     │
  └────┬─────┘    └──────┬───────┘    └──────┬───────┘
       │                 │                    │
       └────────┬────────┘                    │
                ▼                             │
  ┌──────────────────────────┐                │
  │  build_and_launch.sh     │◄───────────────┘
  │  1. python setup.py sdist│
  │  2. gsutil cp dist/*.gz  │
  │  3. python launch_job.py │
  └──────────┬───────────────┘
             ▼
  ┌──────────────────────────────────────┐
  │  Vertex AI Custom Job (T4 GPU)       │
  │  ┌─────────────────────────────────┐ │
  │  │ Contenedor pre-built            │ │
  │  │ TF 2.17 (MobileNet)            │ │
  │  │ PyTorch 2.4 (YOLO)             │ │
  │  │ + tfm-trainer-1.0.0.tar.gz     │ │
  │  └─────────────────────────────────┘ │
  │  12 Bloques: Setup → Train → Eval  │
  │  → Export TFLite → Upload GCS       │
  └──────────┬───────────────────────────┘
             ▼
  ┌──────────────────────────────────────┐
  │  GCS: gs://...tfm-data/output/      │
  │  - model.keras / best.pt            │
  │  - tflite/model_int8.tflite         │
  │  - experiment.json                  │
  │  - training_curves.png              │
  │  - *_evaluation.json                │
  │  - *_confusion_matrix.png           │
  └──────────────────────────────────────┘
```

### 1.3 Familias de Modelos Soportadas

| Familia | Framework | Contenedor Vertex AI | Formato Dataset | Fases de Entrenamiento |
|---|---|---|---|---|
| **YOLO11** | PyTorch + Ultralytics | `pytorch-gpu.2-4.py310` | YOLO (txt/images) | 1 fase |
| **YOLO26** | PyTorch + Ultralytics | `pytorch-gpu.2-4.py310` | YOLO (txt/images) | 1 fase |
| **MobileNetV2** | TensorFlow/Keras | `tf-gpu.2-17.py310` | TFRecord | 2 fases (freeze + fine-tune) |
| **MobileNetV3** | TensorFlow/Keras | `tf-gpu.2-17.py310` | TFRecord | 2 fases (freeze + fine-tune) |

---

## 2. Metodología: Paso a Paso para Nuevos Entrenamientos

### Paso 0 — Preparar el Dataset

1. **Formato YOLO** (para YOLO11/YOLO26):
   - Estructura: `train/images/`, `train/labels/`, `val/images/`, `val/labels/`, `test/images/`, `test/labels/`
   - Labels en formato YOLO: `class_id cx cy w h` (normalizado 0-1)
   - Comprimir como `.zip` y subir a GCS:
     ```bash
     gsutil cp dataset.zip gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/dataset.zip
     ```

2. **Formato TFRecord** (para MobileNetV2/V3):
   - Directorio con `train.tfrecord`, `val.tfrecord`, `test.tfrecord` + `metadata.json`
   - Generados desde `src_colab.utils_data.write_tfrecord()`
   - Subir directorio completo a GCS:
     ```bash
     gsutil -m cp -r tfrecord/ gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/tfrecord_v2/
     ```

> **IMPORTANTE:** Los modelos YOLO y MobileNet usan formatos de dataset diferentes. Si se cambia el dataset, hay que regenerar AMBOS formatos.

### Paso 1 — Crear/Editar el YAML de Configuración

Crear un nuevo archivo en `vertex_ai/configs/` basándose en los existentes:

```yaml
model:
  family: YOLO26                           # YOLO11 | YOLO26 | MobileNetV2 | MobileNetV3
  variant: yolo26n                         # yolo26n | yolo11n | MobileNetV3S_SSDLite | etc.
  version: v1                              # versión del experimento
  description: "Descripción del experimento"

dataset:
  name: yolo26                             # nombre del dataset
  gcs_uri: gs://bucket/datasets/data.zip   # URI en GCS (zip o directorio)
  class_names: [dog, door, obstacle, person, stair]
  img_size: 224                            # restricción ESP32-S3

common:
  batch_size: 32
  patience: 30
  seed: 42
  conf_threshold: 0.25
  iou_threshold: 0.45

# Sección específica de la familia:
yolo:
  epochs: 100
  optimizer: auto
  lr0: 0.01
  # ... (ver YAMLs existentes como referencia)

# O si es MobileNet:
mobilenet:
  phase1_epochs: 50
  phase1_lr: 0.003
  phase2_epochs: 250
  phase2_lr: 0.0001
  # ... (ver YAMLs existentes como referencia)
```

### Paso 2 — Autenticación con Google Cloud

```bash
gcloud auth application-default login
gcloud config set project project-18f58341-12cf-47bc-861
```

### Paso 3 — Lanzar el Entrenamiento

```bash
cd 02_ING_MODELOS/GoogleCloudAI/

# Opción A: Script completo (build + upload + launch)
./vertex_ai/build_and_launch.sh nombre_config

# Opción B: Dry-run (solo muestra configuración, no lanza)
./vertex_ai/build_and_launch.sh nombre_config --dry-run

# Opción C: Con nombre de run personalizado
./vertex_ai/build_and_launch.sh nombre_config --run-name mi-experimento-v2
```

El script hace 3 cosas automáticamente:
1. `python setup.py sdist` → empaqueta `src_colab/` + `trainer/` en `dist/tfm_trainer-1.0.0.tar.gz`
2. `gsutil cp` → sube el paquete a `gs://...tfm-data/packages/`
3. `python launch_job.py` → sube el YAML config a GCS y lanza un `CustomPythonPackageTrainingJob`

### Paso 4 — Monitorear el Job

- **Consola GCP:** `https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=project-18f58341-12cf-47bc-861`
- **Logs en vivo:** Cloud Logging → filtrar por el job ID
- **Vertex AI Experiments:** hiperparámetros, métricas por época, artefactos PNG

### Paso 5 — Descargar Resultados

Los artefactos se suben automáticamente a `gs://...tfm-data/output/{experiment_name}/`:

```bash
# Descargar todo el directorio de un experimento
gsutil -m cp -r gs://...tfm-data/output/yolo26n_v2/ outputs/yolo26n_v2/
```

Artefactos generados por cada experimento:
- `experiment.json` — configuración + todas las métricas
- `training_curves.png` — curvas de loss/mAP por época
- `val_evaluation.json` + `test_evaluation.json` — métricas completas
- `val_confusion_matrix.png` + `test_confusion_matrix.png`
- `val_per_class.png` + `test_per_class.png`
- `inference_samples.png` — predicciones visuales en muestras de val
- `tflite/model_int8.tflite` — modelo cuantizado INT8
- `export_result.json` — detalles del export (tamaño, shapes, errors)
- `comparison_result.json` — agreement FW vs TFLite
- `fw_vs_tflite_metrics.png` + `fw_vs_tflite_samples.png`
- `checkpoints/` — checkpoints intermedios
- `logs/` — CSVs con historial de entrenamiento

---

## 3. Recursos Disponibles (ya creados y reutilizables)

### 3.1 Infraestructura GCP

| Recurso | Detalle | Estado |
|---|---|---|
| Proyecto GCP | `project-18f58341-12cf-47bc-861` | ✅ Activo |
| Región | `us-central1` | ✅ Configurada |
| Bucket GCS | `gs://project-18f58341-12cf-47bc-861-tfm-data` | ✅ Con datos |
| Vertex AI Experiments | `tfm-deteccion-objetos` | ✅ Con runs anteriores |
| Máquina tipo | `n1-standard-8` (8 vCPU, 30 GB RAM) | ✅ Funcional |
| GPU | NVIDIA Tesla T4 × 1 (16 GB VRAM) | ✅ Funcional |

### 3.2 Datasets ya subidos a GCS

| Dataset | URI GCS | Formato | Clases |
|---|---|---|---|
| YOLO | `gs://...tfm-data/datasets/yolo26.zip` | YOLO (txt+images) | dog, door, obstacle, person, stair |
| TFRecord | `gs://...tfm-data/datasets/tfrecord/` | TFRecord (train/val/test) | dog, door, obstacle, person, stair |

> Ambos datasets se generaron a partir del mismo dataset COCO maestro con 5 clases y están en `02_ING_MODELOS/datasets/IODC/` en formatos `coco/`, `tfrecord/`, `yolo/`.

### 3.3 Paquete de Código (`tfm-trainer-1.0.0`)

El paquete sdist incluye todo lo necesario:

- **`trainer/`** — 5 archivos, entry-points específicos para MobileNet y YOLO
- **`src_colab/`** — 13 módulos (~3000+ líneas), toolkit unificado con:
  - Soporte para 4 familias de modelos
  - Pipeline completa de 12 bloques (dataset → train → eval → export → compare)
  - Verificación de dataset, class weights, data augmentation
  - Generación de anclas SSD con soporte para offset regression
  - Export TFLite INT8 con representative dataset
  - Evaluación TFLite sobre test split
  - Comparación automática Framework vs TFLite (agreement, IoU, Δconf)
  - Registro en Vertex AI Experiments (params, metrics, time-series, artifacts)

### 3.4 Configuraciones YAML Pre-existentes

| Config | Familia | Versión | Cambios Clave |
|---|---|---|---|
| `mobilenet_v3s_ssdlite_v1.yaml` | MobileNetV3-Small | Baseline | batch=32, patience=15, medium augmentation |
| `mobilenet_v3s_ssdlite_v2.yaml` | MobileNetV3-Small | Ciclo 2 | AdamW, cosine LR, label_smoothing, cls_weight=2.0, heavy aug |
| `mobilenet_v3s_ssdlite_v3.yaml` | MobileNetV3-Small | Ciclo 3 | Offset regression SSD-standard (Δcx/Δcy/Δw/Δh) |
| `mobilenet_v2_ssdlite_v1.yaml` | MobileNetV2 | Hereda de V3S_v2 | alpha=0.5, AdamW, cosine, label_smoothing |
| `mobilenet_v2_ssdlite_v2.yaml` | MobileNetV2 | Ciclo 2 | + Offset regression |
| `yolo11n_v1.yaml` | YOLO11-Nano | Baseline | 30 épocas, auto optimizer |
| `yolo26n_v1.yaml` | YOLO26-Nano | Baseline | 30 épocas, auto optimizer |

### 3.5 Resultados de 7 Experimentos Completados

Se tienen artefactos completos descargados en `outputs/` para cada uno de los 7 experimentos ejecutados.

---

## 4. Resultados Obtenidos (Ciclos 1-3)

### 4.1 Tabla Comparativa Completa — Test Split (Framework)

| Modelo | Ciclo | Épocas | Tiempo (min) | mAP@50 | mAP@50-95 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| **YOLO11n_v1** | 1 | 30 | 20.4 | **0.8074** | **0.5982** | **0.8367** | **0.7171** | **0.7723** |
| **YOLO26n_v1** | 1 | 30 | 28.1 | 0.7932 | 0.5978 | 0.8228 | 0.6989 | 0.7558 |
| MBNTv3S_v2 | 2 | 241 | 123.1 | 0.4431 | 0.0 | 0.2066 | 0.6123 | 0.3090 |
| MBNTv2_v1 | 1 | 126 | 73.9 | 0.4064 | 0.0 | 0.1838 | 0.5975 | 0.2811 |
| MBNTv3S_v3 | 3 | 300 | 248.2 | 0.3575 | 0.0 | 0.2509 | 0.4962 | 0.3333 |
| MBNTv2_v2 | 2 | 249 | 211.0 | 0.3473 | 0.0 | 0.2465 | 0.5027 | 0.3308 |
| MBNTv3S_v1 | 1 | 80 | 44.9 | 0.0127 | 0.0 | 0.1583 | 0.0141 | 0.0259 |

### 4.2 Tabla Comparativa — TFLite INT8 (Test Split)

| Modelo | TFLite Size | ESP32 OK | TFLite mAP@50 | Δ mAP (drop) | Agreement |
|---|---|---|---|---|---|
| **YOLO11n_v1** | 2.68 MB | ✅ | 0.8064 | -0.0010 (0.1%) | — |
| **YOLO26n_v1** | 2.55 MB | ✅ | 0.7904 | -0.0027 (0.3%) | — |
| MBNTv3S_v2 | 0.75 MB | ✅ | 0.3213 | -0.1218 (27.5%) | 60.6% |
| MBNTv2_v1 | 1.20 MB | ✅ | 0.3872 | -0.0192 (4.7%) | 78.8% |
| MBNTv3S_v3 | 0.75 MB | ✅ | 0.2551 | -0.1025 (28.7%) | 64.3% |
| MBNTv2_v2 | 1.20 MB | ✅ | 0.3394 | -0.0079 (2.3%) | 67.0% |
| MBNTv3S_v1 | 0.75 MB | ✅ | 0.0115 | -0.0013 (10%) | 69.4% |

### 4.3 Métricas por Clase — Los 2 Mejores Modelos (Test, Framework)

| Clase | YOLO11n AP@50 | YOLO26n AP@50 | MBNTv3S_v2 AP@50 | Mejor MBN AP@50 |
|---|---|---|---|---|
| dog | 0.787 | 0.764 | 0.542 | 0.542 (V3S_v2) |
| door | **0.904** | **0.904** | 0.557 | 0.557 (V3S_v2) |
| obstacle | 0.746 | 0.747 | 0.255 | 0.255 (V3S_v2) |
| person | **0.746** | 0.711 | 0.162 | 0.189 (V3S_v3) |
| stair | 0.853 | 0.839 | 0.700 | 0.746 (V2_v1) |

---

## 5. Aprendizajes y Lecciones Aprendidas

### 5.1 YOLO — Lo que Funciona

1. **YOLO domina completamente en rendimiento**: Con solo 30 épocas, YOLO11n alcanza mAP@50=0.807, triplicando al mejor MobileNet (0.443). Incluso es probable que esté subentrenado.

2. **Cuantización INT8 casi sin pérdida**: YOLO pierde <0.3% mAP al cuantizar, lo cual es excepcional para edge deployment. La arquitectura moderna con cabezas end-to-end es robusta.

3. **YOLO11n ligeramente mejor que YOLO26n en 30 épocas**: Contrario a la expectativa, YOLO11n obtuvo 0.807 vs 0.793 del YOLO26n. Esto se debe probablemente a que YOLO11n converge más rápido con pocas épocas. YOLO26n probablemente necesita más épocas para mostrar su ventaja.

4. **Formato YOLO funciona sin fricción**: El dataset en formato YOLO se descarga como zip, se extrae, y Ultralytics lo consume directamente. El script genera automáticamente el `data.yaml` necesario.

### 5.2 MobileNet SSD-Lite — Lo que NO Funciona

1. **Rendimiento inaceptablemente bajo**: Incluso con 300 épocas y múltiples optimizaciones (AdamW, cosine LR, label smoothing, Focal Loss, class weights), el mejor MobileNet solo llega a mAP@50=0.443. Esto es 44% de lo que logra YOLO con 6× menos tiempo de entrenamiento.

2. **mAP@50-95 = 0.0 en TODOS los MobileNet**: Esto indica que las predicciones de bounding box son tan imprecisas que no hay overlap significativo a umbrales IoU > 0.50. Es un fallo arquitectónico fundamental en la cabeza SSD multi-branch (objectness + class + bbox separados).

3. **El offset regression (v3) NO mejoró, empeoró**:
   - V3S_v2 (sigmoid directo): mAP@50 = 0.443
   - V3S_v3 (offset Δcx/Δcy/Δw/Δh): mAP@50 = 0.358
   - La implementación de offset regression con la cabeza SSD-Lite custom fue contraproducente. Posible causa: la loss function no está correctamente calibrada para offsets, o los variances no son apropiados para el rango de tamaños del dataset.

4. **Degradación TFLite severa en MobileNetV3-Small**: 
   - V3S_v2: pierde 27.5% mAP al cuantizar (0.443 → 0.321)
   - V3S_v3: pierde 28.7% mAP al cuantizar (0.358 → 0.255)
   - El agreement FW↔TFLite es solo 60-64%, indicando que la cuantización corrompe las predicciones delicadas de la cabeza SSD multi-output.

5. **MobileNetV2 (alpha=0.5) cuantiza mejor que V3-Small**:
   - V2_v1: solo pierde 4.7% mAP (0.406 → 0.387). Agreement de 78.8%.
   - Probable causa: MobileNetV2 con alpha=0.5 tiene menos parámetros cerca de cero que V3-Small con activaciones SE y h-swish, que son más sensibles a INT8.

6. **Confusión obstacle ↔ stair persiste en MobileNet**: A 224px las texturas de escaleras y obstáculos son demasiado similares para el backbone ligero. YOLO, con cabezas DFL (Distribution Focal Loss) y feature fusion más sofisticada, sí logra distinguirlos.

7. **Person sub-detectada en MobileNet**: Los anchors predefinidos no cubren bien personas de pie (aspect ratio alto). YOLO, que no usa anchors fijos, no tiene este problema.

### 5.3 Infraestructura — Lo que Funciona

1. **El script `build_and_launch.sh` funciona end-to-end**: Empaqueta, sube, lanza. Un solo comando.

2. **El sistema de YAML configs es limpio y extensible**: Permite iterar hiperparámetros sin tocar código. Fácil de versionar y comparar.

3. **Los 12 bloques del pipeline son robustos**: Setup → Verificación → Build → Train → Curvas → Val → Inferencia → Test → Export TFLite → Comparación FW/TFLite → Registro de Experimento → Subida GCS. Todo se ejecuta de forma secuencial con error handling.

4. **Vertex AI Experiments registra todo automáticamente**: Parámetros, métricas por época, evaluaciones, artefactos PNG. Permite comparar runs en la consola GCP.

5. **El logger es tolerante a fallos**: Si Experiments no está disponible, el entrenamiento continúa sin interrupciones (decorator `@_safe_call`).

### 5.4 Infraestructura — Problemas Encontrados y Corregidos

1. **YOLO DDP en Vertex AI**: Vertex AI inyecta variables de entorno (`RANK`, `WORLD_SIZE`, etc.) que hacen que Ultralytics asuma entrenamiento distribuido. **Solución implementada**: limpiar estas vars al inicio de `task_yolo.py`.

2. **`RandomSampler.set_epoch` missing**: PyTorch 2.4 en single-GPU llama `set_epoch()` que no existe en `RandomSampler`. **Solución implementada**: monkey-patch en `task_yolo.py`.

3. **Ultralytics no está en setup.py**: Se instala dinámicamente en `task_yolo.py` con `pip install -q ultralytics>=8.4` para evitar conflictos con el contenedor TF cuando se usa MobileNet. Esto funciona pero es frágil.

4. **protobuf versioning**: Se fija `protobuf>=3.20.3,<4.0.0` en setup.py y requirements para evitar incompatibilidades con TensorFlow y gRPC.

5. **MBNTv3S_v1 tuvo un bug de evaluación**: `n_ground_truths=195,412` para 2066 imágenes (debería ser ~4000). Esto infló la denominación del recall y produjo métricas absurdas (mAP=0.013). El bug fue corregido en v2 donde `n_ground_truths=4,056` es coherente.

### 5.5 Tiempos y Costos

| Modelo | Épocas | Tiempo GPU (min) | Costo Estimado* |
|---|---|---|---|
| YOLO11n_v1 | 30 | 20 | ~$0.20 |
| YOLO26n_v1 | 30 | 28 | ~$0.28 |
| MBNTv3S_v1 | 80 | 45 | ~$0.45 |
| MBNTv2_v1 | 126 | 74 | ~$0.74 |
| MBNTv3S_v2 | 241 | 123 | ~$1.23 |
| MBNTv2_v2 | 249 | 211 | ~$2.11 |
| MBNTv3S_v3 | 300 | 248 | ~$2.48 |

*Estimado a T4 spot pricing ~$0.01/min. Los totales reales incluyen startup + descarga de datos.

---

## 6. Sugerencias para el Próximo Ciclo

### 6.1 Decisión Estratégica: Descartar MobileNet SSD-Lite Custom

**Recomendación fuerte: No invertir más tiempo en la arquitectura MobileNet + SSD-Lite custom.**

Razones:
- Después de 3 ciclos iterativos con mejoras progresivas (optimizador, LR schedule, loss weights, label smoothing, augmentation, offset regression), el mejor MobileNet (V3S_v2) solo alcanza mAP@50=0.443 — **la mitad del rendimiento de YOLO con 6× menos tiempo**.
- La degradación TFLite del 27% en V3-Small invalida el modelo para ESP32.
- El mAP@50-95=0.0 en todos los MobileNet muestra que la localización es fundamentalmente defectuosa.
- El esfuerzo de debug, tuning y tiempo de GPU no justifica seguir este camino.

**Alternativa si se necesita un modelo <1 MB:** Considerar EfficientDet-Lite0 o MobileNetV3 con cabeza de detección oficial (Object Detection API de TensorFlow), no la cabeza SSD-Lite custom que se implementó.

### 6.2 Foco en YOLO — Plan de Acción

1. **Entrenar YOLO con más épocas**: 30 épocas fue muy poco. El loss seguía bajando al final. Subir a 100-200 épocas con patience=30-50.

2. **Probar YOLO26n vs YOLO11n con 100+ épocas**: YOLO26n probablemente superará a YOLO11n con entrenamiento suficiente (backbone más nuevo).

3. **Augmentation más agresiva**: Los configs v1 usaron augmentation estándar. Para mejorar generalización con el nuevo dataset:
   - `mixup: 0.15`, `copy_paste: 0.1`, `degrees: 15.0`, `translate: 0.2`
   - `close_mosaic: 20` para estabilizar las últimas épocas

4. **cls loss weight**: Si obstacle/stair sigue confundiéndose, subir `cls: 0.75` o `cls: 1.0`.

### 6.3 Para el Nuevo Dataset

1. **Subir el nuevo dataset** a GCS en formato YOLO (.zip) y TFRecord (si se mantiene MobileNet).
2. **Actualizar `class_names`** en los YAML si cambian las clases.
3. **Actualizar `DATASET_MASTER_CLASSES`** en `src_colab/config.py` si se usa un nombre de dataset nuevo.
4. **Verificar que el dataset tiene splits train/val/test** correctamente balanceados.
5. **Actualizar `dataset.gcs_uri`** en los nuevos YAML configs.

### 6.4 Mejoras de Infraestructura Sugeridas

1. **Versionar el paquete**: Cambiar `version="1.0.0"` en `setup.py` a `1.1.0` antes de un nuevo ciclo para no cachear un paquete desactualizado en GCS.

2. **Agregar el nombre del dataset al YAML**: Incluir metadata del dataset (versión, número de imágenes, distribución) para trazabilidad completa.

3. **Considerar usar `--machine-type n1-highmem-8`** si los batches grandes de YOLO causan OOM (30 GB RAM puede ser justo con augmentation heavy + batch 64).

4. **Los logs JSON descargados** en `logs/` tienen un parser (`parse_log.py`) útil para diagnosticar errores. Mantener esta práctica para cada job.

5. **Script de descarga masiva**: Crear un helper para descargar todos los artefactos de un experimento de GCS con un solo comando.

### 6.5 Checklist Antes de Lanzar un Nuevo Job

- [ ] Dataset subido a GCS y URI verificada
- [ ] YAML de configuración creado con `class_names` correctas
- [ ] `gcloud auth application-default login` ejecutado
- [ ] Dry-run (`--dry-run`) revisado sin errores
- [ ] `setup.py` version actualizada si se cambió código
- [ ] Suficiente crédito/cuota en el proyecto GCP
- [ ] Logs del job anterior revisados para no repetir errores

---

## Anexo A: Referencia Rápida de Comandos

```bash
# Autenticación
gcloud auth application-default login

# Ver configs disponibles
ls vertex_ai/configs/*.yaml

# Dry-run (no lanza, solo muestra config)
./vertex_ai/build_and_launch.sh yolo26n_v1 --dry-run

# Lanzar entrenamiento
./vertex_ai/build_and_launch.sh yolo26n_v1

# Lanzar con nombre personalizado
./vertex_ai/build_and_launch.sh yolo26n_v1 --run-name yolo26n-dataset-v2

# Ver jobs activos
gcloud ai custom-jobs list --region=us-central1 --format="table(displayName,state,createTime)"

# Descargar resultados
gsutil -m cp -r gs://project-18f58341-12cf-47bc-861-tfm-data/output/yolo26n_v1/ outputs/yolo26n_v1/

# Ver logs de un job (reemplazar JOB_ID)
gcloud logging read "resource.labels.job_id=JOB_ID" --limit=200 --format=json > logs/job_log.json
```

## Anexo B: Parámetros CLI de `launch_job.py`

| Parámetro | Default | Descripción |
|---|---|---|
| `--config` | (requerido) | Ruta al YAML de configuración |
| `--run-name` | autogenerado | Nombre del run en Experiments |
| `--machine-type` | `n1-standard-8` | Tipo de VM (8 vCPU, 30 GB) |
| `--accelerator-type` | `NVIDIA_TESLA_T4` | Tipo de GPU |
| `--accelerator-count` | `1` | Número de GPUs |
| `--dry-run` | `false` | Solo mostrar config sin lanzar |
