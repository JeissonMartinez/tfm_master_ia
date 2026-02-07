# Google Colab — Unified Training Toolkit

> **TFM UNIR — Detección de Objetos para ESP32-S3**

Toolkit modular y unificado para el entrenamiento, evaluación, exportación y comparación de **4 familias de modelos de detección de objetos** orientados al despliegue en microcontroladores **ESP32-S3** (N16R8, 8 MB PSRAM). Diseñado para ejecutarse tanto en **Google Colab** (GPU T4) como en **entorno local** (macOS / Linux con CUDA o Apple MPS).

---

## Tabla de Contenidos

1. [Estructura del Directorio](#estructura-del-directorio)
2. [Paquete src\_colab](#paquete-src_colab)
3. [Familias de Modelos Soportadas](#familias-de-modelos-soportadas)
4. [Pipeline del Notebook (12 Bloques)](#pipeline-del-notebook-12-bloques)
5. [Dataset](#dataset)
6. [Guía de Uso en Google Colab](#guía-de-uso-en-google-colab)
7. [Guía de Uso en Entorno Local](#guía-de-uso-en-entorno-local)
8. [Artefactos Generados por Experimento](#artefactos-generados-por-experimento)
9. [Requisitos y Dependencias](#requisitos-y-dependencias)
10. [Consideraciones de Despliegue ESP32-S3](#consideraciones-de-despliegue-esp32-s3)

---

## Estructura del Directorio

```
Google_Colab/
├── 07_TrainColab.ipynb                                   ← Notebook principal (12 bloques, 27 celdas)
├── README.md                                             ← Este documento
├── Migrating_Notebook.md                                 ← Documentación del proceso de migración
├── yolo11n.pt                                            ← Pesos preentrenados YOLO11 nano
├── yolo26n.pt                                            ← Pesos preentrenados YOLO26 nano
├── calibration_image_sample_data_20x128x128x3_float32.npy  ← Datos de calibración TFLite INT8
└── src_colab/                                            ← Paquete Python (~5.450 líneas)
    ├── __init__.py           (215 loc)  API pública unificada
    ├── config.py             (283 loc)  Entorno, GPU, rutas, familias de modelos
    ├── utils_io.py           (116 loc)  I/O seguro (JSON, YAML, texto, copia de archivos)
    ├── utils_widgets.py      (658 loc)  Panel interactivo con ipywidgets
    ├── utils_data.py         (905 loc)  Verificación de datasets, pipelines YOLO/TFRecord, anclas
    ├── utils_model.py        (317 loc)  Construcción de modelos, especificaciones, resúmenes
    ├── utils_train.py        (544 loc)  Entrenamiento YOLO (1 fase) y MobileNet (2 fases)
    ├── utils_metrics.py      (314 loc)  Historial de entrenamiento, curvas, resúmenes
    ├── utils_eval.py         (516 loc)  Evaluación unificada (mAP@50, mAP@50-95, P, R, F1)
    ├── utils_infer.py        (590 loc)  Inferencia, predicciones y visualización
    ├── utils_export.py       (348 loc)  Exportación TFLite INT8 (YOLO y MobileNet)
    ├── utils_compare.py      (252 loc)  Comparación Framework vs TFLite
    └── utils_experiment.py   (391 loc)  Schema unificado de experimentos y comparativa global
```

---

## Paquete `src_colab`

El paquete `src_colab` encapsula toda la lógica del pipeline en módulos especializados con una API pública limpia. Principales abstracciones:

| Clase / Tipo | Módulo | Descripción |
|---|---|---|
| `ColabEnvironment` | `config` | Detección automática de entorno (Colab / Local), GPU (CUDA, MPS, TF-GPU) |
| `ProjectPaths` | `config` | Resolución unificada de rutas de proyecto con creación automática de directorios |
| `ExperimentSetup` | `utils_widgets` | Configuración completa del experimento seleccionada por el usuario (familia, variante, hiperparámetros) |
| `YoloTrainConfig` | `utils_train` | Dataclass con toda la configuración de entrenamiento YOLO |
| `TrainingHistory` | `utils_metrics` | Historial normalizado de métricas de entrenamiento (YOLO y MobileNet) |
| `EvaluationResults` | `utils_eval` | Resultados estandarizados de evaluación (por split y por clase) |
| `TFLiteExportResult` | `utils_export` | Resultado de exportación: ruta, tamaño, compatibilidad ESP32 |
| `TFLiteVerificationResult` | `utils_export` | Resultado de verificación de modelo TFLite |
| `DetectedObject` | `utils_infer` | Detección individual (bbox, clase, confianza) |
| `UnifiedExperiment` | `utils_experiment` | Schema JSON completo del experimento (config + resultados + metadatos) |

### Funciones auxiliares clave

- **`setup_environment()`** — Configuración completa en una sola llamada: detección de entorno, montaje de Drive, configuración de GPU y resolución de rutas.
- **`create_model_selector()`** — Panel interactivo con ipywidgets para seleccionar familia, variante, dataset e hiperparámetros.
- **`create_manual_setup()`** — Alternativa programática cuando los widgets no están disponibles.
- **`is_yolo_family()` / `is_mobilenet_family()`** — Helpers para ramificación condicional del pipeline.
- **`create_yolo_working_copy()`** — Genera una copia de trabajo del dataset con remapeo de clases cuando se entrena un subconjunto.
- **`compare_framework_vs_tflite()`** — Comparación automática de predicciones entre el modelo original y su versión TFLite INT8.

---

## Familias de Modelos Soportadas

| Familia | Variantes | Framework | Fases de Entrenamiento | Formato Dataset |
|---|---|---|---|---|
| **YOLO11** | `n` · `s` · `m` · `l` · `x` | Ultralytics (PyTorch) | 1 fase | YOLO |
| **YOLO26** | `n` · `s` · `m` · `l` · `x` | Ultralytics (PyTorch) | 1 fase | YOLO |
| **MobileNetV2 + SSD-Lite** | `alpha 0.35` · `0.5` · `1.0` | TensorFlow / Keras | 2 fases | TFRecord |
| **MobileNetV3 + SSD-Lite** | `Small` · `Large` | TensorFlow / Keras | 2 fases | TFRecord |

### Estrategia de entrenamiento

- **YOLO (1 fase):** Entrenamiento end-to-end con Ultralytics `model.train()`. Configuración completa de augmentation (mosaic, mixup, copy-paste), optimizer, learning rate schedule y early stopping.
- **MobileNet (2 fases):**
  - **Phase 1** — Backbone congelado. Se entrena únicamente la cabeza SSD-Lite con learning rate alto (~1e-3).
  - **Phase 2** — Descongelamiento parcial del backbone (capas configurables). Fine-tuning con learning rate reducido (~1e-4) y early stopping.

---

## Pipeline del Notebook (12 Bloques)

El notebook `07_TrainColab.ipynb` implementa un pipeline secuencial de 12 bloques, cada uno con su celda Markdown descriptiva seguida de la celda de código correspondiente.

| # | Bloque | Descripción |
|---|---|---|
| 1 | **Setup** | Detección de entorno (Colab/Local), instalación de dependencias, montaje de Google Drive, configuración de GPU, resolución de rutas del proyecto. |
| 2 | **Selección de Modelo** | Panel interactivo con ipywidgets para elegir familia, variante, dataset, tamaño de imagen, batch size, epochs, patience y configuración específica (augmentation YOLO, fases MobileNet). Incluye fallback manual. |
| 3 | **Verificación del Dataset** | Validación de la estructura del dataset, descompresión automática de `.zip`, creación de working copy con remapeo de clases (si se entrena un subconjunto), distribución de clases, cálculo de pesos para desbalance, generación de `data.yaml`, y visualización de muestras GT por clase. |
| 4 | **Construcción del Modelo** | Carga de pesos preentrenados (YOLO) o construcción de backbone + SSD-Lite head con anclas configurables (MobileNet). Resumen de arquitectura, estimación de tamaño y estimación de latencia en ESP32-S3. |
| 5 | **Entrenamiento** | Ejecución del entrenamiento: fase única (YOLO) o dos fases (MobileNet). Guardado de historial en CSV, modelo final en `.keras` (MobileNet) o `best.pt` (YOLO). Medición del tiempo total. |
| 6 | **Curvas de Entrenamiento** | Visualización estandarizada en panel multi-gráfica: Total Loss, Box Loss, Cls Loss, Obj/DFL Loss, Learning Rate y Métricas. Guardado como PNG. |
| 7 | **Validación (split=val)** | Evaluación sobre el conjunto de validación: mAP@50, mAP@50-95, Precision, Recall, F1. Matriz de confusión y métricas por clase. Guardado en JSON y PNG. |
| 8 | **Inferencia Visual** | Predicciones sobre muestras del conjunto de validación con visualización de bounding boxes para inspección cualitativa. Guardado como PNG. |
| 9 | **Evaluación Final (split=test)** | Métricas definitivas sobre el conjunto de test (no utilizado durante el entrenamiento). Mismas métricas que validación. |
| 10 | **Export TFLite INT8** | Cuantización a INT8 para despliegue en ESP32-S3. YOLO: `model.export(format="tflite", int8=True)`. MobileNet: SavedModel → `TFLiteConverter` con dataset representativo de calibración. Verificación de tamaño < 8 MB. |
| 11 | **Comparación Framework vs TFLite** | Verificación de consistencia entre el modelo original y la versión cuantizada: agreement rate, distribución de IoU y visualización side-by-side. |
| 12 | **Registro y Comparación** | Guardado del experimento completo como JSON unificado (`UnifiedExperiment`). Carga de todos los experimentos previos y generación de tabla comparativa, gráficas de comparación y CSV consolidado. |

---

## Dataset

### Clases de detección

El dataset maestro contiene **5 clases** orientadas a la asistencia de personas con discapacidad visual:

| ID | Clase | Descripción |
|----|-------|-------------|
| 0 | `dog` | Perros |
| 1 | `door` | Puertas |
| 2 | `obstacle` | Obstáculos genéricos |
| 3 | `person` | Personas |
| 4 | `stair` | Escaleras |

> **Nota:** El ordenamiento es alfabético (convención de exportación de Roboflow). El toolkit soporta el entrenamiento con subconjuntos de clases mediante remapeo automático de labels.

### Formatos soportados

| Formato | Estructura esperada | Familias |
|---|---|---|
| **YOLO** | `{split}/images/*.jpg` + `{split}/labels/*.txt` | YOLO11, YOLO26 |
| **TFRecord** | `train.tfrecord`, `val.tfrecord`, `test.tfrecord`, `metadata.json` | MobileNetV2, MobileNetV3 |

### Ubicación de los datasets

```
02_ING_MODELOS/datasets/
├── yolo_v11/       ← YOLO format (train/valid/test con images/ y labels/)
├── yolo_v11.zip    ← Comprimido (descompresión automática si no existe la carpeta)
├── yolo26/         ← YOLO format
├── yolo26.zip
└── tf_records/     ← TFRecord format
```

---

## Guía de Uso en Google Colab

### 1. Subir archivos a Google Drive

Copiar la carpeta completa `Google_Colab/` y los datasets a Google Drive:

```
Google Drive/
└── TFM_UNIR/
    └── 02_ING_MODELOS/
        ├── Google_Colab/        ← Este directorio completo
        │   ├── 07_TrainColab.ipynb
        │   ├── src_colab/
        │   ├── yolo11n.pt
        │   └── yolo26n.pt
        └── datasets/            ← Datasets en los formatos requeridos
            ├── yolo_v11/
            ├── yolo26/
            └── tf_records/
```

### 2. Abrir y configurar el notebook

1. Ir a [colab.research.google.com](https://colab.research.google.com).
2. **Archivo → Abrir notebook → Google Drive** → Navegar a `TFM_UNIR/02_ING_MODELOS/Google_Colab/07_TrainColab.ipynb`.
3. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**.
4. Ejecutar las celdas **secuencialmente** desde el Bloque 1.

### 3. Flujo de trabajo iterativo

El notebook está diseñado para iteración rápida entre experimentos:

1. Ejecutar Bloques 1 (Setup) — solo una vez por sesión.
2. Bloques 2–12 — un ciclo completo por experimento.
3. Para un nuevo experimento: **volver al Bloque 2**, seleccionar nuevos parámetros y re-ejecutar desde allí.

---

## Guía de Uso en Entorno Local

El notebook detecta automáticamente el entorno local y se adapta:

- No monta Google Drive; resuelve las rutas desde la ubicación del notebook (`Google_Colab/ → 02_ING_MODELOS/`).
- Detecta GPU CUDA o Apple MPS automáticamente.
- Compatible con VS Code + Jupyter Extension o JupyterLab.

```bash
# Desde 02_ING_MODELOS/
cd Google_Colab
jupyter notebook 07_TrainColab.ipynb
```

---

## Artefactos Generados por Experimento

Cada ejecución completa del pipeline genera la siguiente estructura dentro de `02_ING_MODELOS/models/{experiment_name}/`:

```
{experiment_name}/
├── experiment.json              ← Schema unificado del experimento completo
├── training_curves.png          ← Visualización multi-panel de curvas de entrenamiento
├── val_evaluation.json          ← Métricas de validación (mAP, P, R, F1 por clase)
├── val_confusion_matrix.png     ← Matriz de confusión (val)
├── val_per_class.png            ← Métricas por clase (val)
├── test_evaluation.json         ← Métricas de test
├── test_confusion_matrix.png    ← Matriz de confusión (test)
├── test_per_class.png           ← Métricas por clase (test)
├── export_result.json           ← Resultado de exportación TFLite
├── comparison_result.json       ← Resultado comparación Framework vs TFLite
├── inference_samples.png        ← Visualización de predicciones
├── tflite/                      ← Modelo TFLite INT8 cuantizado
├── train/                       ← (YOLO) weights/best.pt, results.csv
├── checkpoints/                 ← (MobileNet) .keras checkpoints
└── logs/                        ← CSVs de historial, logs de TensorBoard
```

Adicionalmente, en `02_ING_MODELOS/reports/` se generan:

- `experiments_comparison.png` — Gráfica comparativa de todos los experimentos.
- `experiments_comparison.csv` — Tabla consolidada de métricas.

---

## Requisitos y Dependencias

### Google Colab

Las dependencias se instalan automáticamente en el Bloque 1 del notebook:

```bash
pip install -q ultralytics ipywidgets pyyaml
```

TensorFlow, NumPy, Matplotlib y OpenCV están preinstalados en Colab.

### Entorno Local

```
ultralytics
tensorflow
ipywidgets
pyyaml
numpy
matplotlib
opencv-python
pandas
seaborn
scikit-learn
```

Consultar `02_ING_MODELOS/requirements.txt` para la lista completa con versiones.

---

## Consideraciones de Despliegue ESP32-S3

| Parámetro | Valor |
|---|---|
| **Microcontrolador** | ESP32-S3 (N16R8) |
| **PSRAM disponible** | 8 MB |
| **Tamaño máximo del modelo TFLite** | < 8 MB |
| **Resolución de entrada recomendada** | 128×128 — 224×224 px |
| **Cuantización** | INT8 (post-training quantization) |

El pipeline verifica automáticamente la compatibilidad del modelo exportado con las restricciones de memoria del ESP32-S3 y reporta el tamaño final, la estimación de latencia y si el modelo cabe en PSRAM.

---

## Documentación Adicional

- **[Migrating_Notebook.md](Migrating_Notebook.md)** — Documentación detallada del proceso de migración desde los notebooks locales (`05_TrainMobileNet.ipynb`, `06_TrainYolo.ipynb`) hacia el notebook unificado `07_TrainColab.ipynb`.

---

*Última actualización: Febrero 2026*
