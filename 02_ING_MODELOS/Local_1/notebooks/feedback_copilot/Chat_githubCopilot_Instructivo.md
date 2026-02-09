# 📋 Instrucciones para Replicar Metodología de Entrenamiento de Modelos de Detección

## Contexto del Proyecto

Este notebook implementa un pipeline completo de entrenamiento de modelos de detección de objetos optimizados para despliegue en **ESP32-S3** (microcontrolador con ~2MB de memoria). El objetivo es entrenar modelos ligeros que detecten 4 clases: `door`, `footpath`, `obstacle`, `person`.

---

## Arquitectura del Notebook

### Estructura de Secciones (sin Introducción/Conclusión)

1. **Setup y Configuración** - Imports, paths, configuración GPU
2. **Configuración del Experimento** - Variables editables para cada experimento
3. **Cargar y Analizar Dataset** - Carga COCO JSON, class weights, distribución
4. **Generar Anchors y Data Generators** - Anchors SSD, generadores con augmentación
5. **Construir Modelo** - Arquitectura backbone + detection head
6. **Configurar Losses y Métricas** - Focal Loss, class weights, loss weights
7. **Entrenamiento Fase 1: Warm-up** - Backbone congelado, solo head
8. **Entrenamiento Fase 2: Fine-Tuning** - Backbone parcialmente descongelado
9. **Evaluación del Modelo** - Métricas, visualizaciones con NMS
10. **Exportar a TFLite** - Cuantización INT8 para ESP32-S3
11. **Evaluación Avanzada con mAP@50** - mAP, precision, recall, confusion matrix
12. **Guardar Experimento** - Serializar config y resultados
13. **Comparación de Experimentos** - Tabla comparativa entre experimentos

---

## Modularización en `src_{modelo}/`

El código está organizado en módulos separados dentro de `src_mobilenet/`:

| Módulo | Responsabilidad |
|--------|-----------------|
| `utils_{modelo}_model.py` | Construcción de arquitecturas (backbone + head) |
| `utils_{modelo}_data.py` | Carga COCO, generadores, anchors, augmentación, Copy-Paste |
| `utils_{modelo}_losses.py` | Focal Loss, Smooth L1, GIoU, combined loss |
| `utils_{modelo}_train.py` | Freeze/unfreeze, callbacks, train_two_phase, history |
| `utils_{modelo}_export.py` | Exportación TFLite (float32, float16, int8), verificación |
| `utils_{modelo}_eval.py` | mAP@50, precision/recall, confusion matrix, Keras vs TFLite |
| `utils_{modelo}_infer.py` | Postprocesamiento NMS, visualización de predicciones |
| `utils_{modelo}_experiment.py` | Dataclasses para config/resultados, guardar/cargar experimentos |

---

## Patrón de Configuración del Experimento

Todas las variables editables están centralizadas en una celda de configuración:

## 📋 Instrucciones para Replicar Metodología de Entrenamiento de Modelos de Detección

### Contexto del Proyecto

Este documento describe cómo replicar un pipeline completo de entrenamiento de modelos de detección de objetos optimizados para despliegue en **ESP32-S3** (microcontrolador con ~2MB de memoria para modelo). El objetivo es entrenar modelos ligeros que detecten clases específicas de objetos.

---

## 🏗️ Arquitectura del Notebook

### Estructura de Secciones

| # | Sección | Descripción |
|---|---------|-------------|
| 1 | **Setup y Configuración** | Imports, paths, configuración GPU/memoria |
| 2 | **Configuración del Experimento** | Variables editables centralizadas (nombre, clases, augmentación, backbone) |
| 3 | **Cargar y Analizar Dataset** | Carga COCO JSON, class weights, visualización distribución |
| 4 | **Generar Anchors y Data Generators** | Anchors SSD, generadores con augmentación online |
| 5 | **Construir Modelo** | Arquitectura backbone + detection head |
| 6 | **Configurar Losses y Métricas** | Focal Loss, class weights, loss weights combinados |
| 7 | **Entrenamiento Fase 1: Warm-up** | Backbone congelado, solo head trainable |
| 8 | **Entrenamiento Fase 2: Fine-Tuning** | Backbone parcialmente descongelado, LR muy bajo |
| 9 | **Evaluación del Modelo** | Métricas básicas, visualización predicciones con NMS |
| 10 | **Exportar a TFLite** | Cuantización INT8 para ESP32-S3 |
| 11 | **Evaluación Avanzada con mAP@50** | mAP, precision, recall, F1, confusion matrix |
| 12 | **Guardar Experimento** | Serializar config y resultados en JSON |
| 13 | **Comparación de Experimentos** | Tabla y gráficos comparativos entre versiones |

---

## 📁 Modularización en `src_{modelo}/`

El código está organizado en **8 módulos** separados dentro de un directorio `src_{nombre_modelo}/`:

| Módulo | Responsabilidad | Funciones Principales |
|--------|-----------------|----------------------|
| `utils_{modelo}_model.py` | Construcción de arquitecturas | `build_{backbone}_ssd_lite()`, `print_model_summary()` |
| `utils_{modelo}_data.py` | Carga datos, generadores, augmentación | `load_coco_annotations()`, `COCODataGenerator`, `generate_anchors()`, `ObstacleBank`, `apply_copy_paste_augmentation()` |
| `utils_{modelo}_losses.py` | Funciones de pérdida | `focal_loss()`, `binary_focal_loss()`, `smooth_l1_loss()`, `ssd_combined_loss()` |
| `utils_{modelo}_train.py` | Utilidades de entrenamiento | `freeze_backbone()`, `unfreeze_backbone_layers()`, `create_callbacks()`, `save_training_history()`, `plot_training_history()` |
| `utils_{modelo}_export.py` | Exportación TFLite | `export_tflite()`, `export_tflite_int8()`, `estimate_model_size()`, `verify_tflite_model()` |
| `utils_{modelo}_eval.py` | Evaluación con mAP | `evaluate_model_full()`, `compare_keras_vs_tflite()`, `plot_confusion_matrix()` |
| `utils_{modelo}_infer.py` | Inferencia y visualización | `postprocess_detections()`, `visualize_predictions_nms()`, `run_inference_keras()`, `run_inference_tflite()` |
| `utils_{modelo}_experiment.py` | Gestión de experimentos | `ExperimentConfig`, `ExperimentResults`, `Experiment`, `save_experiment()`, `load_all_experiments()`, `compare_experiments()` |

---

## ⚙️ Patrón de Configuración del Experimento

Todas las variables editables están **centralizadas en una única celda** al inicio del notebook:

```python
# ============================================
# 🔬 CONFIGURACIÓN DEL EXPERIMENTO
# ============================================
EXPERIMENT_NAME = "NombreModelo_version"
EXPERIMENT_DESCRIPTION = "Descripción del objetivo del experimento"

# Clases a entrenar
SELECTED_CLASSES = ["obstacle", "footpath"]  # o None para todas

# Nivel de augmentación online
AUGMENTATION_LEVEL = "heavy"  # none, light, medium, heavy

# Selección de backbone (específico del modelo)
BACKBONE_TYPE = "small"  # Opciones dependen del modelo

# ============================================
# 🧬 TÉCNICAS ESPECIALES (opcionales)
# ============================================
COPY_PASTE_ENABLED = True
COPY_PASTE_CONFIG = {
    "enabled": True,
    "target_class_idx": 1,      # Índice de clase minoritaria
    "num_pastes": 2,            # Intentos por imagen
    "paste_prob": 0.5,          # Probabilidad por intento
    "scale_range": (0.6, 1.4),  # Rango de escalado
    "blend_mode": "direct",
}

# ============================================
# 📊 HIPERPARÁMETROS DE ENTRENAMIENTO
# ============================================
BATCH_SIZE = 16
IMG_SIZE = 224

# Fase 1: Warm-up (backbone congelado)
PHASE1_EPOCHS = 30
PHASE1_LR = 1e-3

# Fase 2: Fine-tuning (backbone parcialmente descongelado)
PHASE2_EPOCHS = 50
PHASE2_LR = 5e-5
NUM_LAYERS_TO_UNFREEZE = 50

# Callbacks
PATIENCE_REDUCE_LR = 7
PATIENCE_EARLY_STOP = 12
REDUCE_LR_FACTOR = 0.5

# ============================================
# 📉 CONFIGURACIÓN DE LOSS
# ============================================
CLS_WEIGHT = 1.5
OBJ_WEIGHT = 2.0
BBOX_WEIGHT = 1.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Class weights (manual o calculado)
class_weights = np.array([0.7, 2.0])  # [clase_mayoritaria, clase_minoritaria]

# ============================================
# 🎯 UMBRALES DE INFERENCIA
# ============================================
SCORE_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.4
```

---

## 🎓 Estrategia de Entrenamiento en 2 Fases

| Fase | Backbone | Learning Rate | Épocas | Early Stopping | Objetivo |
|------|----------|---------------|--------|----------------|----------|
| **1. Warm-up** | 🔒 Congelado | 1e-3 (alto) | 15-30 | Desactivado | Entrenar solo el detection head desde cero |
| **2. Fine-tuning** | 🔓 Últimas N capas | 1e-5 (muy bajo) | 50-100 | Activado | Ajustar backbone + head sin destruir pesos pre-entrenados |

**Justificación:**
- Fase 1 permite que el head aprenda rápidamente sin afectar el backbone pre-entrenado
- Fase 2 con LR muy bajo permite ajuste fino sin "catastrofic forgetting"
- Early stopping en Fase 2 previene overfitting

---

## ⚖️ Manejo de Desbalance de Clases

Se utilizan **3 técnicas complementarias**:

1. **Focal Loss** ($\alpha=0.25$, $\gamma=2.0$)
   - Reduce peso de ejemplos fáciles
   - Enfoca entrenamiento en ejemplos difíciles

2. **Class Weights** (manual o por frecuencia inversa)
   - Aumenta penalización para clases minoritarias
   - Ejemplo: `[0.7, 2.0]` para [mayoritaria, minoritaria]

3. **Copy-Paste Augmentation** (para clases muy minoritarias)
   - Extrae crops de objetos de clase minoritaria
   - Los pega en otras imágenes durante entrenamiento
   - Aumenta variabilidad sin recolectar más datos

---

## 📦 Formato de Dataset

### Estructura COCO JSON

```json
{
  "images": [
    {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}
  ],
  "categories": [
    {"id": 1, "name": "footpath"},
    {"id": 2, "name": "obstacle"}
  ]
}
```

### Estructura Retornada por `load_coco_annotations()`

```python
[
    {
        "image_path": "/path/to/img001.jpg",
        "boxes": [[x, y, w, h], ...],  # Normalizado 0-1
        "classes": [class_idx, ...]     # Índices 0-based
    },
    ...
]
```

---

## 🔲 Sistema de Anchors (SSD)

```python
anchors = generate_anchors(
    feature_map_size=7,              # IMG_SIZE / stride (224/32 = 7)
    scales=[0.03, 0.08, 0.2, 0.45],  # Tamaños relativos
    aspect_ratios=[0.5, 1.0, 2.0],   # Relaciones de aspecto
)
# Total anchors: 7 × 7 × (4 scales × 3 ratios) = 588
```

---

## 🏛️ Arquitectura del Modelo (Patrón General)

```
Input (224×224×3)
    ↓
┌─────────────────────────────────────┐
│  Backbone Pre-entrenado (ImageNet)  │  ← Weights congelados en Fase 1
│  (MobileNet, EfficientNet, etc.)    │
└─────────────────────────────────────┘
    ↓
Feature Maps (7×7×channels)
    ↓
┌─────────────────────────────────────┐
│     Detection Head (SSD-Lite)       │
│  (Convoluciones separables)         │
├─────────────────────────────────────┤
│  ├── objectness: (7,7,num_anchors,1)│  ← ¿Hay objeto?
│  ├── class_out:  (7,7,num_anchors,C)│  ← ¿Qué clase?
│  └── bbox_out:   (7,7,num_anchors,4)│  ← ¿Dónde? (dx,dy,dw,dh)
└─────────────────────────────────────┘
```

---

## 📊 Generador de Datos (`COCODataGenerator`)

```python
class COCODataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        images_data,           # Lista de dicts con paths y anotaciones
        anchors,               # Array de anchors
        batch_size=16,
        img_size=224,
        num_classes=2,
        iou_threshold=0.35,    # Para matching anchor-GT
        augmentation_level="heavy",
        shuffle=True,
        copy_paste_config=None,  # Opcional: Copy-Paste augmentation
    ):
        ...
    
    def __getitem__(self, idx):
        # Retorna (X, Y) donde:
        # X: (batch, 224, 224, 3)
        # Y: {
        #     "objectness": (batch, 7, 7, num_anchors),
        #     "class_out": (batch, 7, 7, num_anchors, num_classes),
        #     "bbox_out": (batch, 7, 7, num_anchors, 4)
        # }
```

---

## 📉 Configuración de Losses

```python
loss_config = ssd_combined_loss(
    cls_weight=1.5,           # Peso para clasificación
    obj_weight=2.0,           # Peso para objectness
    bbox_weight=1.0,          # Peso para regresión de bbox
    focal_alpha=0.25,
    focal_gamma=2.0,
    class_weights=class_weights_tensor,
    use_giou=False,           # True para GIoU loss en bbox
)

# Retorna:
# {
#     "losses": {
#         "objectness": binary_focal_loss,
#         "class_out": focal_loss,
#         "bbox_out": smooth_l1_loss,
#     },
#     "weights": {
#         "objectness": 2.0,
#         "class_out": 1.5,
#         "bbox_out": 1.0,
#     }
# }
```

---

## 🔄 Callbacks de Entrenamiento

```python
callbacks = [
    ModelCheckpoint(
        filepath=f"{model_name}_best.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
    ),
    TensorBoard(log_dir=log_dir),
    LearningRateLogger(),  # Custom callback
]
```

---

## 📤 Exportación TFLite

```python
# Cuantización INT8 (objetivo: < 1MB para ESP32-S3)
def export_tflite_int8(model, output_path, representative_dataset):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
```

---

## 📈 Evaluación con mAP@50

```python
results = evaluate_model_full(
    model=model,                    # Keras model o TFLite interpreter
    test_data=test_data,
    anchors=anchors,
    num_classes=NUM_CLASSES,
    class_names=CLASS_NAMES,
    score_threshold=0.35,           # Umbral para filtrar detecciones
    nms_iou_threshold=0.4,          # IoU para NMS
    eval_iou_threshold=0.5,         # IoU para matching GT-prediction
    cm_score_threshold=0.5,         # Umbral para confusion matrix
    is_tflite=False,
    verbose=True,
)

# Retorna dataclass con:
# - map50: float
# - precision: float
# - recall: float
# - f1_score: float
# - ap_per_class: dict
# - confusion_matrix: np.ndarray
```

---

## 💾 Sistema de Experimentos

### Dataclasses para Serialización

```python
@dataclass
class ExperimentConfig:
    name: str
    description: str
    backbone: str
    num_classes: int
    selected_classes: List[str]
    img_size: int
    augmentation_level: str
    phase1_epochs: int
    phase1_lr: float
    phase2_epochs: int
    phase2_lr: float
    batch_size: int
    use_focal_loss: bool
    focal_alpha: float
    focal_gamma: float
    score_threshold: float
    feature_channels: int
    # ... otros hiperparámetros

@dataclass
class ExperimentResults:
    map50_keras: float
    map50_tflite: float
    precision_keras: float
    recall_keras: float
    f1_keras: float
    ap_per_class_keras: Dict[str, float]
    ap_per_class_tflite: Dict[str, float]
    tflite_model_size_kb: float
    keras_model_path: str
    tflite_model_path: str

@dataclass
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults
    timestamp: str
```

### Guardar y Cargar

```python
# Guardar
exp_path = save_experiment(experiment, base_dir="logs/experiments")

# Cargar todos
all_experiments = load_all_experiments("logs/experiments")

# Comparar
df = compare_experiments(all_experiments, sort_by="mAP@50_keras")
```

---

## 📂 Estructura de Directorios

```
TFM_UNIR/
├── 01_ING_DATOS/
│   └── Dataset/
│       ├── train/augmented2_images/
│       │   ├── *.jpg
│       │   └── train_final2.json
│       ├── valid/
│       │   ├── *.jpg
│       │   └── val_final.json
│       └── test/
│           ├── *.jpg
│           └── test_final.json
│
├── 02_ING_MODELOS/
│   ├── notebooks/
│   │   └── 05_TrainMobileNet.ipynb
│   │
│   ├── src_mobilenet/              # ← Módulos del modelo
│   │   ├── __init__.py
│   │   ├── utils_mobilenet_model.py
│   │   ├── utils_mobilenet_data.py
│   │   ├── utils_mobilenet_losses.py
│   │   ├── utils_mobilenet_train.py
│   │   ├── utils_mobilenet_export.py
│   │   ├── utils_mobilenet_eval.py
│   │   ├── utils_mobilenet_infer.py
│   │   └── utils_mobilenet_experiment.py
│   │
│   ├── models/
│   │   ├── checkpoints/            # Checkpoints durante entrenamiento
│   │   └── final_export/           # Modelos finales (.keras, .tflite)
│   │
│   ├── logs/
│   │   ├── experiments/            # JSONs de experimentos
│   │   └── *.csv                   # Historiales de entrenamiento
│   │
│   ├── reports/
│   │   ├── figures/{EXPERIMENT}/   # Figuras por experimento
│   │   └── experiments_comparison.csv
│   │
│   └── scripts/
│       └── export_tflite_v2.py     # Script standalone de exportación
```

---

## 🎯 Métricas Objetivo para ESP32-S3

| Métrica | Objetivo | Notas |
|---------|----------|-------|
| **Tamaño TFLite INT8** | < 1 MB | Ideal < 800 KB |
| **mAP@50** | ≥ 50% | Balance precision/recall |
| **Gap Keras vs TFLite** | < 5 pp | Cuantización sin degradación |
| **AP clase minoritaria** | ≥ 30% | Problema principal a resolver |
| **Gap entre clases** | < 40 pp | Evitar modelo sesgado |
| **Tiempo inferencia ESP32** | < 500 ms | Depende del hardware |

---

## 🔧 Instrucciones para Adaptar a Nuevo Modelo

1. **Crear directorio** `src_{nuevo_modelo}/` con los 8 módulos

2. **Adaptar `utils_{modelo}_model.py`**:
   - Implementar `build_{backbone}_ssd_lite()` con el nuevo backbone
   - Mantener la misma interfaz de salida (3 heads: objectness, class_out, bbox_out)
   - Usar pesos pre-entrenados de ImageNet si están disponibles

3. **Reutilizar módulos genéricos** (requieren mínima adaptación):
   - `utils_{modelo}_data.py` → Genérico para formato COCO + anchors
   - `utils_{modelo}_losses.py` → Focal Loss es estándar
   - `utils_{modelo}_eval.py` → Evaluación mAP es estándar
   - `utils_{modelo}_infer.py` → NMS es estándar
   - `utils_{modelo}_experiment.py` → Completamente reutilizable

4. **Adaptar `utils_{modelo}_train.py`**:
   - Ajustar `freeze_backbone()` según estructura del nuevo backbone
   - Ajustar `unfreeze_backbone_layers()` para el nuevo modelo

5. **Adaptar `utils_{modelo}_export.py`**:
   - Verificar compatibilidad de operaciones con TFLite
   - Ajustar representative dataset si es necesario

6. **Crear notebook siguiendo la estructura de 13 secciones**

7. **Mantener el patrón de configuración centralizada** en una celda

8. **Mantener el sistema de 2 fases de entrenamiento**

9. **Mantener el sistema de experimentos** para comparación

---

## 📝 Checklist para Nuevo Experimento

- [ ] Definir `EXPERIMENT_NAME` único
- [ ] Escribir `EXPERIMENT_DESCRIPTION` con objetivos claros
- [ ] Seleccionar `SELECTED_CLASSES` si aplica
- [ ] Configurar `AUGMENTATION_LEVEL`
- [ ] Seleccionar `BACKBONE_TYPE`
- [ ] Ajustar `class_weights` según distribución
- [ ] Configurar `COPY_PASTE_CONFIG` si hay clase minoritaria
- [ ] Ejecutar todas las celdas en orden
- [ ] Verificar tamaño del modelo INT8 (< 1MB)
- [ ] Comparar con experimentos anteriores
- [ ] Documentar conclusiones