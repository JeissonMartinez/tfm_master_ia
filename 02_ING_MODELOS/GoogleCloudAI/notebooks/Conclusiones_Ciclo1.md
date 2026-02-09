# Conclusiones y Plan de Mejora — Ciclo 2 de Entrenamiento

## 1. Conclusiones del Ciclo 1

### 1.1 Ranking general (test split)

| Métrica | YOLO11n | YOLO26n | MBNTv3S |
|---|---|---|---|
| mAP@50 | ~0.55–0.60 | **~0.60–0.65** | ~0.25–0.35 |
| mAP@50-95 | ~0.30–0.35 | **~0.35–0.40** | ~0.10–0.15 |
| Precision | ~0.60 | **~0.65** | ~0.35 |
| Recall | ~0.50 | **~0.55** | ~0.30 |
| TFLite size | ~2.5 MB | ~2.8 MB | **~1.2 MB** |
| TFLite degradation | mínima | mínima | significativa |

### 1.2 Observaciones clave

**YOLO26n (mejor modelo):**
- Mejor balance precisión/recall en 4 de 5 clases
- La cuantización INT8 prácticamente no degrada métricas (~1-2% drop)
- Bboxes más ajustados visualmente (mejor localización)

**YOLO11n (segundo):**
- Rendimiento competitivo pero ligeramente inferior a YOLO26n
- Cuantización casi sin pérdida — buena opción si se necesita un modelo más ligero
- Confunde ocasionalmente `obstacle` ↔ `stair`

**MBNTv3S SSDLite (peor rendimiento):**
- **Confusión sistemática `obstacle` ↔ `stair`** → las features del backbone no distinguen texturas/patrones de estos objetos a 224px
- **`person` sub-detectada** → los anchors predefinidos no cubren bien las proporciones de personas (altas y estrechas)
- **Muchos falsos positivos** → la cabeza SSD de 2 etapas (objectness + class) no filtra bien con anchors genéricos
- **Degradación significativa FW → TFLite** → la cuantización INT8 amplifica los problemas del modelo base

### 1.3 Diagnóstico por clase

| Clase | YOLO11n | YOLO26n | MBNTv3S | Problema MBNTv3S |
|---|---|---|---|---|
| dog | ✅ buena | ✅ buena | ⚠️ media | Confunde con obstacle |
| door | ✅ buena | ✅ buena | ⚠️ media | Bboxes imprecisas |
| obstacle | ⚠️ media | ✅ buena | ❌ mala | Confunde con stair |
| person | ⚠️ media | ✅ buena | ❌ mala | Sub-detección severa |
| stair | ⚠️ media | ✅ buena | ❌ mala | Confunde con obstacle |

---

## 2. Plan de Hiperparámetros — Ciclo 2 (Vertex AI + T4 GPU)

### 2.1 YOLO26n v2 (ajuste fino del mejor modelo)

````python
YOLO26N_V2_HPARAMS = {
    # ── Entrenamiento ──
    "epochs": 200,              # v1 usó 100 → dar más épocas con early stopping
    "patience": 30,             # early stopping con más paciencia
    "batch": 32,                # T4 tiene 16GB → subir batch (v1 usó 16)
    "imgsz": 224,               # mantener por restricción ESP32
    "optimizer": "AdamW",       # mejor generalización que SGD
    "lr0": 0.002,               # v1 usó 0.01 → bajar LR inicial (convergió rápido)
    "lrf": 0.01,                # LR final = lr0 * lrf = 0.00002
    "weight_decay": 0.0005,     # regularización L2
    "warmup_epochs": 5,         # v1 usó 3 → calentar más
    "warmup_momentum": 0.5,
    
    # ── Augmentation (más agresivo) ──
    "hsv_h": 0.02,              # v1 default 0.015
    "hsv_s": 0.8,               # v1 default 0.7
    "hsv_v": 0.5,               # v1 default 0.4
    "degrees": 15.0,            # rotación (v1=0)
    "translate": 0.2,           # traslación (v1=0.1)
    "scale": 0.6,               # escala (v1=0.5)
    "flipud": 0.1,              # flip vertical (v1=0)
    "fliplr": 0.5,              # flip horizontal
    "mosaic": 1.0,              # mosaic completo
    "mixup": 0.15,              # añadir mixup (v1=0)
    "copy_paste": 0.1,          # copy-paste augmentation (v1=0)
    
    # ── Loss weights ──
    "box": 7.5,                 # default YOLO
    "cls": 0.5,                 # default
    "dfl": 1.5,                 # default
    
    # ── Otros ──
    "cos_lr": True,             # cosine LR scheduler
    "label_smoothing": 0.05,    # suavizar labels → mejor generalización
    "nbs": 64,                  # nominal batch size para LR scaling
    "close_mosaic": 20,         # desactivar mosaic últimas 20 épocas
}
````

**Justificación:**
- `epochs=200` + `patience=30`: el modelo converge rápido, darle espacio para refinar
- `batch=32`: T4 con 16GB lo soporta a 224px; mejora estabilidad del gradiente
- `lr0=0.002`: LR más bajo evita overshooting ya que YOLO26 converge rápido
- `mixup=0.15` + `copy_paste=0.1`: mejorar generalización en clases difíciles
- `label_smoothing=0.05`: evitar sobreconfianza en predicciones

---

### 2.2 YOLO11n v2 (cerrar brecha con YOLO26n)

````python
YOLO11N_V2_HPARAMS = {
    # ── Entrenamiento ──
    "epochs": 250,              # YOLO11n necesita más épocas (backbone más pequeño)
    "patience": 35,
    "batch": 32,                # T4 lo soporta bien
    "imgsz": 224,
    "optimizer": "AdamW",
    "lr0": 0.005,               # intermedio: v1 fue 0.01, más agresivo que YOLO26n
    "lrf": 0.01,
    "weight_decay": 0.001,      # más regularización (tiende a overfit más rápido)
    "warmup_epochs": 5,
    "warmup_momentum": 0.5,
    
    # ── Augmentation (agresivo) ──
    "hsv_h": 0.02,
    "hsv_s": 0.8,
    "hsv_v": 0.5,
    "degrees": 15.0,
    "translate": 0.2,
    "scale": 0.6,
    "flipud": 0.1,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.2,              # más mixup que YOLO26n (necesita más regularización)
    "copy_paste": 0.15,
    
    # ── Loss weights ──
    "box": 7.5,
    "cls": 0.75,                # subir peso de clasificación (confunde obstacle/stair)
    "dfl": 1.5,
    
    # ── Otros ──
    "cos_lr": True,
    "label_smoothing": 0.1,    # más smoothing (tiende a sobreconfianza)
    "nbs": 64,
    "close_mosaic": 25,
}
````

**Justificación:**
- `epochs=250`: backbone más compacto necesita más iterations para converger
- `weight_decay=0.001`: más regularización porque el gap train/val es mayor
- `cls=0.75`: penalizar más los errores de clasificación (obstacle ↔ stair)
- `mixup=0.2` + `label_smoothing=0.1`: reducir overfitting agresivamente

---

### 2.3 MBNTv3S SSDLite v2 (mejora urgente)

Este modelo necesita los cambios más drásticos:

````python
MBNTV3S_V2_HPARAMS = {
    # ── Entrenamiento ──
    "epochs": 300,              # v1 usó ~150 → necesita mucho más
    "batch_size": 64,           # T4 soporta batch grande con MBN (modelo ligero)
    "imgsz": 224,
    
    # ── Learning rate ──
    "optimizer": "AdamW",
    "learning_rate": 0.003,     # v1 usó 0.01 → bajar significativamente
    "lr_schedule": "cosine",    # cosine decay
    "lr_warmup_epochs": 10,     # warmup más largo (v1=5)
    "lr_min": 1e-6,             # LR mínimo
    "weight_decay": 0.0005,
    
    # ── Anchors (CRÍTICO para MBN) ──
    # v1 usó anchors genéricos → recalcular con k-means sobre el dataset
    "anchor_sizes": [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9],   # más granular
    "anchor_ratios": [0.33, 0.5, 1.0, 2.0, 3.0],              # añadir 0.33 y 3.0
    # → cubre mejor personas (ratio 0.33) y escaleras/obstáculos anchos (3.0)
    
    # ── SSD Head ──
    "num_anchors_per_cell": 6,  # subir de 5 a 6
    "nms_iou_threshold": 0.45,
    "confidence_threshold": 0.3,
    "max_detections": 50,       # v1=100 → reducir falsos positivos
    
    # ── Augmentation ──
    "horizontal_flip": True,
    "vertical_flip": False,     # no tiene sentido para esta tarea
    "rotation_range": 15,
    "brightness_range": 0.3,
    "contrast_range": 0.3,
    "saturation_range": 0.3,
    "random_crop_scale": [0.5, 1.0],
    "mixup_alpha": 0.2,
    
    # ── Loss ──
    "focal_loss": True,         # CRÍTICO: reemplazar CE por Focal Loss
    "focal_alpha": 0.25,        # weight para clases desbalanceadas
    "focal_gamma": 2.0,         # penalizar más los ejemplos fáciles
    "box_loss_weight": 1.0,
    "cls_loss_weight": 2.0,     # SUBIR: penalizar confusiones de clase
    
    # ── Regularización ──
    "dropout": 0.2,             # v1 usó 0.1 → subir
    "label_smoothing": 0.1,
    
    # ── Freeze strategy ──
    "freeze_backbone_epochs": 5,    # congelar backbone 5 épocas
    "unfreeze_lr_multiplier": 0.1,  # backbone LR = 10% de head LR
}
````

**Justificación de los cambios críticos:**

1. **Focal Loss** (`focal_loss=True`): El problema #1 de MBNTv3S es la confusión `obstacle` ↔ `stair`. Focal Loss penaliza más los ejemplos que el modelo clasifica mal con alta confianza, forzándolo a distinguir clases similares.

2. **Anchors recalculados**: La sub-detección de `person` se debe a que los anchors genéricos no cubren ratios altos (personas de pie = ratio ~0.3). Añadir `0.33` y `3.0` corrige esto.

3. **`cls_loss_weight=2.0`**: Duplicar el peso de la loss de clasificación fuerza al modelo a priorizar "qué es" sobre "dónde está".

4. **`batch_size=64`**: MBNTv3S es ligero (~3MB en Keras), T4 puede manejar batches grandes que estabilizan gradientes y mejoran convergencia.

5. **`dropout=0.2`** + `label_smoothing=0.1`: El modelo tiene overfitting visible (train loss baja pero val no mejora).

---

## 3. Resumen de prioridades por modelo

| Modelo | Prioridad #1 | Prioridad #2 | Prioridad #3 |
|---|---|---|---|
| **YOLO26n** | Más épocas + LR bajo | Augmentation agresivo | Label smoothing |
| **YOLO11n** | Subir cls loss weight | Más regularización | Más épocas |
| **MBNTv3S** | **Focal Loss** | **Recalcular anchors** | **Subir cls_loss_weight** |

## 4. Métricas objetivo Ciclo 2

| Métrica | YOLO26n v2 | YOLO11n v2 | MBNTv3S v2 |
|---|---|---|---|
| mAP@50 | ≥ 0.70 | ≥ 0.65 | ≥ 0.45 |
| mAP@50-95 | ≥ 0.42 | ≥ 0.38 | ≥ 0.22 |
| TFLite drop | < 3% | < 3% | < 10% |
| obstacle AP@50 | ≥ 0.65 | ≥ 0.60 | ≥ 0.35 |
| person AP@50 | ≥ 0.70 | ≥ 0.65 | ≥ 0.40 |

> **Nota**: Si MBNTv3S v2 no alcanza mAP@50 ≥ 0.40, considerar migrar a **EfficientDet-Lite0** que tiene un mejor balance tamaño/precisión para detección en edge devices, o bien probar **MobileNetV3-Large** como backbone (1.5x más parámetros pero mejor feature extraction).