# Task FCOS — MobileNetV3-Small + FPN + FCOS Head

> **Entry-point:** `trainer.task_fcos`  
> **YAML:** `vertex_ai/configs/fcos_v3s_v1.yaml`  
> **Contenedor:** `pytorch-gpu.2-4.py310:latest`

---

## Arquitectura del Modelo

```
Input (B, 3, H, W)
    │
    ▼
MobileNetV3-Small (backbone, pretrained ImageNet)
    │
    ├── stride 8  → C3 features
    ├── stride 16 → C4 features
    └── stride 32 → C5 features
    │
    ▼
SimpleFPN (64 channels)
    │
    ├── P3 (H/8,  W/8,  64)
    ├── P4 (H/16, W/16, 64)
    └── P5 (H/32, W/32, 64)
    │
    ▼
FCOSHead (per-level, shared weights)
    ├── cls_tower (2 × Conv3×3 + GN + ReLU) → cls_logits (C)
    ├── reg_tower (2 × Conv3×3 + GN + ReLU) → reg_pred (4: l,t,r,b)
    └── centerness → centerness_logit (1)
```

### Parámetros del Modelo

| Parámetro | Valor |
|---|---|
| Backbone | MobileNetV3-Small (pretrained) |
| FPN channels | 64 |
| Strides | [8, 16, 32] |
| Head convs | 2 |
| Params estimados | ~2.5M |
| Tamaño ONNX est. | ~10 MB |

---

## Estrategia de Entrenamiento

### Fase 1 — Backbone Congelado

- **Epochs:** 30
- **Optimizer:** AdamW (lr=1e-3, wd=1e-4)
- **Scheduler:** Cosine annealing
- **Objetivo:** Entrenar head + FPN sin destruir features de ImageNet

### Fase 2 — Todo Descongelado

- **Epochs:** 60
- **Optimizer:** AdamW (lr=1e-4, wd=1e-5)
- **Scheduler:** Cosine annealing
- **Objetivo:** Fine-tune end-to-end con tasa reducida

### Redimensionado Progresivo

| Epoch | Resolución |
|---|---|
| 0 | 640×640 |
| 10 | 416×416 |
| 20 | 320×320 |
| 30 | 224×224 |

> El modelo se adapta gradualmente a la resolución objetivo del ESP32-S3 (224×224).

---

## Loss Function

```
L_total = λ_cls · L_cls + λ_reg · L_reg + λ_ctr · L_centerness

L_cls       = BCEWithLogitsLoss (focal loss optional)
L_reg       = SmoothL1Loss (l, t, r, b distances)
L_centerness = BCEWithLogitsLoss
```

Pesos por defecto: `cls=1.0, reg=1.5, centerness=1.0`

---

## Augmentation (Albumentations)

| Transformación | Probabilidad |
|---|---|
| HorizontalFlip | 0.5 |
| RandomBrightnessContrast | 0.3 |
| GaussNoise | 0.2 |
| ShiftScaleRotate (±15°) | 0.3 |
| HueSaturationValue | (configurable) |
| Resize + Normalize | siempre |

---

## Pipeline (8 Bloques)

```
Bloque 1 — Setup         : Descarga config YAML + dataset YOLO desde GCS
Bloque 2 — Verify        : Verificación dataset + distribución de clases
Bloque 3 — Build Model   : Construye FCOSModel + freeze backbone
Bloque 4 — Train         : 2 fases (30 + 60 epochs) con progressive resize
Bloque 5 — Curves        : Extrae CSV → gráficas de loss/lr/img_size
Bloque 6 — Val Eval      : mAP@50 en validación + confusion matrix
Bloque 7 — Test Eval     : mAP@50 en test + predicciones visuales
Bloque 8 — Save+Upload   : ONNX export + experiment.json → GCS
```

---

## Lanzamiento

```bash
# Via launch_job.py
python vertex_ai/launch_job.py \
  --family FCOS \
  --config-name fcos_v3s_v1.yaml \
  --run-name fcos_v3s_v1-run1

# Via build_and_launch.sh (incluye sdist)
bash vertex_ai/build_and_launch.sh FCOS fcos_v3s_v1.yaml fcos-run1
```

---

## Métricas Esperadas

| Métrica | Target |
|---|---|
| Val mAP@50 | > 0.30 (baseline) |
| Test mAP@50 | > 0.25 |
| ONNX size | < 15 MB |
| Training time (T4) | ~1.5h |
