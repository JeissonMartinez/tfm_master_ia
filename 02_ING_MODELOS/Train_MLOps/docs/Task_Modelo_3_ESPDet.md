# Task ESPDet-Pico — Anchor-Free Micro-Detector

> **Entry-point:** `trainer.task_espdet`  
> **YAML:** `vertex_ai/configs/espdet_pico_v1.yaml`  
> **Contenedor:** `pytorch-gpu.2-4.py310:latest`

---

## Arquitectura del Modelo

Diseñado desde cero para ESP32-S3 (~0.36M parámetros).

```
Input (B, 3, H, W)
    │
    ▼
ESPDetPicoBackbone (width_mult=0.5)
    │ DepthwiseSeparableConv stages
    │
    ├── Stage 1 (stride 4)  → C2 features [16 ch]
    ├── Stage 2 (stride 8)  → C3 features [32 ch]
    └── Stage 3 (stride 16) → C4 features [64 ch]
    │
    ▼
SimpleFPN (32 channels)
    │
    ├── P2 (H/4,  W/4,  32)
    ├── P3 (H/8,  W/8,  32)
    └── P4 (H/16, W/16, 32)
    │
    ▼
ESPDetPicoHead (anchor-free, reg_max=1)
    ├── cls_tower (2 × DWConv + BN + ReLU) → cls_logits (C)
    └── reg_tower (2 × DWConv + BN + ReLU) → reg_pred (4)
```

### Parámetros del Modelo

| Parámetro | Valor |
|---|---|
| Backbone | ESPDetPicoBackbone (custom) |
| Width multiplier | 0.5 |
| FPN channels | 32 |
| Strides | [4, 8, 16] |
| reg_max | 1 |
| Head convs | 2 (depthwise separable) |
| Params | ~0.36M |
| Tamaño ONNX est. | ~1.5 MB |

> **Nota:** reg_max=1 significa regresión directa de 4 distancias (l, t, r, b),
> sin la distribución integral usada en PP-PicoDet (reg_max=7).

---

## Estrategia de Entrenamiento

### Fase 1 — Backbone Congelado

- **Epochs:** 40
- **Optimizer:** AdamW (lr=1e-3, wd=1e-4)
- **Scheduler:** Cosine
- **Objetivo:** Head convergence rápida

### Fase 2 — Todo Descongelado

- **Epochs:** 80
- **Optimizer:** AdamW (lr=5e-5, wd=1e-5)
- **Scheduler:** Cosine
- **Objetivo:** End-to-end con LR conservador (modelo pequeño)

> **Total:** 120 epochs (40 + 80), patience=20

### Redimensionado Progresivo

| Epoch | Resolución |
|---|---|
| 0 | 640×640 |
| 15 | 416×416 |
| 30 | 320×320 |
| 40 | 224×224 |

---

## Loss Function

```
L_total = λ_cls · L_cls + λ_reg · L_reg

L_cls = BCEWithLogitsLoss
L_reg = SmoothL1Loss (distancias l, t, r, b)
```

Pesos por defecto: `cls=1.0, reg=2.0`

> Pesos de regresión mayores que en FCOS para compensar la menor
> capacidad del backbone.

---

## Augmentation (Albumentations — Agresiva)

| Transformación | Probabilidad |
|---|---|
| HorizontalFlip | 0.5 |
| RandomBrightnessContrast | 0.4 |
| GaussNoise | 0.3 |
| ShiftScaleRotate (±20°) | 0.3 |
| HueSaturationValue | ✅ |
| RandomGamma | ✅ |
| CLAHE | ✅ |
| Resize + Normalize | siempre |

> Augmentation más agresiva que FCOS por menor capacidad del modelo.
> CLAHE y RandomGamma mejoran robustez ante cambios de iluminación
> (escenarios interiores del ESP32-S3).

---

## Pipeline (8 Bloques)

```
Bloque 1 — Setup         : Descarga config + dataset desde GCS
Bloque 2 — Verify        : Verificación dataset + distribución
Bloque 3 — Build Model   : build_espdet_pico() + freeze backbone
Bloque 4 — Train         : 2 fases (40 + 80 epochs) con progressive resize
Bloque 5 — Curves        : CSV → gráficas de loss/lr/img_size
Bloque 6 — Val Eval      : mAP@50 en validación
Bloque 7 — Test Eval     : mAP@50 en test
Bloque 8 — Save+Upload   : ONNX export + GCS upload
```

---

## Lanzamiento

```bash
python vertex_ai/launch_job.py \
  --family ESPDet \
  --config-name espdet_pico_v1.yaml \
  --run-name espdet_pico_v1-run1
```

---

## Métricas Esperadas

| Métrica | Target |
|---|---|
| Val mAP@50 | > 0.15 (model muy pequeño) |
| Test mAP@50 | > 0.12 |
| ONNX size | < 2 MB |
| Training time (T4) | ~2h (más epochs) |
| Latencia ESP32-S3 | < 500ms (objetivo) |

---

## Justificación

ESPDet-Pico es el modelo más pequeño de los 3, diseñado para:

1. **Evaluación de límites:** ¿Cuánto mAP se puede obtener con ~0.36M params?
2. **Despliegue directo:** Cabe holgadamente en la Flash/PSRAM del ESP32-S3
3. **Comparación de arquitecturas:** Backbone custom (DSConv) vs MobileNet (FCOS) vs YOLO
4. **Latencia mínima:** Strides más pequeños [4, 8, 16] para objetos cercanos
