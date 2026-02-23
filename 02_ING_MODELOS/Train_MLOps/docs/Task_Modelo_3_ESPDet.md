# Task ESPDet-Pico — Anchor-Free Micro-Detector

> **Entry-point:** `trainer.task_espdet`  
> **YAML:** `vertex_ai/configs/espdet_pico_v2.yaml`  
> **Contenedor:** `pytorch-gpu.2-4.py310:latest`  
> **Paquete:** `tfm_trainer-2.6.0.tar.gz`

---

## Arquitectura del Modelo

Reimplementación fiel de la arquitectura oficial ESPDet-Pico del repositorio
`espressif/esp-detection` (AGPL-3.0). Scale `n`: [depth=0.50, width=0.25, max_ch=512].

```
Input (B, 3, H, W)
    │
    ▼
Backbone (layers 0-10) — Official Espressif Topology
    │ Conv(3→16,s=2) → DSConv(16→32,s=2) → ESPBlockLite(32→64)
    │ → DSConv(64→64,s=2) → DSC3k2(64→64) → SCDown(64→64)
    │ → DSC3k2(64→64,c3k) → SCDown(64→128) → DSC3k2(128→128,c3k)
    │ → SPPF(128→128) → DSConv(128→128,k=7)
    │
    ├── P3/8  (64 ch)  ← layer4 output
    ├── P4/16 (64 ch)  ← layer6 output
    └── P5/32 (128 ch) ← layer10 output
    │
    ▼
Neck — Top-down + Bottom-up FPN
    │ Top-down: Upsample → Concat → ESPBlock (×2)
    │ Bottom-up: DSConv(s=2) → Concat → ESPBlock (×2)
    │
    ├── P3/8  (32 ch)  ← layer16 output
    ├── P4/16 (128 ch) ← layer19 output
    └── P5/32 (128 ch) ← layer22 output
    │
    ▼
ESPDetectHead (anchor-free, reg_max=1)
    ├── cv2 (box):  DSConv → DSConv → Conv2d(4×reg_max) per level
    └── cv3 (cls):  [DWConv+Conv] → [DWConv+Conv] → Conv2d(nc) per level
```

### Parámetros del Modelo

| Parámetro | Valor |
|---|---|
| Backbone | Official Espressif (DSConv, ESPBlockLite, DSC3k2, SCDown, SPPF) |
| Scale | `n` = [depth=0.50, width=0.25, max_ch=512] |
| Strides | [8, 16, 32] (P3, P4, P5) |
| Detection head | ESPDetectHead (cv2=box, cv3=cls) |
| reg_max | 1 |
| Params | **~0.36M** |
| Pretrained | `espdet_pico_224_224_cat.pt` (cat detection, nc=1) |
| Transfer | ~99.97% params transferidos (strict=False) |
| Tamaño ONNX est. | ~1.5 MB |

> **Nota:** reg_max=1 significa regresión directa de 4 distancias (l, t, r, b),
> sin la distribución integral usada en PP-PicoDet (reg_max=7).

---

## Estrategia de Entrenamiento

### Fase 1 — Backbone Congelado (layers 0-10)

- **Epochs:** 50
- **Optimizer:** AdamW (lr=1e-3, wd=1e-4)
- **Scheduler:** Cosine
- **Objetivo:** Adaptar neck + head a las 5 clases IODC sobre features pretrained

### Fase 2 — Todo Descongelado

- **Epochs:** 100
- **Optimizer:** AdamW (lr=1e-4, wd=1e-5)
- **Scheduler:** Cosine
- **Objetivo:** End-to-end fine-tuning conservador

> **Total:** 150 epochs (50 + 100), patience=25

### Redimensionado

| Epoch | Resolución |
|---|---|
| 0-149 | 224×224 (fijo, match pretrained) |

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
  --config-name espdet_pico_v2.yaml \
  --run-name espdet_pico_v2-run1
```

---

## Métricas Esperadas

| Métrica | Target |
|---|---|
| Val mAP@50 | > 0.25 (transfer learning desde cat-detection) |
| Test mAP@50 | > 0.20 |
| ONNX size | < 2 MB |
| Training time (T4) | ~3h (150 epochs) |
| Latencia ESP32-S3 | < 500ms (objetivo) |

---

## Justificación

ESPDet-Pico es el modelo más pequeño de los 3, diseñado para:

1. **Evaluación de límites:** ¿Cuánto mAP se puede obtener con ~0.36M params?
2. **Despliegue directo:** Cabe holgadamente en la Flash/PSRAM del ESP32-S3
3. **Comparación de arquitecturas:** Backbone oficial Espressif (DSConv) vs MobileNet (FCOS) vs YOLO
4. **Transfer learning:** Cat-detection weights como punto de partida (~99.97% params transferidos)
5. **Compatibilidad nativa:** ONNX interleaved → esp-ppq → .espdl sin conversiones extra
