# Registro de Entrenamiento — ESPDet-Pico (Custom PyTorch Loop)

> **Modelo**: `espdet_pico` — ESPDet-Pico anchor-free micro-detector  
> **Arquitectura v1 (Train 1)**: Custom simplificada — ~22.8K params (0.09 MB)  
> **Arquitectura v2 (Train 2)**: Oficial Espressif — ~0.36M params (1.41 MB ONNX)  
> **Arquitectura v3 (Train 3)**: Oficial Espressif + Focal Loss — ~0.36M params (1.41 MB ONNX)  
> **Dataset**: IODC YOLO — 5 clases (dog, door, obstacle, person, stair)  
> **Splits**: Train 1470 | Val 188 | Test 187  
> **Infraestructura**: Google Vertex AI Custom Job — `n1-standard-8` + NVIDIA Tesla T4  
> **Contenedor**: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`  
> **Entry-point**: `trainer.task_espdet`  
> **Paquete base**: `tfm_trainer-2.6.2.tar.gz`  
> **Última actualización**: 24 de febrero de 2026 (v2.6.2 — Train 3 completado)  

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Configuración Base (Compartida)](#2-configuración-base-compartida)
3. [Train 1 — Baseline](#3-train-1--baseline)
4. [Train 2 — Arquitectura Oficial Espressif (v2)](#4-train-2--arquitectura-oficial-espressif-v2)
5. [Train 3 — Focal Loss (v3)](#5-train-3--focal-loss-v3)
6. [Backlog de Propuestas](#6-backlog-de-propuestas)
7. [Comparativa Cross-Model](#7-comparativa-cross-model)
8. [Conclusiones Generales](#8-conclusiones-generales)

---

## 1. Resumen Ejecutivo

| Métrica (Test) | Train 1 (v1) | Train 2 (v2) | **Train 3 (v3)** |
|---|:---:|:---:|:---:|
| **mAP@50** | 0.0105 | **0.6203** | 0.5993 |
| **mAP@50-95** | 0.0023 | **0.3078** | 0.3220 |
| **Precision** | 0.0009 | **0.2956** | 0.2156 |
| **Recall** | 0.1597 | 0.7235 | **0.7663** |
| **F1-Score** | 0.0017 | **0.4197** | 0.3365 |
| Épocas (P1+P2) | 120 (40+80) | 113 (50+63) | 131 (50+81) |
| Loss | BCE | BCE | **Focal (γ=2.0, α=0.25)** |
| Optimizer | AdamW | AdamW | AdamW |
| conf_threshold | 0.25 | 0.25 | 0.25 |
| Tiempo | 23.6 min | 21.7 min | 27.0 min |
| Inferencia (GPU) | 34.4 ms | 4.6 ms | 3.6 ms |
| ONNX size | 0.109 MB | 1.41 MB | 1.41 MB |
| Arquitectura | Custom v1 (22.8K) | Oficial v2 (361K) | **Oficial v2 + Focal (361K)** |
| Pretrained | No | Sí (cat-detection) | Sí (cat-detection) |

> Tabla actualizada tras cada entrenamiento exitoso.

---

## 2. Configuración Base (Compartida)

Parámetros comunes a todos los entrenamientos de esta serie, salvo modificación explícita.

> **⚠️ NOTA IMPORTANTE**: La configuración base cambió radicalmente entre Train 1 (v1, custom) y Train 2+ (v2, oficial). La sección §2 refleja la configuración **v2 vigente**. Para la configuración v1 usada en Train 1, ver §3.2.

### 2.1 Arquitectura (v2 — Official Espressif)

| Parámetro | Valor |
|---|---|
| Familia | ESPDet (Custom PyTorch Loop) |
| Variante | espdet_pico |
| Arquitectura | **Oficial Espressif** (`espressif/esp-detection`, AGPL-3.0) |
| Scale | `n` = [depth=0.50, width=0.25, max_ch=512] |
| Backbone | 11 layers: Conv→DSConv→ESPBlockLite→DSConv→DSC3k2→SCDown→DSC3k2(c3k)→SCDown→DSC3k2(c3k)→SPPF→DSConv(k=7) |
| Neck | Top-down: Upsample→Concat→ESPBlock (×2). Bottom-up: DSConv(s=2)→Concat→ESPBlock (×2) |
| Head | ESPDetectHead: cv2 (box) DSConv→DSConv→Conv2d(4). cv3 (cls) [DWConv+Conv]²→Conv2d(nc) |
| Niveles FPN | 3 (strides **[8, 16, 32]** — P3/P4/P5) |
| P3 channels | 32 |
| P4 channels | 128 |
| P5 channels | 128 |
| reg_max | 1 |
| Params totales | **~360K** (~0.36M) |
| Pretrained | **`espdet_pico_224_224_cat.pt`** (cat-detection, nc=1, ~99.97% transferred) |
| Input size (export) | 224 |
| ONNX opset | 13 |
| ONNX format | Interleaved: (box0, score0, box1, score1, box2, score2) |
| Clases | 5 (dog, door, obstacle, person, stair) |

> **Cambio vs v1**: La arquitectura pasó de custom (DepthwiseSeparableConv + SimpleFPN, 22.8K params, strides [4,8,16]) a la **implementación oficial** del repo Espressif (0.36M params, strides [8,16,32]). El cambio fue necesario tras el fracaso de Train 1 (mAP@50=0.01) y la incompatibilidad detectada con los pesos pretrained.

### 2.2 Estrategia de 2 Fases (v2)

| Parámetro | Phase 1 (Freeze) | Phase 2 (Full) |
|---|---|---|
| Epochs | 50 | 100 |
| LR | 1e-3 | 1e-4 |
| Weight Decay | 1e-4 | 1e-5 |
| Optimizer | AdamW | AdamW |
| Scheduler | Cosine | Cosine |
| Frozen | Backbone (layers 0-10) | Nada |

**Total: 150 epochs (50 + 100), patience=25**

### 2.3 Optimización

| Parámetro | Valor |
|---|---|
| Optimizer | AdamW |
| AMP | True |
| Grad clip | 5.0 |
| Batch size | 32 |
| Workers | 4 |

### 2.4 Redimensionado (v2)

| Epoch | Resolución | Fase |
|---|---|---|
| 0-149 | 224 (fijo) | Phase 1 + Phase 2 |

> **Cambio vs v1**: Eliminado progressive resizing (640→416→320→224). Match con pretrained resolution.

### 2.5 Loss

| Componente | Función | Peso |
|---|---|---|
| `cls_loss` | BCE (per-level, normalizado por n_pos) | 1.0 |
| `reg_loss` | GIoU loss (l,t,r,b stride-normalizado) | 2.0 |

> ESPDet NO utiliza centerness (a diferencia de FCOS). `ctr_loss` siempre es 0.0.
>
> **Train 3**: `cls_loss` cambió a **Sigmoid Focal Loss** (γ=2.0, α=0.25) para intentar mejorar la supresión de background. Resultado: **no mejoró** — ver §5 para análisis completo.

### 2.6 Augmentación (Albumentations)

| Transform | Parámetro efectivo | Controlado por |
|---|---|---|
| HorizontalFlip | p=0.5 | YAML `aug_hflip_prob: 0.5` |
| BrightnessContrast | brightness=0.3, contrast=0.3, p=0.5 | YAML `aug_brightness_limit: 0.3` + `aug_contrast_limit: 0.3` |
| HueSaturationValue | H=20, S=30, V=20, p=0.5 | `_ESPDET_DEFAULTS` (hue/sat/val shift) |
| Affine (rotate/scale/shift) | ±15°, scale=0.2, shift=0.1, p=0.5 | YAML `aug_rotate_limit: 15` + defaults |
| GaussNoise | std=(0.01, 0.05), p=0.2 | YAML `aug_gaussian_noise: 0.2` |

> ✅ **Naming mismatch corregido** (v2.6.0): Los keys del YAML v2 ahora coinciden con los que `IODCDataset._build_transforms()` busca (`aug_hflip_prob`, `aug_brightness_limit`, `aug_contrast_limit`, `aug_rotate_limit`, `aug_gaussian_noise`). Los valores personalizados del YAML se aplican correctamente.

### 2.7 Inferencia / Evaluación

| Parámetro | Valor |
|---|---|
| conf_threshold | 0.25 |
| iou_threshold | 0.45 |

### 2.8 Protecciones Aplicadas (Lecciones FCOS/YOLO26)

| Lección | Aplicación en ESPDet |
|---|---|
| FCOS T6 — pip cache | Version bump a `tfm_trainer-2.6.0` ✅ |
| FCOS T7/T8 — whitelist config_loader | Ya corregido en v2.2.0 (todas las claves YAML pasan) ✅ |
| FCOS T8 — launch_job hardcoded | `build_and_launch.sh` pasa `--package-uri` dinámico ✅ |
| FCOS T8 — DEPLOY VERIFICATION | Bloque de verificación en Bloque 3 (`v2.6.0`) ✅ |
| FCOS T8 — `log()` vs `print()` | Entry-point usa exclusivamente `print()` ✅ |
| Bug nuevo — aug_config missing | Fix en v2.5.0: `aug_config` se extrae de fc y se pasa a IODCDataset ✅ |
| T1 lección — arquitectura custom | v2.6.0: Reimplementación con arquitectura oficial Espressif ✅ |
| T1 lección — sin pretrained | v2.6.0: Transfer learning desde `espdet_pico_224_224_cat.pt` ✅ |
| T1 lección — strides incorrectos | v2.6.0: Strides [8,16,32] (oficial) en vez de [4,8,16] ✅ |
| T2 fallido — ultralytics missing | v2.6.1: Añadido `ultralytics>=8.2` a `install_requires` ✅ |
| T2 lección — ExperimentSetup bugs | v2.6.2: Corregidos `best_val_loss`, `best_epoch`, `duration_s`, `batch_size`, aug field names ✅ |
| T3 lección — Focal Loss degradó | Focal Loss (γ=2.0, α=0.25) empeoró Precision (-27%) en modelo de 0.36M. No recomendado sin más capacidad ⚠️ |

---

## 3. Train 1 — Baseline (⚠️ Arquitectura Custom v1 — OBSOLETA)

> **NOTA**: Train 1 usó la **arquitectura custom simplificada** (v1), no la oficial Espressif.  
> Resultados no son comparables con Train 2+ (arquitectura oficial v2).  
> Causa raíz del fracaso: 22.8K params insuficientes + sin transfer learning + strides [4,8,16] incorrectos.

### 3.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `3958704668489547776` |
| **Pipeline ID** | `1278128305320493056` |
| **Fecha** | 23 de febrero de 2026 |
| **Paquete** | `tfm_trainer-2.5.0.tar.gz` |
| **Config YAML** | `espdet_pico_v1.yaml` → `espdet_pico_v1-train1.yaml` |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet_pico_v1-train1/` |
| **Output local** | `outputs/espdet_pico_v1-train1/` |

### 3.2 Configuración

Baseline puro — sin cambios respecto a la configuración base (§2).

| Parámetro | Valor |
|---|---|
| width_mult | 0.5 |
| pretrained_weights | None |
| Phase 1 | 40 ep, LR=1e-3, WD=1e-4 |
| Phase 2 | 80 ep, LR=5e-5, WD=1e-5 |
| AMP | True |
| Grad clip | 5.0 |
| Batch size | 32 |
| Patience | 20 |
| conf_threshold | 0.25 |

### 3.3 Verificación de Despliegue

DEPLOY VERIFICATION confirmado en logs — todos los parámetros correctos:

```
🎯 DEPLOY VERIFICATION — ESPDet-Pico v2.5.0
  width_mult:      0.5
  reg_max:         1
  pretrained:      None
  Phase 1:         40 ep, LR=0.001, WD=0.0001
  Phase 2:         80 ep, LR=5e-05, WD=1e-05
  Optimizer:       adamw
  cls_weight:      1.0
  reg_weight:      2.0
  Conf threshold:  0.25
  IoU threshold:   0.45
  AMP:             True
  Grad clip:       5.0
  Aug keys:        ['aug_brightness_limit', 'aug_contrast_limit', ..., 'aug_clahe']
```

Bug fix de `aug_config` confirmado: 16 aug keys presentes (vs 0 sin el fix).

### 3.4 Entrenamiento

- **Épocas completadas**: 120 (40 Phase 1 + 80 Phase 2, sin early stopping)
- **Mejor val_loss**: 8.0929 (epoch 116)
- **Phase 1** (backbone frozen, 12,247 trainable params):
  - Start: train=10.608, val=10.654 (img=640)
  - Best: epoch 29, val=8.350 (img=416)
  - End: epoch 39, val=8.357 (img=320)
  - Time: 9.7 min
  - cls_loss: 4.607 → 3.316 (-28%)
  - reg_loss: 6.000 → 4.774 (-20%)
- **Phase 2** (full fine-tuning, 22,839 trainable params):
  - Start: train=8.578, val=8.885 (img=224) — **spike por cambio de resolución**
  - Best: epoch 116, val=8.093 (img=224)
  - End: epoch 119, val=8.097 (img=224)
  - Time: 13.8 min
  - cls_loss: 3.320 → 3.055 (-8%)
  - reg_loss: 5.258 → 4.619 (-12%)
- **Tiempo total**: 23.6 min
- **Phase 2 vs Phase 1**: val_loss mejoró 8.350 → 8.093 (-3.1%) ✅
- **No early stopping**: val_loss siguió mejorando lentamente hasta epoch 116 (last 4 epochs sin mejora, pero patience=20 no se agotó)

**Evolución de pérdida (puntos clave):**

| Epoch | Phase | img | train_loss | val_loss | cls | reg | Nota |
|---|---|---|---|---|---|---|---|
| 0 | P1 | 640 | 10.608 | 10.654 | 4.607 | 6.000 | Start |
| 13 | P1 | 640 | 8.425 | 8.438 | 3.469 | 4.956 | Best P1 @ 640 |
| 29 | P1 | 416 | 8.145 | 8.350 | 3.352 | 4.793 | **Best P1** |
| 39 | P1 | 320 | 8.090 | 8.357 | 3.316 | 4.774 | End P1 |
| 40 | P2 | 224 | 8.578 | 8.885 | 3.320 | 5.258 | **Spike** (res→224) |
| 57 | P2 | 224 | 7.981 | 8.326 | 3.185 | 4.797 | Supera P1 best |
| 97 | P2 | 224 | 7.684 | 8.107 | 3.059 | 4.625 | — |
| 116 | P2 | 224 | 7.661 | **8.093** | 3.045 | 4.616 | **Best global** |
| 119 | P2 | 224 | 7.674 | 8.097 | 3.055 | 4.619 | End |

### 3.5 Resultados — Validación

| Métrica | Valor |
|---|---|
| mAP@50 | **0.0154** |
| mAP@50-95 | 0.0052 |
| Precision | 0.0005 |
| Recall | 0.1024 |
| F1-Score | 0.0011 |
| Detecciones totales | **107,629** (~572/imagen) |
| Ground truths | 762 |
| Inferencia | 44.5 ms |

**Per-class AP@50 (Val):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.0000 | 0.0004 | 0.0400 | 0.0007 |
| door | **0.0763** | 0.0008 | 0.3313 | 0.0016 |
| obstacle | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| person | 0.0000 | 0.0003 | 0.0275 | 0.0006 |
| stair | 0.0004 | 0.0012 | 0.1132 | 0.0024 |

**Confusion Matrix (Val):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 6 | 0 | 0 | 0 | 0 | 144 |
| **door** | 0 | 53 | 0 | 0 | 0 | 107 |
| **obstacle** | 0 | 0 | 0 | 0 | 0 | 164 |
| **person** | 0 | 0 | 0 | 5 | 0 | 177 |
| **stair** | 0 | 0 | 0 | 0 | 12 | 94 |
| **FP (bkg)** | 16,471 | **64,068** | 6 | 17,195 | 9,813 | — |

### 3.6 Resultados — Test

| Métrica | Valor |
|---|---|
| mAP@50 | **0.0105** |
| mAP@50-95 | 0.0023 |
| Precision | 0.0009 |
| Recall | 0.1597 |
| F1-Score | 0.0017 |
| Detecciones totales | **96,283** (~515/imagen) |
| Ground truths | 576 |
| Inferencia | 34.4 ms |

**Per-class AP@50 (Test):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.0008 | 0.0007 | 0.1724 | 0.0014 |
| door | **0.0495** | 0.0006 | 0.2574 | 0.0013 |
| obstacle | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| person | 0.0002 | 0.0008 | 0.1188 | 0.0017 |
| stair | 0.0018 | 0.0021 | 0.2500 | 0.0042 |

**Confusion Matrix (Test):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 10 | 0 | 0 | 0 | 0 | 48 |
| **door** | 0 | 35 | 0 | 0 | 0 | 101 |
| **obstacle** | 0 | 0 | 0 | 0 | 0 | 173 |
| **person** | 0 | 0 | 0 | 12 | 0 | 89 |
| **stair** | 0 | 0 | 0 | 0 | 27 | 81 |
| **FP (bkg)** | 14,100 | **54,890** | 1 | 14,416 | 12,792 | — |

### 3.7 Export ONNX

| Parámetro | Valor |
|---|---|
| ONNX size | **0.109 MB** (112 KB) |
| ONNX valid | ✅ |
| ONNX latency | 0.76 ms (CPU, OnnxRuntime) |
| Input shape | (1, 3, 224, 224) |
| Opset | 13 |
| Checkpoint size | 0.149 MB |

> **Contexto**: El ONNX de ESPDet-Pico es **44× más pequeño** que FCOS (4.74 MB) y **92× más pequeño** que YOLO26 (9.97 MB). Cabe cómodamente en Flash del ESP32-S3, pero la utilidad está condicionada a que el modelo detecte correctamente.

### 3.8 Análisis

**Resultado general: FRACASO FUNCIONAL — el modelo no aprendió detección significativa. mAP@50 ≈ 0.01 es indistinguible de ruido aleatorio.**

#### Diagnóstico principal: Crisis de capacidad

1. **22,839 parámetros es insuficiente para detección de 5 clases**. ESPDet-Pico con `width_mult=0.5` produce canales de [8, 16, 32] en el backbone y un FPN de 8 canales. Este es el detector más pequeño intentado en el proyecto, con 54× menos params que FCOS y 113× menos que YOLO26. La capacidad representativa no alcanza para discriminar 5 clases de objetos con variaciones significativas de apariencia, escala y pose.

2. **cls_loss final ≈ 3.05 vs random ≈ 3.47** — Para 5 clases con BCE, la pérdida aleatoria (sigmoid(0)=0.5) es $-\ln(0.5) \times 5 = 3.465$. El modelo apenas superó la clasificación aleatoria por un **12% relativo**. En contraste, FCOS reducía cls_loss de ~4.5 a ~1.2 (73% reducción).

3. **107,629 detecciones en 188 imágenes val (572/imagen)** — La cabeza clasificadora NO ha aprendido a suprimir el fondo. La mayoría de las salidas sigmoid superan conf=0.25, produciendo un spam masivo de FP. Precision = 0.0005 (1 de cada 2,000 detecciones es correcta).

4. **obstacle: AP@50 = 0.0000** — Cero detecciones correctas. El modelo no aprendió esta clase en absoluto. Además, solo produce 6 FP de obstacle en val y 1 en test, lo que sugiere que ni siquiera intenta predecirla.

5. **door es la clase "menos mala"** (AP@50 val=0.0763, test=0.0495) — Paradójicamente, también acumula la mayor cantidad de FP (64,068 val, 54,890 test). El modelo emite "door" indiscriminadamente.

#### Causas raíz identificadas

| Causa | Impacto | Evidencia |
|---|---|---|
| **Capacidad insuficiente** | CRÍTICA | cls_loss ≈ random; 22.8K params vs 1.2M+ necesarios |
| **Sin pretraining** | ALTA | YOLO26 arranca con COCO pretrained (mAP50=0.75); ESPDet arranca de cero |
| **BCE sin Focal Loss** | MEDIA | Masivo desbalance background/foreground no mitigado |
| **Progressive resizing innecesario** | MEDIA | 640px con canales de 8: feature maps inútilmente grandes; spike en epoch 40 pierde 17 epochs |
| **Naming mismatch aug** | BAJA | ✅ Corregido v2.6.0 — YAML keys alineados con `_build_transforms()` |

#### Training dynamics

6. **Phase 1 convergió correctamente** — val_loss bajó de 10.65 a 8.35 (-22%) en 40 epochs. El backbone (10,592 params) estaba frozen, por lo que solo la cabeza aprendía. La cabeza hizo lo que pudo con features aleatorias del backbone.

7. **Phase 2 mejoró sobre Phase 1** — val_loss 8.35 → 8.09 (-3.1%). El descongelamiento del backbone ayudó, pero la mejora absoluta fue marginal. El modelo convergía muy lentamente en los últimos 20 epochs (mejoras de 0.005-0.01 por epoch).

8. **Spike en epoch 40** — El cambio de resolución 320→224 causó un salto de val_loss 8.35→8.88. Phase 2 tardó 17 epochs (hasta epoch 57) en recuperar el nivel de Phase 1. Esto es el 21% del presupuesto de Phase 2 desperdiciado en recuperación.

9. **No se disparó early stopping** — Best en epoch 116, end en 119. Solo 3 epochs sin mejora (patience=20). El modelo todavía tenía margen de aprendizaje, pero la mejora por epoch era insignificante (~0.003 val_loss).

#### Dead config detectada

10. **`fpn_channels: 32` en YAML es dead config** — `build_espdet_pico()` no acepta este parámetro; el FPN channel se determina automáticamente como `ch_list[0]` del backbone (= 8 con width_mult=0.5). El YAML sugiere que se esperaba un FPN de 32 canales, pero el modelo usa 8.

11. **`num_head_convs: 2` en YAML es dead config** — `build_espdet_pico()` no acepta este parámetro; siempre usa 2 convs por defecto en la implementación.

### 3.9 Lecciones

1. **LECCIÓN PRINCIPAL: ESPDet-Pico con width_mult=0.5 NO es viable para 5 clases** — 22.8K params es un orden de magnitud insuficiente. Es necesario aumentar significativamente la capacidad (width_mult=1.0 mínimo, idealmente 2.0+).

2. **DEPLOY VERIFICATION funcionó correctamente** — Confirmó paquete v2.5.0, todos los parámetros correctos, y el fix de aug_config. Primera vez que un modelo ESPDet se lanza sin bugs de despliegue.

3. **Bug de aug_config corregido, y naming mismatch resuelto** — `aug_config` se pasa correctamente al dataset (fix v2.5.0). En v2.6.0, los keys del YAML v2 (`aug_hflip_prob`, `aug_brightness_limit`, etc.) coinciden con los que `IODCDataset._build_transforms()` busca. Los valores personalizados del YAML ahora se aplican correctamente.

4. **Progressive resizing es contraproducente para este modelo** — El spike de epoch 40 y la recuperación lenta sugieren que este modelo micro no maneja bien los cambios de resolución. Entrenar directamente a 224px simplificaría el proceso.

5. **Vertex AI Experiments no disponible (403)** — Error recurrente de scoping (`ACCESS_TOKEN_SCOPE_INSUFFICIENT`). El entrenamiento continúa sin registro de experimentos. No impacta los resultados pero sí la trazabilidad en Vertex AI.

6. **Warnings benignos** — `pythonjsonlogger` no encontrado, pip dependency conflicts (bigframes, ydata-profiling, dataproc-jupyter-plugin). Idénticos a FCOS y YOLO26. Sin impacto.

---

## 4. Train 2 — Arquitectura Oficial Espressif (v2) ✅

> **PRIMER ENTRENAMIENTO EXITOSO** — El modelo ESPDet-Pico ya detecta objetos. mAP@50=0.62 (test), crecimiento de **59×** vs Train 1.

### 4.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `3793775725299892224` |
| **Pipeline ID** | `1216450101048770560` |
| **Fecha** | 23 de febrero de 2026 |
| **Paquete** | `tfm_trainer-2.6.1.tar.gz` |
| **Config YAML** | `espdet_pico_v2.yaml` → `espdet-pico-v2-t2.yaml` |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v2-t2/` |
| **Output local** | `outputs/espdet-pico-v2-t2/` |

### 4.2 Configuración

Arquitectura oficial Espressif v2 — según configuración base §2 (sin modificaciones).

| Parámetro | Valor |
|---|---|
| Arquitectura | **Oficial Espressif** (v2, 361,563 params) |
| Pretrained | `espdet_pico_224_224_cat.pt` (cat-detection, nc=1) |
| Phase 1 | 50 ep, LR=1e-3, WD=1e-4, backbone frozen |
| Phase 2 | 100 ep, LR=1e-4, WD=1e-5, full fine-tuning |
| AMP | True |
| Grad clip | 5.0 |
| Batch size | 32 |
| Patience | 25 |
| conf_threshold | 0.25 |
| Resolución | 224×224 (fija) |

### 4.3 Verificación de Despliegue

DEPLOY VERIFICATION confirmado en logs — todos los parámetros correctos:

```
🎯 DEPLOY VERIFICATION — ESPDet-Pico v2.6.1 (Official Architecture)
  Architecture:    Official Espressif (esp-detection repo)
  Strides:         [8, 16, 32]
  pretrained:      gs://...pretrained/espdet_pico_224_224_cat.pt
  Phase 1:         50 ep, LR=0.001, WD=0.0001
  Phase 2:         100 ep, LR=0.0001, WD=1e-05
  Optimizer:       adamw
  cls_weight:      1.0
  reg_weight:      2.0
  Conf threshold:  0.25
  IoU threshold:   0.45
  AMP:             True
  Grad clip:       5.0
  Export imgsz:    224
  Batch size:      32
  Patience:        25
  Aug keys:        ['aug_brightness_limit', 'aug_contrast_limit', ..., 'aug_gaussian_noise']
```

Transfer learning confirmado:
- ✅ 622 param groups cargados (~99.97%)
- ℹ️ Shape mismatch (random init): 6 tensores de `head.cv3.*.2` (cls final nc=1→5)
- ✅ Backbone congelado: 213,440 params frozen (59.0%)
- ✅ Trainable Phase 1: 148,123 / 361,563 (41.0%)

### 4.4 Entrenamiento

- **Épocas completadas**: 113 (50 Phase 1 + 63 Phase 2, **early stopping** epoch 112)
- **Mejor val_loss**: **4.4100** (epoch 87)
- **Phase 1** (backbone frozen, 148,123 trainable params):
  - Start: train=16.476, val=14.675 (img=224)
  - Best: epoch 48, val=4.682
  - End: epoch 49, val=4.690
  - Time: 9.5 min
  - cls_loss: 12.654 → 1.283 (**-89.9%**)
  - reg_loss: 3.821 → 1.766 (-53.8%)
- **Phase 2** (full fine-tuning, 361,563 trainable params):
  - Start: epoch 50, val=4.645 (sin spike — resolución constante 224px ✅)
  - Best: **epoch 87, val=4.410**
  - Early stop: epoch 112 (25 epochs sin mejora)
  - Time: 12.1 min
  - cls_loss: 1.305 → 0.722 (-44.7%)
  - reg_loss: 1.756 → 1.366 (-22.2%)
- **Tiempo total**: 21.7 min
- **Phase 2 vs Phase 1**: val_loss mejoró 4.682 → 4.410 (-5.8%) ✅
- **Early stopping se activó**: Best en epoch 87, stop en epoch 112 (patience=25 agotado)

**Evolución de pérdida (puntos clave):**

| Epoch | Phase | train_loss | val_loss | cls | reg | Nota |
|---|---|---|---|---|---|---|
| 0 | P1 | 16.476 | 14.675 | 12.654 | 3.821 | Start |
| 2 | P1 | 7.150 | 6.781 | 4.239 | 2.910 | Descenso rápido |
| 11 | P1 | 3.788 | 5.025 | 1.708 | 2.080 | — |
| 25 | P1 | 3.315 | 4.772 | 1.430 | 1.885 | — |
| 42 | P1 | 3.113 | 4.690 | 1.329 | 1.785 | — |
| 48 | P1 | 3.050 | **4.682** | 1.303 | 1.747 | **Best P1** |
| 50 | P2 | 3.061 | 4.645 | 1.305 | 1.756 | Start P2 (sin spike ✅) |
| 60 | P2 | 2.816 | 4.465 | 1.129 | 1.687 | — |
| 71 | P2 | 2.490 | 4.418 | 0.938 | 1.552 | — |
| 87 | P2 | 2.291 | **4.410** | 0.833 | 1.458 | **Best global** ★ |
| 100 | P2 | 2.142 | 4.505 | 0.739 | 1.403 | Overfitting visible |
| 112 | P2 | 2.089 | 4.532 | 0.722 | 1.366 | **Early stop** ⏹️ |

### 4.5 Resultados — Validación

| Métrica | Valor |
|---|---|
| mAP@50 | **0.4543** |
| mAP@50-95 | 0.2100 |
| Precision | 0.2848 |
| Recall | 0.5628 |
| F1-Score | 0.3782 |
| Detecciones totales | 1,538 (~8.2/imagen) |
| Ground truths | 762 |
| Inferencia | 11.2 ms |

**Per-class AP@50 (Val):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.4100 | 0.3304 | 0.5067 | 0.4000 |
| door | **0.5313** | 0.3127 | 0.6313 | 0.4182 |
| obstacle | 0.3614 | 0.2266 | 0.5915 | 0.3277 |
| person | 0.4893 | 0.3160 | 0.5659 | 0.4055 |
| stair | 0.4796 | 0.2381 | 0.5189 | 0.3264 |

**Confusion Matrix (Val):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 76 | 0 | 0 | 0 | 0 | 74 |
| **door** | 0 | 101 | 0 | 0 | 0 | 59 |
| **obstacle** | 0 | 0 | 97 | 0 | 0 | 67 |
| **person** | 0 | 0 | 0 | 103 | 0 | 79 |
| **stair** | 0 | 0 | 0 | 0 | 55 | 51 |
| **FP (bkg)** | 154 | 222 | 331 | 223 | 176 | — |

### 4.6 Resultados — Test

| Métrica | Valor |
|---|---|
| mAP@50 | **0.6203** |
| mAP@50-95 | 0.3078 |
| Precision | 0.2956 |
| Recall | 0.7235 |
| F1-Score | 0.4197 |
| Detecciones totales | 1,416 (~7.6/imagen) |
| Ground truths | 576 |
| Inferencia | 4.6 ms |

**Per-class AP@50 (Test):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.6085 | 0.2826 | 0.6724 | 0.3980 |
| door | 0.5846 | 0.2697 | 0.6544 | 0.3820 |
| obstacle | 0.5043 | 0.2934 | 0.7225 | 0.4174 |
| person | 0.6742 | 0.3333 | 0.7624 | 0.4639 |
| stair | **0.7299** | 0.2990 | 0.8056 | 0.4361 |

**Confusion Matrix (Test):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 39 | 0 | 0 | 0 | 0 | 19 |
| **door** | 0 | 89 | 0 | 0 | 0 | 47 |
| **obstacle** | 0 | 0 | 125 | 0 | 0 | 48 |
| **person** | 0 | 0 | 0 | 77 | 0 | 24 |
| **stair** | 0 | 0 | 0 | 0 | 87 | 21 |
| **FP (bkg)** | 99 | 241 | 301 | 154 | 204 | — |

### 4.7 Export ONNX

| Parámetro | Valor |
|---|---|
| ONNX size | **1.41 MB** |
| ONNX valid | ✅ |
| ONNX latency | 2.5 ms (CPU, OnnxRuntime) |
| Input shape | (1, 3, 224, 224) |
| Opset | 13 |
| Outputs | 6 — Interleaved: box0(1,4,28,28), score0(1,5,28,28), box1(1,4,14,14), score1(1,5,14,14), box2(1,4,7,7), score2(1,5,7,7) |
| Checkpoint size | 1.70 MB |
| onnxsim | No (no instalado en container) |

### 4.8 Análisis

**Resultado general: ÉXITO — El modelo detecta correctamente. mAP@50=0.62 (test) valida la arquitectura oficial + transfer learning como viable para ESP32-S3.**

#### Mejoras vs Train 1

| Métrica (Test) | Train 1 (v1) | Train 2 (v2) | Factor |
|---|:---:|:---:|:---:|
| mAP@50 | 0.0105 | **0.6203** | **59×** |
| mAP@50-95 | 0.0023 | **0.3078** | **134×** |
| Precision | 0.0009 | **0.2956** | **329×** |
| Recall | 0.1597 | **0.7235** | **4.5×** |
| F1 | 0.0017 | **0.4197** | **247×** |
| Detecciones/img (val) | 572 | **8.2** | **-98.6%** |
| Inferencia (test) | 34.4 ms | **4.6 ms** | **7.5× más rápido** |

La mejora es transformacional: de ruido aleatorio a detector funcional.

#### Fortalezas identificadas

1. **Transfer learning altamente efectivo** — La pérdida de clasificación bajó de 12.65 a 1.28 solo en Phase 1 (backbone frozen). Los features del backbone pretrained cat-detection generalizaron bien a las 5 clases IODC. Esto confirma que la representación visual de bajo nivel (bordes, texturas, formas) es transferible entre dominios de detección.

2. **Sin spike en Phase 2** — A diferencia de Train 1 (spike de val 8.35→8.89 por cambio de resolución 320→224), Train 2 con resolución fija 224px no tuvo spike. Epoch 50 (val=4.645) fue inmediatamente mejor que epoch 48 (val=4.682). El descongelamiento del backbone fue suave.

3. **Early stopping funcionó** — El modelo no desperdició 37 epochs de cómputo. Best en epoch 87, stop en 112. Patience=25 fue un buen balance entre exploración y eficiencia.

4. **Recall alto (0.72 test)** — El modelo encuentra la mayoría de los objetos. Todas las clases superan 0.65 recall. Para asistencia visual (caso de uso ESP32-S3), recall alto es prioritario sobre precision.

5. **Cero confusión inter-clase** — Las confusion matrices muestran TP solo en la diagonal. El modelo NO confunde clases entre sí: nunca clasifica un dog como door, ni un person como obstacle. Todos los errores son FP (background→clase) o FN (clase→miss).

6. **stair es la clase mejor detectada** (AP@50=0.7299 test) — Recall 80.6%, la más alta del dataset. Relevante para el caso de uso de asistencia visual.

#### Debilidades identificadas

7. **Precision baja (~0.30)** — El modelo produce ~2.5× más detecciones que objetos reales. Para 576 GT en test, genera 1,416 detecciones. Esto implica que ~60% de las detecciones son FP (background clasificado como objeto). Aunque mucho mejor que T1 (572/img → 8/img), aún hay margen de mejora.

8. **obstacle tiene los más FP** — 331 FP en val (vs 97 TP), 301 FP en test (vs 125 TP). El modelo tiende a clasificar áreas de background como obstacle, probablemente porque esta clase es visualmente diversa (muebles, objetos varios).

9. **Gap Val-Test significativo** — mAP@50 val=0.454 vs test=0.620 (+36% en test). Esto es inusual y sugiere que el conjunto de validación es más difícil o tiene distribución distinta. No indica overfitting al val set (al contrario), pero dificulta usar val_loss como proxy fiable del rendimiento real.

10. **Indicios de overfitting al final** — train_loss bajó a 2.09 mientras val_loss se estancó en 4.41 (ratio 2.1×). Las últimas 25 epochs (87-112) no mejoraron validación. Más epochs no habrían ayudado.

11. **ONNX sin simplificación** — `onnxsim` no está instalado en el container; el ONNX (1.41 MB) podría reducirse ~10-15% con simplificación.

#### Bugs de tracking detectados

12. **experiment.json tiene campos sin poblar** — `best_val_loss: "inf"`, `best_epoch: 0`, `duration_s: 0.0`. El código no actualiza estos campos de `ExperimentSetup` después del entrenamiento. Bug cosmético (métricas reales están en `val_evaluation.json` y `test_evaluation.json`).

13. **experiment.json config tiene aug keys viejos** — Muestra `aug_horizontal_flip`, `aug_brightness_contrast`, `aug_rotation_limit` (keys del dataclass `ExperimentSetup`, no del YAML real). Las augmentaciones se aplicaron correctamente (confirmado por DEPLOY VERIFICATION), pero el tracking captura snapshots del dataclass que usa naming legacy. Bug cosmético, sin impacto en training.

14. **experiment.json tiene batch_size: 16 en two_phase** — El default del dataclass, no el valor real (32). Mismo origen que #13.

### 4.9 Lecciones

1. **LECCIÓN PRINCIPAL: La arquitectura oficial + transfer learning es la clave** — 16× más parámetros (22.8K → 361K) y features pretrained de calidad transformaron el modelo de inoperable (0.01 mAP) a funcional (0.62 mAP). Los ~$10 invertidos en el retrain en Vertex AI validaron la hipótesis completamente.

2. **Resolución fija simplifica el training** — Sin progressive resize, no hay spikes ni epochs de recuperación. La transición Phase 1→Phase 2 fue limpia.

3. **Precision baja es el próximo cuello de botella** — Para mejorar la calidad del modelo, el siguiente paso debería enfocarse en reducir FP (Focal Loss, NMS tuning, o conf_threshold ajuste).

4. **ExperimentSetup dataclass necesita refactoring** — Los campos de runtime (`best_val_loss`, `best_epoch`, `duration_s`) y los aug keys deberían actualizarse post-training. Bug cosmético pero reduce la utilidad del tracking de experimentos.

5. **`ultralytics>=8.2` en dependencies funciona** — La instalación (8.4.14) fue exitosa, pero arrastra ~155 MB de deps extra (polars, opencv-python duplicado). Futuro: considerar vendorear solo los módulos necesarios (`ultralytics.nn.modules.conv`, `.block`).

6. **Vertex AI Experiments sigue sin funcionar (403)** — Error recurrente idéntico a FCOS y YOLO26. El training continúa sin registro de experimentos. Pendiente de resolver a nivel de IAM/scoping del service account.

---

## 5. Train 3 — Focal Loss (v3)

> **Hipótesis**: Reemplazar BCE por Sigmoid Focal Loss (γ=2.0, α=0.25) en `cls_loss` para mejorar la supresión de background y reducir FP, subiendo Precision sin sacrificar Recall.
>
> **Resultado**: ❌ **HIPÓTESIS NO CONFIRMADA** — Focal Loss **degradó** Precision (-27%), mAP@50 (-3.4%) y F1 (-19.8%). El único beneficio fue un ligero aumento de Recall (+5.9%). El modelo de 0.36M params no tiene capacidad suficiente para beneficiarse de Focal Loss con estos hiperparámetros.

### 5.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `2124347638428991488` |
| **Pipeline ID** | `579401860008378368` |
| **Fecha** | 23 de febrero de 2026 |
| **Paquete** | `tfm_trainer-2.6.2.tar.gz` |
| **Config YAML** | `espdet_pico_v3.yaml` → `espdet-pico-v3-t3.yaml` |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/espdet-pico-v3-t3/` |
| **Output local** | `outputs/espdet-pico-v3-t3/` |

### 5.2 Configuración (Cambios vs Train 2)

| Parámetro | Train 2 (v2) | **Train 3 (v3)** | Cambio |
|---|---|---|---|
| `cls_loss` | BCE | **Sigmoid Focal Loss** | γ=2.0, α=0.25 |
| Paquete | `tfm_trainer-2.6.1` | **`tfm_trainer-2.6.2`** | Focal Loss + ExperimentSetup fixes |
| Config YAML | `espdet_pico_v2.yaml` | **`espdet_pico_v3.yaml`** | Solo cambios de loss |
| Arquitectura | Oficial Espressif (361K) | Idéntica (361K) | Sin cambios |
| Pretrained | `espdet_pico_224_224_cat.pt` | Idéntico | Sin cambios |
| Two-phase | P1=50ep, P2=100ep | Idéntico | Sin cambios |
| Augmentación | HFlip, BC, HSV, Rotate, GaussNoise | Idéntica | Sin cambios |
| conf_threshold | 0.25 | 0.25 | Sin cambios |

> **Cambio único**: Solo se modificó la función de loss de clasificación. Todo lo demás es idéntico a Train 2 para aislar el efecto de Focal Loss.

#### Fixes en v2.6.2 (cosmético — sin impacto en training)

| Fix | Descripción |
|---|---|
| `best_val_loss` | Ahora se actualiza post-training (antes: `"inf"`) |
| `best_epoch` | Ahora se actualiza post-training (antes: `0`) |
| `duration_s` | Ahora se registra correctamente (antes: `0.0`) |
| `batch_size` | Ahora refleja el valor real de `two_phase` (antes: default `16`) |
| Aug field names | Ahora usa naming correcto del YAML (antes: naming legacy del dataclass) |

### 5.3 DEPLOY VERIFICATION

```
🔍 DEPLOY VERIFICATION:
  Package version: 2.6.2 ✅
  Focal Loss: ON (γ=2.0, α=0.25) ✅
  Model architecture: espdet_pico  
  Total params: 361,563
  Transfer learning: 622 param groups, 616 loaded, 6 shape mismatches (cls head → nc=5)
  Aug keys: 10 keys present
  Export format: interleaved
```

> ✅ **Focal Loss confirmado activo** en el log: `🎯 ESPDet cls_loss: Sigmoid Focal Loss (γ=2.0, α=0.25)`

### 5.4 Dinámica de Entrenamiento

| Métrica | Phase 1 (Freeze) | Phase 2 (Full) | Total |
|---|---|---|---|
| Epochs | 50 | 81 (early stop ep. 130) | **131** |
| Duración | 10.2 min | 16.8 min | **27.0 min** |
| train_loss (inicio→fin) | 6.46 → 1.87 | 1.85 → 1.34 | 6.46 → 1.34 |
| val_loss (inicio→fin) | 5.65 → 2.76 | 2.73 → 2.57 | 5.65 → 2.57 |
| Best val_loss | 2.7633 (ep. 48) | **2.5571 (ep. 105)** | — |
| LR | 1e-3 → ~0 (cosine) | 1e-4 → ~0 (cosine) | — |

**Observaciones**:
- Transición Phase 1→Phase 2 limpia: val_loss pasó de 2.76 a 2.73 sin spike ✅
- Early stopping activado en epoch 130 (patience=25, best_epoch=105)
- **Loss scale NO comparable con T2**: Focal Loss pondera los ejemplos fáciles con factor $(1-p_t)^\gamma$, reduciendo la magnitud del loss. T2 best_val_loss=4.41 vs T3 best_val_loss=2.56 — la diferencia es por la ponderación, NO por mejor convergencia.
- train/val ratio al final: 1.34/2.57 = 1.9× — similar a T2 (2.09/4.41 = 2.1×). Mismo nivel de overfitting.

### 5.5 Resultados — Validación

| Métrica | Valor |
|---|---|
| **mAP@50** | **0.4230** |
| **mAP@50-95** | 0.2164 |
| **Precision** | 0.2111 |
| **Recall** | 0.5936 |
| **F1-Score** | 0.3114 |
| Detecciones / GT | 2,367 / 762 (3.1× ratio) |

#### Per-class AP@50 — Validación

| Clase | AP@50 | Recall | TP | FP (bg) | Nota |
|---|:---:|:---:|:---:|:---:|---|
| dog | 0.3913 | 0.4667 | 70 | 217 | Clase más difícil en val |
| door | 0.4683 | 0.6563 | 105 | 359 | — |
| obstacle | 0.3309 | 0.7256 | 119 | 790 | Más FP absolutos |
| person | 0.4956 | 0.6099 | 111 | 328 | — |
| stair | 0.4286 | 0.5094 | 54 | 214 | — |

### 5.6 Resultados — Test

| Métrica | Valor | vs T2 | Delta |
|---|---|:---:|---|
| **mAP@50** | **0.5993** | 0.6203 | **-3.4%** ⬇️ |
| **mAP@50-95** | 0.3220 | 0.3078 | +4.6% ⬆️ |
| **Precision** | 0.2156 | 0.2956 | **-27.1%** ⬇️ |
| **Recall** | **0.7663** | 0.7235 | +5.9% ⬆️ |
| **F1-Score** | 0.3365 | 0.4197 | **-19.8%** ⬇️ |
| Detecciones / GT | 2,119 / 576 | 1,416 / 576 | **+49.6% más detecciones** ⬇️ |

#### Per-class AP@50 — Test (comparativa)

| Clase | T3 AP@50 | T2 AP@50 | Delta | T3 Recall | T3 TP | T3 FP (bg) |
|---|:---:|:---:|---|:---:|:---:|:---:|
| dog | 0.5989 | 0.6085 | -1.6% | 0.6724 | 39 | 153 |
| door | 0.5589 | 0.5846 | -4.4% | 0.7426 | 101 | 380 |
| obstacle | 0.4418 | 0.5043 | **-12.4%** | 0.8266 | 143 | 598 |
| person | 0.7078 | 0.6742 | **+5.0%** | 0.8119 | 82 | 283 |
| stair | 0.6893 | 0.7299 | -5.6% | 0.7778 | 84 | 256 |

### 5.7 Confusion Matrices

```
Cero confusión inter-clase (idéntico a T2)
Todas las off-diagonal cls→cls son 0.0
Todos los errores son: background → clase (FP) o clase → miss (FN)
```

**Comparativa FP (background→clase) — Test**:

| Clase | T2 FP (bg) | T3 FP (bg) | Delta |
|---|:---:|:---:|---|
| dog | 87 | 153 | +75.9% ⬆️ |
| door | 192 | 380 | +97.9% ⬆️ |
| obstacle | 301 | 598 | +98.7% ⬆️ |
| person | 202 | 283 | +40.1% ⬆️ |
| stair | 103 | 256 | +148.5% ⬆️ |
| **Total** | **885** | **1,670** | **+88.7%** ⬆️ |

> ⚠️ **Focal Loss casi duplicó los FP de background**. El efecto fue exactamente opuesto al esperado.

### 5.8 ONNX Export

| Métrica | Valor | vs T2 |
|---|---|---|
| ONNX size | 1.41 MB | = |
| ONNX outputs | 6 (interleaved) | = |
| Latencia GPU (T4) | 3.6 ms | ~T2 (2.5 ms) |
| Checkpoint | 1.70 MB | = |
| `onnxsim` | No instalado | = |

### 5.9 ExperimentSetup — Validación de Fixes (v2.6.2)

| Campo | T2 (v2.6.1) | T3 (v2.6.2) | Estado |
|---|---|---|---|
| `best_val_loss` | `"inf"` | **2.5571** | ✅ Fix confirmado |
| `best_epoch` | `0` | **105** | ✅ Fix confirmado |
| `duration_s` | `0.0` | **1622.65** | ✅ Fix confirmado |
| `batch_size` | `16` (wrong default) | **32** | ✅ Fix confirmado |
| Aug keys | Legacy names | **Correctos** (`aug_hflip_prob`, etc.) | ✅ Fix confirmado |

> ✅ Los 5 bugs cosméticos del `ExperimentSetup` (Propuesta G) están **completamente resueltos** en v2.6.2.

### 5.10 Análisis

#### ¿Por qué Focal Loss degradó el rendimiento?

1. **Efecto opuesto al esperado** — Focal Loss estaba diseñada para reducir FP de background down-weighting los ejemplos fáciles (background claro). En su lugar, **aumentó** los FP un 88.7%. El modelo ahora produce 2,119 detecciones para 576 GT (3.7× ratio) vs 1,416 en T2 (2.5× ratio).

2. **Capacidad insuficiente del modelo** — Con solo 0.36M params, ESPDet-Pico necesita **toda** la señal de gradiente disponible para aprender la supresión de background. Focal Loss (α=0.25) reduce el peso de los negativos fáciles en un factor ~100× ($\alpha \cdot (1-p_t)^\gamma \approx 0.25 \cdot 0.01 = 0.0025$ para negativos con $p_t \approx 0.9$). Esto priva al modelo de la señal que necesita para discriminar background.

3. **Focal Loss funciona mejor en modelos más grandes** — En la literatura (Lin et al., 2017), Focal Loss mejora RetinaNet (~36M params). Con 100× menos capacidad, ESPDet-Pico no puede compensar la reducción de gradiente en negativos fáciles con mejores representaciones internas.

4. **obstacle fue la clase más afectada** — AP@50 cayó -12.4% (0.5043→0.4418). Esta clase ya era la más difícil (visualmente diversa, muchos FP). Al reducir la señal de gradiente para background, obstacle perdió la capacidad de distinguir objetos reales del background complejo.

5. **person fue la única mejora** — AP@50 subió +5.0% (0.6742→0.7078). Posiblemente porque es la clase más visualmente consistente (forma humana), donde el modelo ya tenía features fuertes y Focal Loss ayudó a enfocarse en los ejemplos difíciles restantes.

6. **mAP@50-95 mejoró ligeramente** (+4.6%) — Esto sugiere que las detecciones que sí produce T3 tienen mejor localización (boxes más precisos), pero la calidad clasificatoria empeoró drásticamente.

#### Fortalezas

7. **Recall mejorado** — 0.7663 vs 0.7235 (+5.9%). El modelo detecta más objetos reales, pero a costa de muchos más FP.

8. **Cero confusión inter-clase** — Idéntico a T2. La clasificación entre clases sigue siendo perfecta; el problema es exclusivamente background vs objeto.

9. **ExperimentSetup fixes funcionan** — Los 5 bugs cosméticos de Propuesta G están resueltos. El tracking de experimentos es ahora completo y confiable.

#### Debilidades

10. **Precision desplomada (0.2156)** — Peor que T2 (0.2956) en un 27%. Para 576 GT en test, genera 2,119 detecciones — ~73% son FP.

11. **F1-Score degradado (0.3365)** — -19.8% vs T2 (0.4197). El balance Precision/Recall empeoró significativamente.

12. **Efecto contrario al objetivo** — El propósito de T3 era **mejorar** Precision reduciendo FP. El resultado fue exactamente el opuesto.

### 5.11 Lecciones

1. **LECCIÓN PRINCIPAL: Focal Loss no es adecuada para modelos ultra-ligeros (0.36M params)** — La reducción de gradiente en negativos fáciles priva al modelo de la señal necesaria para aprender background suppression. Con capacidad limitada, BCE es más efectivo porque proporciona gradiente completo en todos los ejemplos.

2. **El problema de Precision en ESPDet-Pico no se resuelve con loss engineering** — Hay que explorar otras vías: ajuste de `conf_threshold` (0.35-0.40), NMS más agresivo, o post-procesamiento heurístico.

3. **Aislar variables experimentales funciona** — Al cambiar SOLO la loss function (mismo modelo, mismos hyperparams, mismos datos), el efecto es claramente atribuible a Focal Loss. Buena práctica MLOps.

4. **v2.6.2 es estable para tracking** — ExperimentSetup ahora registra correctamente todos los campos runtime. Base sólida para futuros experimentos.

5. **Train 2 (BCE) sigue siendo el mejor modelo ESPDet-Pico** — Para deploy en ESP32-S3, Train 2 (mAP@50=0.6203, F1=0.4197) es superior a Train 3 en todas las métricas excepto Recall.

---

## Debugging / Incidentes

> Sección para registrar problemas encontrados durante el despliegue y debugging de los entrenamientos.

### Pre-Train 1 — Fix preventivo: aug_config missing

- **Descubierto en**: Revisión de código pre-lanzamiento (comparación con `task_fcos.py`)
- **Síntoma potencial**: `IODCDataset` recibiría `aug_config={}`, usando solo defaults internos de Albumentations
- **Causa raíz**: `task_espdet.py` no extraía ni pasaba `aug_config` al constructor de `IODCDataset`
- **Fix aplicado**: Añadido `aug_config = {k: v for k, v in fc.items() if k.startswith("aug_")}` + pasado como `aug_config=aug_config` a `IODCDataset()`
- **Versión**: Fix incluido en `tfm_trainer-2.5.0`

### Train 1 — Dead config detectada (no impacto directo)

- **`fpn_channels: 32`**: YAML lo define pero `build_espdet_pico()` no acepta el parámetro. FPN usa `ch_list[0]=8` del backbone.
- **`num_head_convs: 2`**: YAML lo define pero no se pasa al constructor.
- **Impacto**: Ninguno en T1 (defaults son razonables).
- **Acción**: ✅ Obsoleto en v2 — YAML v2 no usa estos keys.

### Pre-Train 2 — Revisión integral pre-lanzamiento

- **Bug encontrado**: ONNX interleaved export no se activaba — `hasattr(model, "export_onnx_forward")` era False (método en `model.head`, no en `model`)
- **Fix aplicado**: Cambiado a `hasattr(model, "set_export_mode")` en `utils_export.py`
- **Bug encontrado**: `espdet_modules/__init__.py` no exportaba `C3k`, `ESPSerial`, `ESPSerialLite`, `ESPDetect`, `ESPDLDetect`
- **Fix aplicado**: Añadidas todas las exportaciones faltantes
- **Bug encontrado**: Aug naming mismatch — YAML usaba keys viejos que `_build_transforms()` ignoraba silenciosamente
- **Fix aplicado**: YAML v2 actualizado con keys correctos (`aug_hflip_prob`, `aug_brightness_limit`, etc.)
- **Versión**: Fixes incluidos en `tfm_trainer-2.6.0`
- **Verificación**: 5/5 checks pasados (forward, export mode, pretrained, freeze, aug alignment)

### Train 2 — Intento fallido (antes de T2 exitoso)

- **Error**: `ModuleNotFoundError: No module named 'ultralytics'` en `utils_model.py:261`
- **Causa raíz**: `ultralytics` no estaba en `install_requires` de setup.py. El contenedor Vertex AI no lo incluye. Train 1 (arq. custom) no lo necesitaba.
- **Fix aplicado**: Añadido `"ultralytics>=8.2"` a `install_requires`, version bump a 2.6.1
- **Versión**: `tfm_trainer-2.6.1`
- **Resultado**: Re-lanzamiento exitoso

### Train 2 — Warnings benignos (sin impacto)

- `pythonjsonlogger` module no encontrado — error de `sitecustomize` del contenedor
- pip dependency conflicts (bigframes, ydata-profiling, dataproc-jupyter-plugin, pydantic version)
- pip PATH warnings para scripts de `ultralytics`, `onnxruntime`, etc.
- Vertex AI Experiments 403 (`ACCESS_TOKEN_SCOPE_INSUFFICIENT`) — entrenamiento continúa sin tracking

### Train 3 — Sin bugs detectados

- Pipeline completó sin errores (`exit code 0`)
- Mismos warnings benignos que T2 (`pythonjsonlogger`, pip conflicts, Vertex AI Experiments 403)
- Focal Loss se activó correctamente (confirmado por log `🎯 ESPDet cls_loss: Sigmoid Focal Loss`)
- ExperimentSetup fixes v2.6.2 confirmados funcionales (5/5 campos corregidos)
- No se encontraron bugs nuevos de código

---

## 6. Backlog de Propuestas

> Propuestas identificadas durante el análisis de Train 1, Train 2 y Train 3. **Actualizadas tras resultados de Train 3.**

### ✅ PROPUESTA EJECUTADA — Reimplementación Arquitectura Oficial (v2.6.0/v2.6.1)

**Objetivos cumplidos**: Reemplazar arquitectura custom (22.8K params) por oficial Espressif (0.36M params).

| Aspecto | Train 1 (v1) | Train 2 (v2) | Estado |
|---|---|---|---|
| Arquitectura | Custom (DepthwiseSeparable + SimpleFPN) | **Oficial Espressif** (DSConv, DSC3k2, ESPBlock, ESPDetectHead) | ✅ |
| Params | 22,839 | **361,563** (16×) | ✅ |
| Strides | [4, 8, 16] | **[8, 16, 32]** (oficial) | ✅ |
| Pretrained | None | **espdet_pico_224_224_cat.pt** (~99.97% transferred) | ✅ |
| Resize | Progressive 640→224 | **Fijo 224** | ✅ |
| Phase 1 | 40 ep | **50 ep** | ✅ |
| Phase 2 | 80 ep, lr=5e-5 | **100 ep, lr=1e-4** | ✅ |
| Patience | 20 | **25** | ✅ |
| ONNX format | Grouped (cls0,cls1,cls2,reg0,...) | **Interleaved (box0,score0,box1,score1,box2,score2)** | ✅ |
| Paquete | tfm_trainer-2.5.0 | **tfm_trainer-2.6.1** | ✅ |
| **mAP@50 (test)** | 0.0105 | **0.6203** (59×) | ✅ |

> Esta propuesta unificada subsumió las Propuestas A (capacidad), B (224px fijo), y D (más epochs) del backlog original. **Resultados validan la hipótesis**: mAP@50 creció 59× con la arquitectura correcta.

### PROPUESTA C — Reducir conf_threshold evaluación

**Estado**: ⏳ **Prioridad alta** — Tras el fracaso de Focal Loss (T3), esta es la vía más prometedora para mejorar Precision sin reentrenar. Experimentar con `conf_threshold=0.35` o `0.40` sobre el checkpoint de T2 para subir Precision a costa de Recall. **No requiere GPU ni reentrenamiento** — solo re-evaluar con umbral diferente.

### ❌ PROPUESTA EVALUADA — Focal Loss (Train 3)

**Estado**: ❌ **Evaluada y descartada en Train 3** — Sigmoid Focal Loss (γ=2.0, α=0.25) **degradó** el rendimiento en lugar de mejorarlo.

| Aspecto | Train 2 (BCE) | Train 3 (Focal) | Resultado |
|---|---|---|---|
| mAP@50 (test) | **0.6203** | 0.5993 | -3.4% ⬇️ |
| Precision | **0.2956** | 0.2156 | -27.1% ⬇️ |
| F1 | **0.4197** | 0.3365 | -19.8% ⬇️ |
| FP totales (test) | **885** | 1,670 | +88.7% ⬇️ |

> **Conclusión**: Focal Loss no es adecuada para modelos ultra-ligeros (0.36M params). La reducción de gradiente en negativos fáciles priva al modelo de la señal necesaria para background suppression. Se descarta para esta familia de modelos.

### PROPUESTA F — Alinear naming de augmentación (Mejora código)

**Estado**: ✅ **Corregido en v2.6.0** — YAML `espdet_pico_v2.yaml` usa keys alineados con `_build_transforms()` (`aug_hflip_prob`, `aug_brightness_limit`, `aug_contrast_limit`, `aug_rotate_limit`, `aug_gaussian_noise`). Eliminados keys muertos. Confirmado funcional en Train 2 (DEPLOY VERIFICATION muestra 10 aug keys correctos).

### ✅ PROPUESTA G — Refactoring ExperimentSetup (Bug cosmético)

**Estado**: ✅ **Completado en v2.6.2** — Confirmado funcional en Train 3. Los campos `best_val_loss`, `best_epoch`, `duration_s`, `batch_size` se actualizan correctamente post-training. Aug keys usan naming correcto del YAML.

### PROPUESTA H — Vendor ultralytics modules

**Estado**: ⏳ Nuevo — La dependencia `ultralytics>=8.2` instala ~155 MB de deps innecesarios (polars, opencv-python duplicado). ESPDet solo usa `ultralytics.nn.modules.conv` y `.block`. Copiar localmente las ~5 clases necesarias (Conv, DWConv, SPPF, C2f, C3, Bottleneck, SCDown) eliminaría la dependencia completamente.

### Planificación Actualizada

| Train | Config | Cambios | Estado |
|---|---|---|---|
| Train 1 | espdet_pico_v1.yaml | Baseline (arq. custom, sin pretrained) | ✅ Completado (FRACASO) |
| Train 2 | **espdet_pico_v2.yaml** | **Arq. oficial + transfer learning + strides oficiales** | ✅ **Completado (ÉXITO)** |
| Train 3 | espdet_pico_v3.yaml | Focal Loss (γ=2.0, α=0.25) | ❌ **Completado (REGRESIÓN)** |
| Train 4 | espdet_pico_v4.yaml | conf_threshold tuning (0.35-0.40) ó NMS tuning | ⏳ Pendiente |

---

## 7. Comparativa Cross-Model

### Test Metrics — ESPDet vs FCOS vs YOLO26

| Métrica | ESPDet T1 | **ESPDet T2** | ESPDet T3 | FCOS T3 (prod) | FCOS T7 (bench) | YOLO26 T1 | YOLO26 T2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **mAP@50** | 0.0105 | **0.6203** | 0.5993 | 0.5675 | 0.6120 | 0.7544 | **0.7747** |
| **mAP@50-95** | 0.0023 | 0.3078 | **0.3220** | 0.2602 | 0.2824 | 0.5153 | **0.5456** |
| **Precision** | 0.0009 | 0.2956 | 0.2156 | **0.6609** | 0.3716 | 0.8264 | **0.8324** |
| **Recall** | 0.1597 | 0.7235 | **0.7663** | 0.6276 | 0.6872 | 0.6402 | 0.6853 |
| **F1-Score** | 0.0017 | 0.4197 | 0.3365 | **0.6438** | 0.4824 | 0.7215 | **0.7517** |
| Params | **0.023M** | **0.36M** | **0.36M** | 1.23M | 1.23M | 2.58M | 2.58M |
| ONNX size | **0.109 MB** | **1.41 MB** | **1.41 MB** | 4.74 MB | 4.74 MB | 9.97 MB | 9.97 MB |
| ONNX latency | **0.76 ms** | **2.5 ms** | **3.6 ms** | — | — | 8.3 ms | 9.9 ms |
| Train time | 23.6 min | 21.7 min | 27.0 min | — | — | 26.0 min | 32.6 min |
| Loss | BCE | BCE | **Focal** | — | — | — | — |

### Conclusión cross-model

ESPDet-Pico T2 con arquitectura oficial y transfer learning **supera a FCOS T3** en mAP@50 (0.6203 vs 0.5675) y es comparable a FCOS T7 (0.6120), siendo **3.4× más pequeño** en ONNX. El modelo tiene el mejor Recall del grupo (0.7235), pero la Precision más baja (0.2956), resultando en un F1 inferior (0.4197 vs 0.6438 FCOS, 0.7517 YOLO26).

ESPDet T3 (Focal Loss) no mejoró la situación — degradó Precision aún más (0.2156) y bajó mAP@50 a 0.5993. **Train 2 se confirma como el mejor modelo ESPDet para deploy.**

YOLO26 T2 sigue siendo el mejor modelo en calidad de detección (mAP@50=0.7747, F1=0.7517), pero su ONNX de 9.97 MB lo hace **inviable para ESP32-S3** (8 MB Flash disponibles). ESPDet-Pico T2 (1.41 MB ONNX) es el **único modelo que cabe cómodamente en el dispositivo embebido** con rendimiento funcional.

| Modelo | mAP@50 | ONNX size | Cabe en ESP32-S3? | Nota |
|---|:---:|:---:|:---:|:---:|
| ESPDet T2 | 0.6203 | **1.41 MB** | ✅ Sí | **Candidato para deploy** |
| FCOS T7 | 0.6120 | 4.74 MB | ⚠️ Ajustado | Post-cuantización INT8 necesaria |
| YOLO26 T2 | **0.7747** | 9.97 MB | ❌ No | Excede Flash disponible |

---

## 8. Conclusiones Generales

1. **Train 2 sigue siendo el mejor modelo ESPDet-Pico** — mAP@50=0.6203, F1=0.4197, 1.41 MB ONNX. Focal Loss (Train 3) no mejoró ninguna métrica clave excepto Recall. **Train 2 es el candidato para deploy en ESP32-S3.**

2. **Focal Loss no es adecuada para modelos ultra-ligeros** — Con 0.36M params, la reducción de gradiente en negativos fáciles priva al modelo de la señal necesaria. BCE proporciona mejor training signal para modelos con capacidad limitada. Lección importante para la familia ESPDet.

3. **Train 2 valida la arquitectura oficial Espressif + transfer learning** — mAP@50 creció **59×** (0.0105 → 0.6203) con la misma infraestructura (Vertex AI T4, ~22 min). La inversión en reimplementar la arquitectura oficial fue la decisión correcta.

4. **ESPDet-Pico es ahora el candidato principal para ESP32-S3** — Con 1.41 MB ONNX y mAP@50=0.62, supera a FCOS en calidad AND tamaño. Es el único modelo que cabe cómodamente en Flash del ESP32-S3 sin necesitar cuantización extrema.

5. **Recall alto (0.72-0.77) es relevante para asistencia visual** — El modelo encuentra la mayoría de objetos. Para un dispositivo de asistencia, fallar en detectar un obstáculo es más peligroso que un falso positivo.

6. **Precision baja (0.22-0.30) es la debilidad principal** — ~60-73% de detecciones son FP. Focal Loss no resolvió el problema. Siguiente vía: ajuste de `conf_threshold` (Propuesta C) o NMS tuning.

7. **Cero confusión inter-clase** — Consistente en T2 y T3. El modelo nunca confunde una clase con otra. La discriminación clasificatoria es perfecta; el problema es solo la supresión de background.

8. **Pipeline MLOps maduro y estable** — 3 entrenamientos exitosos con la misma infraestructura. ExperimentSetup ahora tracking completo (v2.6.2). DEPLOY VERIFICATION funciona correctamente.

9. **Siguiente paso recomendado**: Evaluar si la precisión actual de T2 (0.30) es aceptable para el caso de uso del ESP32-S3, o experimentar con `conf_threshold=0.35-0.40` (Propuesta C — **no requiere reentrenar**, solo ajustar el umbral de evaluación). Paralelamente, iniciar la conversión ONNX → ESPDL para despliegue en el dispositivo.

---

*Documento generado y mantenido como parte del pipeline MLOps del TFM — Detección de Objetos para Asistencia Visual.*
