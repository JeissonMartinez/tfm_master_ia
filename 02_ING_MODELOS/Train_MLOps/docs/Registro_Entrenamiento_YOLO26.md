# Registro de Entrenamiento — YOLO26 Custom (Ultralytics 2-Phase)

> **Modelo**: `yolo26n_custom` — YOLO26n (nano) via Ultralytics con entrenamiento en 2 fases  
> **Parámetros**: ~2.6M (FP32: ~10 MB | ONNX est.: ~6 MB)  
> **Dataset**: IODC YOLO — 5 clases (dog, door, obstacle, person, stair)  
> **Splits**: Train 1470 | Val 188 | Test 187  
> **Infraestructura**: Google Vertex AI Custom Job — `n1-standard-8` + NVIDIA Tesla T4  
> **Contenedor**: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`  
> **Entry-point**: `trainer.task_yolo26_custom`  
> **Paquete base**: `tfm_trainer-2.4.0.tar.gz`  
> **Última actualización**: 22 de febrero de 2026  

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Configuración Base (Compartida)](#2-configuración-base-compartida)
3. [Train 1 — Baseline](#3-train-1--baseline)
4. [Train 2 — MuSGD Optimizer](#4-train-2--musgd-optimizer)
5. [Backlog de Propuestas](#5-backlog-de-propuestas)
6. [Comparativa Global](#comparativa-global)
7. [Conclusiones Generales](#conclusiones-generales)

---

## 1. Resumen Ejecutivo

| Métrica (Test) | Train 1 | Train 2 |
|---|:---:|:---:|
| **mAP@50** | **0.7544** | — |
| **mAP@50-95** | **0.5153** | — |
| **Precision** | 0.8264 | — |
| **Recall** | 0.6402 | — |
| **F1-Score** | **0.7215** | — |
| Épocas (P1+P2) | 100 (30+70) | 100 (30+70) |
| Optimizer | auto→AdamW | **MuSGD** |
| conf_threshold | 0.25 | 0.15 |
| Tiempo | 26.0 min | — |
| Inferencia | 2.6 ms | — |

> Tabla actualizada tras cada entrenamiento exitoso.

---

## 2. Configuración Base (Compartida)

Parámetros comunes a todos los entrenamientos de esta serie, salvo modificación explícita.

### 2.1 Arquitectura

| Parámetro | Valor |
|---|---|
| Base model | YOLO11n (via Ultralytics) |
| Pretrained | COCO (`yolo11n.pt`) |
| Params | ~2.6M |
| Input size (training) | 640 |
| Input size (export) | 224 |
| ONNX opset | 13 |
| Clases | 5 (dog, door, obstacle, person, stair) |

### 2.2 Estrategia de 2 Fases

| Parámetro | Phase 1 (Freeze) | Phase 2 (Full) |
|---|---|---|
| Epochs | 30 | 70 |
| Freeze layers | 10 | 0 |
| LR0 | 0.01 | 0.001 |
| LRf | 0.01 | 0.001 |
| Cosine LR | Sí | Sí |

**Total: 100 epochs (30 + 70), patience=30**

### 2.3 Optimización

| Parámetro | Valor |
|---|---|
| Optimizer | auto (Ultralytics) |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| Warmup momentum | 0.8 |
| Warmup bias LR | 0.1 |
| AMP | True |
| Batch size | 16 |

### 2.4 Augmentación (Ultralytics Built-in)

| Param | Valor | Descripción |
|---|---|---|
| mosaic | 1.0 | Mosaic augmentation |
| mixup | 0.1 | MixUp alpha |
| close_mosaic | 10 | Disable mosaic last N epochs |
| hsv_h | 0.015 | Hue shift |
| hsv_s | 0.7 | Saturation shift |
| hsv_v | 0.4 | Value shift |
| fliplr | 0.5 | Horizontal flip |
| scale | 0.5 | Scale factor |
| translate | 0.1 | Translation |
| degrees | 0.0 | Rotation |
| shear | 0.0 | Shear |
| perspective | 0.0 | Perspective |
| copy_paste | 0.0 | Copy-paste |
| erasing | 0.0 | Random erasing |

### 2.5 Loss Weights

| Componente | Peso |
|---|---|
| Box (CIoU) | 7.5 |
| Cls (BCE) | 0.5 |

### 2.6 Inferencia / Evaluación

| Parámetro | Valor |
|---|---|
| conf_threshold | 0.25 |
| iou_threshold | 0.45 |
| max_det | 300 |

### 2.7 DDP Cleanup

Vertex AI inyecta variables de entorno DDP que rompen Ultralytics en single-GPU (lección Ciclo 1). Limpieza aplicada en el entry-point:

```python
for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
    os.environ.pop(var, None)
```

### 2.8 Protecciones Aplicadas (Lecciones FCOS)

| Lección FCOS | Aplicación en YOLO26 |
|---|---|
| T6 — pip cache | Version bump a `tfm_trainer-2.3.0` |
| T7/T8 — whitelist config_loader | Ya corregido en v2.2.0 (todas las claves YAML pasan) |
| T8 — launch_job hardcoded | `build_and_launch.sh` pasa `--package-uri` dinámico |
| T8 — DEPLOY VERIFICATION | Bloque de verificación en Bloque 3 del entry-point |
| T8 — `log()` vs `print()` | Entry-point usa exclusivamente `print()` |

---

## 3. Train 1 — Baseline

### 3.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `1614616246815293440` |
| **Fecha** | 22 de febrero de 2026 |
| **Paquete** | `tfm_trainer-2.3.0.tar.gz` |
| **Config YAML** | `yolo26n_custom_v1.yaml` → `yolo26n_custom_v1-run1.yaml` |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/yolo26n_custom_v1-run1/` |
| **Output local** | `outputs/yolo26n_custom_v1-run1/` |

### 3.2 Configuración

Baseline puro — sin cambios respecto a la configuración base (§2).

| Parámetro | Valor |
|---|---|
| Phase 1 | 30 ep, freeze=10, LR0=0.01 |
| Phase 2 | 70 ep, freeze=0, LR0=0.001 |
| Patience | 30 |
| Mosaic | 1.0, close_mosaic=10 |
| Mixup | 0.1 |
| Box/Cls weights | 7.5 / 0.5 |
| conf_threshold | 0.25 |

### 3.3 Entrenamiento

- **Épocas completadas**: 100 (30 Phase 1 + 70 Phase 2, sin early stopping)
- **Mejor val mAP@50 (durante training, conf=0.001)**: 0.789 (epoch 16, Phase 1) / 0.774 (epoch 70, Phase 2)
- **Mejor val mAP@50 (best.pt, conf=0.001)**: 0.788 (Phase 1 best)
- **Tiempo total**: 26.0 min (Phase 1: ~6 min, Phase 2: ~19 min)
- **GPU memory peak**: 2.71 GB
- **Observaciones**:
  - Phase 2 NO superó a Phase 1 en mAP@50 durante training (0.771 vs 0.788; best.pt)
  - Optimizer auto → AdamW(lr=0.001111) para Phase 2
  - Phase 1 congeló 10 layers; Phase 2 solo congeló `model.23.dfl.conv.weight` (auto Ultralytics)

### 3.4 Resultados — Validación

| Métrica | Valor |
|---|---|
| mAP@50 | **0.6556** |
| mAP@50-95 | 0.4298 |
| Precision | 0.8215 |
| Recall | 0.4556 |
| F1-Score | 0.5861 |
| Inferencia | 2.8 ms |

**Per-class AP@50 (Val):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.6759 | 0.7411 | 0.5533 | 0.6336 |
| door | 0.6282 | 0.8769 | 0.3563 | 0.5067 |
| obstacle | 0.6071 | 0.7117 | 0.4817 | 0.5745 |
| person | 0.7075 | 0.8667 | 0.5000 | 0.6341 |
| stair | 0.6590 | 0.9111 | 0.3868 | 0.5430 |

### 3.5 Resultados — Test

| Métrica | Valor |
|---|---|
| mAP@50 | **0.7544** |
| mAP@50-95 | **0.5153** |
| Precision | 0.8264 |
| Recall | 0.6402 |
| F1-Score | **0.7215** |
| Inferencia | 2.6 ms |

**Per-class AP@50 (Test):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.7740 | 0.7692 | 0.6897 | 0.7273 |
| door | 0.6531 | 0.7952 | 0.4853 | 0.6027 |
| obstacle | 0.7011 | 0.7609 | 0.6069 | 0.6752 |
| person | 0.8301 | 0.8837 | 0.7525 | 0.8128 |
| stair | 0.8138 | 0.9231 | 0.6667 | 0.7742 |

**Confusion Matrix (Test):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 40 | 1 | 0 | 6 | 0 | 5 |
| **door** | 0 | 65 | 1 | 0 | 1 | 16 |
| **obstacle** | 0 | 3 | 108 | 1 | 1 | 25 |
| **person** | 1 | 1 | 2 | 73 | 0 | 9 |
| **stair** | 0 | 1 | 0 | 0 | 72 | 5 |
| **FP (bkg)** | 17 | 65 | 62 | 21 | 34 | — |

### 3.6 Export ONNX

| Parámetro | Valor |
|---|---|
| ONNX size | 9.97 MB |
| ONNX valid | ✅ |
| ONNX latency | 8.3 ms (CPU, OnnxRuntime) |
| Input shape | (1, 3, 224, 224) |
| Output shape | (1, 9, 1029) |
| Opset | 13 |

### 3.7 Análisis

**Resultado general: EXCELENTE — superó ampliamente al FCOS en todas las métricas clave.**

1. **mAP@50 (test) = 0.7544** — Supera al FCOS T3 producción (+33%) y FCOS T7 benchmark (+23%). Primer entrenamiento ya supera el objetivo (>0.50 val / >0.45 test) por un margen muy amplio.

2. **Precision alta (0.826) pero Recall moderado (0.640)** — El modelo es conservador: cuando detecta, acierta, pero pierde ~36% de los objetos reales. El `conf_threshold=0.25` contribuye a este comportamiento.

3. **Mejor clase: person** (AP@50=0.830, F1=0.813) — Beneficiada por el pretraining COCO (clase mayoritaria en COCO). **Peor clase: door** (AP@50=0.653, F1=0.603) — Recall=0.485, muchos FN (65 FP desde background).

4. **Val vs Test gap**: Val mAP@50=0.656 < Test mAP@50=0.754. Inusual (test > val). Posible causa: diferente distribución en los splits, o que val tiene algunos ejemplos más difíciles. Requiere vigilancia en entrenamientos futuros.

5. **Phase 2 NO mejoró Phase 1** en mAP@50 durante training (0.771 vs 0.788 best.pt). Posibles causas: (a) el backbone ya estaba suficientemente adaptado en 30 epochs con backbone congelado, (b) LR0=0.001 más el auto-optimizer (AdamW lr=0.001111) puede ser demasiado agresivo para fine-tuning.

6. **Discrepancia training vs evaluation**: Durante training, best.pt reporta mAP@50=0.788 (conf=0.001), pero la evaluación final con conf=0.25 da 0.656. Esto es esperado — conf=0.25 filtra detecciones de baja confianza, reduciendo recall pero aumentando precision.

### 3.8 Lecciones

1. **BUG encontrado: Curvas solo muestran Phase 1 (30 de 100 epochs)** — El código de Bloque 5 usaba `results_csvs[-1]` que tomaba solo un CSV. Phase 2 results.csv no se concatenaba ni se subía a GCS. **FIX aplicado en v2.3.0-post1**: concatenar todos los CSVs por fase y subir cada uno.

2. **Progressive Resizing panel vacío — correcto**: YOLO26 entrena a 640px fijo (Ultralytics maneja internamente). El panel existe porque `plot_training_curves()` es código compartido con FCOS. No es un bug.

3. **Phase 2 podría beneficiarse de LR más bajo**: Phase 2 no mejoró Phase 1. Para Train 2, considerar `phase2.lr0=0.0005` o `phase2.lr0=0.0001`.

4. **Recall bajo es la métrica a mejorar**: Para asistencia visual, es preferible detectar más objetos (incluso con algo de ruido) que perder detecciones. Opciones: bajar `conf_threshold` a 0.15, o aumentar augmentación (scale, translate).

5. **ONNX size (9.97 MB) es el doble que FCOS (4.74 MB)** — Esperado por el mayor número de parámetros (2.6M vs 1.23M). Para ESP32-S3 (8MB PSRAM) habrá que convertir a INT8 vía ESP-DL.

---

## Debugging / Incidentes

> Sección para registrar problemas encontrados durante el despliegue y debugging de los entrenamientos, antes de obtener un Train exitoso.

### Train 1 — Bug: Curvas incompletas (solo Phase 1)

- **Síntoma**: `training_curves.png` muestra solo 30 epochs. Resumen dice "Épocas: 30".
- **Causa raíz**: `results_csvs[-1]` tomaba un solo CSV. Phase 2 `results.csv` existía en la VM pero no se concatenaba ni subía a GCS.
- **Impacto**: Curvas no representan el entrenamiento completo (100 epochs). Métricas de training_metrics logger solo reflejan Phase 1.
- **Fix**: Modificado Bloque 4/5 en `task_yolo26_custom.py` para: (a) ordenar y concatenar todos los CSVs, (b) generar `results_combined.csv`, (c) establecer labels de fase para shading en plots, (d) subir cada CSV de fase individual a GCS.
- **Versión**: Fix incluido en próximo build (`tfm_trainer-2.4.0`)

### Train 1 — Warnings benignos (no impacto)

- `pythonjsonlogger` module no encontrado — error de `sitecustomize` del contenedor, sin impacto
- pip dependency conflicts (bigframes, ydata-profiling, dataproc-jupyter-plugin) — paquetes del contenedor base, sin impacto en el entrenamiento
- `WARNING: Retry 1/2 failed: 'ascii' codec` durante ONNX export — retry automático exitoso

---

## 4. Train 2 — MuSGD Optimizer

**Propuesta**: A — Cambio de optimizer a MuSGD (nativo YOLO26)

### 4.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | *(pendiente)* |
| **Fecha** | *(pendiente)* |
| **Paquete** | `tfm_trainer-2.4.0.tar.gz` |
| **Config YAML** | `yolo26n_custom_v2.yaml` |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/yolo26n_custom_v2-run1/` |
| **Output local** | `outputs/yolo26n_custom_v2-run1/` |

### 4.2 Cambios vs Train 1

| Parámetro | Train 1 (v1) | Train 2 (v2) | Justificación |
|---|---|---|---|
| `optimizer` | `auto` (→AdamW) | **`MuSGD`** | Optimizer nativo YOLO26 (SGD+Muon). Con `auto`, nuestro dataset (1470 imgs) produce <10k iterations → siempre elige AdamW, ignorando lr0/momentum |
| `momentum` | 0.937 (ignorado) | **0.9** | Valor que usa `auto` cuando elige MuSGD |
| `phase2_lr0` | 0.001 (ignorado) | **0.0005** | Más conservador para fine-tuning (Phase 2 no mejoró Phase 1 en T1) |
| `phase2_lrf` | 0.001 | **0.01** | Ajustado al lr0 más bajo (final_lr = 0.0005×0.01 = 5e-6 vs 0.0005×0.001 = 5e-7) |
| `warmup_bias_lr` | 0.1 | **0.0** | ⚠️ CRÍTICO: con 0.1 y lr0=0.01, biases arrancarían a 10× del target LR. `auto` siempre fuerza 0.0 |
| `conf_threshold` | 0.25 | **0.15** | Mejorar recall (0.640 en T1): más detecciones, menos FN |

### 4.3 Análisis de Riesgo — MuSGD

**Flujo del optimizer** (verificado en Ultralytics `engine/trainer.py`):

1. Cuando `optimizer != "auto"`, Ultralytics **usa directamente** los `lr0` y `momentum` configurados
2. MuSGD clasifica parámetros en 4 grupos:
   - **Muon** (ndim≥2): Pesos convolucionales — ortogonalización matricial (20% Muon + 80% SGD)
   - **Weight+decay**: Otros pesos con weight_decay
   - **BN**: BatchNorm (sin decay)
   - **Bias**: Sesgos (sin decay)
3. Head params (`.23.cv3`, `proto.semseg`) reciben **3× LR** automáticamente
4. Phase 2 carga `phase1/best.pt` → crea optimizer nuevo (no resume), correcto

**Riesgo identificado y mitigado**: `warmup_bias_lr=0.0` (en T1 era 0.1, pero `auto` lo overrideaba a 0.0; con MuSGD explícito ya no hay override automático).

### 4.4 Qué se espera comparar

| Aspecto | Hipótesis |
|---|---|
| mAP@50 | Mejora +3-5% por convergencia más estable (Muon ortogonalización) |
| Phase 2 vs Phase 1 | Phase 2 debería mejorar ahora (LR verdaderamente diferente: 0.01→0.0005) |
| Recall | Mejora significativa con conf_threshold=0.15 |
| Precision | Posible caída leve (más detecciones → más FP) |
| Tiempo | Similar (~26 min). MuSGD puede ser ligeramente más lento |

### 4.5 Entrenamiento

*(Se completará tras ejecución)*

### 4.6 Resultados — Validación

*(Se completará tras ejecución)*

### 4.7 Resultados — Test

*(Se completará tras ejecución)*

### 4.8 Export ONNX

*(Se completará tras ejecución)*

### 4.9 Análisis

*(Se completará tras ejecución)*

### 4.10 Lecciones

*(Se completará tras ejecución)*

---

## 5. Backlog de Propuestas

> Propuestas identificadas durante el análisis de Train 1, pendientes de implementar en futuros entrenamientos.

### PROPUESTA B — Augmentación Reforzada (Pendiente)

**Objetivo**: Mejorar generalización, especialmente en door (AP@50=0.653) y obstacle (0.701).

| Parámetro | Actual (v1/v2) | Propuesto | Justificación |
|---|---|---|---|
| `degrees` | 0.0 | **10.0** | Rotación leve, robustez angular |
| `shear` | 0.0 | **2.0** | Perspectiva leve |
| `perspective` | 0.0 | **0.0005** | Variación 3D sutil |
| `flipud` | 0.0 | **0.1** | 10% flips verticales |
| `erasing` | 0.0 | **0.2** | Random erasing → forzar features alternativas |
| `mixup` | 0.1 | **0.15** | Ligero aumento |
| `multi_scale` | *(no incluido)* | **0.5** | ±50% imgsz por batch. Requiere añadir campo a `Yolo26CustomConfig` |

**Prerequisito**: Requiere que `Yolo26CustomConfig` soporte el campo `multi_scale` (actualmente no está en el dataclass).

### PROPUESTA C — Loss Balance + Paciencia (Pendiente)

**Objetivo**: Reducir confusión door/obstacle y dar más tiempo de convergencia.

| Parámetro | Actual (v1/v2) | Propuesto | Justificación |
|---|---|---|---|
| `cls` | 0.5 | **1.0** | Mayor peso clasificación → reducir confusión door/obstacle |
| `patience` | 30 | **40** | MuSGD puede necesitar más epochs para converger |
| `close_mosaic` | 10 | **15** | Más epochs sin mosaic para estabilizar |

### Planificación

| Train | Propuestas | Estado |
|---|---|---|
| Train 1 | Baseline (ninguna) | ✅ Completado |
| Train 2 | **A** — MuSGD Optimizer | 🔄 Preparado |
| Train 3 | A + **B** — Augmentación | ⏳ Pendiente |
| Train 4 | A + B + **C** — Loss Balance | ⏳ Pendiente |

> Esta planificación es orientativa. Se ajustará según los resultados de cada Train.

---

## Comparativa Global

*(Se actualizará cuando existan ≥2 entrenamientos)*

---

## Conclusiones Generales

*(Se actualizará al finalizar la serie de entrenamientos)*

---

## Referencia Cruzada — FCOS vs YOLO26

| Métrica (Test) | FCOS T3 (producción) | FCOS T7 (benchmark) | YOLO26 T1 |
|---|:---:|:---:|:---:|
| mAP@50 | 0.5675 | 0.6120 | **0.7544** (+33%/+23%) |
| mAP@50-95 | 0.2602 | 0.2824 | **0.5153** (+98%/+82%) |
| Precision | 0.6609 | 0.3716 | **0.8264** |
| Recall | 0.6276 | **0.6872** | 0.6402 |
| F1-Score | 0.6438 | 0.4824 | **0.7215** (+12%) |
| Params | 1.23M | 1.23M | 2.58M |
| ONNX size | 4.74 MB | 4.74 MB | 9.97 MB |
| Inference (GPU) | — | — | 2.6 ms |

> **Conclusión**: YOLO26 T1 baseline supera al mejor FCOS en todas las métricas salvo recall (vs T7). El trade-off es el doble de parámetros y tamaño ONNX. Para ESP32-S3, la cuantización INT8 será crítica.

---

*Documento generado y mantenido como parte del pipeline MLOps del TFM — Detección de Objetos para Asistencia Visual.*
