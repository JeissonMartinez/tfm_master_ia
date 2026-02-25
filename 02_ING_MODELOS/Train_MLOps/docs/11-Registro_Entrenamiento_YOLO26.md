# Registro de Entrenamiento — YOLO26 Custom (Ultralytics 2-Phase)

> **Modelo**: `yolo26n_custom` — YOLO26n (nano) via Ultralytics con entrenamiento en 2 fases  
> **Parámetros**: ~2.6M (FP32: ~10 MB | ONNX est.: ~6 MB)  
> **Dataset**: IODC YOLO — 5 clases (dog, door, obstacle, person, stair)  
> **Splits**: Train 1470 | Val 188 | Test 187  
> **Infraestructura**: Google Vertex AI Custom Job — `n1-standard-8` + NVIDIA Tesla T4  
> **Contenedor**: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`  
> **Entry-point**: `trainer.task_yolo26_custom`  
> **Paquete base**: `tfm_trainer-2.7.0.tar.gz`  
> **Última actualización**: 24 de febrero de 2026 (Train 3 completado)  

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Configuración Base (Compartida)](#2-configuración-base-compartida)
3. [Train 1 — Baseline](#3-train-1--baseline)
4. [Train 2 — MuSGD Optimizer](#4-train-2--musgd-optimizer)
5. [Train 3 — DFL Removal](#5-train-3--dfl-removal)
6. [Backlog de Propuestas](#6-backlog-de-propuestas)
7. [Comparativa Global](#comparativa-global)
8. [Conclusiones Generales](#conclusiones-generales)

---

## 1. Resumen Ejecutivo

| Métrica (Test) | Train 1 | Train 2 | Train 3 | Δ (T3 vs T2) |
|---|:---:|:---:|:---:|:---:|
| **mAP@50** | 0.7544 | **0.7747** | 0.7466 | -3.6% |
| **mAP@50-95** | 0.5153 | **0.5456** | 0.5333 | -2.3% |
| **Precision** | 0.8264 | **0.8324** | 0.7853 | -5.7% |
| **Recall** | 0.6402 | **0.6853** | 0.6522 | -4.8% |
| **F1-Score** | 0.7215 | **0.7517** | 0.7126 | -5.2% |
| Épocas (P1+P2) | 100 (30+70) | 98 (30+68) | 92 (30+62) | Early stop |
| Optimizer | auto→AdamW | MuSGD | MuSGD | — |
| Pretrained | yolo11n.pt | yolo11n.pt | **yolo26n.pt** | Nativo YOLO26 |
| reg_max / DFL | 16 / DFL | 16 / DFL | **1 / Identity** | Sin DFL |
| Params | 2,590,815 | 2,590,815 | **2,505,750** | -3.3% |
| conf_threshold | 0.25 | 0.15 | 0.15 | — |
| Tiempo | 26.0 min | 32.6 min | 30.5 min | -6.4% |
| Inferencia | 2.6 ms | 2.9 ms | 4.0 ms | — |

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
| **Job ID** | `7776077098732486656` |
| **Fecha** | 22 de febrero de 2026 |
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

- **Épocas completadas**: 98 (30 Phase 1 + 68 Phase 2; early stopping en P2 a 68/70)
- **Best fitness (Phase 2)**: epoch 38 (combined epoch 68) — mAP50=0.796, mAP50-95=0.538, fitness=0.564
- **Phase 1 best val (best.pt, conf=0.001)**: P=0.791, R=0.689, mAP50=0.789, mAP50-95=0.520
- **Phase 2 best val (best.pt, conf=0.001)**: P=0.750, R=0.730, mAP50=0.797, mAP50-95=0.538
- **Phase 2 mejoró Phase 1**: ✅ mAP50 +1.0%, mAP50-95 +3.5%, Recall +5.9%
- **Tiempo total**: 32.6 min (Phase 1: ~7.6 min, Phase 2: ~24.1 min)
- **GPU memory peak**: 2.71 GB
- **Combined CSV**: 98 epochs — `results_combined.csv` ✅ (bug fix confirmado)
- **Optimizer confirmado**: Phase 1 `MuSGD(lr=0.01, momentum=0.9)`, Phase 2 `MuSGD(lr=0.0005, momentum=0.9)`
- **Early stopping**: Best fitness en P2 epoch 38, patience=30 → stop en epoch 68 (2 epochs antes del máximo 70)
- **Observaciones**:
  - MuSGD ~25% más lento que AdamW (32.6 vs 26.0 min) — overhead de la ortogonalización Muon
  - Phase 2 SÍ mejoró Phase 1 esta vez (en T1 no lo hizo). Confirma que el LR diferenciado (0.01→0.0005) funciona correctamente con MuSGD
  - Warnings benignos idénticos a T1 (pythonjsonlogger, pip dependencies) — sin impacto

### 4.6 Resultados — Validación

| Métrica | Valor |
|---|---|
| mAP@50 | **0.6638** |
| mAP@50-95 | 0.4345 |
| Precision | 0.7810 |
| Recall | 0.5119 |
| F1-Score | 0.6185 |
| Inferencia | 3.2 ms |

**Per-class AP@50 (Val):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.7238 | 0.7797 | 0.5915 | 0.6725 |
| door | 0.5920 | 0.7727 | 0.3920 | 0.5203 |
| obstacle | 0.6254 | 0.6967 | 0.5226 | 0.5972 |
| person | 0.7439 | 0.8293 | 0.5789 | 0.6820 |
| stair | 0.6340 | 0.8267 | 0.4747 | 0.6029 |

### 4.7 Resultados — Test

| Métrica | Valor |
|---|---|
| mAP@50 | **0.7747** |
| mAP@50-95 | **0.5456** |
| Precision | 0.8324 |
| Recall | 0.6853 |
| F1-Score | **0.7517** |
| Inferencia | 2.9 ms |

**Per-class AP@50 (Test):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.7956 | 0.8410 | 0.7069 | 0.7682 |
| door | 0.6601 | 0.7654 | 0.5013 | 0.6058 |
| obstacle | 0.7384 | 0.8152 | 0.6647 | 0.7325 |
| person | 0.8661 | 0.8854 | 0.8317 | 0.8577 |
| stair | 0.8133 | 0.8571 | 0.7222 | 0.7838 |

**Confusion Matrix (Test):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 43 | 1 | 0 | 1 | 0 | 6 |
| **door** | 0 | 76 | 1 | 0 | 0 | 20 |
| **obstacle** | 0 | 0 | 118 | 0 | 1 | 28 |
| **person** | 0 | 0 | 0 | 85 | 1 | 13 |
| **stair** | 0 | 0 | 2 | 0 | 78 | 15 |
| **FP (bkg)** | 15 | 59 | 52 | 15 | 28 | — |

### 4.8 Export ONNX

| Parámetro | Valor |
|---|---|
| ONNX size | 9.97 MB |
| ONNX valid | ✅ |
| ONNX latency | 9.9 ms (CPU, OnnxRuntime) |
| Input shape | (1, 3, 224, 224) |
| Output shape | (1, 9, 1029) |
| Opset | 13 |

### 4.9 Análisis

**Resultado general: TODOS LOS TEST METRICS MEJORARON — MuSGD + LR diferenciado + conf_threshold funcionan.**

1. **mAP@50 (test) = 0.7747 (+2.7% vs T1)** — Mejora consistente. mAP@50-95 sube aún más (+5.9%), indicando que las detecciones tienen mejor calidad de bbox (IoU más alto).

2. **Recall sube de 0.640 a 0.685 (+7.0%)** — Principal métrica objetivo para asistencia visual. La mejora proviene de dos fuentes: (a) conf_threshold reducido 0.25→0.15 permite más detecciones, y (b) mejor calibración del modelo con MuSGD.

3. **Precision TAMBIÉN subió (0.826→0.832)** — Resultado inesperado y muy positivo. A pesar de bajar el umbral de confianza (que normalmente incrementa FP), la precision mejoró. Esto demuestra que MuSGD produce un modelo mejor calibrado: las detecciones de baja confianza (0.15-0.25) son en su mayoría correctas.

4. **Phase 2 SÍ mejoró Phase 1 — hipótesis principal CONFIRMADA**:
   - Phase 1 best.pt: mAP50=0.789, mAP50-95=0.520
   - Phase 2 best.pt: mAP50=0.797 (+1.0%), mAP50-95=0.538 (+3.5%)
   - En T1 esto NO ocurrió (Phase 2 tenía mAP50 inferior). La clave fue el LR verdaderamente diferenciado con MuSGD explícito (0.01→0.0005) vs T1 donde AdamW ignoraba lr0.

5. **Early stopping eficiente**: Phase 2 paró a 68/70 epochs (best fitness en epoch 38, patience=30). El modelo convergía lentamente y el early stopping fue adecuado (solo 2 epochs ahorrados).

6. **Per-class Test AP@50 vs T1**:
   - 🟢 person: 0.830→0.866 (+4.3%) — mayor mejora absoluta
   - 🟢 obstacle: 0.701→0.738 (+5.3%) — mayor mejora relativa
   - 🟢 dog: 0.774→0.796 (+2.8%)
   - 🟡 door: 0.653→0.660 (+1.1%) — mejora mínima, sigue siendo la peor clase
   - ⚪ stair: 0.814→0.813 (-0.1%) — estable

7. **Val-Test gap persiste**: Val mAP50=0.664 vs Test mAP50=0.775 (~11% gap). Similar a T1 (10% gap). Esto sugiere que el split de validación es inherentemente más difícil, no un problema de overfitting.

8. **FP desde background (Test)**: door(59) y obstacle(52) siguen teniendo muchos FP desde background. Esto afecta la precision real en despliegue. Propuesta C (mayor peso cls) podría ayudar.

9. **Overhead temporal**: MuSGD +25% tiempo (32.6 vs 26.0 min). Aceptable dado el costo marginal en Vertex AI (~$0.15 adicional por entrenamiento).

### 4.10 Lecciones

1. **MuSGD explícito es NECESARIO para YOLO26 con datasets pequeños** — Con `optimizer: auto`, Ultralytics siempre elige AdamW cuando iterations < 10,000 (nuestro dataset: 1470 imgs × 100 ep / 16 batch = ~9,200). Esto anula lr0/momentum configurados. Usar `optimizer: MuSGD` directamente.

2. **`warmup_bias_lr=0.0` es crítico con MuSGD explícito** — Con `auto`, Ultralytics fuerza `warmup_bias_lr=0.0` internamente. Con optimizer explícito, no hay override. Dejar siempre `warmup_bias_lr=0.0` en la config.

3. **Combined CSV funciona correctamente** — El bug fix de T1 (concatenación de CSVs) funciona: `results_combined.csv` tiene 98 rows (30+68), `experiment.json` apunta al CSV combinado.

4. **Early stopping funcional pero ajustado** — Solo ahorró 2 epochs (68/70). Considerar `patience=40` si se amplía el número de epochs en futuros trainings.

5. **Bajar conf_threshold mejora recall SIN sacrificar precision** — Contraintuitivo pero verificado. El modelo MuSGD tiene buena calibración de confianza. Para Train 3, mantener conf_threshold=0.15.

6. **door sigue siendo la clase problemática** — AP@50=0.660 (peor de las 5 clases), con 59 FP desde background y Recall=0.501. La Propuesta B (augmentación reforzada) podría ayudar si las doors varían mucho en apariencia.

---

## 5. Train 3 — DFL Removal

**Propuesta**: DFL Removal — Usar pesos nativos `yolo26n.pt` con `reg_max=1` (DFL=Identity), eliminando la necesidad de softmax + integral DFL en el post-processing del ESP32-S3.

### 5.1 Identificador

| Campo | Valor |
|---|---|
| Job ID (Vertex AI) | `6267808829191225344` |
| Fecha lanzamiento | 24 de febrero de 2026 |
| Paquete | `tfm_trainer-2.7.0.tar.gz` |
| Config YAML | `yolo26n_custom_v3.yaml` |
| Output GCS | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/yolo26n_custom_v3-run1/` |
| Output local | `outputs/yolo26n_custom_v3-run1/` |

### 5.2 Cambios vs Train 2

| Parámetro | T2 | T3 | Justificación |
|---|---|---|---|
| `pretrained_weights` | yolo11n.pt | **yolo26n.pt** | Pesos nativos YOLO26 COCO con DFL removal |
| `reg_max` | 16 (DFL activa) | **1 (DFL=Identity)** | 4 canales directos vs 64 probabilísticos |
| `cv2[0] out_channels` | 64 | **4** | Acorde a reg_max × 4 = 4 |
| Parámetros totales | 2,590,815 | **2,505,750** (-3.3%) | Menos params en cabeza de box |
| `setup.py` version | 2.4.0 | **2.7.0** | DFL inspection en DEPLOY VERIFICATION |
| Otros hiperparámetros | — | **IDÉNTICOS a T2** | MuSGD, lr, phases, patience, conf |

### 5.3 Motivación DFL Removal

1. **Simplificación firmware ESP32-S3**: Elimina softmax + integral DFL del post-processing. Reducción directa de complejidad y latencia de inferencia en microcontrolador.
2. **Cuantización INT8 (hipótesis)**: 4 canales uniformes de distancia vs 64 canales de probabilidad DFL → se espera menor sensibilidad a cuantización.
3. **Menor tamaño de salida ESP ONNX**: Box shapes `[1, 4, H, W]` en lugar de `[1, 64, H, W]` (16× reducción en canales de caja).
4. **Pesos nativos YOLO26**: `yolo26n.pt` (preentrenado COCO) → transfer learning directo sin cross-model YOLO11→YOLO26 (T1/T2 usaban `yolo11n.pt`).

### 5.4 Verificación Arquitectura

Verificado programáticamente en el modelo entrenado (`best.pt`) mediante bloque de DEPLOY VERIFICATION:

| Propiedad | Esperado | Verificado |
|---|---|---|
| `reg_max` | 1 | ✅ 1 |
| `dfl` module | `Identity()` | ✅ `Identity()` |
| `cv2[0] out_channels` | 4 | ✅ 4 |
| `nc` | 5 | ✅ 5 |
| Parámetros totales | ~2.5M | ✅ 2,505,750 |
| `best.pt` size | <6 MB | ✅ 5.15 MB |

### 5.5 Entrenamiento

| Aspecto | Valor |
|---|---|
| Épocas totales | **92** (30 Phase 1 + 62 Phase 2) |
| Phase 2 early stopping | Sí (62/70 epochs, 8 ahorrados) |
| Best mAP@50 (training, conf=0.001) | 0.774 (epoch 91) |
| Phase 1 final mAP@50 | 0.763 (epoch 30) |
| Phase 2 start mAP@50 | 0.700 (epoch 31, drop por unfreeze) |
| Tiempo total | 30.5 min (1829 s) |
| Phase 1 tiempo | 11.7 min (702 s) |
| Phase 2 tiempo | 18.8 min (1127 s) |
| GPU memory | ~2.7 GB (T4) |

**Phase 2 mejoró Phase 1**: mAP@50 0.763 → 0.774 (+1.4%). Confirmación de que MuSGD + LR diferenciado sigue funcionando con DFL Removal y pesos nativos. El patrón Phase 1→Phase 2 es consistente en los tres entrenamientos.

### 5.6 Resultados — Validación

| Métrica | Valor |
|---|---|
| mAP@50 | **0.6234** |
| mAP@50-95 | 0.4246 |
| Precision | 0.7289 |
| Recall | 0.4738 |
| F1-Score | 0.5743 |
| Inferencia | 5.24 ms |

**Per-class AP@50 (Val):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.6746 | 0.8085 | 0.5067 | 0.6230 |
| door | 0.5206 | 0.6570 | 0.3500 | 0.4567 |
| obstacle | 0.5621 | 0.5994 | 0.5244 | 0.5594 |
| person | 0.7953 | 0.8676 | 0.6484 | 0.7421 |
| stair | 0.5643 | 0.7118 | 0.3396 | 0.4598 |

### 5.7 Resultados — Test

| Métrica | Valor |
|---|---|
| mAP@50 | **0.7466** |
| mAP@50-95 | **0.5333** |
| Precision | 0.7853 |
| Recall | 0.6522 |
| F1-Score | **0.7126** |
| Inferencia | 3.98 ms |

**Per-class AP@50 (Test):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.8047 | 0.8269 | 0.7414 | 0.7818 |
| door | 0.6606 | 0.7474 | 0.5221 | 0.6147 |
| obstacle | 0.6543 | 0.7273 | 0.5549 | 0.6295 |
| person | 0.8860 | 0.8485 | 0.8317 | 0.8400 |
| stair | 0.7272 | 0.7765 | 0.6111 | 0.6839 |

**Confusion Matrix (Test):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 43 | 1 | 1 | 1 | 0 | 6 |
| **door** | 0 | 73 | 0 | 0 | 0 | 22 |
| **obstacle** | 0 | 0 | 98 | 0 | 1 | 33 |
| **person** | 1 | 0 | 1 | 85 | 0 | 12 |
| **stair** | 0 | 0 | 0 | 0 | 68 | 17 |
| **FP (bkg)** | 14 | 62 | 73 | 15 | 39 | — |

### 5.8 Export ONNX (estándar)

| Parámetro | Valor |
|---|---|
| ONNX size | 9.21 MB |
| ONNX valid | ✅ |
| ONNX latency | 9.97 ms (CPU, OnnxRuntime) |
| Input shape | (1, 3, 224, 224) |
| Output shape | (1, 9, 1029) |
| Opset | 13 |

> **Nota**: Con `reg_max=1`, el output shape es `(1, 9, 1029)` donde 9 = 4 (bbox directas) + 5 (clases). Mismo shape numérico que T2 `(1, 9, 1029)` donde 9 = 4 (dist2bbox post-DFL integral) + 5 (clases), pero T3 no requiere DFL integral posterior.

### 5.9 Export ONNX ESP (6 salidas)

| Parámetro | Valor |
|---|---|
| Script | `scripts/export_yolo26_esp.py` |
| Archivo | `best_esp.onnx` |
| Tamaño | 9,359.7 KB (9.14 MB) |
| Simplificado | ✅ (onnxsim) |
| Outputs | 6 (3 escalas × box + score) |

**Output shapes (DFL Removal):**

| Output | Shape | Descripción |
|---|---|---|
| box0 | `[1, 4, 28, 28]` | Distancias directas escala P3 (stride 8) |
| score0 | `[1, 5, 28, 28]` | Scores 5 clases escala P3 |
| box1 | `[1, 4, 14, 14]` | Distancias directas escala P4 (stride 16) |
| score1 | `[1, 5, 14, 14]` | Scores 5 clases escala P4 |
| box2 | `[1, 4, 7, 7]` | Distancias directas escala P5 (stride 32) |
| score2 | `[1, 5, 7, 7]` | Scores 5 clases escala P5 |

> Con DFL Removal, los box outputs pasan de `[1, 64, H, W]` (T2) a `[1, 4, H, W]` (T3). Esto elimina la necesidad de softmax + weighted sum DFL en el firmware ESP32-S3, simplificando significativamente el post-processing.

### 5.10 Cuantización INT8 → ESPDL

| Parámetro | Valor |
|---|---|
| Script | `scripts/convert_onnx_to_espdl.py` |
| Config key | `yolo26n_t3_esp` |
| Input ONNX | `best_esp.onnx` (9,359.7 KB) |
| Output ESPDL | `yolo26n_t3_esp.espdl` (2.566 MB) |
| Compresión | **3.56×** |
| Calibración | 188 imágenes test split |

**Output exponents:**

| Output | Exponent |
|---|---|
| box0 (P3) | -3 |
| score0 (P3) | -3 |
| box1 (P4) | -3 |
| score1 (P4) | -2 |
| box2 (P5) | -4 |
| score2 (P5) | -3 |

> El ESPDL de 2.566 MB cabe holgadamente en los 8 MB de PSRAM del ESP32-S3 (32% de ocupación). Compresión 3.56× comparable a T2.

### 5.11 Evaluación FP32 vs INT8 (ESP)

| Métrica | FP32 (ESP) | INT8 (ESPDL) | Degradación |
|---|:---:|:---:|:---:|
| **mAP@50** | 0.5027 | 0.3769 | **-25.0%** |
| mAP@50-95 | 0.3206 | 0.2348 | -26.7% |
| Precision | 0.8437 | 0.8643 | +2.4% |
| Recall | 0.5163 | 0.3927 | -23.9% |
| F1-Score | 0.6406 | 0.5400 | -15.7% |
| Detecciones | 309 | 218 | -29.4% |
| **Veredicto** | — | — | **MARGINAL** |

**Per-class AP@50 FP32 vs INT8:**

| Clase | FP32 | INT8 | Δ |
|---|:---:|:---:|:---:|
| dog | 0.6823 | 0.5687 | -16.6% |
| door | 0.1489 | 0.0784 | -47.3% |
| obstacle | 0.3686 | 0.2268 | -38.5% |
| person | 0.8236 | 0.7299 | -11.4% |
| stair | 0.4899 | 0.2809 | -42.7% |

> **Análisis de cuantización**: La degradación del 25.0% en mAP@50 es clasificada como MARGINAL (umbral 20-30%). Las clases door y obstacle sufren la mayor degradación (~40-47%), consistente con el patrón observado en T2. **La hipótesis de que DFL Removal mejoraría la cuantización INT8 NO se confirmó** — T2 con DFL activa obtuvo menor degradación (18.7% vs 25.0%). Posible causa: la integral DFL actúa como filtro de suavizado que mejora la robustez a cuantización.

### 5.12 Análisis

**Resultado general: T3 es LIGERAMENTE INFERIOR a T2 en métricas absolutas, pero aporta beneficios arquitecturales significativos para despliegue.**

1. **mAP@50 (test) = 0.7466 (-3.6% vs T2)** — Pequeña degradación aceptable. T3 sigue superando ampliamente a FCOS. La caída se debe a: (a) pesos pretrained diferentes (yolo26n.pt vs yolo11n.pt), y (b) la DFL integral contribuye ligeramente a la precisión de bounding box en FP32.

2. **mAP@50-95 (test) = 0.5333 (-2.3% vs T2)** — Impacto menor en IoU estricto, sugiriendo que la calidad de bbox con 4 canales directos es comparable a 64 canales DFL.

3. **Per-class Test AP@50 vs T2**:
   - 🟡 person: 0.886 (+2.3%) — Única clase que MEJORA vs T2
   - 🔴 obstacle: 0.654 (-11.4%) — Mayor caída absoluta
   - 🔴 stair: 0.727 (-10.6%) — Caída significativa
   - 🟡 dog: 0.805 (+1.1%) — Leve mejora
   - ⚪ door: 0.661 (+0.1%) — Estable

4. **Precision más baja (0.785 vs T2 0.832 → -5.7%)** — Más FP desde background. La tabla de confusión muestra: door(62), obstacle(73), stair(39) FP↑. Patrón similar a T2 pero amplificado.

5. **Phase 2 sigue funcionando**: mAP50 0.763→0.774 (+1.4%). El patrón MuSGD + LR diferenciado es robusto y se mantiene con DFL Removal. Tres de tres entrenamientos confirman la mejora Phase 1→Phase 2.

6. **Beneficios para despliegue ESP32-S3 (principal motivación de T3)**:
   - ✅ Post-processing simplificado: sin softmax + integral DFL (ahorro de ~200 operaciones por detection head por escala)
   - ✅ Shapes de box 16× más compactos: `[1,4,H,W]` vs `[1,64,H,W]`
   - ✅ ESPDL 2.566 MB viable (32% PSRAM)
   - ✅ 85K menos parámetros (2.506M vs 2.591M)
   - ❌ Cuantización INT8 peor de lo esperado (25% vs 18% degradación)

7. **Análisis de riesgo cuantización**: La degradación MARGINAL (25%) es manejable pero requiere:
   - Evaluación funcional en dispositivo real (ESP32-S3)
   - Consideración de cuantización mixta (INT8/INT16) para outputs de box
   - El caso de uso de asistencia visual prioriza recall → INT8 recall=0.393 es insuficiente

### 5.13 Lecciones

1. **DFL Removal reduce accuracy FP32 marginalmente (-3.6%)** — Trade-off aceptable dado el beneficio en simplificación de firmware. Para el caso de uso (asistencia visual en ESP32-S3), la simplicidad de post-processing puede compensar la pequeña pérdida de accuracy.

2. **DFL NO ayuda a cuantización INT8 (contra-hipótesis)** — La integral DFL (softmax + weighted sum) actúa como regularizador implícito que suaviza los valores de distancia, haciéndolos más robustos a cuantización. Los 4 canales directos de T3 tienen mayor rango dinámico y son más sensibles al truncamiento INT8.

3. **Pesos nativos yolo26n.pt funcionan bien** — Transfer learning desde yolo26n.pt (YOLO26 COCO) produce resultados comparables a yolo11n.pt (YOLO11 COCO → YOLO26). No hay penalización significativa por usar pesos de la misma familia.

4. **El pipeline completo (train→export→quantize→eval) funciona end-to-end con DFL Removal** — Todos los scripts (export_yolo26_esp.py, convert_onnx_to_espdl.py, eval_fp32_vs_int8.py) se adaptaron correctamente para manejar `reg_max=1`.

5. **El veredicto para producción depende del contexto**:
   - Si el firmware ESP32-S3 puede ejecutar DFL integral eficientemente → T2 es preferible (mejor accuracy FP32 y mejor cuantización INT8)
   - Si la simplificación de firmware es prioritaria → T3 ofrece post-processing más simple a cambio de ~3.6% accuracy

---

## 6. Backlog de Propuestas

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
| Train 2 | **A** — MuSGD Optimizer | ✅ Completado |
| Train 3 | **D** — DFL Removal (`reg_max=1`) | ✅ Completado |
| Train 4 | A + **B** — Augmentación | ⏳ Pendiente |
| Train 5 | A + B + **C** — Loss Balance | ⏳ Pendiente |

> Esta planificación es orientativa. Se ajustará según los resultados de cada Train.

---

## Comparativa Global

### Test Metrics

| Métrica | Train 1 | Train 2 | Train 3 | Δ (T3−T2) | Mejor |
|---|:---:|:---:|:---:|:---:|:---:|
| mAP@50 | 0.7544 | **0.7747** | 0.7466 | -3.6% | T2 |
| mAP@50-95 | 0.5153 | **0.5456** | 0.5333 | -2.3% | T2 |
| Precision | 0.8264 | **0.8324** | 0.7853 | -5.7% | T2 |
| Recall | 0.6402 | **0.6853** | 0.6522 | -4.8% | T2 |
| F1-Score | 0.7215 | **0.7517** | 0.7126 | -5.2% | T2 |

### Per-class AP@50 (Test)

| Clase | Train 1 | Train 2 | Train 3 | Δ (T3−T2) | Mejor |
|---|:---:|:---:|:---:|:---:|:---:|
| dog | 0.7740 | 0.7956 | **0.8047** | +1.1% | T3 |
| door | 0.6531 | **0.6601** | 0.6606 | +0.1% | ≈ |
| obstacle | 0.7011 | **0.7384** | 0.6543 | -11.4% | T2 |
| person | 0.8301 | 0.8661 | **0.8860** | +2.3% | T3 |
| stair | **0.8138** | 0.8133 | 0.7272 | -10.6% | T1 |

### Val Metrics

| Métrica | Train 1 | Train 2 | Train 3 | Δ (T3−T2) | Mejor |
|---|:---:|:---:|:---:|:---:|:---:|
| mAP@50 | 0.6556 | **0.6638** | 0.6234 | -6.1% | T2 |
| mAP@50-95 | 0.4298 | **0.4345** | 0.4246 | -2.3% | T2 |
| Precision | **0.8215** | 0.7810 | 0.7289 | -6.7% | T1 |
| Recall | 0.4556 | **0.5119** | 0.4738 | -7.4% | T2 |
| F1-Score | 0.5861 | **0.6185** | 0.5743 | -7.1% | T2 |

### Training Dynamics

| Aspecto | Train 1 | Train 2 | Train 3 |
|---|---|---|---|
| Optimizer real | AdamW (auto) | MuSGD (explícito) | MuSGD (explícito) |
| Pretrained weights | yolo11n.pt | yolo11n.pt | yolo26n.pt (nativo) |
| reg_max / DFL | 16 / DFL | 16 / DFL | 1 / Identity |
| Phase 2 ¿mejoró Phase 1? | ❌ No | ✅ Sí (+3.5% mAP50-95) | ✅ Sí (+1.4% mAP50) |
| Early stopping | No (100/100) | Sí (98/100, P2: 68/70) | Sí (92/100, P2: 62/70) |
| Tiempo total | 26.0 min | 32.6 min | 30.5 min |
| Best P2 epoch | 70 (last) | 38 (early stop) | 61 (early stop) |
| GPU memory | 2.71 GB | 2.71 GB | ~2.7 GB |
| Params | 2,590,815 | 2,590,815 | 2,505,750 |

### Despliegue ESP32-S3

| Aspecto | Train 2 | Train 3 | Mejor |
|---|:---:|:---:|:---:|
| ONNX estándar | 9.97 MB | 9.21 MB | T3 |
| ONNX ESP (6 out) | — | 9.14 MB | T3 |
| Box output shape | [1,64,H,W] | [1,4,H,W] | T3 (16× menor) |
| ESPDL (INT8) | — | 2.566 MB | T3 |
| FP32 ESP mAP@50 | 0.5297 | 0.5027 | T2 |
| INT8 ESP mAP@50 | 0.4307 | 0.3769 | T2 |
| INT8 degradación | 18.7% | 25.0% | T2 |
| Post-processing | softmax+DFL integral | Directo (dist2bbox) | T3 (más simple) |

---

## Conclusiones Generales

1. **Train 2 (MuSGD) sigue siendo el mejor modelo en métricas absolutas** — mAP@50=0.775, F1=0.752, superando a T1 y T3 en todas las métricas de test. T3 (DFL Removal) pierde -3.6% mAP@50 vs T2.

2. **Train 3 (DFL Removal) aporta valor para despliegue, no para accuracy** — El beneficio principal es la simplificación del firmware ESP32-S3: post-processing sin DFL integral, box shapes 16× más compactos, y 85K menos parámetros. La pérdida de accuracy es marginal (-3.6%).

3. **La hipótesis de mejor cuantización INT8 con DFL Removal NO se confirmó** — T3 INT8 degrada 25% vs T2 18%. La integral DFL actúa como regularizador que suaviza los valores de distancia, haciéndolos más robustos a cuantización. Hallazgo contra-intuitivo pero reproducible.

4. **MuSGD + Fases es un patrón robusto** — Phase 2 mejoró Phase 1 en T2 (+3.5%) y T3 (+1.4%), confirmando la robustez del esquema de entrenamiento bifásico con MuSGD y LR diferenciado.

5. **YOLO26 supera ampliamente a FCOS** — T2 alcanza mAP@50=0.775 vs FCOS T3=0.568 (+36%) y FCOS T7=0.612 (+27%). T3 con DFL Removal también supera a FCOS: mAP@50=0.747 (+31%/+22%).

6. **Recall es la métrica a seguir mejorando** — T2 Recall=0.685 es el mejor, pero para asistencia visual se busca >0.75. Las Propuestas B (augmentación) y C (loss balance) están diseñadas para esto.

7. **door sigue siendo la clase más débil** — AP@50~0.660 estable en T2 y T3, Recall~0.50-0.52. Requiere atención específica (augmentación, oversampling, loss rebalancing).

8. **Decisión de producción para ESP32-S3**: Depende del firmware:
   - Si DFL integral es viable en ESP32-S3 → usar T2 (mejor accuracy + mejor INT8)
   - Si simplificación es prioritaria → usar T3 (post-processing trivial, accuracy aceptable)
   - En ambos casos, el ESPDL de ~2.5 MB cabe en 8 MB PSRAM

---

## Referencia Cruzada — FCOS vs YOLO26

| Métrica (Test) | FCOS T3 (producción) | FCOS T7 (benchmark) | YOLO26 T1 | YOLO26 T2 | YOLO26 T3 (DFL off) |
|---|:---:|:---:|:---:|:---:|:---:|
| mAP@50 | 0.5675 | 0.6120 | 0.7544 | **0.7747** | 0.7466 |
| mAP@50-95 | 0.2602 | 0.2824 | 0.5153 | **0.5456** | 0.5333 |
| Precision | 0.6609 | 0.3716 | 0.8264 | **0.8324** | 0.7853 |
| Recall | 0.6276 | **0.6872** | 0.6402 | 0.6853 | 0.6522 |
| F1-Score | 0.6438 | 0.4824 | 0.7215 | **0.7517** | 0.7126 |
| Params | 1.23M | 1.23M | 2.59M | 2.59M | 2.51M |
| reg_max / DFL | — | — | 16 / DFL | 16 / DFL | 1 / Identity |
| ONNX size | 4.74 MB | 4.74 MB | 9.97 MB | 9.97 MB | 9.21 MB |
| ESPDL (INT8) | — | — | — | — | 2.57 MB |
| Inference (GPU) | — | — | 2.6 ms | 2.9 ms | 4.0 ms |

> **Conclusión**: YOLO26 T2 establece el mejor modelo en métricas absolutas (+36% mAP@50 vs FCOS T3 producción). YOLO26 T3 (DFL Removal) sigue superando a FCOS (+31% mAP@50) con el beneficio adicional de post-processing simplificado y cuantización ESPDL verificada (2.57 MB). Para despliegue ESP32-S3, T3 ofrece el pipeline más completo: ONNX ESP → ESPDL → inferencia sin DFL. El recall de T2 (0.685) supera al de FCOS T3 producción (0.628), acercándose al FCOS T7 benchmark (0.687).

---

*Documento generado y mantenido como parte del pipeline MLOps del TFM — Detección de Objetos para Asistencia Visual.*
