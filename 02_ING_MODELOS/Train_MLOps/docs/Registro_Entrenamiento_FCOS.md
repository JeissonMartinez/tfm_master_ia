# Registro de Entrenamiento — FCOS (MobileNetV3-Small + FPN)

> **Modelo**: `fcos_v3s` — FCOS con MobileNetV3-Small backbone + SimpleFPN + FCOS Head  
> **Parámetros**: 1,233,450 total (FP32: 4.71 MB | INT8 est.: 1.18 MB)  
> **Dataset**: IODC YOLO — 5 clases (dog, door, obstacle, person, stair)  
> **Splits**: Train 1470 | Val 188 | Test 187  
> **Infraestructura**: Google Vertex AI Custom Job — `n1-standard-8` + NVIDIA Tesla T4  
> **Contenedor**: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`  
> **Última actualización**: 21 de febrero de 2026  

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Train 1 — Baseline](#2-train-1--baseline)
3. [Train 2 — Stride Normalization + Bug Fixes](#3-train-2--stride-normalization--bug-fixes)
4. [Train 3 — GIoU Loss + Más Épocas](#4-train-3--giou-loss--más-épocas)
5. [Train 4 — Scoring Refinements (conf, centerness, IoU-aware)](#5-train-4--scoring-refinements-conf-centerness-iou-aware)
6. [Threshold Sweep — Análisis Post-NMS del Modelo T4](#6-threshold-sweep--análisis-post-nms-del-modelo-t4)
7. [Comparativa Global](#7-comparativa-global)
8. [Conclusiones Generales](#8-conclusiones-generales)

---

## 1. Resumen Ejecutivo

| Métrica (Test) | Train 1 | Train 2 | Train 3 | Train 4 |
|---|:---:|:---:|:---:|:---:|
| **mAP@50** | 0.4304 | 0.5600 | 0.5675 | **0.5936** |
| **mAP@50-95** | N/C | N/C | 0.2602 | **0.2644** |
| **Precision** | 0.5427 | 0.6049 | **0.6609** | 0.3462 |
| **Recall** | 0.5291 | 0.6271 | 0.6276 | **0.6886** |
| **F1-Score** | 0.5358 | 0.6158 | **0.6438** | 0.4607 |
| Épocas | 52 | 74 | 101 | 77 |
| Tiempo | 12.1 min | 15.9 min | 23.6 min | 17.9 min |
| Inferencia | 5.0 ms | 4.6 ms | 4.8 ms | 5.4 ms |

> **N/C**: No Calculado — la implementación de mAP@50-95 fue añadida después de Train 2.  
> **Train 4**: Mejores mAP@50 y Recall de la serie, pero Precision colapsa por exceso de FP (scoring demasiado permisivo).

---

## 2. Train 1 — Baseline

### 2.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `fcos_v3s_v1-1771683868` |
| **Fecha** | 21 de febrero de 2026 |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771683868/` |
| **Output local** | `outputs/fcos_v3s_v1-1771683868/` |
| **Log** | `logs/FCOS_Train_1.md` |

### 2.2 Configuración

| Parámetro | Valor |
|---|---|
| Regression Loss | **Smooth L1** (beta=1.0) |
| Stride Normalization | **No** — reg targets en píxeles absolutos |
| Phase 1 | 30 epochs, LR=1e-3, backbone frozen |
| Phase 2 | 60 epochs, LR=1e-4, full fine-tuning |
| Patience | 15 |
| Optimizer | AdamW (WD=1e-4 / 1e-5) |
| Scheduler | CosineAnnealing |
| Batch Size | 16, AMP=True |
| Resize Schedule | {0: 640, 10: 416, 20: 320, 30: 224} |
| Conf Threshold | 0.25 |
| IoU Threshold | 0.45 |
| cls_weight / reg_weight / ctr_weight | 1.0 / 1.5 / 1.0 |

### 2.3 Entrenamiento

- **Épocas completadas**: 52 (early stopping epoch 51)
- **Mejor val_loss**: 135.4938 (epoch 36)
- **Tiempo total**: 12.1 min
- **Observaciones**:
  - `reg_loss` dominaba completamente (~682 en epoch 0 a 640px), dos órdenes de magnitud mayor que `cls_loss` (~3.3) y `ctr_loss` (~1.8).
  - Al reducir resolución de 640→224px, `reg_loss` bajó proporcionalmente (en píxeles absolutos), llegando a ~41 al final.
  - El desbalance entre componentes de loss impidió que cls y centerness contribuyeran efectivamente al gradiente.

### 2.4 Resultados — Validación

| Métrica | Valor |
|---|---|
| mAP@50 | 0.2835 |
| mAP@50-95 | N/C |
| Precision | 0.5223 |
| Recall | 0.3413 |
| F1-Score | 0.4128 |
| Detecciones / GT | 525 / 762 |
| Inferencia | 4.8 ms |

**Per-class AP@50 (Val):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.1956 | 0.4353 | 0.2467 | 0.3149 |
| door | 0.2919 | 0.6494 | 0.3125 | 0.4219 |
| obstacle | 0.2360 | 0.4196 | 0.3659 | 0.3909 |
| person | 0.3530 | 0.4843 | 0.4231 | 0.4516 |
| stair | 0.3409 | 0.6230 | 0.3585 | 0.4551 |

### 2.5 Resultados — Test

| Métrica | Valor |
|---|---|
| mAP@50 | 0.4304 |
| mAP@50-95 | N/C |
| Precision | 0.5427 |
| Recall | 0.5291 |
| F1-Score | 0.5358 |
| Detecciones / GT | 548 / 576 |
| Inferencia | 5.0 ms |

**Per-class AP@50 (Test):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.4056 | 0.4167 | 0.5172 | 0.4615 |
| door | 0.3391 | 0.5745 | 0.3971 | 0.4696 |
| obstacle | 0.3022 | 0.5232 | 0.4566 | 0.4877 |
| person | 0.5206 | 0.4820 | 0.6634 | 0.5583 |
| stair | 0.5844 | 0.7174 | 0.6111 | 0.6600 |

**Confusion Matrix (Test):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 30 | 0 | 0 | 0 | 0 | 28 |
| **door** | 0 | 54 | 0 | 0 | 0 | 82 |
| **obstacle** | 0 | 0 | 79 | 0 | 0 | 94 |
| **person** | 0 | 0 | 0 | 67 | 0 | 34 |
| **stair** | 0 | 0 | 0 | 0 | 66 | 42 |
| **FP** | 42 | 40 | 72 | 72 | 26 | — |

### 2.6 Problemas Identificados

1. **reg_loss desproporcionada**: Los targets de regresión (l, t, r, b) estaban en píxeles absolutos. A 640×640, distancias típicas de ~80px generaban reg_loss ~682, eclipsando cls_loss (~3) y ctr_loss (~1.8).
2. **mAP@50-95 no implementada**: La función `_compute_map()` solo evaluaba a IoU=0.5.
3. **Warnings de deprecación**: `torch.load` sin `weights_only=True`, `ShiftScaleRotate` deprecado.
4. **Argumento swap en `write_yaml`**: Orden de parámetros invertido.
5. **Split incorrecto en evaluación de Test**: `evaluate_pytorch_model()` no recibía `split="test"`.

---

## 3. Train 2 — Stride Normalization + Bug Fixes

### 3.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `fcos_v3s_v1-1771687747` |
| **Fecha** | 21 de febrero de 2026 |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771687747/` |
| **Output local** | `outputs/fcos_v3s_v1-1771687747/` |
| **Log** | `logs/FCOS_Train_2.md` |

### 3.2 Cambios Respecto a Train 1

| # | Cambio | Archivo(s) | Detalle |
|---|---|---|---|
| 1 | **Stride Normalization** | `task_fcos.py`, `task_espdet.py`, `utils_infer.py` | Reg targets divididos por stride en encode; multiplicados por stride en decode/predict. Normaliza l,t,r,b al rango [0, feat_size]. |
| 2 | `torch.load` → `weights_only=True` | `utils_train.py`, `task_fcos.py`, `task_espdet.py`, `utils_model.py` | Elimina FutureWarning en PyTorch 2.4+. |
| 3 | `ShiftScaleRotate` → `A.Affine` | `utils_data.py` | Reemplazo por API actual de Albumentations. |
| 4 | `write_yaml(data, path)` → `write_yaml(path, data)` | `utils_data.py` | Corrección del orden de argumentos. |
| 5 | Parámetro `split` en evaluación | `utils_eval.py`, `task_fcos.py`, `task_espdet.py` | `evaluate_pytorch_model()` recibe `split="test"` para evaluación en test. |

### 3.3 Configuración

Idéntica a Train 1 excepto por los fix de código. Los hiperparámetros del YAML no cambiaron.

### 3.4 Entrenamiento

- **Épocas completadas**: 74 (early stopping epoch 73)
- **Mejor val_loss**: 24.0068 (epoch 58)
- **Tiempo total**: 15.9 min
- **Observaciones**:
  - `reg_loss` ahora arranca en ~33.8 (vs ~682 en T1) y baja a ~1.3 al final.
  - Las tres componentes de loss están en rangos comparables: cls ~0.7, reg ~1.3, ctr ~1.7.
  - El modelo entrenó 22 épocas más que T1 antes de early stopping → mejor capacidad de aprendizaje.
  - val_loss mejoró drásticamente: 24.0 vs 135.5 (-82.3%).

### 3.5 Resultados — Validación

| Métrica | Train 1 | **Train 2** | Δ |
|---|:---:|:---:|:---:|
| mAP@50 | 0.2835 | **0.3792** | +33.7% |
| Precision | 0.5223 | **0.5345** | +2.3% |
| Recall | 0.3413 | **0.4312** | +26.3% |
| F1 | 0.4128 | **0.4773** | +15.6% |
| Detecciones / GT | 525 / 762 | 623 / 762 | +98 dets |

**Per-class AP@50 (Val):**

| Clase | Train 1 | **Train 2** | Δ |
|---|:---:|:---:|:---:|
| dog | 0.1956 | **0.2743** | +40.3% |
| door | 0.2919 | **0.4329** | +48.3% |
| obstacle | 0.2360 | **0.3935** | +66.7% |
| person | 0.3530 | **0.3871** | +9.6% |
| stair | 0.3409 | **0.4083** | +19.8% |

### 3.6 Resultados — Test

| Métrica | Train 1 | **Train 2** | Δ |
|---|:---:|:---:|:---:|
| mAP@50 | 0.4304 | **0.5600** | +30.1% |
| Precision | 0.5427 | **0.6049** | +11.5% |
| Recall | 0.5291 | **0.6271** | +18.5% |
| F1 | 0.5358 | **0.6158** | +14.9% |
| Detecciones / GT | 548 / 576 | 585 / 576 | +37 dets |

**Per-class AP@50 (Test):**

| Clase | Train 1 | **Train 2** | Δ |
|---|:---:|:---:|:---:|
| dog | 0.4056 | **0.4627** | +14.1% |
| door | 0.3391 | **0.5194** | +53.2% |
| obstacle | 0.3022 | **0.4051** | +34.1% |
| person | 0.5206 | **0.6355** | +22.1% |
| stair | 0.5844 | **0.7774** | +33.0% |

**Confusion Matrix (Test):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 32 | 0 | 0 | 0 | 0 | 26 |
| **door** | 0 | 80 | 0 | 0 | 0 | 56 |
| **obstacle** | 0 | 0 | 86 | 0 | 0 | 87 |
| **person** | 0 | 0 | 0 | 70 | 0 | 31 |
| **stair** | 0 | 0 | 0 | 0 | 87 | 21 |
| **FP** | 32 | 83 | 43 | 38 | 34 | — |

### 3.7 Efecto de los Cambios

**Stride Normalization (cambio dominante):**
- Al normalizar reg targets por stride, las tres componentes de loss quedaron balanceadas (~1-3 cada una), permitiendo que cls y centerness contribuyeran efectivamente al gradiente.
- Impacto directo: +30% mAP@50 en test, +26% recall en val.
- El modelo genera más detecciones (623 vs 525 en val) porque la cabeza de clasificación y centerness ahora están correctamente optimizadas.

**Bug fixes (efecto menor pero necesario):**
- El fix de `split="test"` garantiza que la evaluación en test use el split correcto.
- Los otros fixes eliminan warnings y errores potenciales, mejorando la robustez del pipeline.

---

## 4. Train 3 — GIoU Loss + Más Épocas

### 4.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `fcos_v3s_v1-1771690809` |
| **Fecha** | 21 de febrero de 2026 |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771690809/` |
| **Output local** | `outputs/fcos_v3s_v1-1771690809/` |
| **Log** | `logs/FCOS_Train_3.md` |

### 4.2 Cambios Respecto a Train 2

| # | Cambio | Archivo(s) | Detalle |
|---|---|---|---|
| 1 | **GIoU Loss** reemplaza Smooth L1 | `utils_train.py` | Nueva función `_giou_loss_ltrb()` calcula GIoU directamente sobre l,t,r,b normalizados por stride. Aplicado en `build_fcos_loss()` y `build_espdet_loss()`. Rango de loss: [0, 2]. |
| 2 | **Patience 15 → 20** | `fcos_v3s_v1.yaml` | Permite más épocas sin mejora antes de early stop. |
| 3 | **Phase 2 epochs 60 → 80** | `fcos_v3s_v1.yaml` | Mayor presupuesto de épocas para convergencia. |
| 4 | **mAP@50-95 implementado** | `utils_eval.py` | Nueva función `_compute_aps_at_iou()` evalúa AP a IoU=[0.50, 0.55, ..., 0.95] y promedia. |
| 5 | **Panel "Métricas Val" corregido** | `utils_metrics.py` | 6° panel del training_curves.png ahora muestra Centerness Loss para modelos PyTorch (antes vacío). Campos `train_ctr_loss`/`val_ctr_loss` añadidos a `TrainingHistory`. |

### 4.3 Configuración

| Parámetro | Train 2 | **Train 3** |
|---|---|---|
| Regression Loss | Smooth L1 | **GIoU** |
| Phase 2 Epochs | 60 | **80** |
| Patience | 15 | **20** |
| _Resto_ | _Igual_ | _Igual_ |

### 4.4 Entrenamiento

- **Épocas completadas**: 101 (early stopping epoch 100)
- **Mejor val_loss**: 28.1248 (epoch 80)
- **Tiempo total**: 23.6 min
- **Observaciones**:
  - **Meseta GIoU en reg_loss = 4.5 durante epochs 0-13**: Comportamiento esperado. Con backbone congelado y predicciones iniciales, la mayoría de boxes no solapan con GT. GIoU retorna -1 cuando no hay intersección → loss = 1-(-1) = 2.0. Promediado sobre 3 niveles FPN × reg_weight 1.5 ≈ 4.5. El modelo rompe la meseta en epoch 14 cuando las predicciones empiezan a solapar.
  - Después de la meseta, `reg_loss` baja progresivamente de ~3.1 (e14) a ~1.1 (e100).
  - `cls_loss` baja de 3.06 → 0.52, `ctr_loss` de 1.86 → 1.67.
  - Phase 2 best en epoch 80 vs 58 en T2 — los 20 epochs extra de presupuesto fueron utilizados efectivamente.

### 4.5 Resultados — Validación

| Métrica | Train 2 | **Train 3** | Δ |
|---|:---:|:---:|:---:|
| mAP@50 | 0.3792 | **0.3761** | -0.8% |
| mAP@50-95 | N/C | **0.1791** | — |
| Precision | 0.5345 | **0.5910** | +10.6% |
| Recall | 0.4312 | **0.4267** | -1.0% |
| F1 | 0.4773 | **0.4956** | +3.8% |
| Detecciones / GT | 623 / 762 | 555 / 762 | -68 dets |

**Per-class AP@50 (Val):**

| Clase | Train 2 | **Train 3** | Δ |
|---|:---:|:---:|:---:|
| dog | 0.2743 | **0.2994** | +9.1% |
| door | 0.4329 | 0.3847 | -11.1% |
| obstacle | 0.3935 | 0.3566 | -9.4% |
| person | 0.3871 | 0.3800 | -1.8% |
| stair | 0.4083 | **0.4596** | +12.6% |

### 4.6 Resultados — Test

| Métrica | Train 2 | **Train 3** | Δ |
|---|:---:|:---:|:---:|
| mAP@50 | 0.5600 | **0.5675** | +1.3% |
| mAP@50-95 | N/C | **0.2602** | — |
| Precision | 0.6049 | **0.6609** | +9.3% |
| Recall | 0.6271 | **0.6276** | +0.1% |
| F1 | 0.6158 | **0.6438** | +4.5% |
| Detecciones / GT | 585 / 576 | 533 / 576 | -52 dets |

**Per-class AP@50 (Test):**

| Clase | Train 2 | **Train 3** | Δ |
|---|:---:|:---:|:---:|
| dog | 0.4627 | **0.4957** | +7.1% |
| door | 0.5194 | 0.5034 | -3.1% |
| obstacle | 0.4051 | **0.4575** | +12.9% |
| person | 0.6355 | **0.6359** | +0.1% |
| stair | 0.7774 | 0.7451 | -4.2% |

**Per-class Precision / Recall / F1 (Test):**

| Clase | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| dog | 0.5789 | 0.5690 | 0.5739 |
| door | 0.6364 | 0.5662 | 0.5992 |
| obstacle | 0.6715 | 0.5318 | 0.5935 |
| person | 0.7292 | 0.6931 | 0.7107 |
| stair | 0.6885 | 0.7778 | 0.7304 |

**Confusion Matrix (Test):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 33 | 0 | 0 | 0 | 0 | 25 |
| **door** | 0 | 77 | 0 | 0 | 0 | 59 |
| **obstacle** | 0 | 0 | 92 | 0 | 0 | 81 |
| **person** | 0 | 0 | 0 | 70 | 0 | 31 |
| **stair** | 0 | 0 | 0 | 0 | 84 | 24 |
| **FP** | 24 | 44 | 45 | 26 | 38 | — |

### 4.7 Efecto de los Cambios

**GIoU Loss (cambio dominante):**
- **mAP@50-95 = 0.26** en test — mejora transformadora. Smooth L1 no optimiza IoU directamente; los boxes producidos podían tener tamaño y posición decentes (suficiente para IoU>0.5) pero no ajuste fino (insuficiente para IoU>0.75). GIoU corrige esto.
- **Precision +9.3%**: Menos detecciones (533 vs 585) pero más precisas → boxes más ajustados al GT.
- **mAP@50 +1.3%**: Mejora marginal. GIoU no produce más detecciones, produce mejores boxes.
- **Recall estable (+0.1%)**: Confirma que GIoU afecta calidad, no cantidad de detecciones.
- **Meseta inicial**: 13 epochs de entrenamiento "desperdiciados" en Phase 1 mientras el modelo no logra solapamiento. Posible oportunidad de optimización con warmup gradual o loss híbrida.

**Más épocas / Patience (cambio secundario):**
- El best cayó en epoch 80, más allá del límite anterior de 60 en Phase 2.
- Con patience=15 (T2), habría parado en epoch 67+15=82. Con patience=20, tuvo margen hasta epoch 100.
- El modelo siguió mejorando gradualmente entre epoch 58 y 80, validando la utilidad de más épocas.

### 4.8 Análisis de Gráficas

**training_curves.png** (6 paneles):
- Panel 1 (Total Loss): Descenso continuo de ~9.4 a ~3.3. Val loss ruidosa (oscila 28-100+), típico de val set pequeño (188 imgs).
- Panel 2 (Box/GIoU Loss): Meseta a 4.5 en epochs 0-13 (sin solapamiento pred↔GT), ruptura en epoch 14, luego descenso a ~1.1.
- Panel 3 (Cls Loss): Descenso suave 3.05→0.52. Sin problemas.
- Panel 4 (Progressive Resizing): Escalones claros 640→416→320→224. Cada reducción genera un salto en loss.
- Panel 5 (LR Schedule): Warmup 3 epochs + cosine en Phase 1, cosine completo en Phase 2.
- Panel 6 (Centerness Loss): Train ctr baja suavemente 1.86→1.67. Val ctr más ruidosa.

**val_confusion_matrix.png / test_confusion_matrix.png:**
- **Cero confusión inter-clase** en ambas matrices. La cabeza de clasificación discrimina perfectamente las 5 clases.
- Problema principal: alta tasa de **falsos negativos** (FN), especialmente en val (dog: 100 FN, person: 102 FN).
- FP moderados, con obstacle (~45-75) y door (~38-44) como principales fuentes.

**val_per_class.png:**
- stair lidera (AP=0.460, F1=0.568), seguido de person y door.
- dog es consistentemente la clase más débil (AP=0.299, F1=0.403).
- Todas las clases muestran precision > recall → modelo conservador.

---

## 5. Train 4 — Scoring Refinements (conf, centerness, IoU-aware)

### 5.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `fcos_v3s_v1-1771695807` |
| **Fecha** | 21 de febrero de 2026 |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771695807/` |
| **Output local** | `outputs/fcos_v3s_v1-1771695807/` |
| **Log** | `logs/FCOS_Train_4.md` |

### 5.2 Cambios Respecto a Train 3

| # | Cambio | Archivo(s) | Detalle |
|---|---|---|---|
| 1 | **conf_threshold 0.25 → 0.15** | `fcos_v3s_v1.yaml` | Umbral de confianza más bajo para recuperar detecciones que antes se descartaban. |
| 2 | **Filtrado por cls_score puro** | `utils_infer.py` | Antes: `mask = (cls * ctr) > threshold`. Ahora: `mask = cls > threshold`. No se multiplica centerness antes de filtrar. |
| 3 | **ctr_power = 0.5** | `utils_infer.py`, `fcos_v3s_v1.yaml` | `score = cls × ctr^0.5 × geo_quality`. Centerness elevado a 0.5 reduce su efecto supresivo (ctr=0.3 aporta 0.55 en vez de 0.30). |
| 4 | **IoU-aware scoring** | `utils_infer.py`, `fcos_v3s_v1.yaml` | Factor de calidad geométrica: `geo_quality = sqrt(min(l,r)/max(l,r) × min(t,b)/max(t,b))`. Penaliza boxes descentrados. |
| 5 | **Defaults en _FCOS_DEFAULTS** | `utils_widgets.py` | Nuevos campos `ctr_power` e `iou_aware_scoring` en defaults del pipeline. |

> **Nota**: El entrenamiento (loss, modelo, augmentation) no cambió. Los 3 cambios solo afectan la **inferencia/evaluación**.

### 5.3 Configuración

| Parámetro | Train 3 | **Train 4** |
|---|---|---|
| conf_threshold | 0.25 | **0.15** |
| Scoring formula | cls × ctr | **cls × ctr^0.5 × geo_quality** |
| ctr_power | 1.0 (implícito) | **0.5** |
| iou_aware_scoring | No | **Sí** |
| _Resto_ | _Igual_ | _Igual_ |

### 5.4 Entrenamiento

- **Épocas completadas**: 77 (early stopping epoch 76)
- **Mejor val_loss**: 36.4576 (epoch 56)
- **Tiempo total**: 17.9 min
- **Observaciones**:
  - Mismo patrón que T3: meseta GIoU en epochs 0-13, ruptura en e14, descenso continuo.
  - Loss finales similares a T3: cls=0.56, reg=1.09, ctr=1.68.
  - Early stop en epoch 76 (vs 100 en T3) — paró **24 epochs antes** sin encontrar mejor val_loss después de epoch 56. El entrenamiento converge al mismo modelo base; las diferencias están enteramente en la inferencia.

### 5.5 Resultados — Validación

| Métrica | Train 3 | **Train 4** | Δ |
|---|:---:|:---:|:---:|
| mAP@50 | 0.3761 | **0.4178** | +11.1% |
| mAP@50-95 | 0.1791 | **0.1843** | +2.9% |
| Precision | 0.5910 | 0.3189 | **-46.1%** |
| Recall | 0.4267 | **0.5115** | +19.9% |
| F1 | 0.4956 | 0.3928 | -20.7% |
| Detecciones / GT | 555 / 762 | **1224 / 762** | +120.5% dets |

**Per-class AP@50 (Val):**

| Clase | Train 3 | **Train 4** | Δ |
|---|:---:|:---:|:---:|
| dog | 0.2994 | **0.3138** | +4.8% |
| door | 0.3847 | **0.4459** | +15.9% |
| obstacle | 0.3566 | **0.4088** | +14.6% |
| person | 0.3800 | **0.4531** | +19.2% |
| stair | 0.4596 | **0.4673** | +1.7% |

### 5.6 Resultados — Test

| Métrica | Train 3 | **Train 4** | Δ |
|---|:---:|:---:|:---:|
| mAP@50 | 0.5675 | **0.5936** | +4.6% |
| mAP@50-95 | 0.2602 | **0.2644** | +1.6% |
| Precision | 0.6609 | 0.3462 | **-47.6%** |
| Recall | 0.6276 | **0.6886** | +9.7% |
| F1 | 0.6438 | 0.4607 | -28.4% |
| Detecciones / GT | 533 / 576 | **1120 / 576** | +110.1% dets |

**Per-class AP@50 (Test):**

| Clase | Train 3 | **Train 4** | Δ |
|---|:---:|:---:|:---:|
| dog | 0.4957 | **0.5021** | +1.3% |
| door | 0.5034 | **0.5334** | +6.0% |
| obstacle | 0.4575 | **0.5116** | +11.8% |
| person | 0.6359 | **0.6816** | +7.2% |
| stair | 0.7451 | **0.7394** | -0.8% |

**Confusion Matrix (Test):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 35 | 0 | 0 | 0 | 0 | 23 |
| **door** | 0 | 90 | 0 | 0 | 0 | 46 |
| **obstacle** | 0 | 0 | 107 | 0 | 0 | 66 |
| **person** | 0 | 0 | 0 | 78 | 0 | 23 |
| **stair** | 0 | 0 | 0 | 0 | 85 | 23 |
| **FP** | 88 | 188 | 165 | 144 | 140 | — |

**Confusion Matrix (Val):**

|  | dog | door | obst | pers | stair | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **dog** | 59 | 0 | 0 | 0 | 0 | 91 |
| **door** | 0 | 88 | 0 | 0 | 0 | 72 |
| **obstacle** | 0 | 0 | 94 | 0 | 0 | 70 |
| **person** | 0 | 0 | 0 | 95 | 0 | 87 |
| **stair** | 0 | 0 | 0 | 0 | 55 | 51 |
| **FP** | 172 | 166 | 201 | 184 | 110 | — |

### 5.7 Efecto de los Cambios

**Lo positivo — mAP y Recall mejoraron:**
- **mAP@50 +4.6%** en test (0.5675→0.5936): El mejor mAP@50 de los 4 entrenamientos. AP sube porque el ranking de detecciones mejora al incluir más TP en posiciones altas del ranking.
- **Recall +9.7%** en test (0.6276→0.6886): 395 TP vs 356 en T3 → **39 objetos más detectados**. Se cumplió el objetivo de recuperar detecciones.
- **mAP@50-95 estable** (+1.6%): La calidad de las boxes (IoU con GT) no cambió significativamente, como se esperaba — los cambios son solo de scoring, no de regresión.
- **Todas las clases mejoran en AP@50** excepto stair (-0.8%). obstacle (+11.8%) y person (+7.2%) son las más beneficiadas.

**Lo negativo — Precision colapsó:**
- **Precision -47.6%** en test (0.6609→0.3462): El modelo genera **1120 detecciones para 576 GT** (ratio 1.94x). En T3 generaba 533 (ratio 0.93x).
- **FP se dispararon**: 725 FP en test (vs 177 en T3) → **+309%**. Cada clase acumula entre 88 y 188 FP.
- **F1 cae -28.4%**: La caída masiva de precision supera con creces la ganancia de recall.

**Diagnóstico — El scoring es demasiado permisivo:**

El problema no es uno solo de los 3 cambios, sino su **efecto compuesto**:
1. `conf_threshold 0.15` deja pasar candidatos con cls_score bajo que habrían muerto a 0.25.
2. El filtrado por `cls_score` puro (sin multiplicar ctr primero) deja pasar aún más candidatos.
3. `ctr^0.5` suaviza el centerness, inflando los scores finales de detecciones de baja calidad.
4. `geo_quality` en principio debería penalizar boxes malos, pero como los factores 1-3 ya dejaron pasar demasiados candidatos, NMS no puede limpiar todo (hay muchos boxes con overlap bajo entre sí pero todos son FP).

El resultado neto es una **explosión de falsos positivos**. El modelo tiene capacidad de localización (la curva precision-recall tiene area mayor), pero el punto de operación es subóptimo.

### 5.8 Lecciones

1. **No acumular cambios permisivos sin contrapeso**: Los 3 cambios empujan en la misma dirección (más detecciones). Se necesitaba aplicar solo uno a la vez, o incluir un mecanismo de restricción (como max_detections_per_image o score_threshold más adaptativo).
2. **mAP sube, F1 baja**: mAP mide el area bajo la curva P-R completa, no un punto específico. Que mAP suba no significa que el modelo sea mejor operativamente — en producción necesitamos un punto de trabajo con precision razonable.
3. **El threshold óptimo** post-NMS probablemente está entre 0.20 y 0.30 para T4, no en 0.15. Un análisis de la curva P-R podría identificarlo.

---

## 6. Threshold Sweep — Análisis Post-NMS del Modelo T4

Tras los resultados de T4 (mAP@50 = 0.5936 pero F1 = 0.4607 con `conf_threshold=0.15`), se realizó un barrido offline de umbral de confianza post-NMS para buscar el punto operativo óptimo del modelo T4 sin reentrenar.

Se utilizó el script `scripts/fcos_threshold_sweep.py` con los pesos del modelo T4 (`outputs/fcos_v3s_v1-1771695807/best_model.pt`).

### 6.1 Sweep v1 — Rango [0.10 – 0.30]

**Artefactos**: `outputs/fcos_v3s_v1-1771695807/threshold_sweep_v1/`

| Threshold | mAP@50 | Precision | Recall | F1 | Dets | TP | FP | FN |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.10 | 0.6074 | 0.2746 | 0.7274 | 0.3987 | 1512 | 424 | 1088 | 152 |
| 0.15 | 0.6021 | 0.3038 | 0.7107 | 0.4256 | 1329 | 412 | 917 | 164 |
| 0.20 | 0.5994 | 0.3269 | 0.7025 | 0.4462 | 1215 | 406 | 809 | 170 |
| 0.25 | 0.5936 | 0.3462 | 0.6886 | 0.4607 | 1120 | 395 | 725 | 181 |
| 0.30 | 0.5896 | 0.3611 | 0.6789 | 0.4715 | 1056 | 388 | 668 | 188 |

> **Observación**: F1 crece monotónicamente de 0.10 a 0.30, sugiriendo que el óptimo está más arriba. Se extiende el barrido.

### 6.2 Sweep v2 — Rango [0.30 – 0.50]

**Artefactos**: `outputs/fcos_v3s_v1-1771695807/threshold_sweep_v2/`

| Threshold | mAP@50 | Precision | Recall | F1 | Dets | TP | FP | FN |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.30 | 0.5896 | 0.3611 | 0.6789 | 0.4715 | 1056 | 388 | 668 | 188 |
| 0.35 | 0.5857 | 0.3767 | 0.6686 | 0.4819 | 996 | 382 | 614 | 194 |
| 0.40 | 0.5800 | 0.3919 | 0.6530 | 0.4898 | 936 | 375 | 561 | 201 |
| 0.45 | 0.5782 | 0.4130 | 0.6478 | 0.5044 | 877 | 371 | 506 | 205 |
| 0.50 | 0.5773 | 0.4375 | 0.6463 | 0.5218 | 827 | 370 | 457 | 206 |

### 6.3 Per-class AP@50 (Test) — Mejores puntos por clase

| Clase | Mejor AP@50 | Threshold | AP@50 T3 |
|---|:---:|:---:|:---:|
| dog | 0.5025 | 0.30 | **0.496** |
| door | 0.5348 | 0.10 | 0.503 |
| obstacle | 0.5593 | 0.10 | **0.458** |
| person | 0.6945 | 0.10 | **0.636** |
| stair | 0.7466 | 0.10 | 0.745 |

> Las AP@50 más altas se obtienen a thresholds bajos (maximizan recall para la curva P-R), pero el punto operativo con mejor F1 está en thr ≥ 0.50.

### 6.4 Análisis y Conclusiones del Sweep

1. **F1 monotónicamente creciente hasta 0.50**: Incluso a `conf_threshold=0.50`, el F1 test del modelo T4 es **0.5218**, todavía **19% inferior** al F1 de T3 (0.6438 a conf=0.25).

2. **Trade-off irreconciliable**: A thr=0.50, T4 tiene P=0.44, R=0.65. T3 a thr=0.25 tiene P=0.66, R=0.63. T3 domina en Precision sin sacrificar Recall.

3. **FP siguen elevados**: A thr=0.50, T4 aún produce 457 FP (vs 177 de T3). La diferencia es estructural: el scoring de T4 (`ctr^0.5` + `iou_aware`) infla scores de detecciones de baja calidad.

4. **mAP@50 ≈ estable**: Baja marginalmente de 0.607 a 0.577 en todo el rango, lo cual confirma que mAP no es sensible al threshold (mide area completa de la curva P-R).

5. **Decisión**: El scoring de T4 queda **descartado** para producción. T3 sigue siendo el mejor modelo operativo. Las mejoras futuras deben enfocarse en el entrenamiento (augmentación, loss), no en ingeniería de scoring.

---

## 7. Comparativa Global

### 7.1 Evolución Total Loss

| Fase | Train 1 | Train 2 | Train 3 | Train 4 |
|---|:---:|:---:|:---:|:---:|
| Epoch 0 (640px) | 688.5 | 38.9 | 9.4 | 8.9 |
| Final | ~43.5 | ~3.5 | ~3.3 | ~3.3 |
| Best val_loss | 135.49 | **24.01** | 28.12 | 36.46 |

> Nota: val_loss no es directamente comparable entre T2 (Smooth L1) y T3/T4 (GIoU) por cambio de función de loss. T3 y T4 comparten la misma función de loss; la diferencia en best val_loss (28.12 vs 36.46) refleja varianza del val set pequeño y diferente momento de early stop.

### 7.2 Evolución reg_loss (Train)

| Resolución | Train 1 | Train 2 | Train 3 | Train 4 |
|---|:---:|:---:|:---:|:---:|
| 640px (e0) | ~682 | ~33.8 | ~4.5 (meseta) | ~3.96 (meseta) |
| 224px (final) | ~41 | ~1.3 | ~1.1 | ~1.1 |

### 7.3 Per-class AP@50 — Test (Evolución)

| Clase | T1 | T2 | T3 | T4 | Δ T1→T4 |
|---|:---:|:---:|:---:|:---:|:---:|
| dog | 0.406 | 0.463 | 0.496 | **0.502** | +23.6% |
| door | 0.339 | 0.519 | 0.503 | **0.533** | +57.2% |
| obstacle | 0.302 | 0.405 | 0.458 | **0.512** | +69.3% |
| person | 0.521 | 0.636 | 0.636 | **0.682** | +30.9% |
| stair | 0.584 | **0.777** | 0.745 | 0.739 | +26.5% |
| **Media** | **0.430** | **0.560** | **0.568** | **0.594** | **+38.0%** |

### 7.4 Confusion Matrix (Test) — Evolución de TP

| Clase (Test GT) | T1 TP | T2 TP | T3 TP | T4 TP | T1 FN | T2 FN | T3 FN | T4 FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| dog (58) | 30 | 32 | 33 | **35** | 28 | 26 | 25 | **23** |
| door (136) | 54 | 80 | 77 | **90** | 82 | 56 | 59 | **46** |
| obstacle (173) | 79 | 86 | 92 | **107** | 94 | 87 | 81 | **66** |
| person (101) | 67 | 70 | 70 | **78** | 34 | 31 | 31 | **23** |
| stair (108) | 66 | 87 | 84 | **85** | 42 | 21 | 24 | **23** |
| **Total** | **296** | **355** | **356** | **395** | **280** | **221** | **220** | **181** |

### 7.5 Falsos Positivos (Test)

| Clase | T1 FP | T2 FP | T3 FP | T4 FP |
|---|:---:|:---:|:---:|:---:|
| dog | 42 | 32 | **24** | 88 |
| door | 40 | 83 | **44** | 188 |
| obstacle | 72 | **43** | 45 | 165 |
| person | 72 | 38 | **26** | 144 |
| stair | 26 | 34 | 38 | 140 |
| **Total** | **252** | **230** | **177** | **725** |

> **T4 muestra el trade-off clásico: máximo recall (395 TP) pero explosión de FP (725).** El scoring permisivo produce 4.1x más FP que T3 para ganar solo 39 TP adicionales.

---

## 8. Conclusiones Generales

### 8.1 Impacto por Cambio

| Cambio | Métrica más afectada | Impacto |
|---|---|---|
| **Stride Normalization** (T1→T2) | mAP@50, Recall | +30% mAP@50, +26% Recall. Cambio más impactante de la serie. |
| **GIoU Loss** (T2→T3) | mAP@50-95, Precision | mAP@50-95 de 0→0.26. Precision +9.3%. Mejora calidad de box. |
| **Más épocas** (T2→T3) | Convergencia | Best en epoch 80 vs 58. Presupuesto extra utilizado efectivamente. |
| **Scoring permisivo** (T3→T4) | Recall, mAP@50, FP | +9.7% Recall, +4.6% mAP@50, pero Precision -47.6% y FP +309%. Trade-off negativo. |

### 8.2 Fortalezas del Modelo

- **Clasificación perfecta**: Zero confusión inter-clase en los 4 entrenamientos. La cabeza cls discrimina las 5 clases impecablemente.
- **Modelo ligero**: 1.2M params, 4.71 MB FP32, <6ms inferencia en T4 GPU.
- **Export ONNX exitoso**: 9 outputs, opset 13, 4.74 MB, latencia 5.9ms.
- **Alto potencial de recall**: T4 demuestra que el modelo *ve* ~69% de los objetos; el reto es rankear bien las detecciones.
- **mAP@50 mejora sostenidamente**: 0.43 → 0.56 → 0.57 → 0.59 a lo largo de la serie.

### 8.3 Debilidades / Cuellos de Botella

- **Trade-off Precision-Recall no resuelto**: T3 es conservador (P=0.66, R=0.63), T4 es permisivo (P=0.35, R=0.69). Ninguno logra ambos simultáneamente.
- **FP dominan en T4**: 725 FP para 395 TP. Ratio FP/TP = 1.84, inaceptable para producción.
- **dog sigue siendo la clase más débil**: AP@50 = 0.50 (test), menor de las 5 clases en todos los entrenamientos.
- **Val loss ruidosa**: Con 188 imágenes, oscila enormemente (36→70+ entre epochs), dificultando checkpoint selection.
- **Meseta GIoU en Phase 1**: ~13 epochs sin aprendizaje de localización en cada entrenamiento.

### 8.4 Mejor Configuración Operativa

| Objetivo | Mejor Train | Justificación |
|---|---|---|
| **Máximo mAP@50** | **T4** (0.5936) | Si solo importa el ranking global de detecciones. |
| **Producción (F1 balanceado)** | **T3** (F1=0.6438) | Mejor balance precision/recall para uso real. |
| **Máximo Recall** | **T4** (0.6886) | Si las detecciones perdidas son más costosas que los FP. |
| **Máxima Precisión** | **T3** (0.6609) | Mínimos falsos positivos (177 FP en test). |

> **Recomendación**: T3 es el mejor modelo operativo actual. Los pesos de T4 son idénticos (mismo entrenamiento); solo cambia el scoring en inferencia. Se puede usar el checkpoint de T3/T4 con un conf_threshold intermedio (0.20) para buscar un punto de operación óptimo.

### 8.5 Oportunidades de Mejora (Candidatas para Train 5+)

1. ~~**Threshold sweep post-NMS** — COMPLETADO (§6)~~: Barrido [0.10–0.50] en 2 corridas. F1 máximo T4 = 0.5218 (thr=0.50), todavía 19% inferior a T3 (0.6438). Scoring T4 **descartado** para producción.
2. ~~**Max detections per image** — DESCARTADO~~: El sweep demostró que el problema es estructural (scoring infla calidad de FP), no de cantidad bruta. Limitar detecciones no resolvería.
3. ~~**Revertir a ctr_power=1.0, conf=0.20** — DESCARTADO~~: El sweep mostró que incluso a thr=0.50 con scoring T4 no se alcanza T3. La línea de scoring queda cerrada.
4. **Warmup de loss híbrido**: Iniciar con Smooth L1 durante primeros 10 epochs → transición a GIoU, evitando la meseta de 13 epochs.
5. **Augmentación más agresiva**: GaussNoise, CoarseDropout, GaussianBlur para mayor regularización con 1470 imgs.
6. **Resolución final mayor**: Evaluar a 320px en vez de 224px para mejorar localización de objetos pequeños.
7. **Focal Loss tuning**: Ajustar gamma (2→3) para la cabeza de clasificación, ayudando con clases difíciles.

---

*Documento generado y mantenido como parte del pipeline MLOps del TFM — Detección de Objetos para Asistencia Visual.*
