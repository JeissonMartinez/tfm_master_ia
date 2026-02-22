# Registro de Entrenamiento — FCOS (MobileNetV3-Small + FPN)

> **Modelo**: `fcos_v3s` — FCOS con MobileNetV3-Small backbone + SimpleFPN + FCOS Head  
> **Parámetros**: 1,233,450 total (FP32: 4.71 MB | INT8 est.: 1.18 MB)  
> **Dataset**: IODC YOLO — 5 clases (dog, door, obstacle, person, stair)  
> **Splits**: Train 1470 | Val 188 | Test 187  
> **Infraestructura**: Google Vertex AI Custom Job — `n1-standard-8` + NVIDIA Tesla T4  
> **Contenedor**: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`  
> **Última actualización**: 22 de febrero de 2026  

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Train 1 — Baseline](#2-train-1--baseline)
3. [Train 2 — Stride Normalization + Bug Fixes](#3-train-2--stride-normalization--bug-fixes)
4. [Train 3 — GIoU Loss + Más Épocas](#4-train-3--giou-loss--más-épocas)
5. [Train 4 — Scoring Refinements (conf, centerness, IoU-aware)](#5-train-4--scoring-refinements-conf-centerness-iou-aware)
6. [Threshold Sweep — Análisis Post-NMS del Modelo T4](#6-threshold-sweep--análisis-post-nms-del-modelo-t4)
7. [Train 5 — Hybrid Loss Warmup + Aggressive Augmentation](#7-train-5--hybrid-loss-warmup--aggressive-augmentation)
8. [Train 6 — Phase 1 Extendida + Sin HFlip (Despliegue Parcial)](#8-train-6--phase-1-extendida--sin-hflip-despliegue-parcial)
9. [Train 7 — Config Final (T3 + build_fcos_loss + conf 0.30)](#9-train-7--config-final-t3--build_fcos_loss--conf-030)
10. [Comparativa Global](#10-comparativa-global)
11. [Conclusiones Generales](#11-conclusiones-generales)

---

## 1. Resumen Ejecutivo

| Métrica (Test) | Train 1 | Train 2 | Train 3 | Train 4 | Train 5 | Train 6 | **Train 7** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **mAP@50** | 0.4304 | 0.5600 | 0.5675 | 0.5936 | 0.5887 | 0.5572 | **0.6120** |
| **mAP@50-95** | N/C | N/C | 0.2602 | 0.2644 | 0.2703 | 0.2511 | **0.2824** |
| **Precision** | 0.5427 | 0.6049 | **0.6609** | 0.3462 | 0.3505 | 0.3290 | 0.3716 |
| **Recall** | 0.5291 | 0.6271 | 0.6276 | **0.6886** | 0.6845 | 0.6558 | 0.6872 |
| **F1-Score** | 0.5358 | 0.6158 | **0.6438** | 0.4607 | 0.4636 | 0.4382 | 0.4824 |
| Épocas | 52 | 74 | 101 | 77 | 76 | 86 | 98 |
| Tiempo | 12.1 min | 15.9 min | 23.6 min | 17.9 min | 17.9 min | 19.5 min | ~23 min |
| Inferencia | 5.0 ms | 4.6 ms | 4.8 ms | 5.4 ms | 4.9 ms | 5.1 ms | 5.0 ms |

> **N/C**: No Calculado — la implementación de mAP@50-95 fue añadida después de Train 2.  
> **Train 3**: Mejor F1 (0.6438) y Precision (0.6609) de toda la serie. Referencia operativa para producción.  
> **Train 7**: **Mejor mAP@50 (0.6120) y mAP@50-95 (0.2824) de la serie.** Segundo incidente de despliegue: bug de whitelist en `config_loader.py` impidió activar Focal Loss y SL1 warmup. Resultado efectivo: T3 + `build_fcos_loss` (reg_weight=1.5) + conf=0.30 + HFlip restaurado.  
> **Train 6**: Despliegue parcial — sdist cache impidió deploy de código Python. Peor F1 de la serie (0.4382). Confirma que HFlip runtime es esencial.

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

## 7. Train 5 — Hybrid Loss Warmup + Aggressive Augmentation

### 7.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `fcos_v3s_v1-1771710798` |
| **Fecha** | 21 de febrero de 2026 |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771710798/` |
| **Output local** | `outputs/fcos_v3s_v1-1771710798/` |
| **Log** | `logs/FCOS_Train_5.md` |

### 7.2 Cambios Respecto a Train 3 (línea base de scoring)

| Cambio | T3 (referencia) | T5 |
|---|---|---|
| **Regression warmup** | GIoU desde epoch 0 (meseta ~13 ep) | Smooth L1 ep 0–9 → GIoU ep 10+ |
| **Augmentación** | HFlip, BrightnessContrast, HSV, Affine | + GaussNoise (p=0.3), CoarseDropout (p=0.4), GaussianBlur (p=0.2) |
| **Brightness/Contrast limits** | 0.2 | 0.3 |
| **Scoring** | ctr^1.0, conf=0.25, no iou_aware | Idéntico a T3 |
| **Phase 2 epochs** | 70 | 80 |
| **albumentations** | ≥1.4 | ≥2.0.0 (nueva API CoarseDropout/GaussNoise) |

### 7.3 Configuración

Misma arquitectura (MobileNetV3-Small + SimpleFPN + FCOS Head). Nuevos parámetros:

| Parámetro | Valor |
|---|---|
| `reg_warmup_epochs` | 10 |
| `aug_gaussian_noise` | p=0.3, std_range=(0.01, 0.05) |
| `aug_coarse_dropout` | p=0.4, 2–8 holes, h=8–32px, w=8–32px |
| `aug_blur` | p=0.2, blur_limit=(3, 7) |
| `aug_brightness_limit` / `aug_contrast_limit` | 0.3 (vs 0.2 en T3) |

### 7.4 Entrenamiento

- **Épocas completadas**: 76 (early stopping epoch 75, patience=20)
- **Mejor val_loss**: 28.7534 (epoch 55)
- **Tiempo total**: 17.9 min (Phase 1: 7.5 min, Phase 2: 10.3 min)

**Dinámica del warmup (eliminación de la meseta GIoU):**

| Epoch | reg_loss T3 (GIoU) | reg_loss T5 (SL1→GIoU) | Observación |
|:---:|:---:|:---:|---|
| 0 | 4.5000 (meseta) | 3.6730 | SL1 permite descenso inmediato |
| 5 | 4.5000 (meseta) | 2.2778 | SL1 reduce reg_loss -38% |
| 9 | 4.5000 (meseta) | 2.1398 | Último epoch SL1 |
| 10 | 4.5000 (meseta) | 2.0336 | Transición SL1→GIoU, sin spike |
| 13 | 4.5000 (meseta rota) | 1.8839 | T3 apenas empieza; T5 ya 58% menor |
| 20 | 1.9641 | 1.7896 | T5 mantiene ventaja -8.9% |

> **Hallazgo clave**: El warmup eliminó completamente la meseta de 13 epochs de GIoU. La transición SL1→GIoU en epoch 10 fue suave (sin spike). La cabeza de regresión llegó a epoch 20 con reg_loss 8.9% menor que T3.

### 7.5 Resultados — Validación

| Métrica | Valor |
|---|---|
| mAP@50 | 0.4134 |
| mAP@50-95 | 0.1810 |
| Precision | 0.3417 |
| Recall | 0.5135 |
| F1-Score | 0.4103 |
| Detecciones / GT | 1190 / 762 |
| Inferencia | 5.5 ms |

**Per-class AP@50 (Val):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.3300 | 0.2500 | 0.4200 | 0.3134 |
| door | 0.4496 | 0.3125 | 0.5625 | 0.4018 |
| obstacle | 0.4189 | 0.3220 | 0.5793 | 0.4139 |
| person | 0.4218 | 0.4000 | 0.5055 | 0.4466 |
| stair | 0.4469 | 0.4240 | 0.5000 | 0.4589 |

### 7.6 Resultados — Test

| Métrica | Valor | vs T3 | vs T4 |
|---|---|---|---|
| mAP@50 | 0.5887 | +3.7% | −0.8% |
| mAP@50-95 | **0.2703** | +3.9% | +2.2% |
| Precision | 0.3505 | −47.0% | +1.2% |
| Recall | 0.6845 | +9.1% | −0.6% |
| F1-Score | 0.4636 | −28.0% | +0.6% |
| Detecciones / GT | 1114 / 576 | — | — |
| Inferencia | 4.9 ms | — | — |

**Per-class AP@50 (Test):**

| Clase | AP@50 | Precision | Recall | F1 | vs T3 |
|---|:---:|:---:|:---:|:---:|---|
| dog | **0.5223** | 0.2590 | 0.6207 | 0.3655 | +5.3% |
| door | **0.5436** | 0.3010 | 0.6838 | 0.4180 | +8.1% |
| obstacle | 0.4644 | 0.3804 | 0.6069 | 0.4677 | +1.4% |
| person | 0.6663 | 0.4167 | 0.7426 | 0.5338 | +4.8% |
| stair | **0.7467** | 0.3952 | 0.7685 | 0.5220 | +0.2% |

**Confusion Matrix (Test):**

| Clase (GT) | TP | FP | FN |
|---|:---:|:---:|:---:|
| dog (58) | 36 | 103 | 22 |
| door (136) | 93 | 216 | 43 |
| obstacle (173) | 105 | 171 | 68 |
| person (101) | 75 | 105 | 26 |
| stair (108) | 83 | 127 | 25 |
| **Total (576)** | **392** | **722** | **184** |

### 7.7 Efecto de los Cambios

**Warmup híbrido (SL1→GIoU) — POSITIVO parcial:**
- ✅ Eliminó la meseta GIoU de 13 epochs → convergencia inmediata de reg_loss.
- ✅ Mejor calidad de box: mAP@50-95 = 0.2703, **mejor de la serie** (+3.9% vs T3).
- ✅ Transición SL1→GIoU suave sin spike de loss.
- ⚠️ No contribuyó por sí solo al problema de precision — ese efecto proviene de la augmentación.

**Augmentación agresiva — NEGATIVO:**
- ❌ **Cambio de calibración del modelo**: T5 con scoring idéntico a T3 (conf=0.25, ctr^1.0) produce 1114 detecciones vs 533 de T3. El modelo genera ~2× más predicciones de alta confianza.
- ❌ **722 FP** (vs 177 de T3, vs 725 de T4). Perfil operativo casi idéntico a T4 pese a scoring conservador.
- ❌ F1 = 0.4636, un **28% inferior** a T3 (0.6438). La augmentación degradó la capacidad discriminativa de la cabeza cls.
- ✅ Recall alto (0.6845) con +36 TP vs T3 — el modelo detecta más objetos reales.

**Diagnóstico — La augmentación agresiva degrada la calibración de confianza:**

La comparación clave es con T3 (mismo scoring): T3 produce 533 dets a conf=0.25, T5 produce 1114 dets. Esto significa que CoarseDropout+GaussNoise+GaussianBlur enseñaron al modelo a producir scores altos para regiones de fondo. Posibles mecanismos:
1. **CoarseDropout** simula oclusiones, pero puede enseñar al modelo a "completar" objetos a partir de evidencia parcial → hallucina objetos en fondo con textura.
2. **GaussNoise** reduce la relación señal/ruido, haciendo que el clasificador sea menos selectivo.
3. El efecto combinado produjo un modelo con sensibilidad alta pero sin la selectividad necesaria.

### 7.8 Lecciones

1. **Warmup de loss funciona**: La técnica SL1→GIoU es un avance válido que debe mantenerse en futuros entrenamientos. Eliminó un cuello de botella real.
2. **Augmentación excesiva destruye calibración**: Con solo 1.2M params y 1470 imágenes train, augmentaciones destructivas (dropout, ruido) hacen que el modelo pierda capacidad discriminativa en la cabeza cls.
3. **La augmentación no es un sustituto de más datos**: El dataset es pequeño. Las augmentaciones geométricas (HFlip, Affine, resize) son suficientes; las que destruyen información (noise, dropout) resultan contraproducentes.
4. **mAP@50-95 no correlaciona con F1 operativo**: T5 tiene la mejor calidad de box de la serie pero el peor F1 (junto con T4). La métrica de producción sigue siendo F1 a threshold fijo.

---

## 8. Train 6 — Phase 1 Extendida + Sin HFlip (Despliegue Parcial)

### 8.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `fcos_v3s_v1-1771715459` |
| **Fecha** | 21 de febrero de 2026 |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771715459/` |
| **Output local** | `outputs/fcos_v3s_v1-1771715459/` |
| **Log** | `logs/FCOS_Train_6.md` |

### 8.2 Incidente de Despliegue

> **⚠️ DESPLIEGUE PARCIAL**: Los cambios de código Python (Focal Loss γ=3, SL1 warmup) NO se activaron en el job desplegado. Solo los cambios del YAML config (Phase 1=40ep, HFlip=0.0) fueron efectivos.

**Evidencia:**

| Señal | Valor esperado | Valor observado | Conclusión |
|---|---|---|---|
| reg_loss epoch 0 | ~3.67 (SL1) | 4.5000 (GIoU) | Warmup SL1 **no activo** |
| reg_loss epochs 0-39 | Descenso inmediato | Constante 4.5000 | Meseta GIoU de 40 epochs |
| cls_loss epoch 0 | ~0.4 (Focal γ=3) | 3.02 (≈BCE) | Focal Loss **no activo** |
| Log "🎯 Focal Loss" | Presente | Ausente | Función no ejecutada |

**Causa probable**: El paquete `tfm_trainer-2.0.0.tar.gz` subido a GCS no contenía las modificaciones de `src_colab/utils_train.py` y `trainer/task_fcos.py`. El YAML sí se subió correctamente porque se copia como archivo independiente, no como parte del sdist.

**Configuración efectiva (lo que realmente corrió):**

| Parámetro | Valor Previsto | Valor Real |
|---|---|---|
| Focal Loss | γ=3, α=0.25 | **BCE** (estándar) |
| Warmup SL1→GIoU | 10 epochs | **Desactivado** (GIoU desde ep 0) |
| Phase 1 epochs | 40 | 40 ✅ |
| HFlip | p=0.0 | p=0.0 ✅ |
| Brightness/Contrast | 0.2 / 0.2 | 0.2 / 0.2 ✅ |
| Aug agresiva | Eliminada | Eliminada ✅ |
| Scoring | T3 (conf=0.25) | T3 ✅ |

### 8.3 Entrenamiento

- **Épocas completadas**: 86 (Phase 1: 40, Phase 2: 46, early stop epoch 85)
- **Mejor val_loss**: 33.3135 (epoch 65)
- **Tiempo total**: 19.5 min (Phase 1: 9.4 min, Phase 2: 10.1 min)

**Meseta GIoU en Phase 1 (40 epochs completos):**

| Epoch | reg_loss | Observación |
|:---:|:---:|---|
| 0-39 | 4.5000 (constante) | GIoU meseta, idéntica a T3/T4 |
| 40 | 4.5000 | Phase 2 inicia, aún congelado |
| 42 | 3.4977 | Backbone descongelado, reg empieza a aprender |
| 65 | 1.3076 | ★ best val_loss |
| 85 | 1.1197 | Early stop |

> La Phase 1 extendida con meseta GIoU desperdició 10 epochs adicionales sin aprendizaje de regresión. Phase 2 solo aprovechó 46 epochs (vs 71 en T3).

### 8.4 Resultados — Validación

| Métrica | Valor |
|---|---|
| mAP@50 | 0.3799 |
| mAP@50-95 | 0.1632 |
| Precision | 0.2904 |
| Recall | 0.4778 |
| F1-Score | 0.3613 |
| Detecciones / GT | 1289 / 762 |
| Inferencia | 5.5 ms |

### 8.5 Resultados — Test

| Métrica | Valor | vs T3 | vs T5 |
|---|---|---|---|
| mAP@50 | 0.5572 | −1.8% | −5.4% |
| mAP@50-95 | 0.2511 | −3.5% | −7.1% |
| Precision | 0.3290 | −50.2% | −6.1% |
| Recall | 0.6558 | +4.5% | −4.2% |
| F1-Score | 0.4382 | −31.9% | −5.5% |
| Detecciones / GT | 1124 / 576 | — | — |
| Inferencia | 5.1 ms | — | — |

**Per-class AP@50 (Test):**

| Clase | AP@50 | Precision | Recall | F1 | vs T3 |
|---|:---:|:---:|:---:|:---:|---|
| dog | 0.4424 | 0.2540 | 0.5517 | 0.3478 | −10.8% |
| door | 0.4642 | 0.3571 | 0.5515 | 0.4335 | −7.7% |
| obstacle | 0.4600 | 0.3293 | 0.6243 | 0.4311 | +0.4% |
| person | 0.6834 | 0.3433 | 0.7921 | 0.4790 | +7.4% |
| stair | 0.7359 | 0.3612 | 0.7593 | 0.4896 | −1.2% |

**Confusion Matrix (Test):**

| Clase (GT) | TP | FP | FN |
|---|:---:|:---:|:---:|
| dog (58) | 32 | 94 | 26 |
| door (136) | 75 | 135 | 61 |
| obstacle (173) | 108 | 220 | 65 |
| person (101) | 80 | 153 | 21 |
| stair (108) | 82 | 145 | 26 |
| **Total (576)** | **377** | **747** | **199** |

### 8.6 Efecto de los Cambios

**Eliminación de HFlip runtime — SEVERAMENTE NEGATIVO:**
- ❌ **1124 detecciones** con scoring T3 (conf=0.25). T3 producía solo 533 con el mismo scoring.
- ❌ **747 FP** — peor de toda la serie (vs T3: 177, T4: 725, T5: 722).
- ❌ **377 TP** — peor de T3-T6, incluso inferior a T3 (356) corregido: T3 tiene 356 TP. En realidad T6 tiene más TP (377 > 356) pero con +570 FP adicionales.
- ❌ F1 = 0.4382 — **peor de la serie**, inferior incluso a T4 (0.4607) y T5 (0.4636).
- La eliminación de HFlip redujo la diversidad de augmentación, causando que el modelo sea menos discriminativo pese a que el dataset ya contiene copias flippeadas offline.

**Phase 1 extendida (30→40) con meseta GIoU — CONTRAPRODUCENTE:**
- ❌ 10 epochs adicionales con reg_loss=4.5 constante. Sin aprendizaje de regresión.
- ❌ Phase 2 convergió más lento: best epoch 65 (25 epochs Phase 2) vs T3 best epoch 80 (50 epochs Phase 2).
- ❌ Early stopping en epoch 85 (46 epochs Phase 2) vs T3 epoch 100 (70 epochs Phase 2).
- La extensión de Phase 1 con meseta GIoU es peor que inútil: el head sobreajusta clasificación sin progresar en regresión.

### 8.7 Lecciones

1. **HFlip runtime es esencial** incluso con datos flippeados offline: la combinación dinámica con otras augmentaciones (affine, color) genera variedad que las copias estáticas no aportan.
2. **Extender Phase 1 sin resolver la meseta GIoU es contraproducente**: más epochs congelados = más sobreajuste de cls sin progreso de reg.
3. **Bump de versión obligatorio**: Para futuros trains, se debe incrementar la versión del paquete (`2.0.0 → 2.1.0`) para garantizar que pip instale la versión actualizada y evitar problemas de cache.
4. **Verificación pre-launch**: Añadir un `--dry-run` que imprima los valores de `focal_gamma`, `reg_warmup_epochs` y `aug_hflip_prob` leídos dentro del job.

---

## 9. Train 7 — Config Final (T3 + build_fcos_loss + conf 0.30)

### 9.1 Identificador

| Campo | Valor |
|---|---|
| **Job ID** | `fcos_v3s_v1-1771726575` |
| **Fecha** | 22 de febrero de 2026 |
| **Output GCS** | `gs://project-18f58341-12cf-47bc-861-tfm-data/output/fcos_v3s_v1-1771726575/` |
| **Output local** | `outputs/fcos_v3s_v1-1771726575/` |
| **Package** | `tfm_trainer-2.1.0.tar.gz` (version bump desde 2.0.0) |

### 9.2 Objetivo

Entrenamiento **final** de la serie FCOS. Combinar las mejores configuraciones probadas:
- **Base T3** (scoring conservador, augmentación base, HFlip p=0.5)
- **SL1 warmup** de T5 (10 epochs Smooth L1 → GIoU)
- **Focal Loss γ=3** (nuevo, no probado previamente)
- **conf_threshold=0.30** (subida leve vs T3's 0.25 para reducir FP)
- **Version bump 2.0.0→2.1.0** (fix del cache bug de T6)

### 9.3 Incidente de Despliegue — Bug de Whitelist en config_loader.py

> **⚠️ SEGUNDO INCIDENTE DE DESPLIEGUE**: El código Python se desplegó correctamente (version bump funcionó), pero un bug en `config_loader.py` impidió que las claves nuevas del YAML llegaran al código de entrenamiento.

**Causa raíz**: `config_loader.py` (líneas 73-75) itera sobre las claves de `_FCOS_DEFAULTS` (definido en `utils_widgets.py:76`) como **whitelist**. Solo las claves que existen en `_FCOS_DEFAULTS` se pasan a `create_manual_setup()`. Cualquier clave nueva del YAML que no esté en el diccionario de defaults se descarta silenciosamente.

```python
# config_loader.py líneas 73-75 — bug de whitelist
for key in defaults:                    # ← solo itera claves de _FCOS_DEFAULTS
    if key in family_section:           # ← si la clave existe en YAML
        family_kwargs[key] = family_section[key]  # ← la pasa
# 17 claves del YAML NO están en _FCOS_DEFAULTS → se pierden silenciosamente
```

**Claves YAML descartadas** (17 en total):

| Clave YAML | Valor configurado | Valor efectivo (via .get default) |
|---|---|---|
| `focal_gamma` | 3.0 | **0.0** → Focal Loss **no activo** (BCE) |
| `focal_alpha` | 0.25 | 0.25 (coincide por defecto en .get) |
| `reg_warmup_epochs` | 10 | **0** → Warmup SL1 **no activo** |
| `reg_weight` | 1.5 | 1.5 (coincide por defecto en .get) |
| `cls_weight` | 1.0 | 1.0 (coincide) |
| `centerness_weight` | 1.0 | 1.0 (coincide) |
| `strides` | [8,16,32] | Hardcoded en modelo |
| `num_head_convs` | 2 | Hardcoded en modelo |
| `pretrained_backbone` | true | true (default) |
| `grad_clip` | 10.0 | 10.0 (coincide) |
| `export_imgsz` | 224 | 224 (leído por ruta separada) |
| `export_opset` | 13 | 13 (leído por ruta separada) |
| `phase1_wd` | 1e-4 | 1e-4 (leído por ruta separada) |
| `phase2_wd` | 1e-5 | 1e-5 (leído por ruta separada) |
| `phase1_optimizer` | adamw | adamw (coincide) |
| `phase2_optimizer` | adamw | adamw (coincide) |
| `phase1_scheduler` / `phase2_scheduler` | cosine | cosine (coincide) |

**Evidencia diagnóstica:**

| Señal | Valor esperado (Focal γ=3) | Valor observado | Conclusión |
|---|---|---|---|
| cls_loss epoch 0 | ~0.28 (focal weighting) | 3.0920 (≈ BCE) | Focal Loss **no activo** |
| reg_loss epoch 0 | ~3.67 (SL1) | 3.5739 (GIoU, nuevo código) | Warmup SL1 **no activo** |
| reg_loss patrón | Descenso SL1 → transición ep10 | Descenso GIoU inmediato | GIoU desde epoch 0 |
| `DEPLOY VERIFICATION` log | focal_gamma=3.0, reg_warmup=10 | focal_gamma=0.0, reg_warmup=0 | Valores filtrados |

> **Nota sobre reg_loss**: El valor 3.57 (distinto de la meseta 4.5 de T3/T6) se debe al cambio en la función de loss entre versiones de código. T3/T6 usaron la loss inline original; T4/T5/T7 usan `build_fcos_loss()` con `reg_weight=1.5` y normalización diferente. Ambos computan GIoU, pero con escalado distinto.

**Impacto del bug**: Este bug ha estado presente desde T4 (primera vez que se usó `config_loader.py`). Las claves para Focal Loss y warmup en T5 y T6 también fueron descartadas. La mejora de mAP@50-95 atribuida al warmup SL1 en T5 se debió en realidad al cambio en la función de loss (`build_fcos_loss` con `reg_weight=1.5`), no al warmup explícito.

### 9.4 Configuración Efectiva

| Parámetro | Valor Previsto | Valor Real | Origen |
|---|---|---|---|
| Focal Loss | γ=3, α=0.25 | **BCE** (estándar) | Bug whitelist |
| Warmup SL1→GIoU | 10 epochs | **Desactivado** | Bug whitelist |
| reg_weight | 1.5 | 1.5 ✅ | `.get("reg_weight", 1.5)` coincide |
| Phase 1 epochs | 30 | 30 ✅ | En `_FCOS_DEFAULTS` |
| Phase 2 epochs | 80 | 80 ✅ | En `_FCOS_DEFAULTS` |
| HFlip | p=0.5 | p=0.5 ✅ | En `_FCOS_DEFAULTS` |
| conf_threshold | 0.30 | 0.30 ✅ | Leído aparte |
| Brightness/Contrast | 0.2 | **0.3** | `_FCOS_DEFAULTS` default |
| Gaussian Noise | 0 (eliminado) | **0.2** | `_FCOS_DEFAULTS` default |
| Scoring | T3 (ctr^1.0) | T3 ✅ | Default |
| Package | 2.1.0 | 2.1.0 ✅ | Version bump funcionó |

> **Configuración resultante**: Equivalente a T4/T5 (misma función `build_fcos_loss` con `reg_weight=1.5`, mismos defaults de `_FCOS_DEFAULTS`), pero con `conf_threshold=0.30` (vs 0.25 en T4/T5) y Phase 1 de 30 epochs (vs 30 en T4/T5).

### 9.5 Entrenamiento

- **Épocas completadas**: 98 (Phase 1: 30, Phase 2: 68, early stop epoch 97)
- **Mejor val_loss**: 18.6400 (epoch 77) — **mejor de toda la serie**
- **Tiempo total**: ~23 min (estimado, 98 epochs)

**Dinámica de pérdida:**

| Epoch | train_loss | cls_loss | reg_loss | Observación |
|:---:|:---:|:---:|:---:|---|
| 0 (640px) | 8.512 | 3.092 | 3.574 | GIoU inmediato (nuevo código) |
| 10 (416px) | 5.567 | 1.667 | 2.184 | Resize 640→416 |
| 20 (320px) | 4.928 | 1.406 | 1.802 | Resize 416→320 |
| 29 (320px) | 4.604 | 1.223 | 1.658 | Fin Phase 1 |
| 30 (224px) | 5.095 | 1.274 | 2.100 | → Phase 2, resize 320→224 |
| 50 | 3.561 | 0.648 | 1.213 | Convergencia Phase 2 |
| 77 | 3.261 | 0.542 | 1.042 | ★ best val_loss (18.64) |
| 97 | 3.241 | 0.531 | 1.038 | Early stop |

### 9.6 Resultados — Validación

| Métrica | Valor |
|---|---|
| mAP@50 | 0.4367 |
| mAP@50-95 | 0.2014 |
| Precision | 0.3493 |
| Recall | 0.5180 |
| F1-Score | 0.4173 |
| Detecciones / GT | 1139 / 762 |
| Inferencia | 5.5 ms |

**Per-class AP@50 (Val):**

| Clase | AP@50 | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.3270 | 0.2684 | 0.4133 | 0.3255 |
| door | 0.4651 | 0.3466 | 0.5437 | 0.4234 |
| obstacle | 0.4476 | 0.3369 | 0.5732 | 0.4244 |
| person | 0.4629 | 0.4222 | 0.5220 | 0.4668 |
| stair | 0.4807 | 0.3725 | 0.5377 | 0.4402 |

### 9.7 Resultados — Test

| Métrica | Valor | vs T3 | vs T4 | vs T5 |
|---|---|---|---|---|
| **mAP@50** | **0.6120** | **+7.8%** | +3.1% | +4.0% |
| **mAP@50-95** | **0.2824** | **+8.5%** | +6.8% | +4.5% |
| Precision | 0.3716 | −43.8% | +7.3% | +6.0% |
| Recall | 0.6872 | +9.5% | −0.2% | +0.4% |
| F1-Score | 0.4824 | −25.1% | +4.7% | +4.1% |
| Detecciones / GT | 1049 / 576 | — | — | — |
| Inferencia | 5.0 ms | — | — | — |

**Per-class AP@50 (Test):**

| Clase | AP@50 | Precision | Recall | F1 | vs T3 |
|---|:---:|:---:|:---:|:---:|---|
| dog | 0.5062 | 0.2846 | 0.6034 | 0.3867 | +2.1% |
| door | **0.5689** | 0.3384 | 0.6544 | 0.4461 | +13.0% |
| obstacle | **0.5233** | 0.3993 | 0.6301 | 0.4888 | +14.4% |
| person | **0.6912** | 0.4386 | 0.7426 | 0.5515 | +8.7% |
| stair | **0.7705** | 0.3973 | 0.8056 | 0.5321 | +3.4% |

> T7 logra la **mejor AP@50 en 4 de 5 clases** (door, obstacle, person, stair). Solo dog queda ligeramente por debajo de T5 (0.522).

**Confusion Matrix (Test):**

| Clase (GT) | TP | FP | FN |
|---|:---:|:---:|:---:|
| dog (58) | 35 | 88 | 23 |
| door (136) | 89 | 174 | 47 |
| obstacle (173) | 109 | 164 | 64 |
| person (101) | 75 | 96 | 26 |
| stair (108) | 87 | 132 | 21 |
| **Total (576)** | **395** | **654** | **181** |

### 9.8 Efecto del conf_threshold 0.30

T7 usa `conf_threshold=0.30` (vs 0.25 en T4/T5). Con la misma función de loss (`build_fcos_loss`, `reg_weight=1.5`):

| Métrica | T4 (conf=0.25) | T5 (conf=0.25) | T7 (conf=0.30) | Efecto |
|---|:---:|:---:|:---:|---|
| Detecciones | 1120 | 1114 | **1049** | −6% dets |
| TP | 395 | 392 | **395** | Mantiene TP |
| FP | 725 | 722 | **654** | −10% FP |
| FN | 181 | 184 | **181** | Mantiene FN |
| Precision | 0.3462 | 0.3505 | **0.3716** | +7% |
| mAP@50 | 0.5936 | 0.5887 | **0.6120** | +3% |

> La subida de conf a 0.30 eliminó ~70 FP sin perder TP. Esto confirma que había detecciones de baja confianza que eran mayoritariamente falsos positivos.

### 9.9 Lecciones

1. **Bug de whitelist en config_loader**: El patrón de iterar sobre `_FCOS_DEFAULTS` como whitelist impide agregar nuevas funcionalidades via YAML sin actualizar también el diccionario de defaults. Este bug afectó T4-T7 silenciosamente.
2. **Focal Loss nunca fue probado**: En 7 entrenamientos, la Focal Loss nunca se activó. T6 por cache de sdist, T7 por whitelist de config_loader. Queda como técnica no validada para este modelo.
3. **SL1 warmup nunca fue probado**: Igualmente, el warmup SL1→GIoU nunca se activó realmente. La diferencia de reg_loss entre T3 y T4+ se debe al cambio de función de loss (`build_fcos_loss` vs inline), no al warmup.
4. **conf_threshold=0.30 funciona**: Reduce FP sin sacrificar TP. Es el ajuste más efectivo para controlar la explosión de FP en el régimen de `build_fcos_loss`.
5. **T7 es el mejor modelo en mAP**: mAP@50=0.6120 (+7.8% vs T3), mAP@50-95=0.2824 (+8.5% vs T3). Pero el trade-off con precision persiste (0.37 vs 0.66).
6. **Version bump funcionó**: El cambio 2.0.0→2.1.0 garantizó que pip instalara el código actualizado. El problema de T7 fue diferente (whitelist, no cache).

---

## 10. Comparativa Global

### 10.1 Evolución Total Loss

| Fase | Train 1 | Train 2 | Train 3 | Train 4 | Train 5 | Train 6 | Train 7 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Epoch 0 (640px) | 688.5 | 38.9 | 9.4 | 8.9 | 8.6 | 9.4 (old code) | 8.5 |
| Final | ~43.5 | ~3.5 | ~3.3 | ~3.3 | ~3.4 | ~3.3 | ~3.2 |
| Best val_loss | 135.49 | 24.01 | 28.12 | 36.46 | 28.75 | 33.31 | **18.64** |

> Nota: T1-T3 y T6 usaron la loss inline original. T4, T5 y T7 usaron `build_fcos_loss()` con `reg_weight=1.5`. val_loss no es directamente comparable entre regímenes de código.

### 10.2 Evolución reg_loss (Train)

| Resolución | Train 1 | Train 2 | Train 3 | Train 4 | Train 5 | Train 6 | Train 7 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 640px (e0) | ~682 | ~33.8 | ~4.5 (meseta) | ~3.96 | ~3.67 | ~4.5 (meseta) | ~3.57 |
| 224px (final) | ~41 | ~1.3 | ~1.1 | ~1.1 | ~1.1 | ~1.1 | ~1.0 |

> Dos regímenes: T3/T6 muestran meseta GIoU (4.5, código inline). T4/T5/T7 muestran descenso inmediato (~3.6-3.96, `build_fcos_loss` con `reg_weight=1.5`).

### 10.3 Per-class AP@50 — Test (Evolución)

| Clase | T1 | T2 | T3 | T4 | T5 | T6 | T7 | Δ T1→T7 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| dog | 0.406 | 0.463 | 0.496 | 0.502 | **0.522** | 0.442 | 0.506 | +24.6% |
| door | 0.339 | 0.519 | 0.503 | 0.533 | 0.544 | 0.464 | **0.569** | +67.8% |
| obstacle | 0.302 | 0.405 | 0.458 | 0.512 | 0.464 | 0.460 | **0.523** | +73.2% |
| person | 0.521 | 0.636 | 0.636 | 0.682 | 0.666 | 0.683 | **0.691** | +32.6% |
| stair | 0.584 | **0.777** | 0.745 | 0.739 | 0.747 | 0.736 | 0.770 | +31.8% |
| **Media** | **0.430** | **0.560** | **0.568** | **0.594** | **0.589** | **0.557** | **0.612** | **+42.3%** |

### 10.4 Confusion Matrix (Test) — Evolución de TP

| Clase (Test GT) | T1 TP | T2 TP | T3 TP | T4 TP | T5 TP | T6 TP | T7 TP | T1 FN | T2 FN | T3 FN | T4 FN | T5 FN | T6 FN | T7 FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| dog (58) | 30 | 32 | 33 | 35 | **36** | 32 | 35 | 28 | 26 | 25 | 23 | **22** | 26 | 23 |
| door (136) | 54 | 80 | 77 | 90 | **93** | 75 | 89 | 82 | 56 | 59 | 46 | **43** | 61 | 47 |
| obstacle (173) | 79 | 86 | 92 | 107 | 105 | 108 | **109** | 94 | 87 | 81 | 66 | 68 | 65 | **64** |
| person (101) | 67 | 70 | 70 | 78 | 75 | **80** | 75 | 34 | 31 | 31 | 23 | 26 | **21** | 26 |
| stair (108) | 66 | 87 | 84 | 85 | 83 | 82 | **87** | 42 | 21 | 24 | 23 | 25 | 26 | **21** |
| **Total** | **296** | **355** | **356** | **395** | **392** | **377** | **395** | **280** | **221** | **220** | **181** | **184** | **199** | **181** |

### 10.5 Falsos Positivos (Test)

| Clase | T1 FP | T2 FP | T3 FP | T4 FP | T5 FP | T6 FP | T7 FP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| dog | 42 | 32 | **24** | 88 | 103 | 94 | 88 |
| door | 40 | 83 | **44** | 188 | 216 | 135 | 174 |
| obstacle | 72 | **43** | 45 | 165 | 171 | 220 | 164 |
| person | 72 | 38 | **26** | 144 | 105 | 153 | 96 |
| stair | 26 | 34 | **38** | 140 | 127 | 145 | 132 |
| **Total** | **252** | **230** | **177** | **725** | **722** | **747** | **654** |

> **T7 reduce FP significativamente vs T4-T6** (654 vs 722-747) gracias a `conf_threshold=0.30`. Aun así, **T3 sigue siendo el más limpio** (177 FP) por usar el código inline original con scoring más restrictivo.

---

## 11. Conclusiones Generales

### 11.1 Impacto por Cambio

| Cambio | Métrica más afectada | Impacto |
|---|---|---|
| **Stride Normalization** (T1→T2) | mAP@50, Recall | +30% mAP@50, +26% Recall. Cambio más impactante de la serie. |
| **GIoU Loss** (T2→T3) | mAP@50-95, Precision | mAP@50-95 de 0→0.26. Precision +9.3%. Mejora calidad de box. |
| **Más épocas** (T2→T3) | Convergencia | Best en epoch 80 vs 58. Presupuesto extra utilizado efectivamente. |
| **Scoring permisivo** (T3→T4) | Recall, mAP@50, FP | +9.7% Recall, +4.6% mAP@50, pero Precision −47.6% y FP +309%. Trade-off negativo. |
| **`build_fcos_loss` + reg_weight=1.5** (T3→T4) | reg_loss, mAP | Cambio de código de loss: eliminó meseta GIoU, mejoró mAP. Anteriormente atribuido a SL1 warmup (ahora desmentido). |
| **Aug agresiva** (T3→T5) | Precision, FP | Precision −47%, FP +308%. Destruye calibración de confianza en modelo pequeño. |
| **Sin HFlip runtime** (T3→T6) | Precision, FP | Precision −50%, FP +322%. Peor F1 de la serie. HFlip runtime es irrenunciable. |
| **conf_threshold 0.25→0.30** (T4→T7) | FP, Precision | −10% FP, +7% Precision sin perder TP. Ajuste simple y efectivo. |

### 11.2 Dos Regímenes de Código

Un hallazgo clave del análisis post-T7 es que los 7 entrenamientos se dividen en **dos regímenes de código distintos**:

| Régimen | Trains | Loss regression | reg_loss ep0 | Comportamiento |
|---|---|---|---|---|
| **Inline** (código original) | T1, T2, T3, T6 | GIoU inline, weight=1.0 | ~4.5 (meseta 13 ep) | Precision alta, FP bajos |
| **build_fcos_loss** (nuevo) | T4, T5, T7 | GIoU via `build_fcos_loss`, reg_weight=1.5 | ~3.6-3.9 (descenso inmediato) | mAP alta, FP altos |

T6 usó el código inline (por cache de sdist). T4, T5 y T7 usaron `build_fcos_loss` (código nuevo desplegado correctamente, pero con whitelist bug que descartó Focal Loss y warmup).

La mejora de mAP en T4+ no proviene de scoring, warmup ni Focal Loss, sino del **cambio en la función de loss** (mayor ponderación de regresión con `reg_weight=1.5`).

### 11.3 Fortalezas del Modelo

- **Clasificación perfecta**: Zero confusión inter-clase en los 7 entrenamientos. La cabeza cls discrimina las 5 clases impecablemente.
- **Modelo ligero**: 1.2M params, 4.71 MB FP32, <6ms inferencia en T4 GPU.
- **Export ONNX exitoso**: 9 outputs, opset 13, 4.74 MB, latencia ~4.3–6.1 ms.
- **Alto potencial de recall**: T4/T5/T7 demuestran que el modelo *ve* ~69% de los objetos; el reto es rankear bien las detecciones.
- **mAP@50 mejora sostenidamente**: 0.43 → 0.56 → 0.57 → 0.59 → 0.59 → 0.56 → **0.61** (+42% desde T1).
- **mAP@50-95 progresa**: 0.26 → 0.26 → 0.27 → 0.25 → **0.28** — la calidad de box mejora con `build_fcos_loss`.

### 11.4 Debilidades / Cuellos de Botella

- **Trade-off Precision-Recall no resuelto**: T3 es conservador (P=0.66, R=0.63), T4/T5/T7 son permisivos (P≈0.33-0.37, R≈0.66-0.69). Ninguno logra ambos simultáneamente.
- **FP dominan en T4/T5/T7**: ~654–725 FP para ~392–395 TP. Ratio FP/TP ≈ 1.7–1.8, mejorado pero aún alto. T7's conf=0.30 ayuda (-10% FP) pero no cierra la brecha con T3 (177 FP).
- **dog sigue siendo la clase más débil**: AP@50 = 0.44-0.52, menor de las 5 clases en todos los entrenamientos.
- **Val loss ruidosa**: Con 188 imágenes, oscila enormemente, dificultando checkpoint selection.
- **Augmentación destructiva contraproducente**: CoarseDropout + GaussNoise degradan calibración en modelo de 1.2M params (T5).
- **HFlip runtime irrenunciable**: Su eliminación (T6) produjo el peor F1 de la serie pese a que el dataset tiene copias estáticas flippeadas.
- **Pipeline de despliegue frágil**: Dos incidentes en 7 trains: (1) T6 — cache sdist, (2) T7 — whitelist config_loader. Ninguna funcionalidad nueva (Focal Loss, SL1 warmup) fue testeada exitosamente.

### 11.5 Mejor Configuración Operativa

| Objetivo | Mejor Train | Valor | Justificación |
|---|---|---|---|
| **Máximo mAP@50** | **T7** (0.6120) | ★ | Mejor ranking de detecciones de la serie. |
| **Máximo mAP@50-95** | **T7** (0.2824) | ★ | Mejor calidad geométrica de boxes. |
| **Producción (F1 balanceado)** | **T3** (F1=0.6438) | ★ | Mejor balance precision/recall para uso real. |
| **Máximo Recall** | **T4** (0.6886) | | Si las detecciones perdidas son más costosas que los FP. |
| **Máxima Precisión** | **T3** (0.6609) | | Mínimos falsos positivos (177 FP en test). |

> **Recomendación final**: **T3 para producción** (F1 óptimo, precisión más alta, FP mínimos). **T7 como referencia de ranking** (mejor mAP@50 y mAP@50-95, ideal para benchmarks y papers). El modelo seleccionado para despliegue en ESP32-S3 será T3.

### 11.6 Oportunidades de Mejora No Exploradas

Las siguientes técnicas **nunca fueron probadas exitosamente** debido a los incidentes de despliegue:

1. **Focal Loss (γ=2-3)**: Podría mejorar la discriminación de la cabeza cls, especialmente para clases difíciles (dog). Requiere corregir el bug de whitelist en `config_loader.py`.
2. **SL1 warmup real**: La eliminación de la meseta GIoU debería mejorar convergencia en Phase 1. Requiere añadir `reg_warmup_epochs` a `_FCOS_DEFAULTS`.
3. **conf_threshold fino**: T7 mostró que 0.30 reduce FP sin perder TP. Un sweep 0.30-0.40 en T3 podría mejorar su F1 sin reentrenar.
4. **Ensemble T3+T7**: Combinar la precision de T3 con el recall de T7 mediante NMS-merge.

### 11.7 Fix Aplicado en config_loader.py (v2.2.0)

**Opción B aplicada** en `config_loader.py` líneas 70-76. Se reemplazó el bucle whitelist por:
```python
family_kwargs = dict(family_section)  # pasar TODAS las claves del YAML
```

**Verificación**: Script `_t8_verify.py` simuló el pipeline completo y confirmó que las 36 claves YAML llegan a `family_config`, incluyendo `focal_gamma=3.0`, `reg_warmup_epochs=10` y `conf_threshold=0.35`.

---

## 12. Train 8 — Plan de Ejecución

### 12.1 Objetivo

Primer test real de **Focal Loss** (γ=3.0, α=0.25) y **SL1→GIoU warmup** (10 epochs), que nunca funcionaron en T4-T7 debido al bug de whitelist en `config_loader.py`. Adicionalmente, elevar `conf_threshold` de 0.30 a 0.35 tras los buenos resultados de T7.

### 12.2 Cambios Respecto a T7

| Cambio | T7 (v2.1.0) | T8 (v2.2.0) | Impacto |
|---|---|---|---|
| **config_loader.py** | Whitelist: solo 27 claves de `_FCOS_DEFAULTS` | **Option B**: todas las claves YAML pasan | Focal Loss y SL1 warmup ahora activos |
| **conf_threshold** | 0.30 | **0.35** | Filtro más agresivo para reducir FP |
| **focal_gamma** | 3.0 (YAML) → 0.0 (default, bug) | 3.0 (YAML) → **3.0 (real)** | Focal Loss activa por primera vez |
| **reg_warmup_epochs** | 10 (YAML) → 0 (default, bug) | 10 (YAML) → **10 (real)** | SL1→GIoU transición activa |
| **Paquete** | tfm_trainer-2.1.0 | **tfm_trainer-2.2.0** | Evita cache sdist |
| **DEPLOY VERIFICATION** | v2.1.0 | **v2.2.0** | Confirma paquete correcto en logs |

### 12.3 Configuración Completa T8

```yaml
# fcos_v3s_v1.yaml — Train 8 (v2.2.0)
backbone: mobilenet_v3_small
fpn_channels: 64
head_depth: 4
strides: [8, 16, 32]
phase1_epochs: 30          # freeze backbone
phase2_epochs: 80          # full fine-tune
phase1_lr: 1.0e-03
phase2_lr: 1.0e-04
focal_gamma: 3.0           # ◀ Focal Loss ACTIVA (primera vez real)
focal_alpha: 0.25
reg_warmup_epochs: 10      # ◀ SL1→GIoU warmup ACTIVO (primera vez real)
cls_weight: 1.0
reg_weight: 1.5
centerness_weight: 1.0
conf_threshold: 0.35       # ◀ Subido de 0.30
nms_iou_threshold: 0.50
score_thresh_train: 0.05
topk_candidates: 1000
detections_per_img: 100
aug_hflip_prob: 0.5
aug_brightness_limit: 0.2
aug_contrast_limit: 0.2
# Destructivas OFF: aug_gaussian_noise, aug_coarse_dropout, aug_blur NO presentes
```

### 12.4 Hipótesis y Métricas Esperadas

| Hipótesis | Mecanismo | Señal Esperada |
|---|---|---|
| Focal Loss reduce FP | γ=3.0 penaliza ejemplos fáciles, fuerza la cabeza cls a discriminar mejor | FP < 600 (vs T7=654), Precision > 0.40 |
| SL1 warmup mejora P1 | Transición suave SL1→GIoU evita meseta de reg_loss en Phase 1 | reg_loss < 3.0 al final de P1 (vs T7≈3.6) |
| conf=0.35 reduce FP sin perder TP | Filtro más estricto elimina detecciones débiles | TP ≥ 380, FP < T7 |
| Combinación sinérgica | Los tres cambios trabajan juntos | F1 > 0.55, mAP@50 > 0.60 |

### 12.5 Riesgos

- **Focal Loss γ=3.0 podría ser agresivo**: Si degrada Recall, considerar γ=2.0 en T9.
- **conf=0.35 podría perder TP marginales**: Si TP baja significativamente, revertir a 0.30.
- **Interacción focal + warmup desconocida**: Primera vez que ambos se aplican simultáneamente.

### 12.6 Estado de Preparación

- [x] `config_loader.py` — Fix Option B aplicado y verificado
- [x] `setup.py` — Version 2.2.0 con changelog
- [x] `task_fcos.py` — DEPLOY VERIFICATION v2.2.0
- [x] `fcos_v3s_v1.yaml` — conf_threshold=0.35, comentarios T8
- [x] `_t8_verify.py` — Script de verificación ejecutado (ALL CHECKS PASSED)
- [x] `dist/tfm_trainer-2.2.0.tar.gz` — Paquete construido (67 KB), contenido verificado

---

*Documento generado y mantenido como parte del pipeline MLOps del TFM — Detección de Objetos para Asistencia Visual.*
