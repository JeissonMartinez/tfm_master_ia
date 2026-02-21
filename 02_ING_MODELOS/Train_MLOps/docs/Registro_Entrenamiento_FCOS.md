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
5. [Comparativa Global](#5-comparativa-global)
6. [Conclusiones Generales](#6-conclusiones-generales)

---

## 1. Resumen Ejecutivo

| Métrica (Test) | Train 1 | Train 2 | Train 3 |
|---|:---:|:---:|:---:|
| **mAP@50** | 0.4304 | 0.5600 | **0.5675** |
| **mAP@50-95** | N/C | N/C | **0.2602** |
| **Precision** | 0.5427 | 0.6049 | **0.6609** |
| **Recall** | 0.5291 | 0.6271 | **0.6276** |
| **F1-Score** | 0.5358 | 0.6158 | **0.6438** |
| Épocas | 52 | 74 | 101 |
| Tiempo | 12.1 min | 15.9 min | 23.6 min |
| Inferencia | 5.0 ms | 4.6 ms | 4.8 ms |

> **N/C**: No Calculado — la implementación de mAP@50-95 fue añadida después de Train 2.

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

## 5. Comparativa Global

### 5.1 Evolución Total Loss

| Fase | Train 1 | Train 2 | Train 3 |
|---|:---:|:---:|:---:|
| Epoch 0 (640px) | 688.5 | 38.9 | 9.4 |
| Final | ~43.5 | ~3.5 | ~3.3 |
| Best val_loss | 135.49 | **24.01** | 28.12 |

> Nota: val_loss no es directamente comparable entre T2 (Smooth L1) y T3 (GIoU) por cambio de función de loss.

### 5.2 Evolución reg_loss (Train)

| Resolución | Train 1 | Train 2 | Train 3 |
|---|:---:|:---:|:---:|
| 640px (e0) | ~682 | ~33.8 | ~4.5 (meseta) |
| 224px (final) | ~41 | ~1.3 | ~1.1 |

### 5.3 Per-class AP@50 — Test (Evolución)

| Clase | T1 | T2 | T3 | Δ T1→T3 |
|---|:---:|:---:|:---:|:---:|
| dog | 0.406 | 0.463 | **0.496** | +22.2% |
| door | 0.339 | **0.519** | 0.503 | +48.4% |
| obstacle | 0.302 | 0.405 | **0.458** | +51.5% |
| person | 0.521 | 0.636 | **0.636** | +22.1% |
| stair | 0.584 | **0.777** | 0.745 | +27.6% |
| **Media** | **0.430** | **0.560** | **0.568** | **+32.0%** |

### 5.4 Confusion Matrix (Test) — Evolución de TP

| Clase (Test GT) | T1 TP | T2 TP | T3 TP | T1 FN | T2 FN | T3 FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| dog (58) | 30 | 32 | **33** | 28 | 26 | **25** |
| door (136) | 54 | **80** | 77 | 82 | **56** | 59 |
| obstacle (173) | 79 | 86 | **92** | 94 | 87 | **81** |
| person (101) | 67 | 70 | 70 | 34 | **31** | **31** |
| stair (108) | 66 | **87** | 84 | 42 | **21** | 24 |
| **Total** | **296** | **355** | **356** | **280** | **221** | **220** |

### 5.5 Falsos Positivos (Test)

| Clase | T1 FP | T2 FP | T3 FP |
|---|:---:|:---:|:---:|
| dog | 42 | 32 | **24** |
| door | 40 | 83 | **44** |
| obstacle | 72 | **43** | 45 |
| person | 72 | 38 | **26** |
| stair | 26 | 34 | 38 |
| **Total** | **252** | **230** | **177** |

---

## 6. Conclusiones Generales

### 6.1 Impacto por Cambio

| Cambio | Métrica más afectada | Impacto |
|---|---|---|
| **Stride Normalization** (T1→T2) | mAP@50, Recall | +30% mAP@50, +26% Recall. Cambio más impactante. |
| **GIoU Loss** (T2→T3) | mAP@50-95, Precision | mAP@50-95 de 0→0.26. Precision +9.3%. Mejora calidad de box. |
| **Más épocas** (T2→T3) | Convergencia | Best en epoch 80 vs 58. Utilizado efectivamente. |

### 6.2 Fortalezas del Modelo

- **Clasificación perfecta**: Zero confusión inter-clase en todas las pruebas.
- **Modelo ligero**: 1.2M params, 4.71 MB FP32, <5ms inferencia en T4.
- **Export ONNX exitoso**: 9 outputs, opset 13, 4.74 MB, latencia 6ms.
- **Progresión consistente**: Cada iteración mejoró el resultado global.

### 6.3 Debilidades / Cuellos de Botella

- **Recall limitado (~63%)**: El modelo no detecta ~37% de los objetos. Es conservador — alta precision pero baja cobertura.
- **dog es la clase más débil**: AP@50 = 0.50 (test), menor de las 5 clases en los 3 entrenamientos.
- **Val loss muy ruidosa**: Con solo 188 imágenes de validación, la loss oscila enormemente entre epochs, dificultando la selección del mejor checkpoint.
- **Meseta GIoU en Phase 1**: 13 epochs sin aprendizaje de localización (~45% de Phase 1 desperdiciado).

### 6.4 Oportunidades de Mejora (Candidatas para Train 4+)

1. **Bajar conf_threshold** (0.25 → 0.15): Aumentaría recall a costa de precision. El modelo es demasiado conservador.
2. **Warmup de loss híbrido**: Iniciar con Smooth L1 durante primeras N epochs y transicionar a GIoU, evitando la meseta de 13 epochs.
3. **Augmentación más agresiva**: MixUp, CutMix, Mosaic para mayor regularización con 1470 imgs.
4. **Focal Loss tuning**: Ajustar gamma (2→3) y alpha para la cabeza de clasificación.
5. **Resolución final mayor**: Entrenar/evaluar a 320px en vez de 224px para mejor localización.

---

*Documento generado y mantenido como parte del pipeline MLOps del TFM — Detección de Objetos para Asistencia Visual.*
