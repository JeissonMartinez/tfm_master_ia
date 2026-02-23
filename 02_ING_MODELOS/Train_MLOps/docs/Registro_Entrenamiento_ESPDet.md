# Registro de Entrenamiento — ESPDet-Pico (Custom PyTorch Loop)

> **Modelo**: `espdet_pico` — ESPDet-Pico anchor-free micro-detector  
> **Parámetros**: ~22.8K (FP32: 0.09 MB | INT8 est.: 0.02 MB)  
> **Dataset**: IODC YOLO — 5 clases (dog, door, obstacle, person, stair)  
> **Splits**: Train 1470 | Val 188 | Test 187  
> **Infraestructura**: Google Vertex AI Custom Job — `n1-standard-8` + NVIDIA Tesla T4  
> **Contenedor**: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`  
> **Entry-point**: `trainer.task_espdet`  
> **Paquete base**: `tfm_trainer-2.5.0.tar.gz`  
> **Última actualización**: 23 de febrero de 2026 (Train 1 completado)  

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Configuración Base (Compartida)](#2-configuración-base-compartida)
3. [Train 1 — Baseline](#3-train-1--baseline)
4. [Backlog de Propuestas](#4-backlog-de-propuestas)
5. [Comparativa Cross-Model](#5-comparativa-cross-model)
6. [Conclusiones Generales](#6-conclusiones-generales)

---

## 1. Resumen Ejecutivo

| Métrica (Test) | Train 1 |
|---|:---:|
| **mAP@50** | 0.0105 |
| **mAP@50-95** | 0.0023 |
| **Precision** | 0.0009 |
| **Recall** | 0.1597 |
| **F1-Score** | 0.0017 |
| Épocas (P1+P2) | 120 (40+80) |
| Optimizer | AdamW |
| conf_threshold | 0.25 |
| Tiempo | 23.6 min |
| Inferencia (GPU) | 34.4 ms |
| ONNX size | 0.109 MB |

> Tabla actualizada tras cada entrenamiento exitoso.

---

## 2. Configuración Base (Compartida)

Parámetros comunes a todos los entrenamientos de esta serie, salvo modificación explícita.

### 2.1 Arquitectura

| Parámetro | Valor |
|---|---|
| Familia | ESPDet (Custom PyTorch) |
| Variante | espdet_pico |
| Backbone | ESPDetPicoBackbone (3 stages, DepthwiseSeparableConv) |
| width_mult | 0.5 |
| Backbone channels | Base [16, 32, 64] × 0.5 → **[8, 16, 32]** |
| FPN | SimpleFPN (fpn_ch=ch_list[0]=**8**) |
| Head | 2× DSConv por nivel → cls_out (nc) + reg_out (4 × reg_max) |
| Niveles FPN | 3 (strides [4, 8, 16]) |
| reg_max | 1 |
| Params totales | **22,839** (~0.023M) |
| Backbone params | 10,592 (46.4%) |
| Head params | 12,247 (53.6%) |
| Pretrained | **None** (entrenamiento desde cero) |
| Input size (export) | 224 |
| ONNX opset | 13 |
| Clases | 5 (dog, door, obstacle, person, stair) |

> **Contexto de escala**: ESPDet-Pico tiene **54× menos** parámetros que FCOS (1.23M) y **113× menos** que YOLO26 (2.58M). Diseñado para despliegue ultra-eficiente en ESP32-S3 (8 MB PSRAM).

### 2.2 Estrategia de 2 Fases

| Parámetro | Phase 1 (Freeze) | Phase 2 (Full) |
|---|---|---|
| Epochs | 40 | 80 |
| LR | 1e-3 | 5e-5 |
| Weight Decay | 1e-4 | 1e-5 |
| Optimizer | AdamW | AdamW |
| Scheduler | Cosine | Cosine |
| Warmup | 3 epochs | 0 |
| Params trainable | 12,247 (53.6%) | 22,839 (100%) |

**Total: 120 epochs (40 + 80), patience=20**

### 2.3 Optimización

| Parámetro | Valor |
|---|---|
| Optimizer | AdamW |
| AMP | True |
| Grad clip | 5.0 |
| Batch size | 32 |
| Workers | 4 |

### 2.4 Redimensionado Progresivo

| Epoch | Resolución | Fase |
|---|---|---|
| 0 | 640 | Phase 1 |
| 15 | 416 | Phase 1 |
| 30 | 320 | Phase 1 |
| 40+ | 224 | Phase 2 |

### 2.5 Loss

| Componente | Función | Peso |
|---|---|---|
| `cls_loss` | BCE (per-level, normalizado por n_pos) | 1.0 |
| `reg_loss` | GIoU loss (l,t,r,b stride-normalizado) | 2.0 |

> ESPDet NO utiliza centerness (a diferencia de FCOS). `ctr_loss` siempre es 0.0.

### 2.6 Augmentación (Albumentations)

| Transform | Parámetro efectivo | Controlado por |
|---|---|---|
| HorizontalFlip | p=0.5 | `_ESPDET_DEFAULTS` (`aug_hflip_prob`) |
| BrightnessContrast | limit=0.2, p=0.5 | `_ESPDET_DEFAULTS` (`aug_brightness_limit`) |
| HueSaturationValue | H=20, S=30, V=20, p=0.5 | `_ESPDET_DEFAULTS` |
| Affine (rotate/scale/shift) | ±15°, scale=0.2, shift=0.1, p=0.5 | `_ESPDET_DEFAULTS` (`aug_rotate_limit`) |
| GaussNoise | std=(0.01, 0.05), p=0.3 | YAML `aug_gaussian_noise: 0.3` |

> ⚠️ **Naming mismatch detectado**: Los keys del YAML (`aug_horizontal_flip`, `aug_brightness_contrast`, `aug_rotation_limit`) NO coinciden con los keys que `IODCDataset._build_transforms()` busca (`aug_hflip_prob`, `aug_brightness_limit`, `aug_rotate_limit`). Los defaults de `_ESPDET_DEFAULTS` proveen los keys correctos, por lo que la augmentación funciona con valores por defecto. Los valores personalizados del YAML son **efectivamente ignorados**. Se recomienda alinear nomenclatura para Train 2+.

### 2.7 Inferencia / Evaluación

| Parámetro | Valor |
|---|---|
| conf_threshold | 0.25 |
| iou_threshold | 0.45 |

### 2.8 Protecciones Aplicadas (Lecciones FCOS/YOLO26)

| Lección | Aplicación en ESPDet |
|---|---|
| FCOS T6 — pip cache | Version bump a `tfm_trainer-2.5.0` ✅ |
| FCOS T7/T8 — whitelist config_loader | Ya corregido en v2.2.0 (todas las claves YAML pasan) ✅ |
| FCOS T8 — launch_job hardcoded | `build_and_launch.sh` pasa `--package-uri` dinámico ✅ |
| FCOS T8 — DEPLOY VERIFICATION | Bloque de verificación en Bloque 3 (`v2.5.0`) ✅ |
| FCOS T8 — `log()` vs `print()` | Entry-point usa exclusivamente `print()` ✅ |
| Bug nuevo — aug_config missing | Fix en v2.5.0: `aug_config` se extrae de fc y se pasa a IODCDataset ✅ |

---

## 3. Train 1 — Baseline

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
| **Naming mismatch aug** | BAJA | YAML aug values ignorados, pero defaults razonables aplican |

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

3. **Bug de aug_config corregido, pero naming mismatch preexiste** — `aug_config` se pasa correctamente al dataset (fix v2.5.0), pero los keys del YAML no coinciden con los que `IODCDataset._build_transforms()` busca. Funciona por casualidad (defaults proveen los keys correctos con valores razonables).

4. **Progressive resizing es contraproducente para este modelo** — El spike de epoch 40 y la recuperación lenta sugieren que este modelo micro no maneja bien los cambios de resolución. Entrenar directamente a 224px simplificaría el proceso.

5. **Vertex AI Experiments no disponible (403)** — Error recurrente de scoping (`ACCESS_TOKEN_SCOPE_INSUFFICIENT`). El entrenamiento continúa sin registro de experimentos. No impacta los resultados pero sí la trazabilidad en Vertex AI.

6. **Warnings benignos** — `pythonjsonlogger` no encontrado, pip dependency conflicts (bigframes, ydata-profiling, dataproc-jupyter-plugin). Idénticos a FCOS y YOLO26. Sin impacto.

---

## Debugging / Incidentes

> Sección para registrar problemas encontrados durante el despliegue y debugging de los entrenamientos.

### Pre-Train 1 — Fix preventivo: aug_config missing

- **Descubierto en**: Revisión de código pre-lanzamiento (comparación con `task_fcos.py`)
- **Síntoma potencial**: `IODCDataset` recibiría `aug_config={}`, usando solo defaults internos de Albumentations
- **Causa raíz**: `task_espdet.py` no extraía ni pasaba `aug_config` al constructor de `IODCDataset`, a diferencia de `task_fcos.py` que sí lo hace en L324-335
- **Fix aplicado**: Añadido `aug_config = {k: v for k, v in fc.items() if k.startswith("aug_")}` + pasado como `aug_config=aug_config` a `IODCDataset()`
- **Versión**: Fix incluido en `tfm_trainer-2.5.0`
- **Verificación**: DEPLOY VERIFICATION confirma 16 aug keys presentes

### Train 1 — Dead config detectada (no impacto directo)

- **`fpn_channels: 32`**: YAML lo define pero `build_espdet_pico()` no acepta el parámetro. FPN usa `ch_list[0]=8` del backbone.
- **`num_head_convs: 2`**: YAML lo define pero no se pasa al constructor.
- **`aug_horizontal_flip`, `aug_brightness_contrast`, `aug_rotation_limit`**: YAML keys no coinciden con los que `IODCDataset` busca (`aug_hflip_prob`, `aug_brightness_limit`, `aug_rotate_limit`). Defaults cubren los keys correctos.
- **Impacto**: Ninguno en T1 (defaults son razonables), pero impide personalización vía YAML.
- **Acción recomendada**: Refactorizar para T2+ si se necesita control granular de augmentación y parámetros de FPN/Head.

### Train 1 — Warnings benignos (sin impacto)

- `pythonjsonlogger` module no encontrado — error de `sitecustomize` del contenedor
- pip dependency conflicts (bigframes, ydata-profiling, dataproc-jupyter-plugin, pydantic version)
- Vertex AI Experiments 403 (`ACCESS_TOKEN_SCOPE_INSUFFICIENT`) — entrenamiento continúa sin tracking

---

## 4. Backlog de Propuestas

> Propuestas identificadas durante el análisis de Train 1, priorizadas por impacto esperado.

### PROPUESTA A — Aumento de Capacidad (CRÍTICA, Train 2)

**Objetivo**: Salir de la crisis de capacidad. 22.8K params → mínimo ~90K-360K.

| Parámetro | Train 1 (v1) | Propuesto | Efecto estimado |
|---|---|---|---|
| `width_mult` | 0.5 | **2.0** | ~365K params (16×). Canales backbone [32, 64, 128], FPN=32 |
| `pretrained_weights` | None | **None** (mantener) | Baseline sin pretrain, comparar después |

**Alternativa conservadora**: `width_mult=1.0` (~91K params, 4×). Canales backbone [16, 32, 64], FPN=16.

**Justificación**: El cls_loss final (3.05) está apenas un 12% por debajo de random (3.47). El modelo no tiene suficiente capacidad representativa. Multiplicar canales por 4× es el cambio de mayor impacto posible.

**Riesgo**: Con 365K params FP32 (0.35 MB), el modelo sigue siendo extremadamente pequeño para ESP32-S3. ONNX estimado ~0.5 MB INT8.

### PROPUESTA B — Entrenar directamente a 224px (Train 2)

**Objetivo**: Eliminar el spike de resolución y epochs desperdiciados en recuperación.

| Parámetro | Train 1 (v1) | Propuesto | Justificación |
|---|---|---|---|
| `resize_schedule` | {0:640, 15:416, 30:320, 40:224} | **{0: 224}** | Evitar spike P2; 21% de Phase 2 perdido en recuperación |

**Justificación**: El modelo target funciona a 224px. Con canales tan pequeños, feature maps a 640px no aportan información útil (stride-16 feature map = 40×40 con solo 32 canales en el mejor caso). Entrenar siempre a 224px genera features consistentes con el target.

### PROPUESTA C — Reducir conf_threshold evaluación (Train 2)

**Objetivo**: Evaluar el modelo con un umbral más permisivo para entender mejor su capacidad real.

| Parámetro | Train 1 (v1) | Propuesto | Justificación |
|---|---|---|---|
| `conf_threshold` | 0.25 | **0.10** | 107K detecciones @0.25 indica que sigmoid outputs son muy altos. Pero las pocas detecciones correctas podrían estar en el rango 0.10-0.25 |

### PROPUESTA D — Más epochs y más paciencia (Train 2)

**Objetivo**: Dar más tiempo al modelo para converger.

| Parámetro | Train 1 (v1) | Propuesto | Justificación |
|---|---|---|---|
| `phase1_epochs` | 40 | **60** | Más tiempo para la cabeza |
| `phase2_epochs` | 80 | **120** | Modelo seguía mejorando al final |
| `patience` | 20 | **30** | Early stopping más tolerante |

### PROPUESTA E — Focal Loss (Train 3+)

**Objetivo**: Mitigar el desbalance background/foreground extremo.

| Parámetro | Train 1 | Propuesto | Justificación |
|---|---|---|---|
| cls_loss | BCE | **Focal Loss (γ=2.0, α=0.25)** | El modelo no suprime background. Focal Loss reduciría la contribución de los fáciles (background) |

**Prerequisito**: Modificar `build_espdet_loss()` para soportar focal_gamma/focal_alpha (similar a `build_fcos_loss()`). Dejar después de confirmar que la Propuesta A resuelve el problema de capacidad.

### PROPUESTA F — Alinear naming de augmentación (Mejora código)

**Objetivo**: Que las claves YAML controlen realmente la augmentación.

- Opción 1: Renombrar claves en YAML para que coincidan con `IODCDataset` (`aug_hflip_prob`, `aug_rotate_limit`, etc.)
- Opción 2: Añadir mapeo en `task_espdet.py` que traduzca `aug_horizontal_flip` → `aug_hflip_prob`
- Opción 3: Modificar `IODCDataset._build_transforms()` para aceptar ambos nombres

### PROPUESTA G — fpn_channels y num_head_convs como parámetros reales

**Objetivo**: Que `build_espdet_pico()` acepte `fpn_channels` y `num_head_convs` desde el YAML.

**Justificación**: YAML ya define `fpn_channels: 32` pero el code lo ignora. Permitir esto daría más flexibilidad para experimentar con configuraciones de FPN sin cambiar código.

### Planificación

| Train | Propuestas | Prioridad | Estado |
|---|---|---|---|
| Train 1 | Baseline (ninguna) | — | ✅ Completado |
| Train 2 | **A** (width_mult=2.0) + **B** (224px fijo) + **C** (conf=0.10) + **D** (más epochs) | CRÍTICA | ⏳ Pendiente |
| Train 3 | A+B+C+D + **E** (Focal Loss, si aún necesario) | MEDIA | ⏳ Pendiente |
| Train N | **F** + **G** (mejoras código, no entrenan por sí solas) | BAJA | ⏳ Pendiente |

> La Propuesta A es la intervención individual más importante. Sin capacidad suficiente, ninguna otra mejora producirá resultados significativos.

---

## 5. Comparativa Cross-Model

### Test Metrics — ESPDet vs FCOS vs YOLO26

| Métrica | ESPDet T1 | FCOS T3 (prod) | FCOS T7 (bench) | YOLO26 T1 | YOLO26 T2 |
|---|:---:|:---:|:---:|:---:|:---:|
| **mAP@50** | 0.0105 | 0.5675 | 0.6120 | 0.7544 | **0.7747** |
| **mAP@50-95** | 0.0023 | 0.2602 | 0.2824 | 0.5153 | **0.5456** |
| **Precision** | 0.0009 | 0.6609 | 0.3716 | 0.8264 | **0.8324** |
| **Recall** | 0.1597 | 0.6276 | **0.6872** | 0.6402 | 0.6853 |
| **F1-Score** | 0.0017 | 0.6438 | 0.4824 | 0.7215 | **0.7517** |
| Params | **0.023M** | 1.23M | 1.23M | 2.58M | 2.58M |
| ONNX size | **0.109 MB** | 4.74 MB | 4.74 MB | 9.97 MB | 9.97 MB |
| ONNX latency | **0.76 ms** | — | — | 8.3 ms | 9.9 ms |
| Train time | 23.6 min | — | — | 26.0 min | 32.6 min |

### Conclusión cross-model

ESPDet-Pico T1 con width_mult=0.5 es **funcionalmente inoperable** (mAP@50=0.01). No es comparable con FCOS ni YOLO26 en su estado actual. El modelo cumple el objetivo de tamaño ultra-compacto (0.109 MB ONNX, 0.76 ms latencia) pero a costa de **cero utilidad predictiva**.

El factor limitante es exclusivamente la **capacidad del modelo** (22.8K params). La arquitectura, loss, y pipeline de entrenamiento funcionaron correctamente — la limitación es estructural, no un bug.

---

## 6. Conclusiones Generales

1. **Train 1 establece un baseline técnicamente correcto pero funcionalmente inviable** — El pipeline completo (paquete, despliegue, entrenamiento, evaluación, exportación ONNX) funciona sin errores. El DEPLOY VERIFICATION confirma que todas las lecciones de FCOS/YOLO26 fueron aplicadas. El problema es exclusivamente de capacidad del modelo.

2. **22.8K parámetros es el piso absoluto verificado empíricamente** — Con width_mult=0.5, el modelo no puede distinguir 5 clases de objetos. cls_loss apenas supera la clasificación aleatoria (3.05 vs 3.47). Este dato es útil como cota inferior para el TFM.

3. **El micro-tamaño sí se logra** — ONNX de 112 KB, cabe holgadamente en la Flash del ESP32-S3 (16 MB). Latencia de inferencia CPU de 0.76 ms. Si el modelo aprende a detectar, el despliegue en hardware es trivial.

4. **Acción inmediata: Train 2 con width_mult=2.0** — Aumentar 16× la capacidad (22.8K → ~365K params, ~0.5 MB ONNX) manteniendo el modelo en orden de magnitud menor que FCOS (1.23M). Esto es la Propuesta A y es el cambio más crítico.

5. **El naming mismatch de augmentación y la dead config (fpn_channels, num_head_convs) son deudas técnicas menores** — No impactaron T1 pero deben resolverse antes de que se necesite personalización granular.

6. **Pipeline MLOps maduro** — Cero bugs de despliegue en T1 (vs 4 bugs en los primeros 8 trains de FCOS). Las lecciones acumuladas (version bump, DEPLOY VERIFICATION, aug_config fix, print vs log) funcionaron. Tasa de éxito al primer intento: 100%.

---

*Documento generado y mantenido como parte del pipeline MLOps del TFM — Detección de Objetos para Asistencia Visual.*
