# Instructivo de Despliegue — Modelos ESPDL INT8 para ESP32-S3

> **Objetivo**: Documentar las especificaciones técnicas, formatos de salida, post-procesado y consideraciones de integración de los 2 modelos viables seleccionados para despliegue en ESP32-S3.  
> **Target**: ESP32-S3 WROOM N16R8 (Xtensa LX7, 512 KB SRAM, 8 MB Flash, 8 MB PSRAM)  
> **Placa**: Freenove ESP32-S3 CAM Board + OV5640  
> **Framework firmware**: ESP-IDF v5.x + esp-dl v3.x  
> **Origen de los modelos**: `02_ING_MODELOS/Train_MLOps` — cuantización PTQ INT8 con esp-ppq v1.2.6  
> **Referencia técnica**: `docs/Registro_Cuantizacion_Modelos.md` (§9.4, §9.5, §10)  
> **Fecha**: 24 de febrero de 2026  

---

## Índice

1. [Resumen de Modelos Seleccionados](#1-resumen-de-modelos-seleccionados)
2. [Artefactos y Ubicación](#2-artefactos-y-ubicación)
3. [Especificación Técnica — ESPDet Pico T4](#3-especificación-técnica--espdet-pico-t4)
4. [Especificación Técnica — YOLO26n T2 ESP](#4-especificación-técnica--yolo26n-t2-esp)
5. [Preprocesado de Entrada (Común)](#5-preprocesado-de-entrada-común)
6. [Post-procesado — ESPDet Pico T4](#6-post-procesado--espdet-pico-t4)
7. [Post-procesado — YOLO26n T2 ESP](#7-post-procesado--yolo26n-t2-esp)
8. [Dequantización de Salidas INT8](#8-dequantización-de-salidas-int8)
9. [Integración en Firmware ESP-IDF](#9-integración-en-firmware-esp-idf)
10. [Consideraciones de Memoria y Rendimiento](#10-consideraciones-de-memoria-y-rendimiento)
11. [Diferencias con Modelos Previos (v1)](#11-diferencias-con-modelos-previos-v1)
12. [Checklist de Validación](#12-checklist-de-validación)

---

## 1. Resumen de Modelos Seleccionados

| Propiedad | ESPDet Pico T4 | YOLO26n T2 ESP |
|---|:---:|:---:|
| **Archivo ESPDL** | `espdet_pico_t4.espdl` | `yolo26n_t2_esp.espdl` |
| **Tamaño ESPDL** | **0.52 MB** (545 KB) | **2.71 MB** (2,847 KB) |
| **ONNX original** | 1.42 MB | 9.92 MB |
| **Compresión** | 2.72× | 3.65× |
| **Parámetros** | ~361K | ~2.6M |
| **Familia** | ESPDet (Espressif nativo) | YOLO26 (Ultralytics, re-exportado) |
| **N.º salidas** | 6 tensores | 6 tensores |
| **Canales box** | 4 (l, t, r, b directos) | 64 (DFL, reg_max=16) |
| **Canales score** | 5 (nc=5, raw logits) | 5 (nc=5, raw logits) |
| **Post-procesado** | Sigmoid + dist2box + NMS | Sigmoid + DFL integral + dist2box + NMS |
| **mAP@50 FP32** | 0.5985 | 0.5297 |
| **mAP@50 INT8** | 0.5319 (−11.1%) | 0.4343 (−18.0%) |
| **Errores de exponente** | 0 | 0 |
| **Viabilidad** | 🟢 Candidato ideal | 🟡 Viable con reservas (SRAM) |

### 1.1 Clases de Detección

| Índice | Clase | Descripción |
|:---:|---|---|
| 0 | `dog` | Perro guía |
| 1 | `door` | Puerta |
| 2 | `obstacle` | Obstáculo genérico |
| 3 | `person` | Persona |
| 4 | `stair` | Escalera |

> **Orden idéntico al firmware existente**: coincide con `CLASS_NAMES[]` en `app_config.h`.

---

## 2. Artefactos y Ubicación

### 2.1 Archivos para copiar a `03_ING_DESPLIEGUE/models/`

```
Origen (02_ING_MODELOS/Train_MLOps/):

outputs/espdl/espdet_pico_t4/
├── espdet_pico_t4.espdl          # 545,792 bytes — modelo para Flash
├── espdet_pico_t4.info           # Grafo ESPDL legible (debug/verificación)
└── espdet_pico_t4.json           # Configuración de cuantización

outputs/espdl/yolo26n_t2_esp/
├── yolo26n_t2_esp.espdl          # 2,847,120 bytes — modelo para Flash
├── yolo26n_t2_esp.info           # Grafo ESPDL legible (debug/verificación)
└── yolo26n_t2_esp.json           # Configuración de cuantización
```

### 2.2 Scripts de referencia

| Script | Ubicación | Función |
|---|---|---|
| `convert_onnx_to_espdl.py` | `scripts/` | Conversión ONNX → ESPDL (re-ejecutable) |
| `eval_fp32_vs_int8.py` | `scripts/` | Evaluación FP32 vs INT8 + visualizaciones |
| `export_yolo26_esp.py` | `scripts/` | Re-exportación YOLO26 con 6 salidas ESP |

### 2.3 Archivos ONNX fuente (para re-cuantización)

| Modelo | ONNX | Tamaño |
|---|---|---|
| ESPDet Pico T4 | `outputs/espdet-pico-v4-t4/export/espdet_pico.onnx` | 1.42 MB |
| YOLO26n T2 ESP | `outputs/yolo26n_custom_v2-run1/export/best_esp.onnx` | 9.92 MB |

---

## 3. Especificación Técnica — ESPDet Pico T4

### 3.1 Entrada

| Propiedad | Valor |
|---|---|
| **Nombre del tensor** | `input` |
| **Shape ESPDL** | `[1, 224, 224, 3]` (NHWC) |
| **Tipo** | INT8 |
| **Exponente** | −7 (escala = $2^{-7}$ = 0.0078125) |
| **Rango representable** | $[-128 \times 2^{-7},\ +127 \times 2^{-7}]$ = [−1.0, +0.9921875] |

### 3.2 Salidas

El modelo devuelve **6 tensores** en el siguiente orden (según grafo ESPDL):

```
return %score1, %score2, %score0, %box0, %box1, %box2
```

| Índice retorno | Tensor | Shape (NHWC) | Tipo | Exponente | FPN Level | Stride | Contenido |
|:---:|---|---|---|:---:|:---:|:---:|---|
| 0 | `score1` | `[1, 14, 14, 5]` | INT8 | −3 | P4 | 16 | Logits de clasificación (5 clases) |
| 1 | `score2` | `[1, 7, 7, 5]` | INT8 | −3 | P5 | 32 | Logits de clasificación (5 clases) |
| 2 | `score0` | `[1, 28, 28, 5]` | INT8 | −3 | P3 | 8 | Logits de clasificación (5 clases) |
| 3 | `box0` | `[1, 28, 28, 4]` | INT8 | −3 | P3 | 8 | Distancias l, t, r, b |
| 4 | `box1` | `[1, 14, 14, 4]` | INT8 | −3 | P4 | 16 | Distancias l, t, r, b |
| 5 | `box2` | `[1, 7, 7, 4]` | INT8 | −4 | P5 | 32 | Distancias l, t, r, b |

> ⚠️ **El orden de retorno NO es secuencial**: score1, score2, score0, box0, box1, box2. Al acceder via `get_output(index)`, usar esta tabla para mapear correctamente cada tensor.

### 3.3 Candidatos totales por nivel

| Nivel | Grid | Candidatos |
|:---:|:---:|:---:|
| P3 (stride 8) | 28 × 28 | 784 |
| P4 (stride 16) | 14 × 14 | 196 |
| P5 (stride 32) | 7 × 7 | 49 |
| **Total** | | **1,029** |

### 3.4 Arquitectura interna

- Backbone: arquitectura nativa Espressif (ligera, sin SE blocks)
- Neck: FPN de 3 niveles
- Head: tipo FCOS anchor-free con predicción directa l, t, r, b
- Activaciones: 46 LUTs Swish (pre-computadas en INT8)
- Sin InstanceNorm, sin bloques SE, sin operaciones problemáticas para INT8

---

## 4. Especificación Técnica — YOLO26n T2 ESP

### 4.1 Entrada

| Propiedad | Valor |
|---|---|
| **Nombre del tensor** | `images` |
| **Shape ESPDL** | `[1, 224, 224, 3]` (NHWC) |
| **Tipo** | INT8 |
| **Exponente** | −7 (escala = $2^{-7}$ = 0.0078125) |
| **Rango representable** | $[-128 \times 2^{-7},\ +127 \times 2^{-7}]$ = [−1.0, +0.9921875] |

### 4.2 Salidas

El modelo devuelve **6 tensores** en el siguiente orden (según grafo ESPDL):

```
return %box2, %score2, %score1, %score0, %box1, %box0
```

| Índice retorno | Tensor | Shape (NHWC) | Tipo | Exponente | FPN Level | Stride | Contenido |
|:---:|---|---|---|:---:|:---:|:---:|---|
| 0 | `box2` | `[1, 7, 7, 64]` | INT8 | −3 | P5 | 32 | DFL logits (reg_max=16 × 4) |
| 1 | `score2` | `[1, 7, 7, 5]` | INT8 | −3 | P5 | 32 | Logits de clasificación (5 clases) |
| 2 | `score1` | `[1, 14, 14, 5]` | INT8 | −2 | P4 | 16 | Logits de clasificación (5 clases) |
| 3 | `score0` | `[1, 28, 28, 5]` | INT8 | −3 | P3 | 8 | Logits de clasificación (5 clases) |
| 4 | `box1` | `[1, 14, 14, 64]` | INT8 | −3 | P4 | 16 | DFL logits (reg_max=16 × 4) |
| 5 | `box0` | `[1, 28, 28, 64]` | INT8 | −3 | P3 | 8 | DFL logits (reg_max=16 × 4) |

> ⚠️ **El orden de retorno NO es secuencial**: box2, score2, score1, score0, box1, box0. Mapear cuidadosamente al acceder por índice.

> ⚠️ **Nota sobre score1**: su exponente es **−2** (no −3 como los demás). Esto implica que su escala de dequantización es diferente: $2^{-2}$ = 0.25 vs $2^{-3}$ = 0.125.

### 4.3 Candidatos totales por nivel

Idéntico a ESPDet: **1,029 candidatos** distribuidos en P3 (784) + P4 (196) + P5 (49).

### 4.4 Arquitectura interna

- Backbone: YOLO26 (Ultralytics) con bloques C2f + PSA (Partial Self-Attention)
- Neck: PAN-FPN de 3 niveles
- Head: Detect head con 6 salidas separadas (re-exportado para ESP-DL)
- Activaciones: 76 LUTs Swish (pre-computadas en INT8)
- Módulo de atención: `Attention` con operación `matmul` (reemplaza `einsum` del original)
- `reg_max=16`: cada coordenada de box se predice como distribución sobre 16 bins → requiere DFL integral

### 4.5 Diferencia clave: DFL (Distribution Focal Loss)

A diferencia de ESPDet que predice distancias directamente (4 canales), YOLO26 predice una **distribución de probabilidad** sobre 16 bins para cada una de las 4 coordenadas (total: 64 canales). Esto requiere un paso adicional de **DFL integral** antes de obtener las distancias finales.

```
Box raw [64 canales] → reshape [4, 16] → softmax(dim=1) → Σ(prob × bin_index) → [4 distancias]
```

---

## 5. Preprocesado de Entrada (Común)

Ambos modelos comparten idéntico preprocesado de entrada. Este es **compatible con el pipeline existente** en el firmware v1 (`preprocess.cpp`).

### 5.1 Pipeline de preprocesado

```
OV5640 (320×240 RGB565)
    ↓ Conversión a RGB888 (si no nativo)
    ↓ Center crop: offset_x=48, offset_y=8 → 224×224
    ↓ Normalización a [0.0, 1.0]: pixel / 255.0
    ↓ Cuantización INT8: value_int8 = round(value_float * 2^7)
       = round(value_float * 128)
    → Tensor INT8 [1, 224, 224, 3] (NHWC)
```

### 5.2 Fórmula de cuantización del input

$$
x_{int8} = \text{clamp}\left(\text{round}\left(\frac{x_{float}}{2^{-7}}\right),\ -128,\ 127\right) = \text{clamp}\left(\text{round}(x_{float} \times 128),\ -128,\ 127\right)
$$

Dado que los píxeles normalizados están en $[0, 1]$, los valores INT8 resultantes estarán en $[0, 127]$ (mitad positiva del rango).

### 5.3 Layout de memoria

- **ESPDL espera NHWC**: `[batch, height, width, channels]`
- Los píxeles se almacenan en orden: fila 0 col 0 (R,G,B), fila 0 col 1 (R,G,B), ..., fila 223 col 223 (R,G,B)
- Tamaño total del buffer de entrada: $1 \times 224 \times 224 \times 3 = 150{,}528$ bytes

### 5.4 Nombres de tensor de entrada

| Modelo | Nombre tensor | Nota |
|---|---|---|
| ESPDet T4 | `input` | |
| YOLO26 T2 ESP | `images` | Nombre diferente — adaptar en `set_input()` |

---

## 6. Post-procesado — ESPDet Pico T4

### 6.1 Parámetros recomendados

| Parámetro | Valor | Origen |
|---|---|---|
| `conf_threshold` | **0.35** | Tuning NMS en `Registro_Entrenamiento_ESPDet.md` §6 |
| `iou_threshold` | **0.40** | Tuning NMS en `Registro_Entrenamiento_ESPDet.md` §6 |

### 6.2 Pipeline de decodificación por nivel FPN

Para cada nivel $l \in \{0, 1, 2\}$ con stride $s_l \in \{8, 16, 32\}$ y grid $H_l \times W_l$:

**Paso 1 — Dequantizar** (ver §8):

$$
\text{score\_float}[i] = \text{score\_int8}[i] \times 2^{e_{score_l}}
$$

$$
\text{box\_float}[i] = \text{box\_int8}[i] \times 2^{e_{box_l}}
$$

**Paso 2 — Sigmoid** sobre scores (convierte logits → probabilidades):

$$
p_c = \sigma(\text{score\_float}) = \frac{1}{1 + e^{-\text{score\_float}}}
$$

**Paso 3 — Filtrado** por confianza:

$$
\text{conf} = \max_{c=0}^{4}(p_c) > \text{conf\_threshold}
$$

**Paso 4 — Grid centers** (coordenadas del centro de cada celda en píxeles):

$$
cx = (x + 0.5) \times s_l, \quad cy = (y + 0.5) \times s_l
$$

donde $x \in [0, W_l)$ y $y \in [0, H_l)$.

**Paso 5 — Decode box** (distancias l, t, r, b → coordenadas xyxy normalizadas):

$$
x_1 = \frac{cx - \text{ReLU}(l) \times s_l}{224}, \quad y_1 = \frac{cy - \text{ReLU}(t) \times s_l}{224}
$$

$$
x_2 = \frac{cx + \text{ReLU}(r) \times s_l}{224}, \quad y_2 = \frac{cy + \text{ReLU}(b) \times s_l}{224}
$$

Clamp final a $[0, 1]$.

**Paso 6 — NMS** per-class greedy (estándar, ya implementado en `postprocess.cpp`).

### 6.3 Pseudocódigo C++

```cpp
// ESPDet: 6 outputs — score0/1/2 [1,H,W,5], box0/1/2 [1,H,W,4]
// NHWC layout in ESPDL

static const int STRIDES[3] = {8, 16, 32};
static const int GRID_H[3]  = {28, 14, 7};
static const int GRID_W[3]  = {28, 14, 7};
static const int NC = 5;

// Exponents por tensor (de .info):
static const int SCORE_EXP[3] = {-3, -3, -3};  // score0, score1, score2
static const int BOX_EXP[3]   = {-3, -3, -4};  // box0, box1, box2

DetectionResult postprocess_espdet_espdl(
    const int8_t* score_data[3],  // score0, score1, score2
    const int8_t* box_data[3],    // box0,   box1,   box2
    float conf_thr,
    float iou_thr)
{
    DetectionResult result;
    result.clear();

    for (int lvl = 0; lvl < 3; ++lvl) {
        const int H = GRID_H[lvl];
        const int W = GRID_W[lvl];
        const int stride = STRIDES[lvl];
        const float score_scale = exp2f(SCORE_EXP[lvl]);  // 2^exp
        const float box_scale   = exp2f(BOX_EXP[lvl]);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = y * W + x;

                // --- Scores: NHWC → offset = idx * NC + c ---
                float best_score = -1e9f;
                int   best_cls   = 0;
                for (int c = 0; c < NC; ++c) {
                    float logit = score_data[lvl][idx * NC + c] * score_scale;
                    float prob  = 1.0f / (1.0f + expf(-logit));  // sigmoid
                    if (prob > best_score) {
                        best_score = prob;
                        best_cls = c;
                    }
                }

                if (best_score < conf_thr) continue;

                // --- Box: NHWC → offset = idx * 4 + d ---
                float cx = (x + 0.5f) * stride;
                float cy = (y + 0.5f) * stride;

                float l = fmaxf(0.0f, box_data[lvl][idx * 4 + 0] * box_scale);
                float t = fmaxf(0.0f, box_data[lvl][idx * 4 + 1] * box_scale);
                float r = fmaxf(0.0f, box_data[lvl][idx * 4 + 2] * box_scale);
                float b = fmaxf(0.0f, box_data[lvl][idx * 4 + 3] * box_scale);

                Detection det;
                det.x1 = clamp01((cx - l * stride) / 224.0f);
                det.y1 = clamp01((cy - t * stride) / 224.0f);
                det.x2 = clamp01((cx + r * stride) / 224.0f);
                det.y2 = clamp01((cy + b * stride) / 224.0f);
                det.confidence = best_score;
                det.class_id   = best_cls;

                result.add(det);
            }
        }
    }

    nms_per_class(result, iou_thr);
    return result;
}
```

---

## 7. Post-procesado — YOLO26n T2 ESP

### 7.1 Parámetros recomendados

| Parámetro | Valor | Origen |
|---|---|---|
| `conf_threshold` | **0.25** | Default Ultralytics optimizado |
| `iou_threshold` | **0.45** | Default Ultralytics |

### 7.2 Pipeline de decodificación por nivel FPN

Para cada nivel $l \in \{0, 1, 2\}$ con stride $s_l \in \{8, 16, 32\}$ y grid $H_l \times W_l$:

**Paso 1 — Dequantizar** (ver §8).

**Paso 2 — Sigmoid** sobre scores.

**Paso 3 — Filtrado** por confianza (igual que ESPDet).

**Paso 4 — Grid centers** (igual que ESPDet).

**Paso 5 — DFL Integral** (paso exclusivo de YOLO26):

Para cada candidato con 64 canales de box raw:

$$
\text{reshape}: [64] \rightarrow [4, 16]
$$

$$
\text{softmax por fila}: p_{d,k} = \frac{e^{x_{d,k}}}{\sum_{j=0}^{15} e^{x_{d,j}}}, \quad d \in \{l, t, r, b\},\ k \in [0, 15]
$$

$$
\text{distancia}_d = \sum_{k=0}^{15} k \times p_{d,k}
$$

**Paso 6 — dist2bbox**:

$$
x_1 = \frac{cx - \text{dist}_l \times s_l}{224}, \quad y_1 = \frac{cy - \text{dist}_t \times s_l}{224}
$$

$$
x_2 = \frac{cx + \text{dist}_r \times s_l}{224}, \quad y_2 = \frac{cy + \text{dist}_b \times s_l}{224}
$$

**Paso 7 — NMS** per-class greedy.

### 7.3 Pseudocódigo C++

```cpp
// YOLO26 ESP: 6 outputs — score0/1/2 [1,H,W,5], box0/1/2 [1,H,W,64]
// NHWC layout in ESPDL

static const int REG_MAX = 16;
static const int BOX_CH  = REG_MAX * 4;  // 64

// Exponents por tensor (de .info):
static const int SCORE_EXP_Y[3] = {-3, -2, -3};  // score0, score1, score2
static const int BOX_EXP_Y[3]   = {-3, -3, -3};  // box0, box1, box2

// DFL integral: [REG_MAX] logits → 1 distance value
static float dfl_single(const float* logits, int reg_max) {
    // Softmax
    float max_val = logits[0];
    for (int k = 1; k < reg_max; ++k)
        if (logits[k] > max_val) max_val = logits[k];

    float sum_exp = 0.0f;
    float probs[16];
    for (int k = 0; k < reg_max; ++k) {
        probs[k] = expf(logits[k] - max_val);
        sum_exp += probs[k];
    }

    // Weighted sum: Σ(k × prob_k)
    float dist = 0.0f;
    for (int k = 0; k < reg_max; ++k) {
        dist += k * (probs[k] / sum_exp);
    }
    return dist;
}

DetectionResult postprocess_yolo26_espdl(
    const int8_t* score_data[3],  // score0, score1, score2
    const int8_t* box_data[3],    // box0,   box1,   box2
    float conf_thr,
    float iou_thr)
{
    DetectionResult result;
    result.clear();

    for (int lvl = 0; lvl < 3; ++lvl) {
        const int H = GRID_H[lvl];
        const int W = GRID_W[lvl];
        const int stride = STRIDES[lvl];
        const float score_scale = exp2f(SCORE_EXP_Y[lvl]);
        const float box_scale   = exp2f(BOX_EXP_Y[lvl]);

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = y * W + x;

                // --- Scores: NHWC → offset = idx * NC + c ---
                float best_score = -1e9f;
                int   best_cls   = 0;
                for (int c = 0; c < NC; ++c) {
                    float logit = score_data[lvl][idx * NC + c] * score_scale;
                    float prob  = 1.0f / (1.0f + expf(-logit));
                    if (prob > best_score) {
                        best_score = prob;
                        best_cls = c;
                    }
                }

                if (best_score < conf_thr) continue;

                // --- Box DFL: NHWC → offset = idx * 64 + ch ---
                float dequant_box[64];
                for (int ch = 0; ch < BOX_CH; ++ch) {
                    dequant_box[ch] = box_data[lvl][idx * BOX_CH + ch] * box_scale;
                }

                // DFL integral: [64] → [4] distances
                float dist_l = dfl_single(&dequant_box[0 * REG_MAX], REG_MAX);
                float dist_t = dfl_single(&dequant_box[1 * REG_MAX], REG_MAX);
                float dist_r = dfl_single(&dequant_box[2 * REG_MAX], REG_MAX);
                float dist_b = dfl_single(&dequant_box[3 * REG_MAX], REG_MAX);

                // dist2bbox
                float cx = (x + 0.5f) * stride;
                float cy = (y + 0.5f) * stride;

                Detection det;
                det.x1 = clamp01((cx - dist_l * stride) / 224.0f);
                det.y1 = clamp01((cy - dist_t * stride) / 224.0f);
                det.x2 = clamp01((cx + dist_r * stride) / 224.0f);
                det.y2 = clamp01((cy + dist_b * stride) / 224.0f);
                det.confidence = best_score;
                det.class_id   = best_cls;

                result.add(det);
            }
        }
    }

    nms_per_class(result, iou_thr);
    return result;
}
```

### 7.4 Coste computacional del DFL

El DFL integral requiere, **por cada candidato que pase el filtro de confianza**:
- 64 dequantizaciones (multiplicación × escala)
- 4 softmax sobre 16 elementos (4 × 16 = 64 exp + 64 div)
- 4 sumas ponderadas (4 × 16 = 64 MAC)

Esto es significativamente más costoso que ESPDet (que solo necesita 4 ReLU + 4 multiplicaciones). En ESP32-S3, estimar ~2-5× más tiempo de post-procesado para YOLO26 vs ESPDet.

**Optimización posible**: ejecutar el filtrado de scores **antes** de decodificar boxes. Solo calcular DFL para candidatos que superan el umbral (~5-20 de 1,029 típicamente).

---

## 8. Dequantización de Salidas INT8

### 8.1 Fórmula general

Los tensores de salida ESPDL usan cuantización simétrica POWER_OF_2:

$$
x_{float} = x_{int8} \times 2^{exponent}
$$

No hay zero-point (es simétrica), y la escala es siempre una potencia de 2, lo que permite implementar la dequantización como un **bit-shift** en lugar de una multiplicación float:

```cpp
// Opción 1: multiplicación (genérica)
float val = (float)raw_int8 * exp2f(exponent);

// Opción 2: bit-shift (más eficiente en MCU cuando exp < 0)
// Para exp = -3: val = raw_int8 / 8.0  (o equivalente: raw_int8 >> 3 con ajuste)
```

### 8.2 Tabla de exponentes por tensor

| Modelo | Tensor | Exponente | Escala ($2^{exp}$) |
|---|---|:---:|---|
| **ESPDet** | input | −7 | 0.0078125 |
| | score0, score1, score2 | −3 | 0.125 |
| | box0, box1 | −3 | 0.125 |
| | box2 | **−4** | 0.0625 |
| **YOLO26** | images | −7 | 0.0078125 |
| | score0, score2 | −3 | 0.125 |
| | **score1** | **−2** | **0.25** |
| | box0, box1, box2 | −3 | 0.125 |

> ⚠️ **Exponentes no uniformes**: `box2` de ESPDet y `score1` de YOLO26 tienen exponentes diferentes al resto. La dequantización debe usar el exponente correcto para cada tensor, no un valor fijo.

### 8.3 Rango de valores dequantizados

| Exponente | Rango float | Resolución |
|:---:|---|---|
| −2 | [−32.0, +31.75] | 0.25 |
| −3 | [−16.0, +15.875] | 0.125 |
| −4 | [−8.0, +7.9375] | 0.0625 |
| −7 | [−1.0, +0.9922] | 0.0078125 |

---

## 9. Integración en Firmware ESP-IDF

### 9.1 Cambios necesarios en `app_config.h`

Añadir dos nuevos `ModelType`:

```cpp
enum class ModelType : uint8_t {
    MOBILENET_SSD,   // MBNTv2_ssdlite_v1: 3 tensores (existente)
    YOLO11N,         // yolo11n_v1: [1,9,1029] (existente)
    YOLO26N,         // yolo26n_v1: [1,300,6] end-to-end (existente)
    ESPDET_PICO,     // NEW: espdet_pico_t4: 6 tensores FCOS-style
    YOLO26N_ESP,     // NEW: yolo26n_t2_esp: 6 tensores DFL
};
```

### 9.2 Embebido de modelos en Flash

Los archivos `.espdl` se incluyen como binarios embebidos via `CMakeLists.txt`:

```cmake
# En main/CMakeLists.txt
set(EMBED_FILES
    "${PROJECT_DIR}/models/espdet_pico_t4.espdl"
    "${PROJECT_DIR}/models/yolo26n_t2_esp.espdl"
)

idf_component_register(
    SRCS ...
    EMBED_FILES ${EMBED_FILES}
)
```

Acceso en C++:

```cpp
extern const uint8_t espdet_pico_t4_espdl_start[] asm("_binary_espdet_pico_t4_espdl_start");
extern const uint8_t espdet_pico_t4_espdl_end[]   asm("_binary_espdet_pico_t4_espdl_end");

extern const uint8_t yolo26n_t2_esp_espdl_start[] asm("_binary_yolo26n_t2_esp_espdl_start");
extern const uint8_t yolo26n_t2_esp_espdl_end[]   asm("_binary_yolo26n_t2_esp_espdl_end");
```

### 9.3 Carga e inferencia con esp-dl

```cpp
#include "dl_model.h"

// Cargar modelo
dl::Model model;
model.load(espdet_pico_t4_espdl_start,
           espdet_pico_t4_espdl_end - espdet_pico_t4_espdl_start);

// Preparar input (ya cuantizado a INT8, NHWC)
int8_t input_buffer[1 * 224 * 224 * 3];
// ... fill con imagen preprocesada ...

model.set_input(0, input_buffer);  // tensor index 0 = "input"
model.run();

// Obtener outputs (¡usar el orden correcto de §3.2 / §4.2!)
const int8_t* score1 = (const int8_t*)model.get_output(0);  // ESPDet: score1
const int8_t* score2 = (const int8_t*)model.get_output(1);  // ESPDet: score2
const int8_t* score0 = (const int8_t*)model.get_output(2);  // ESPDet: score0
const int8_t* box0   = (const int8_t*)model.get_output(3);  // ESPDet: box0
const int8_t* box1   = (const int8_t*)model.get_output(4);  // ESPDet: box1
const int8_t* box2   = (const int8_t*)model.get_output(5);  // ESPDet: box2
```

> ⚠️ **Verificar la API de esp-dl v3.x**: la interfaz `get_output(index)` puede variar. Consultar la documentación de esp-dl para la versión exacta utilizada en `03_ING_DESPLIEGUE`. Alternativamente, acceder por nombre del tensor: `model.get_output("score0")` si la API lo soporta.

### 9.4 Nuevas funciones de post-procesado

Añadir en `postprocess.cpp`:

```cpp
// Declarar en postprocess.h
DetectionResult postprocess_espdet_espdl(
    const int8_t* score_data[3],
    const int8_t* box_data[3],
    float conf_thr, float iou_thr);

DetectionResult postprocess_yolo26_espdl(
    const int8_t* score_data[3],
    const int8_t* box_data[3],
    float conf_thr, float iou_thr);
```

### 9.5 Partición de Flash

```
Flash total:         8 MB (8,388,608 bytes)
─────────────────────────────────────────
Firmware ESP-IDF:    ~3.5 MB (estimado)
ESPDet T4 ESPDL:     0.52 MB
YOLO26 T2 ESP ESPDL: 2.71 MB
─────────────────────────────────────────
Ocupación total:     ~6.7 MB de 8 MB
Margen restante:     ~1.3 MB
```

> **Si se despliegan ambos modelos simultáneamente**, caben holgadamente. Si se necesita margen adicional, desplegar solo uno es viable.

---

## 10. Consideraciones de Memoria y Rendimiento

### 10.1 SRAM (512 KB internos)

| Recurso | ESPDet T4 | YOLO26 T2 ESP |
|---|---|---|
| Input buffer | 150 KB | 150 KB |
| Activaciones intermedias (estimado) | ~150-200 KB | ~300-400 KB |
| Output buffers (6 tensores) | ~12 KB | ~55 KB |
| **Total estimado** | **~310-360 KB** | **~505-605 KB** |
| ¿Cabe en SRAM? | 🟢 Sí | 🔴 **No** — requiere PSRAM |

> **YOLO26 T2 ESP probablemente requiere PSRAM** para las activaciones intermedias (2.6M params implican feature maps grandes). Configurar esp-dl para usar PSRAM como arena de activaciones. Verificar empíricamente en dispositivo.

### 10.2 PSRAM (8 MB)

- PSRAM tiene ancho de banda ~5× menor que SRAM interno
- Si YOLO26 cae a PSRAM para activaciones, esperar **2-5× mayor latencia** de inferencia respecto a ESPDet que podría correr enteramente en SRAM
- Las LUTs Swish (76 para YOLO26 vs 46 para ESPDet) consumen espacio adicional pero están pre-computadas

### 10.3 Latencia estimada

| Componente | ESPDet T4 (est.) | YOLO26 T2 ESP (est.) |
|---|---|---|
| Preprocesado (crop + quant) | ~5-10 ms | ~5-10 ms |
| Inferencia INT8 | ~50-150 ms | ~200-500 ms |
| Post-procesado | ~1-5 ms | ~5-15 ms (DFL) |
| **Total estimado** | **~60-165 ms** | **~210-525 ms** |
| **FPS estimado** | ~6-16 FPS | ~2-5 FPS |

> ⚠️ Estos son estimados teóricos. La **latencia real debe medirse en dispositivo** — es una de las tareas prioritarias de `03_ING_DESPLIEGUE`.

### 10.4 Dual-core

El firmware actual usa core 0 para inferencia y core 1 para WiFi/HTTP. Esta configuración se mantiene para ambos modelos nuevos.

---

## 11. Diferencias con Modelos Previos (v1)

Los modelos previamente desplegados en `03_ING_DESPLIEGUE/models/` eran:

| Modelo v1 | Formato salida | Post-procesado |
|---|---|---|
| `MBNTv3S_ssdlite_v1_p2_best.espdl` | 3 tensores (obj, cls, bbox) | Anchors + NMS |
| `yolo11n_v1_best.espdl` | 1 tensor `[1, 9, 1029]` | Transposición + NMS |
| `yolo26n_v1_best.espdl` | 1 tensor `[1, 300, 6]` | Solo filtrado (end-to-end NMS) |

### 11.1 Cambios clave respecto a v1

| Aspecto | v1 | v2 (nuevos modelos) |
|---|---|---|
| **N.º salidas** | 1-3 tensores | **6 tensores** (ambos) |
| **Tipo de salida** | Float32 (post-dequant ESP-DL) | **INT8 raw** — requiere dequantización manual |
| **Formato box** | xywh o xyxy (ya decodificado) | Distancias l,t,r,b (ESPDet) o DFL logits (YOLO26) |
| **Sigmoid** | Ya aplicado internamente o no necesitado | **Hay que aplicarlo** a los scores |
| **NMS** | Integrado (YOLO26 v1) o estándar | **Siempre necesario** — greedy per-class |
| **Grid generation** | No necesario (ya calculado) | **Necesario** — generar centros de grid |

> **Implicación principal**: el post-procesado de los nuevos modelos es más complejo que el de v1. Los modelos v1 devolvían detecciones semi-procesadas; los v2 devuelven tensores de features crudas que requieren decodificación completa.

### 11.2 Ventaja de los v2

A cambio de mayor complejidad de post-procesado:
- **Cuantización más robusta**: al separar scores de boxes, cada tensor tiene su propia escala INT8
- **Mejor mAP**: ESPDet 0.5319 mAP@50, YOLO26 ESP 0.4343 — superiores a MBNTv2 SSD en el mismo dataset
- **Modelos re-entrenados** específicamente para el dataset IODC de 5 clases

---

## 12. Checklist de Validación

### 12.1 Antes de flashear

- [ ] Copiar `.espdl` a `03_ING_DESPLIEGUE/models/`
- [ ] Actualizar `CMakeLists.txt` con `EMBED_FILES` para los nuevos modelos
- [ ] Añadir `ModelType::ESPDET_PICO` y `ModelType::YOLO26N_ESP` a `app_config.h`
- [ ] Implementar `postprocess_espdet_espdl()` en `postprocess.cpp`
- [ ] Implementar `postprocess_yolo26_espdl()` en `postprocess.cpp` (con DFL)
- [ ] Actualizar `espdl_engine.cpp` para esp-dl v3.x API (cargar modelo, set_input, run, get_output)
- [ ] Verificar nombres de tensor de entrada (`input` vs `images`)
- [ ] Verificar orden de outputs (`get_output(0)` = qué tensor — ver §3.2 / §4.2)
- [ ] Configurar arena de activaciones (SRAM para ESPDet, PSRAM para YOLO26)

### 12.2 Validación funcional en dispositivo

- [ ] ESPDet T4: inferencia end-to-end → ¿detecta objetos correctamente?
- [ ] YOLO26 T2 ESP: inferencia end-to-end → ¿detecta objetos correctamente?
- [ ] Comparar detecciones en dispositivo vs simulación CPU (`eval_fp32_vs_int8.py`)
- [ ] Medir latencia: preprocesado + inferencia + post-procesado
- [ ] Medir consumo de heap (internal SRAM + PSRAM)
- [ ] Verificar estabilidad (100+ frames sin crash)
- [ ] Test con imágenes conocidas del dataset de test IODC

### 12.3 Métricas objetivo

| Métrica | Mínimo aceptable | Ideal |
|---|---|---|
| mAP@50 (on-device vs sim.) | ±2% respecto a simulación | Idéntico |
| Latencia total (ESPDet) | < 500 ms | < 200 ms |
| Latencia total (YOLO26) | < 1000 ms | < 500 ms |
| Heap free (internal) | > 50 KB | > 100 KB |
| Crash rate | 0 en 100 frames | 0 en 1000 frames |

---

> **Documento generado**: 24 de febrero de 2026  
> **Autor**: Pipeline `02_ING_MODELOS/Train_MLOps`  
> **Referencia**: `docs/Registro_Cuantizacion_Modelos.md` (§9.4, §9.5, §10)  
> **Destino**: Fase `03_ING_DESPLIEGUE` — integración firmware ESP32-S3  
