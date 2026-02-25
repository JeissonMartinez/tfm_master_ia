# Instructivo de Despliegue — YOLO26n T3 (DFL Removal) para ESP32-S3

> **Objetivo**: Integrar el modelo YOLO26n T3 (`reg_max=1`, sin DFL integral) en el firmware ESP32-S3 para comparar tiempos de inferencia/latencia frente al YOLO26 T2 existente.  
> **Target**: ESP32-S3 WROOM N16R8 (Xtensa LX7, 512 KB SRAM, 8 MB PSRAM, 16 MB Flash)  
> **Placa**: Freenove ESP32-S3 CAM Board + OV5640  
> **Framework firmware**: ESP-IDF v5.x + esp-dl v3.x  
> **Origen del modelo**: `02_ING_MODELOS/Train_MLOps/outputs/espdl/yolo26n_t3_esp/`  
> **Referencia de entrenamiento**: `02_ING_MODELOS/Train_MLOps/docs/Registro_Entrenamiento_YOLO26.md` (§5 Train 3)  
> **Fecha**: 24 de febrero de 2026  

---

## Índice

1. [Contexto y Motivación](#1-contexto-y-motivación)
2. [Resumen Comparativo T2 vs T3](#2-resumen-comparativo-t2-vs-t3)
3. [Artefactos y Ubicación](#3-artefactos-y-ubicación)
4. [Especificación Técnica — YOLO26n T3 ESP](#4-especificación-técnica--yolo26n-t3-esp)
5. [Preprocesado de Entrada](#5-preprocesado-de-entrada)
6. [Post-procesado — YOLO26n T3 ESP (Sin DFL)](#6-post-procesado--yolo26n-t3-esp-sin-dfl)
7. [Diferencias Clave: T3 (Sin DFL) vs T2 (Con DFL)](#7-diferencias-clave-t3-sin-dfl-vs-t2-con-dfl)
8. [Integración en Firmware — Paso a Paso](#8-integración-en-firmware--paso-a-paso)
9. [Flash del Modelo](#9-flash-del-modelo)
10. [Consideraciones de Memoria y Rendimiento](#10-consideraciones-de-memoria-y-rendimiento)
11. [Plan de Pruebas de Latencia/Inferencia](#11-plan-de-pruebas-de-latenciainferencia)
12. [Checklist de Validación](#12-checklist-de-validación)
13. [Rollback — Restaurar T2 sin Afectar Código Existente](#13-rollback--restaurar-t2-sin-afectar-código-existente)

---

## 1. Contexto y Motivación

### 1.1 ¿Qué es YOLO26 T3?

YOLO26n T3 es una variante del modelo YOLO26 entrenada con **DFL Removal** (`reg_max=1`), lo que elimina la Distribution Focal Loss integral del post-procesado. En lugar de predecir una distribución de probabilidad sobre 16 bins para cada coordenada de bounding box (64 canales), T3 predice directamente las 4 distancias `(l, t, r, b)` en solo 4 canales.

### 1.2 ¿Por qué desplegar T3?

| Motivación | Detalle |
|---|---|
| **Post-processing simplificado** | Sin softmax + weighted sum DFL → menos operaciones en ESP32-S3 |
| **Box output 16× más compacto** | `[1, 4, H, W]` vs `[1, 64, H, W]` → menos memoria de activaciones |
| **Comparar latencia real** | Medir cuánto tiempo ahorra la eliminación del DFL en dispositivo |
| **Menos parámetros** | 2,505,750 vs 2,590,815 (-3.3%) → modelo ligeramente más pequeño |

### 1.3 Principio de no-regresión

> ⚠️ **IMPORTANTE**: La integración de T3 **NO debe alterar, eliminar ni dañar** el código existente para YOLO26 T2 (`postprocess_yolo26_espdl`) ni para ESPDet Pico T4 (`postprocess_espdet_espdl`). T3 se añade como un nuevo `ModelType` adicional y una nueva función de post-procesado independiente.

---

## 2. Resumen Comparativo T2 vs T3

| Propiedad | YOLO26n T2 ESP | YOLO26n T3 ESP |
|---|:---:|:---:|
| **Archivo ESPDL** | `yolo26n_t2_esp.espdl` | `yolo26n_t3_esp.espdl` |
| **Tamaño ESPDL** | 2.71 MB (2,847,120 B) | **2.57 MB** (2,691,056 B) |
| **Parámetros** | 2,590,815 | **2,505,750** (-3.3%) |
| **reg_max** | 16 (DFL activa) | **1** (DFL = Identity) |
| **Canales box** | 64 (DFL, 4 × 16 bins) | **4** (l, t, r, b directos) |
| **Canales score** | 5 | 5 |
| **N.º salidas ESPDL** | 6 tensores | 6 tensores |
| **Post-procesado** | Sigmoid + **DFL integral** + dist2box + NMS | Sigmoid + dist2box + NMS |
| **mAP@50 (Test, FP32)** | **0.7747** | 0.7466 (-3.6%) |
| **mAP@50 (ESP FP32)** | **0.5297** | 0.5027 (-5.1%) |
| **mAP@50 (ESP INT8)** | **0.4343** | 0.3769 (-13.2%) |
| **Degradación INT8** | **18.0%** | 25.0% |

### 2.1 Mejora esperada en latencia

El post-procesado de T3 elimina, **por cada candidato**, las siguientes operaciones que T2 sí requiere:

| Operación eliminada | Por candidato | Total (1,029 candidatos) |
|---|---|---|
| 64 dequantizaciones de box | 64 MUL | 65,856 MUL |
| 4× softmax sobre 16 bins | 64 EXP + 64 DIV | 65,856 EXP + 65,856 DIV |
| 4× weighted sum | 64 MAC | 65,856 MAC |

En la práctica, solo ~5-20 candidatos pasan el filtro de scores, pero el **delta de memoria** (64 vs 4 canales por punto del grid) es significativo.

---

## 3. Artefactos y Ubicación

### 3.1 Archivos a copiar a `03_ING_DESPLIEGUE/models/espdl/`

```
Origen: 02_ING_MODELOS/Train_MLOps/outputs/espdl/yolo26n_t3_esp/

yolo26n_t3_esp/
├── yolo26n_t3_esp.espdl    # 2,691,056 bytes — modelo para Flash
├── yolo26n_t3_esp.info     # Grafo ESPDL legible (debug/verificación)
└── yolo26n_t3_esp.json     # Configuración de cuantización

Destino: 03_ING_DESPLIEGUE/models/espdl/
```

**Comando para copiar:**

```bash
cp 02_ING_MODELOS/Train_MLOps/outputs/espdl/yolo26n_t3_esp/yolo26n_t3_esp.* \
   03_ING_DESPLIEGUE/models/espdl/
```

### 3.2 ONNX fuente (para re-cuantización si es necesario)

| Archivo | Ubicación | Tamaño |
|---|---|---|
| `best_esp.onnx` | `02_ING_MODELOS/Train_MLOps/outputs/yolo26n_custom_v3-run1/export/` | 9.14 MB |
| `best.onnx` | `02_ING_MODELOS/Train_MLOps/outputs/yolo26n_custom_v3-run1/export/` | 9.21 MB |

### 3.3 Scripts de referencia (en `02_ING_MODELOS/Train_MLOps/scripts/`)

| Script | Función |
|---|---|
| `export_yolo26_esp.py` | Re-exportación YOLO26 T3 con 6 salidas ESP |
| `convert_onnx_to_espdl.py` | Conversión ONNX → ESPDL (config: `yolo26n_t3_esp`) |
| `eval_fp32_vs_int8.py` | Evaluación FP32 vs INT8 (config: `yolo26n_t3_esp`) |

---

## 4. Especificación Técnica — YOLO26n T3 ESP

### 4.1 Entrada

| Propiedad | Valor |
|---|---|
| **Nombre del tensor** | `images` |
| **Shape ESPDL** | `[1, 224, 224, 3]` (NHWC) |
| **Tipo** | INT8 |
| **Exponente** | −7 (escala = $2^{-7}$ = 0.0078125) |
| **Rango representable** | $[-128 \times 2^{-7},\ +127 \times 2^{-7}]$ = [−1.0, +0.9921875] |

> **Idéntico a YOLO26 T2 ESP y ESPDet T4.** No requiere cambios en el preprocesado.

### 4.2 Salidas

El modelo devuelve **6 tensores** en el siguiente orden (según grafo ESPDL `.info`):

```
return %box0, %box1, %score0, %box2, %score1, %score2
```

| Índice retorno | Tensor | Shape (NHWC) | Tipo | Exponente | FPN Level | Stride | Contenido |
|:---:|---|---|---|:---:|:---:|:---:|---|
| 0 | `box0` | `[1, 28, 28, 4]` | INT8 | −3 | P3 | 8 | **Distancias directas** l, t, r, b |
| 1 | `box1` | `[1, 14, 14, 4]` | INT8 | −3 | P4 | 16 | **Distancias directas** l, t, r, b |
| 2 | `score0` | `[1, 28, 28, 5]` | INT8 | −3 | P3 | 8 | Logits de clasificación (5 clases) |
| 3 | `box2` | `[1, 7, 7, 4]` | INT8 | **−4** | P5 | 32 | **Distancias directas** l, t, r, b |
| 4 | `score1` | `[1, 14, 14, 5]` | INT8 | **−2** | P4 | 16 | Logits de clasificación (5 clases) |
| 5 | `score2` | `[1, 7, 7, 5]` | INT8 | −3 | P5 | 32 | Logits de clasificación (5 clases) |

> ⚠️ **El orden de retorno NO es secuencial**: box0, box1, score0, box2, score1, score2. El código accede por **nombre** (`get_output_by_name("score0")`), no por índice, por lo que el orden no afecta al post-procesado.

> ⚠️ **Exponentes no uniformes**: `box2` tiene exponente **−4** (no −3) y `score1` tiene exponente **−2** (no −3). Idéntico patrón al T2.

### 4.3 Diferencia crítica vs T2: Box channels

| Modelo | Box shape por nivel | Canales | Interpretación |
|---|---|:---:|---|
| **T2** (DFL) | `[1, H, W, 64]` | 64 | 4 direcciones × 16 bins DFL |
| **T3** (sin DFL) | `[1, H, W, 4]` | **4** | 4 distancias directas (l, t, r, b) |

Esto significa que T3 produce outputs de box **16 veces más pequeños** en memoria, y no requiere el paso de DFL integral.

### 4.4 Candidatos totales por nivel

| Nivel | Grid | Candidatos |
|:---:|:---:|:---:|
| P3 (stride 8) | 28 × 28 | 784 |
| P4 (stride 16) | 14 × 14 | 196 |
| P5 (stride 32) | 7 × 7 | 49 |
| **Total** | | **1,029** |

Idéntico a T2 y ESPDet.

---

## 5. Preprocesado de Entrada

El preprocesado es **100% idéntico** al de T2 y ESPDet. **No requiere ningún cambio** en `image_proc.cpp`.

### 5.1 Pipeline

```
OV5640 (320×240 RGB565)
    ↓ rgb565_to_rgb888()
    ↓ Center crop: offset_x=48, offset_y=8 → 224×224
    ↓ Cuantización INT8 para ESP-DL:
        int8_val = clamp((pixel * 128 + 127) / 255, 0, 127)
    → Tensor INT8 [1, 224, 224, 3] (NHWC), exponent=-7
```

### 5.2 Función existente a reutilizar

```cpp
// image_proc.cpp — ya implementada, sin cambios
int8_t* image_preprocess_espdl(const camera_fb_t* fb, int8_t* output);
```

### 5.3 Nombre del tensor de entrada

| Modelo | Tensor input |
|---|---|
| ESPDet T4 | `input` |
| YOLO26 T2 ESP | `images` |
| **YOLO26 T3 ESP** | **`images`** ← mismo que T2 |

> El `EspDlEngine` asigna el input vía `model->get_input()->assign(...)`, por lo que el nombre del tensor no afecta al código.

---

## 6. Post-procesado — YOLO26n T3 ESP (Sin DFL)

### 6.1 Parámetros recomendados

| Parámetro | Valor | Origen |
|---|---|---|
| `conf_threshold` | **0.25** | Entrenamiento T3 (conf_threshold=0.15 de training, 0.25 para deploy) |
| `iou_threshold` | **0.45** | Default Ultralytics |

### 6.2 Pipeline de decodificación por nivel FPN

Para cada nivel $l \in \{0, 1, 2\}$ con stride $s_l \in \{8, 16, 32\}$ y grid $H_l \times W_l$:

**Paso 1 — Dequantizar**:

$$
\text{score\_float}[i] = \text{score\_int8}[i] \times 2^{e_{score_l}}
$$

$$
\text{box\_float}[i] = \text{box\_int8}[i] \times 2^{e_{box_l}}
$$

**Paso 2 — Sigmoid** sobre scores (logits → probabilidades):

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

**Paso 5 — Decode box** (distancias directas l, t, r, b → coordenadas xyxy normalizadas):

$$
x_1 = \frac{cx - l \times s_l}{224}, \quad y_1 = \frac{cy - t \times s_l}{224}
$$

$$
x_2 = \frac{cx + r \times s_l}{224}, \quad y_2 = \frac{cy + b \times s_l}{224}
$$

Clamp final a $[0, 1]$.

> **NO hay Paso "DFL Integral"** — Esta es la diferencia fundamental respecto a T2. Las distancias ya son directas tras la dequantización.

**Paso 6 — NMS** per-class greedy (estándar, reutilizar `nms_per_class()` existente).

### 6.3 Pseudocódigo C++

```cpp
// YOLO26 T3 ESP: 6 outputs — score0/1/2 [1,H,W,5], box0/1/2 [1,H,W,4]
// NHWC layout in ESPDL
// Sin DFL — distancias directas (idéntico a ESPDet, diferente stride decode)

static const int STRIDES[3] = {8, 16, 32};
static const int GRID_H[3]  = {28, 14, 7};
static const int GRID_W[3]  = {28, 14, 7};
static const int NC = 5;
static const int BOX_CH = 4;  // ← Solo 4 canales (vs 64 en T2)

DetectionResult postprocess_yolo26_t3_espdl(
    const InferenceEngine* engine,
    float conf_thr = 0.25f,
    float iou_thr  = 0.45f)
{
    DetectionResult result;
    result.clear();

    static const char* score_names[3] = {"score0", "score1", "score2"};
    static const char* box_names[3]   = {"box0", "box1", "box2"};

    const float inv_dim = 1.0f / 224.0f;

    for (int s = 0; s < 3; ++s) {
        const int8_t* score_data = (const int8_t*)engine->get_output_by_name(score_names[s]);
        const int8_t* box_data   = (const int8_t*)engine->get_output_by_name(box_names[s]);

        if (!score_data || !box_data) continue;

        int score_exp = engine->get_output_exponent(score_names[s]);
        int box_exp   = engine->get_output_exponent(box_names[s]);
        int grid_h    = GRID_H[s];
        int grid_w    = GRID_W[s];
        int stride    = STRIDES[s];

        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                int offset = (gy * grid_w + gx) * NC;

                // --- Scores: dequant + sigmoid → find best class ---
                int best_cls = 0;
                float best_score = -1e9f;
                for (int c = 0; c < NC; ++c) {
                    float s_val = dequant(score_data[offset + c], score_exp);
                    if (s_val > best_score) {
                        best_score = s_val;
                        best_cls = c;
                    }
                }

                float conf = sigmoid(best_score);
                if (conf < conf_thr) continue;

                // --- Box: dequant → distancias DIRECTAS (sin DFL) ---
                int box_offset = (gy * grid_w + gx) * BOX_CH;  // 4 canales
                float l = dequant(box_data[box_offset + 0], box_exp);
                float t = dequant(box_data[box_offset + 1], box_exp);
                float r = dequant(box_data[box_offset + 2], box_exp);
                float b = dequant(box_data[box_offset + 3], box_exp);

                // Grid center
                float cx = (gx + 0.5f) * stride;
                float cy = (gy + 0.5f) * stride;

                // dist2bbox (normalizado)
                Detection det;
                det.x1 = clamp01((cx - l * stride) * inv_dim);
                det.y1 = clamp01((cy - t * stride) * inv_dim);
                det.x2 = clamp01((cx + r * stride) * inv_dim);
                det.y2 = clamp01((cy + b * stride) * inv_dim);
                det.confidence = conf;
                det.class_id   = best_cls;

                result.add(det);
                if (result.count >= MAX_DETECTIONS) break;
            }
            if (result.count >= MAX_DETECTIONS) break;
        }
        if (result.count >= MAX_DETECTIONS) break;
    }

    if (result.count > 1) {
        nms_per_class(result, iou_thr);
    }

    return result;
}
```

### 6.4 Tabla de exponentes por tensor

| Tensor | Exponente | Escala ($2^{exp}$) | Rango float |
|---|:---:|---|---|
| `images` (input) | −7 | 0.0078125 | [−1.0, +0.99] |
| `score0` | −3 | 0.125 | [−16.0, +15.88] |
| `score1` | **−2** | **0.25** | [−32.0, +31.75] |
| `score2` | −3 | 0.125 | [−16.0, +15.88] |
| `box0` | −3 | 0.125 | [−16.0, +15.88] |
| `box1` | −3 | 0.125 | [−16.0, +15.88] |
| `box2` | **−4** | **0.0625** | [−8.0, +7.94] |

> El patrón de exponentes variables en `score1` y `box2` es **idéntico al observado en T2**. No implica problemas; el código lee los exponentes dinámicamente vía `get_output_exponent()`.

---

## 7. Diferencias Clave: T3 (Sin DFL) vs T2 (Con DFL)

### 7.1 Comparativa lado a lado del post-procesado

| Aspecto | T2 (Con DFL) | T3 (Sin DFL) |
|---|---|---|
| **Box channels** | 64 (4 × 16 bins) | **4** (l, t, r, b directos) |
| **Dequant box** | 64 MUL por candidato | **4 MUL** por candidato |
| **Softmax DFL** | 4 × softmax(16) = 64 EXP + 64 DIV | **No necesario** |
| **Weighted sum** | 4 × weighted_sum(16) = 64 MAC | **No necesario** |
| **dist2bbox** | Idéntico | Idéntico |
| **Sigmoid scores** | Idéntico | Idéntico |
| **NMS** | Idéntico | Idéntico |
| **Función existente** | `postprocess_yolo26_espdl()` | **NUEVA**: `postprocess_yolo26_t3_espdl()` |

### 7.2 Diagrama de flujo comparativo

```
T2 (DFL):
  score[5] ─→ dequant ─→ sigmoid ─→ filter ─┐
  box[64]  ─→ dequant ─→ reshape[4,16] ─→ softmax ─→ weighted_sum ─→ [4 dist] ─┐
                                                                                  ├─→ dist2bbox ─→ NMS
                                                                                  │
T3 (Sin DFL):
  score[5] ─→ dequant ─→ sigmoid ─→ filter ─┐
  box[4]   ─→ dequant ─→ [4 dist directas] ─┘─→ dist2bbox ─→ NMS
```

### 7.3 Código existente que NO se modifica

| Archivo | Función/Sección | Estado |
|---|---|---|
| `postprocess.cpp` | `postprocess_espdet_espdl()` | ✅ Sin cambios |
| `postprocess.cpp` | `postprocess_yolo26_espdl()` | ✅ Sin cambios |
| `postprocess.cpp` | `postprocess_mobilenet()` | ✅ Sin cambios |
| `postprocess.cpp` | `postprocess_yolo11()` | ✅ Sin cambios |
| `postprocess.cpp` | `postprocess_yolo26()` | ✅ Sin cambios |
| `postprocess.cpp` | `nms_per_class()`, `iou_xyxy()`, `sigmoid()`, `dequant()` | ✅ Sin cambios, reutilizados |
| `espdl_engine.h/.cpp` | Motor ESP-DL completo | ✅ Sin cambios |
| `image_proc.cpp` | `image_preprocess_espdl()` | ✅ Sin cambios |
| `main.cpp` | Cases `ESPDET_PICO`, `YOLO26N_ESP` | ✅ Sin cambios |

---

## 8. Integración en Firmware — Paso a Paso

### 8.1 Paso 1 — Copiar modelo ESPDL

```bash
# Desde la raíz del proyecto TFM_UNIR
cp 02_ING_MODELOS/Train_MLOps/outputs/espdl/yolo26n_t3_esp/yolo26n_t3_esp.* \
   03_ING_DESPLIEGUE/models/espdl/
```

Verificar que queden los 3 archivos:

```
03_ING_DESPLIEGUE/models/espdl/
├── espdet_pico_t4.espdl     # Existente (545 KB)
├── espdet_pico_t4.info      # Existente
├── espdet_pico_t4.json      # Existente
├── yolo26n_t2_esp.espdl     # Existente (2.71 MB)
├── yolo26n_t2_esp.info      # Existente
├── yolo26n_t2_esp.json      # Existente
├── yolo26n_t3_esp.espdl     # ← NUEVO (2.57 MB)
├── yolo26n_t3_esp.info      # ← NUEVO
└── yolo26n_t3_esp.json      # ← NUEVO
```

### 8.2 Paso 2 — Actualizar `partitions.csv`

El ESPDL de T3 (2.57 MB) cabe en la misma partición de 3 MB que T2. Se necesita una **nueva partición** si se quieren tener T2 y T3 simultáneamente, o **reusar la partición** `model_yolo26` si se quiere intercambiar.

**Opción A — Reusar partición existente (simple, recomendada para pruebas):**

No se modifica `partitions.csv`. Se flashea T3 en la misma partición `model_yolo26` (offset `0xB10000`). Solo un YOLO26 estará disponible a la vez.

**Opción B — Añadir nueva partición (para comparativa simultánea):**

```csv
# Name,          Type,  SubType,  Offset,     Size,       Flags
nvs,             data,  nvs,      0x9000,     0x6000,
phy_init,        data,  phy,      0xf000,     0x1000,
factory,         app,   factory,  0x10000,    0xA00000,
model_espdet,    data,  0x40,     0xA10000,   0x100000,
model_yolo26,    data,  0x40,     0xB10000,   0x300000,
model_yolo26t3,  data,  0x40,     0xE10000,   0x200000,
```

> **Viabilidad de Opción B**: Flash total = 16 MB (0x1000000). Fin de `model_yolo26t3` = 0xE10000 + 0x200000 = **0x1010000** → **excede 16 MB por 0x10000 (64 KB)**. Para resolver: reducir `model_yolo26` a 0x280000 (2.5 MB, sigue cabiendo T2 de 2.71 MB: ❌ no cabe) o reducir `factory` a 0x980000 (~9.5 MB).

**Opción B revisada — Ajustar factory:**

```csv
# Name,          Type,  SubType,  Offset,     Size,       Flags
nvs,             data,  nvs,      0x9000,     0x6000,
phy_init,        data,  phy,      0xf000,     0x1000,
factory,         app,   factory,  0x10000,    0x900000,
model_espdet,    data,  0x40,     0x910000,   0x100000,
model_yolo26,    data,  0x40,     0xA10000,   0x300000,
model_yolo26t3,  data,  0x40,     0xD10000,   0x2F0000,
```

> Factory = 9 MB (0x900000) → sigue siendo amplio para el firmware (~3-4 MB).  
> `model_yolo26t3` = 0x2F0000 = 3,014,656 B → ESPDL de 2,691,056 B cabe (89% uso).  
> Fin = 0xD10000 + 0x2F0000 = **0x1000000** = exactamente 16 MB ✅

**Recomendación**: Usar **Opción A** para pruebas iniciales (sin tocar `partitions.csv`). Migrar a **Opción B** solo si se necesitan ambos modelos YOLO26 en flash simultáneamente.

### 8.3 Paso 3 — Añadir `ModelType` y constantes en `app_config.h`

Añadir al enum `ModelType`:

```cpp
enum class ModelType : uint8_t {
    MOBILENET_SSD,   // MBNTv2_ssdlite_v1: 3 tensores
    YOLO11N,         // yolo11n_v1: [1,9,1029]
    YOLO26N,         // yolo26n_v1: [1,300,6] end-to-end (TFLite)
    ESPDET_PICO,     // ESPDet Pico T4: FCOS 3-scale (ESP-DL)
    YOLO26N_ESP,     // YOLO26n T2 ESP: DFL 3-scale (ESP-DL)
    YOLO26N_T3_ESP,  // ← NUEVO: YOLO26n T3 ESP: direct 3-scale (ESP-DL, sin DFL)
};
```

Añadir umbral de confianza:

```cpp
// Umbrales por modelo ESPDL
#define ESPDET_CONF_THRESHOLD   0.6f
#define YOLO26ESP_CONF_THRESHOLD 0.25f
#define YOLO26T3ESP_CONF_THRESHOLD 0.25f   // ← NUEVO: mismo umbral que T2
#define YOLO26T3ESP_IOU_THRESHOLD  0.45f   // ← NUEVO
```

### 8.4 Paso 4 — Añadir config en `main.cpp`

En `make_model_config()`, añadir un nuevo `case`:

```cpp
case ModelType::YOLO26N_T3_ESP:
    cfg.name            = "YOLO26n_T3_ESP";
    cfg.engine          = EngineType::ESP_DL;
    // Opción A: misma partición que T2
    cfg.espdl_partition = "model_yolo26";
    // Opción B: partición dedicada
    // cfg.espdl_partition = "model_yolo26t3";
    cfg.tflite_data     = nullptr;
    cfg.tflite_size     = 0;
    cfg.arena_size      = 0;
    cfg.conf_threshold  = YOLO26T3ESP_CONF_THRESHOLD;
    cfg.iou_threshold   = YOLO26T3ESP_IOU_THRESHOLD;
    break;
```

En `run_postprocess()`, añadir un nuevo `case`:

```cpp
case ModelType::YOLO26N_T3_ESP:
    return postprocess_yolo26_t3_espdl(engine, cfg.conf_threshold, cfg.iou_threshold);
```

### 8.5 Paso 5 — Implementar post-procesado en `postprocess.h` / `postprocess.cpp`

**En `postprocess.h`**, añadir la declaración (NO modificar las existentes):

```cpp
/// YOLO26n T3 ESP: Direct distance detector — 3 scales, 6 output tensors.
/// Each scale has score[HxWx5] + box[HxWx4] (direct distances l,t,r,b).
/// Pipeline: dequant → sigmoid(scores) → filter → dist2bbox → NMS.
/// NO DFL integral required (reg_max=1, Identity).
/// @param engine  Pointer to EspDlEngine (for get_output_by_name)
/// @param conf_thr  Confidence threshold after sigmoid
/// @param iou_thr   IoU threshold for NMS
DetectionResult postprocess_yolo26_t3_espdl(
    const InferenceEngine* engine,
    float conf_thr = YOLO26T3ESP_CONF_THRESHOLD,
    float iou_thr  = YOLO26T3ESP_IOU_THRESHOLD);
```

**En `postprocess.cpp`**, añadir la implementación **después** de `postprocess_yolo26_espdl()` (sin tocar nada existente):

```cpp
// ═══════════════════════════════════════════════════════════════════════════
//  YOLO26n T3 ESP — Direct distances (3-scale, 4 channels, NO DFL)
//
//  Output tensors:
//    score0 [1,28,28,5] exp=-3   score1 [1,14,14,5] exp=-2   score2 [1,7,7,5] exp=-3
//    box0   [1,28,28,4] exp=-3   box1   [1,14,14,4] exp=-3   box2   [1,7,7,4] exp=-4
//
//  Box layout: [l, t, r, b] — direct predicted distances (like ESPDet)
//  NO DFL required: reg_max=1, dfl=Identity
//
//  Decode identical to ESPDet:
//    l,t,r,b = dequant(box) (no ReLU needed — values already positive from training)
//    x1 = (cx - l * stride) / 224, etc.
// ═══════════════════════════════════════════════════════════════════════════
DetectionResult postprocess_yolo26_t3_espdl(
    const InferenceEngine* engine,
    float conf_thr,
    float iou_thr)
{
    DetectionResult result;
    result.clear();

    static const char* score_names[NUM_SCALES] = {"score0", "score1", "score2"};
    static const char* box_names[NUM_SCALES]   = {"box0", "box1", "box2"};

    const float inv_dim = 1.0f / static_cast<float>(INPUT_WIDTH);

    for (int s = 0; s < NUM_SCALES; ++s) {
        const int8_t* score_data = static_cast<const int8_t*>(
            engine->get_output_by_name(score_names[s]));
        const int8_t* box_data = static_cast<const int8_t*>(
            engine->get_output_by_name(box_names[s]));

        if (!score_data || !box_data) {
            ESP_LOGW(TAG, "YOLO26T3: output '%s' o '%s' no encontrado",
                     score_names[s], box_names[s]);
            continue;
        }

        int score_exp = engine->get_output_exponent(score_names[s]);
        int box_exp   = engine->get_output_exponent(box_names[s]);
        int grid_h    = GRID_SIZES[s];
        int grid_w    = GRID_SIZES[s];
        int stride    = GRID_STRIDES[s];

        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                int score_offset = (gy * grid_w + gx) * NUM_CLASSES;

                // Find best class (dequant + compare in logit space)
                int best_cls = 0;
                float best_score = -1e9f;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    float s_val = dequant(score_data[score_offset + c], score_exp);
                    if (s_val > best_score) {
                        best_score = s_val;
                        best_cls = c;
                    }
                }

                float conf = sigmoid(best_score);
                if (conf < conf_thr) continue;

                // Direct distance decode (NO DFL) — only 4 channels
                int box_offset = (gy * grid_w + gx) * 4;
                float l = dequant(box_data[box_offset + 0], box_exp);
                float t = dequant(box_data[box_offset + 1], box_exp);
                float r = dequant(box_data[box_offset + 2], box_exp);
                float b = dequant(box_data[box_offset + 3], box_exp);

                // Grid center
                float cx = (static_cast<float>(gx) + 0.5f) * stride;
                float cy = (static_cast<float>(gy) + 0.5f) * stride;

                // dist2bbox normalised
                Detection det;
                det.x1 = clamp01((cx - l * stride) * inv_dim);
                det.y1 = clamp01((cy - t * stride) * inv_dim);
                det.x2 = clamp01((cx + r * stride) * inv_dim);
                det.y2 = clamp01((cy + b * stride) * inv_dim);
                det.confidence = conf;
                det.class_id   = best_cls;

                result.add(det);
                if (result.count >= MAX_DETECTIONS) break;
            }
            if (result.count >= MAX_DETECTIONS) break;
        }
        if (result.count >= MAX_DETECTIONS) break;
    }

    if (result.count > 1) {
        nms_per_class(result, iou_thr);
    }

    return result;
}
```

### 8.6 Paso 6 — Actualizar constante de compilación

Para activar T3, cambiar en `main.cpp` (o vía `sdkconfig`):

```cpp
#ifndef ACTIVE_MODEL_TYPE
#define ACTIVE_MODEL_TYPE   ModelType::YOLO26N_T3_ESP   // ← Cambiar aquí
#endif
```

### 8.7 Paso 7 — Actualizar `postprocess_init()` log

En `postprocess_init()`, añadir una línea de log:

```cpp
ESP_LOGI(TAG, "   YOLO26 T3: 3 scales, direct dist (no DFL, box_ch=4)");
```

### 8.8 Paso 8 — Actualizar `flash_models.sh` (si Opción A)

Si se usa la Opción A (reusar partición), actualizar el script para flashear T3:

```bash
# En scripts/flash_models.sh, agregar variable condicional:
# Para flashear T3 en lugar de T2:
YOLO26_FILE="$PROJECT_DIR/models/espdl/yolo26n_t3_esp.espdl"
```

O crear un script separado `flash_yolo26_t3.sh` para no modificar el existente.

---

## 9. Flash del Modelo

### 9.1 Opción A — Reusar partición (recomendada para pruebas)

```bash
cd 03_ING_DESPLIEGUE

# Flashear solo el modelo T3 en la partición model_yolo26
python -m esptool --chip esp32s3 --port /dev/tty.usbmodem* --baud 921600 \
    write_flash --flash_mode dio --flash_size 16MB \
    0xB10000 models/espdl/yolo26n_t3_esp.espdl

# Compilar y flashear firmware
idf.py build && idf.py -p /dev/tty.usbmodem* flash monitor
```

> **IMPORTANTE**: Al flashear T3 sobre la partición de T2, el firmware lee `model_yolo26` como partición label. No es necesario cambiar `partitions.csv` ni el label — solo el contenido binario de la partición cambia.

### 9.2 Opción B — Partición dedicada

```bash
# Actualizar partitions.csv (ver §8.2 Opción B revisada)
# Luego flashear ambos modelos YOLO26:
python -m esptool --chip esp32s3 --port /dev/tty.usbmodem* --baud 921600 \
    write_flash --flash_mode dio --flash_size 16MB \
    0xA10000 models/espdl/espdet_pico_t4.espdl \
    0xB10000 models/espdl/yolo26n_t2_esp.espdl \
    0xD10000 models/espdl/yolo26n_t3_esp.espdl
```

### 9.3 Intercambio rápido T2 ↔ T3 (Opción A)

Para comparar tiempos de latencia entre T2 y T3:

```bash
# 1. Flashear T2 → medir latencia
python -m esptool --chip esp32s3 --port $PORT --baud 921600 \
    write_flash 0xB10000 models/espdl/yolo26n_t2_esp.espdl
# Cambiar ACTIVE_MODEL_TYPE → YOLO26N_ESP
idf.py build flash monitor

# 2. Flashear T3 → medir latencia
python -m esptool --chip esp32s3 --port $PORT --baud 921600 \
    write_flash 0xB10000 models/espdl/yolo26n_t3_esp.espdl
# Cambiar ACTIVE_MODEL_TYPE → YOLO26N_T3_ESP
idf.py build flash monitor
```

> **Nota**: Cada intercambio requiere recompilar el firmware porque el `ModelType` activo y la función de post-procesado cambian. Solo el flash del modelo (`esptool write_flash`) es independiente.

---

## 10. Consideraciones de Memoria y Rendimiento

### 10.1 Memoria de salida output buffers

| Recurso | T2 (DFL) | T3 (Sin DFL) | Δ |
|---|---|---|---|
| box0 `[28,28,64]` | 50,176 B | **box0 `[28,28,4]`** → 3,136 B | **−94%** |
| box1 `[14,14,64]` | 12,544 B | **box1 `[14,14,4]`** → 784 B | −94% |
| box2 `[7,7,64]` | 3,136 B | **box2 `[7,7,4]`** → 196 B | −94% |
| score0-2 | 4,655 B | 4,655 B | 0% |
| **Total outputs** | **70,511 B** | **8,771 B** | **−88%** |

**Ahorro**: ~60 KB en buffers de salida. Esto reduce la presión sobre SRAM/PSRAM.

### 10.2 PSRAM consumo estimado

| Modelo | ESPDL size | Model RAM (estimado) | Output buffers |
|---|---|---|---|
| ESPDet T4 | 0.52 MB | ~1.5 MB | ~12 KB |
| YOLO26 T2 | 2.71 MB | ~4.0 MB | ~70 KB |
| **YOLO26 T3** | **2.57 MB** | **~3.8 MB** | **~9 KB** |

> T3 es ligeramente más ligero que T2 en RAM total (~200 KB menos entre modelo + outputs).

### 10.3 Latencia estimada de post-procesado

| Componente | T2 (DFL) | T3 (Sin DFL) | Estimación |
|---|---|---|---|
| Dequant scores | ~0.1 ms | ~0.1 ms | Idéntico |
| Sigmoid + filter | ~0.2 ms | ~0.2 ms | Idéntico |
| Dequant box | ~0.3 ms (64 ch) | **~0.02 ms** (4 ch) | **15× menos** |
| DFL softmax + wsum | ~0.5-1.0 ms | **0 ms** | **Eliminado** |
| dist2bbox | ~0.05 ms | ~0.05 ms | Idéntico |
| NMS | ~0.1 ms | ~0.1 ms | Idéntico |
| **Total post-procesado** | **~1.3-1.7 ms** | **~0.5 ms** | **2-3× más rápido** |

> ⚠️ Estas son estimaciones teóricas basadas en la complejidad algorítmica. La **latencia real debe medirse en dispositivo**.

### 10.4 Latencia de inferencia (hipótesis)

La inferencia ESP-DL debería ser ligeramente más rápida para T3 porque:
- La cabeza de detección tiene menos parámetros (cv2 con 4 canales output vs 64)
- Los feature maps de box son 16× más pequeños
- Menor presión de caché PSRAM

Estimación conservadora:

| Componente | T2 | T3 (est.) |
|---|---|---|
| Preprocesado | 5-10 ms | 5-10 ms |
| Inferencia ESP-DL | 200-500 ms | **180-450 ms** |
| Post-procesado | 1.5 ms | **0.5 ms** |
| **Total** | 210-515 ms | **190-465 ms** |

> **El beneficio principal de T3 es en post-procesado**, no en inferencia del backbone (que domina el tiempo total).

---

## 11. Plan de Pruebas de Latencia/Inferencia

### 11.1 Métricas a capturar

El firmware ya mide y reporta vía WebSocket (dashboard HTML):

| Métrica | Campo en `InferenceMetrics` | Cómo se mide |
|---|---|---|
| **Preprocesado** | `preprocess_ms` | `metrics_frame_begin()` → `metrics_preprocess_end()` |
| **Inferencia** | `inference_ms` | `metrics_preprocess_end()` → `metrics_inference_end()` |
| **Post-procesado** | `postprocess_ms` | `metrics_inference_end()` → `metrics_postprocess_end()` |
| **Total** | `total_ms` | Suma de los tres |
| **FPS** | `fps` | `1000 / total_ms` |
| **EMA** | `ema_*` | Promedios exponenciales (α=0.065, ventana ~30 frames) |
| **Heap** | `heap_internal_free`, `psram_free` | `heap_caps_get_free_size()` |

### 11.2 Protocolo de comparación T2 vs T3

**Condiciones controladas:**
1. Misma imagen de entrada (usar modo `ON_DEMAND` con imagen fija)
2. Misma temperatura ambiental (el ESP32-S3 puede thermal-throttle)
3. Warm-up de 50 frames antes de medir
4. Recolectar 100 frames consecutivos
5. Reportar: media, mediana, P95, P99, desviación estándar

**Procedimiento:**

```
PASO 1 — Medir T2:
  a. Flash model T2 + compilar con YOLO26N_ESP
  b. Conectar al dashboard (http://<IP>/)
  c. Warm-up: 50 frames en modo CONTINUOUS
  d. Registrar 100 frames → extraer preprocess, inference, postprocess, total
  e. Anotar heap interno y PSRAM libre

PASO 2 — Medir T3:
  a. Flash model T3 + compilar con YOLO26N_T3_ESP
  b. Repetir pasos b-e

PASO 3 — Comparar:
  a. Tabla comparativa por componente
  b. Delta absoluto y porcentual
  c. Verificar hipótesis: ¿T3 post-procesado es 2-3× más rápido?
  d. Documentar en Registro de Despliegue
```

### 11.3 Métricas de accuracy en dispositivo

| Métrica | Cómo verificar |
|---|---|
| Deteciones correctas | Capturar 10+ escenas con objetos conocidos, comparar vs simulación |
| Falsos positivos | Contar FP en escenas sin objetos del dataset |
| Confianza típica | ¿Los scores en dispositivo son coherentes con eval FP32/INT8? |

### 11.4 Tabla de resultados esperada (template)

| Componente | T2 (medido) | T3 (medido) | Δ | Δ% |
|---|---|---|---|---|
| Preprocesado (ms) | | | | |
| Inferencia (ms) | | | | |
| Post-procesado (ms) | | | | |
| **Total (ms)** | | | | |
| **FPS** | | | | |
| Heap interno libre (KB) | | | | |
| PSRAM libre (KB) | | | | |
| Detecciones/frame avg | | | | |

---

## 12. Checklist de Validación

### 12.1 Pre-flash

- [ ] Copiar `yolo26n_t3_esp.espdl` a `03_ING_DESPLIEGUE/models/espdl/`
- [ ] Verificar tamaño: 2,691,056 bytes (2.57 MB)
- [ ] Decidir estrategia de partición: Opción A (reusar) o Opción B (nueva partición)
- [ ] Si Opción B: actualizar `partitions.csv`

### 12.2 Código firmware

- [ ] Añadir `YOLO26N_T3_ESP` a `ModelType` en `app_config.h`
- [ ] Añadir `YOLO26T3ESP_CONF_THRESHOLD` y `YOLO26T3ESP_IOU_THRESHOLD` en `app_config.h`
- [ ] Añadir case `YOLO26N_T3_ESP` en `make_model_config()` de `main.cpp`
- [ ] Añadir case `YOLO26N_T3_ESP` en `run_postprocess()` de `main.cpp`
- [ ] Declarar `postprocess_yolo26_t3_espdl()` en `postprocess.h`
- [ ] Implementar `postprocess_yolo26_t3_espdl()` en `postprocess.cpp` (código completo en §8.5)
- [ ] Actualizar log en `postprocess_init()` para incluir T3
- [ ] Cambiar `ACTIVE_MODEL_TYPE` a `ModelType::YOLO26N_T3_ESP`

### 12.3 Verificación de no-regresión

- [ ] `postprocess_espdet_espdl()` NO fue modificada
- [ ] `postprocess_yolo26_espdl()` NO fue modificada
- [ ] `image_preprocess_espdl()` NO fue modificada
- [ ] `EspDlEngine` NO fue modificado
- [ ] Compilar con `ACTIVE_MODEL_TYPE = ESPDET_PICO` → funciona igual
- [ ] Compilar con `ACTIVE_MODEL_TYPE = YOLO26N_ESP` → funciona igual
- [ ] Compilar con `ACTIVE_MODEL_TYPE = YOLO26N_T3_ESP` → funciona con T3

### 12.4 Validación funcional en dispositivo

- [ ] T3: inferencia end-to-end → detecta objetos correctamente
- [ ] T3: scores de confianza coherentes (no todos 0.0 ni todos 1.0)
- [ ] T3: bounding boxes razonables (no fuera de rango, no colapsadas)
- [ ] T3: al menos 50 frames sin crash
- [ ] T3: dashboard WebSocket muestra métricas correctamente
- [ ] T2→T3→T2: intercambio no introduce corrupción

### 12.5 Comparativa de latencia

- [ ] T2: 100 frames medidos (pre/inf/post/total)
- [ ] T3: 100 frames medidos (pre/inf/post/total)
- [ ] Tabla §11.4 completada con datos reales
- [ ] Resultado documentado en registro de despliegue

---

## 13. Rollback — Restaurar T2 sin Afectar Código Existente

Si T3 tiene problemas en dispositivo, el rollback es trivial:

### 13.1 Opción A (partición compartida)

```bash
# Re-flashear T2 sobre la misma partición
python -m esptool --chip esp32s3 --port $PORT --baud 921600 \
    write_flash 0xB10000 models/espdl/yolo26n_t2_esp.espdl

# Cambiar ACTIVE_MODEL_TYPE de vuelta a YOLO26N_ESP
# Recompilar y flashear firmware
idf.py build flash monitor
```

### 13.2 Opción B (partición dedicada)

Solo cambiar `ACTIVE_MODEL_TYPE` y recompilar — el modelo T2 sigue en su partición original.

### 13.3 Limpieza de código (opcional)

Si T3 se descarta definitivamente, el código añadido es completamente aislado:
- Borrar `YOLO26N_T3_ESP` del enum
- Borrar el case en `make_model_config()` y `run_postprocess()`
- Borrar `postprocess_yolo26_t3_espdl()` de `postprocess.h/cpp`
- Borrar `yolo26n_t3_esp.*` de `models/espdl/`

Ninguno de estos cambios afecta a T2 ni ESPDet.

---

## Resumen de Archivos a Modificar

| Archivo | Cambio | Impacto en existente |
|---|---|:---:|
| `models/espdl/yolo26n_t3_esp.*` | NUEVO — copiar 3 archivos | Ninguno |
| `main/app_config.h` | AÑADIR `YOLO26N_T3_ESP` + thresholds | Ninguno |
| `main/main.cpp` | AÑADIR 2 cases (config + postprocess) | Ninguno |
| `main/postprocess.h` | AÑADIR declaración función | Ninguno |
| `main/postprocess.cpp` | AÑADIR implementación (~50 líneas) | Ninguno |
| `partitions.csv` | SIN CAMBIO (Opción A) o AÑADIR partición (Opción B) | Depende |
| `scripts/flash_models.sh` | OPCIONAL — añadir T3 | Ninguno |

> **Total de código nuevo**: ~60 líneas de C++ + 2 constantes + 1 enum value.  
> **Código existente modificado**: 0 líneas.

---

> **Documento preparado**: 24 de febrero de 2026  
> **Origen del modelo**: `02_ING_MODELOS/Train_MLOps` — YOLO26n T3 (DFL Removal, reg_max=1)  
> **Job ID Vertex AI**: `6267808829191225344`  
> **Referencia**: `docs/Registro_Entrenamiento_YOLO26.md` §5, `docs/Instructivo_Despliegue_ESPDL.md`
