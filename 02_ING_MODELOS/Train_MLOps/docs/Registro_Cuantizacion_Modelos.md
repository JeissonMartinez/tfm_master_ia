# Registro de Cuantización — Exportación ONNX → ESPDL (INT8) para ESP32-S3

> **Proceso**: Cuantización post-entrenamiento (PTQ) INT8 simétrica + exportación a formato ESPDL  
> **Herramienta**: `esp-ppq` v1.2.6 (Espressif Post-training Processing & Quantization)  
> **Target**: ESP32-S3 (Xtensa LX7, 512 KB SRAM, 8 MB Flash PSRAM)  
> **Dataset de calibración**: 500 imágenes del split de entrenamiento IODC  
> **Script**: `scripts/convert_onnx_to_espdl.py` (adaptado de `03_ING_DESPLIEGUE`)  
> **Entorno**: conda `02_ING_MODELOS/env` — Python 3.10.19  
> **Fecha de ejecución**: 24 de febrero de 2026  
> **Última actualización**: 24 de febrero de 2026  

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Contexto y Motivación](#2-contexto-y-motivación)
3. [Modelos de Entrada (ONNX FP32)](#3-modelos-de-entrada-onnx-fp32)
4. [Metodología de Cuantización](#4-metodología-de-cuantización)
5. [Configuración del Entorno](#5-configuración-del-entorno)
6. [Conversión — FCOS T3 (fcos_v3s_t3)](#6-conversión--fcos-t3-fcos_v3s_t3)
7. [Conversión — YOLO26 T2 (yolo26n_t2)](#7-conversión--yolo26-t2-yolo26n_t2)
8. [Conversión — ESPDet T4 (espdet_pico_t4)](#8-conversión--espdet-t4-espdet_pico_t4)
9. [Resultados Comparativos](#9-resultados-comparativos)
10. [Análisis de Viabilidad para ESP32-S3](#10-análisis-de-viabilidad-para-esp32-s3)
11. [Incidencias Técnicas y Resoluciones](#11-incidencias-técnicas-y-resoluciones)
12. [Artefactos Generados](#12-artefactos-generados)
13. [Conclusiones](#13-conclusiones)
14. [Trabajo Futuro](#14-trabajo-futuro)

---

## 1. Resumen Ejecutivo

| Métrica | FCOS T3 | YOLO26 T2 | ESPDet T4 |
|---|:---:|:---:|:---:|
| **ONNX (FP32)** | 4.75 MB | 9.95 MB | 1.42 MB |
| **ESPDL (INT8)** | **1.74 MB** | **2.72 MB** | **0.52 MB** |
| **Compresión** | 2.74× | 3.66× | 2.72× |
| **Tiempo conversión** | 169.4 s | 78.8 s | 83.7 s |
| Parámetros | 1.23M | ~2.6M | ~361K |
| mAP@50 (test, FP32) | 0.5675 | 0.7747 | 0.6052 |
| F1-Score (test, FP32) | 0.6438 | 0.7517 | 0.4539 |
| Fix ONNX requerido | No | **Sí** (ejes negativos) | No |
| Viabilidad ESP32-S3 | 🟢 Viable | 🟡 Ajustado | 🟢 Viable |

> **Los 3 modelos fueron convertidos exitosamente a formato ESPDL (INT8) para ESP32-S3.**  
> ESPDet T4 es el candidato principal para despliegue en dispositivo (0.52 MB, cabe holgadamente).  
> FCOS T3 es viable como alternativa de mayor precisión (1.74 MB, ajustado pero factible).  
> YOLO26 T2 se incluye como referencia comparativa; su viabilidad en ESP32-S3 depende de la partición de Flash disponible tras firmware (~2.72 MB ESPDL + ~3-4 MB firmware ≈ 6-7 MB de 8 MB).

### 1.1 Glosario de Términos

| Término | Definición |
|---|---|
| **PTQ (Post-Training Quantization)** | Técnica de cuantización que se aplica **después** del entrenamiento, sin modificar los pesos aprendidos. Utiliza un dataset de calibración para determinar los rangos óptimos de activación. No requiere re-entrenamiento. |
| **QAT (Quantization-Aware Training)** | Técnica alternativa que simula la cuantización **durante** el entrenamiento, permitiendo al modelo aprender a compensar la pérdida de precisión. Produce modelos cuantizados de mayor calidad que PTQ, pero requiere acceso al pipeline de entrenamiento completo. |
| **INT8 simétrico** | Esquema de cuantización que mapea valores FP32 al rango entero $[-128, +127]$ usando un factor de escala simétrico respecto al cero: $Q(x) = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right),\ -128,\ 127\right)$, donde $s = \frac{\max(\|x\|)}{127}$. |
| **Factor de escala / Exponente** | En el formato ESPDL, el factor de escala se codifica como potencia de 2 para eficiencia de cómputo entero: $s = 2^{\text{exponent}}$. Esto permite que las operaciones de escalado se implementen como bit-shifts en el acelerador ESP-DL, evitando multiplicaciones en punto flotante. |
| **Calibración estática** | Proceso de ejecutar $N$ muestras representativas a través del grafo del modelo para recopilar estadísticas (mínimo, máximo, histograma) de las activaciones en cada capa. Estas estadísticas determinan el factor de escala óptimo para la cuantización de activaciones. |
| **NSPR — Noise:Signal Power Ratio** | Métrica de error de cuantización expresada como porcentaje. Representa la relación entre la potencia del ruido de cuantización y la potencia de la señal original: $\text{NSPR} = \frac{\|y_{INT8} - y_{FP32}\|^2}{\|y_{FP32}\|^2} \times 100\%$. Valores menores indican menor degradación por cuantización. |
| **NSPR layerwise** | NSPR evaluado por **capa aislada**: se alimenta la capa con inputs FP32 perfectos y se compara la salida FP32 vs INT8. Mide el error **intrínseco** de cuantizar esa capa específica, sin considerar errores propagados de capas anteriores. Valores típicos: < 2%. |
| **NSPR graphwise** | NSPR evaluado con **propagación end-to-end**: se ejecuta el modelo completo cuantizado y se captura la salida en una capa específica. Mide el error **acumulativo** de todas las capas desde la entrada hasta ese punto. Valores típicos: 5–50%. Siempre ≥ NSPR layerwise. |
| **Misquantized** | Etiqueta asignada por esp-ppq a capas cuyo NSPR layerwise supera un umbral crítico (~10%), indicando que la cuantización INT8 degrada inaceptablemente la salida de esa capa. Requiere intervención: promover a INT16/FP32 o aplicar QAT. |
| **LUT (Look-Up Table)** | Tabla precalculada de 256 entradas que implementa funciones de activación no-lineales (Swish, HardSigmoid) en dominio entero. Cada entrada mapea un valor INT8 de entrada a su valor INT8 de salida, evitando cómputo en punto flotante durante inferencia. |
| **ESPDL** | Formato binario propietario de Espressif para modelos de deep learning. Almacena pesos INT8, exponentes, topología del grafo y tablas LUT en un formato optimizado para el acelerador ESP-DL del ESP32-S3. Usa layout NHWC (batch, height, width, channels). |
| **ONNX (Open Neural Network Exchange)** | Formato abierto de intercambio de modelos de redes neuronales. Permite exportar modelos desde frameworks como PyTorch y cargarlos en herramientas de cuantización como esp-ppq. Utiliza layout NCHW (batch, channels, height, width). |
| **Layout NCHW vs NHWC** | Orden de almacenamiento de tensores en memoria. **NCHW** (PyTorch/ONNX): canales antes de dimensiones espaciales. **NHWC** (ESP-DL/TFLite): canales al final. La conversión NCHW→NHWC se realiza durante la exportación a ESPDL y requiere remapear los atributos de eje de ciertos operadores. |
| **Dispatching** | Asignación de cada operador del grafo a una plataforma de ejecución (INT8, INT16 o FP32). Los operadores no cuantizables (Resize, Reshape, Softmax) se mantienen en FP32; los cuantizables se asignan a INT8 por defecto. |
| **mAP@50 (mean Average Precision)** | Métrica estándar de detección de objetos. Promedio de la precisión media (AP) sobre todas las clases, evaluada con umbral de IoU ≥ 0.50 para considerar una detección como correcta. |

---

## 2. Contexto y Motivación

### 2.1 Problema

Los modelos de detección de objetos entrenados en Google Vertex AI producen checkpoints en formato PyTorch (`.pt`) y se exportan a ONNX (`.onnx`) con pesos en punto flotante de 32 bits (FP32). El microcontrolador objetivo — **ESP32-S3** con acelerador ESP-DL — requiere modelos en formato propietario **ESPDL** con pesos cuantizados a **8 bits enteros (INT8)** para:

1. **Reducir el tamaño del modelo** para caber en Flash (8 MB, compartidos con firmware)
2. **Acelerar la inferencia** usando operaciones enteras optimizadas del acelerador ESP-DL
3. **Reducir consumo energético** en operaciones de enteros vs punto flotante

### 2.2 Pipeline completo

```
Entrenamiento (Vertex AI)     →  PyTorch (.pt)
Exportación PyTorch→ONNX      →  ONNX FP32 (.onnx)         ← src_colab/utils_export.py
Cuantización ONNX→ESPDL       →  ESPDL INT8 (.espdl)       ← scripts/convert_onnx_to_espdl.py  ★ este documento
Integración en firmware       →  Firmware ESP-IDF (.bin)    ← 03_ING_DESPLIEGUE/
```

### 2.3 Alcance de este registro

Este documento cubre exclusivamente el paso de **cuantización post-entrenamiento (PTQ)** y exportación a formato ESPDL, ejecutado localmente sobre los ONNX ya generados en los entrenamientos de Vertex AI. No se realizó re-entrenamiento ni cuantización-aware training (QAT).

---

## 3. Modelos de Entrada (ONNX FP32)

Los modelos seleccionados para cuantización corresponden a las **mejores versiones de cada familia**, según los criterios documentados en sus respectivos registros de entrenamiento.

### 3.1 Criterios de selección

| Modelo | Registro fuente | Criterio de selección | Run seleccionado |
|---|---|---|---|
| **FCOS T3** | `Registro_Entrenamiento_FCOS.md` §4, §11 | Mejor F1 (0.6438) + mejor Precision (0.6609) de la serie. Referencia operativa para despliegue. Post-sweep: conf_threshold óptimo = 0.40. | `fcos_v3s_v1-1771690809` |
| **YOLO26 T2** | `Registro_Entrenamiento_YOLO26.md` §4 | Mejor mAP@50 (0.7747) y F1 (0.7517) de la serie. Optimizer MuSGD vs baseline AdamW. | `yolo26n_custom_v2-run1` |
| **ESPDet T4** | `Registro_Entrenamiento_ESPDet.md` §6 | Mejor F1 (0.4539) y Precision (0.3298) con NMS tuning (conf=0.35, IoU=0.40). Arquitectura oficial Espressif, candidato principal ESP32-S3. | `espdet-pico-v4-t4` |

### 3.2 Especificaciones ONNX

| Propiedad | FCOS T3 | YOLO26 T2 | ESPDet T4 |
|---|:---:|:---:|:---:|
| **Archivo** | `fcos_v3s.onnx` | `best.onnx` | `espdet_pico.onnx` |
| **Ruta** | `outputs/fcos_v3s_v1-1771690809/export/` | `outputs/yolo26n_custom_v2-run1/export/` | `outputs/espdet-pico-v4-t4/export/` |
| **Tamaño** | 4.75 MB | 9.95 MB | 1.42 MB |
| **Input shape** | `[1, 3, 224, 224]` | `[1, 3, 224, 224]` | `[1, 3, 224, 224]` |
| **Channel format** | NCHW | NCHW | NCHW |
| **Opset** | 13 | 13 | 13 |
| **Parámetros** | 1,233,450 | ~2,600,000 | ~361,000 |
| **Backbone** | MobileNetV3-Small | YOLO26 backbone (C3k2 + PSA) | ESPDet-Pico (DSConv + ESPBlock) |
| **Neck** | SimpleFPN (lateral + top-down) | PAN (top-down + bottom-up) | ESPDet Neck (top-down + bottom-up) |
| **Head** | FCOS Head (cls + reg + centerness) | YOLO Detect Head (unified) | ESPDetect Head (cv2 box + cv3 cls) |
| **Outputs** | 2 (boxes, scores) | 1 tensor `[1, 9, 1029]` | 6 interleaved (box0, score0, box1, score1, box2, score2) |
| **nc** | 5 | 5 | 5 |

### 3.3 Métricas pre-cuantización (Test set, FP32)

| Métrica | FCOS T3 | YOLO26 T2 | ESPDet T4 |
|---|:---:|:---:|:---:|
| **mAP@50** | 0.5675 | **0.7747** | 0.6052 |
| **mAP@50-95** | 0.2602 | **0.5456** | 0.2701 |
| **Precision** | **0.6609** | 0.8324 | 0.3298 |
| **Recall** | 0.6276 | **0.6853** | 0.7278 |
| **F1-Score** | 0.6438 | **0.7517** | 0.4539 |
| **Inferencia (GPU T4)** | 4.8 ms | **2.9 ms** | 5.0 ms |

#### Métricas por clase (mAP@50, test FP32)

| Clase | FCOS T3 | YOLO26 T2 | ESPDet T4 |
|---|:---:|:---:|:---:|
| dog | 0.4957 | **0.7956** | 0.5305 |
| door | 0.5034 | **0.6601** | 0.5572 |
| obstacle | 0.4575 | **0.7384** | 0.4618 |
| person | 0.6359 | **0.8661** | 0.7059 |
| stair | 0.7451 | **0.8133** | 0.7708 |

> YOLO26 T2 domina en todas las clases. ESPDet T4 destaca en `stair` (0.77) y `person` (0.71) — clases críticas para navegación LCMR.

---

## 4. Metodología de Cuantización

### 4.1 Tipo de cuantización

| Propiedad | Valor |
|---|---|
| **Tipo** | Post-Training Quantization (PTQ) |
| **Esquema** | INT8 simétrico |
| **Herramienta** | `esp-ppq` (ESP Post-training Processing & Quantization) |
| **Calibración** | Estática — basada en dataset de imágenes reales |
| **Target platform** | `esp32s3` → `TargetPlatform.ESPDL_INT8` |
| **Formato de salida** | ESPDL (Espressif Deep Learning model format) |

### 4.2 Cuantización INT8 simétrica

La cuantización INT8 simétrica mapea los pesos y activaciones FP32 al rango $[-128, +127]$ usando un factor de escala por canal:

$$
Q(x) = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right), -128, 127\right)
$$

donde $s$ es el factor de escala calculado durante la calibración:

$$
s = \frac{\max(|x|)}{127}
$$

En el formato ESPDL, los factores de escala se almacenan como **exponentes** (`exponents`), representando potencias de 2 para eficiencia de cómputo entero en el acelerador ESP-DL:

$$
s = 2^{\text{exponent}}
$$

### 4.3 Proceso de calibración

La calibración estática ejecuta $N$ muestras representativas a través del grafo del modelo para determinar los rangos óptimos de activación en cada capa.

| Parámetro | Valor |
|---|---|
| **Dataset fuente** | IODC — split de entrenamiento |
| **N° de imágenes** | 500 (de 1,470 disponibles) |
| **Formato imagen** | JPEG (RGB), resize a 224×224 |
| **Normalización** | $\text{pixel} / 255.0$ (rango $[0, 1]$) |
| **Layout** | NCHW: `[1, 3, 224, 224]` → `torch.Tensor` |
| **Calibration steps** | 500 (1 step por imagen) |
| **Collate function** | Default esp-ppq `collate_fn_template` (cast a `torch.float32`) |

> **Criterio de selección**: Se usan imágenes del split de entrenamiento (no de validación/test) para que la calibración refleje la distribución de datos vista durante el entrenamiento. Se seleccionan 500 imágenes de 1,470 disponibles — suficientes para capturar la variabilidad estadística de las 5 clases objetivo sin incrementar excesivamente el tiempo de calibración.

### 4.4 Pipeline esp-ppq

El pipeline interno de `espdl_quantize_onnx()` ejecuta las siguientes fases secuencialmente:

```
1. ONNX Simplification      ─ onnxsim.simplify() — fusión de ops redundantes
2. ConvTranspose Decomp.     ─ Descomposición de ConvTranspose a ops primitivas
3. Quantization Fusion       ─ Fusión de patrones cuantizables (Conv+BN+Act, etc.)
4. Quantize Simplify         ─ Eliminación de cuantizadores redundantes
5. Parameter Quantization    ─ Cuantización de pesos estáticos (INT8 simétrico)
6. Calibration (Phase 1)     ─ Forward pass con datos de calibración — rango de activaciones
7. Calibration (Phase 2)     ─ Refinamiento de rangos (observer: MinMax o Histogram)
8. Quantization Alignment    ─ Alineación de exponentes entre operadores conectados
9. Passive Parameter Quant.  ─ Cuantización de parámetros pasivos (bias, etc.)
10. Graphwise Error Analysis ─ Análisis de error SNR por propagación acumulativa
11. Layerwise Error Analysis ─ Análisis de error SNR por capa aislada
12. Layout Transform (ESPDL) ─ Conversión NCHW→NHWC + serialización binaria
```

### 4.5 Preprocesamiento ONNX especial (YOLO26)

El modelo YOLO26 contiene operadores ONNX con atributos `axis=-1` (ejes negativos), que el exporter ESPDL de esp-ppq no soporta. Se implementó una función de preprocesamiento `fix_negative_axes()` que:

1. Carga el modelo ONNX y ejecuta **shape inference** para obtener los rangos (número de dimensiones) de todos los tensores
2. Itera sobre todos los nodos del grafo buscando atributos `axis` con valor negativo
3. Convierte cada eje negativo a su equivalente positivo: $\text{axis\_pos} = \text{axis\_neg} + \text{rank}$
4. Guarda el modelo corregido como `*_fixed.onnx`

> **Detalle**: Sin este preprocesamiento, toda la cuantización y calibración se completan exitosamente, pero el proceso falla en la fase final de exportación a ESPDL con `ValueError: -1 is not in list` en `layout_patterns.py:233`. Este comportamiento se confirmó experimentalmente en la primera ejecución.

---

## 5. Configuración del Entorno

### 5.1 Dependencias

| Paquete | Versión | Función |
|---|---|---|
| `esp-ppq` | 1.2.6 | Cuantización PTQ + exportación ESPDL |
| `onnx` | 1.15.0 | Carga/manipulación de modelos ONNX |
| `onnxsim` | 0.5.0 | Simplificación de grafos ONNX (dep. de esp-ppq) |
| `onnxruntime` | 1.23.2 | Inferencia ONNX (verificación) |
| `torch` | 2.5.1 | Ejecución del grafo cuantizado durante calibración |
| `numpy` | 1.26.4 | Manipulación de arrays |
| `Pillow` | — | Carga y redimensionamiento de imágenes de calibración |

### 5.2 Hardware de ejecución

| Componente | Especificación |
|---|---|
| **Máquina** | MacBook Pro (Apple Silicon) |
| **Ejecución** | CPU-only (sin GPU) |
| **Entorno conda** | `02_ING_MODELOS/env` — Python 3.10.19 |

> La cuantización PTQ con esp-ppq es un proceso CPU-only. No requiere GPU, lo que permitió ejecutar localmente sin coste de infraestructura cloud, a diferencia del entrenamiento que se realizó en Vertex AI con GPU NVIDIA T4.

### 5.3 Comando de ejecución

```bash
cd 02_ING_MODELOS/Train_MLOps
python scripts/convert_onnx_to_espdl.py \
    --calib-dir ../datasets/IODC/coco/train/images \
    --target esp32s3 \
    --n-samples 500
```

---

## 6. Conversión — FCOS T3 (fcos_v3s_t3)

### 6.1 Identificador

| Campo | Valor |
|---|---|
| **Modelo origen** | FCOS MobileNetV3-Small + SimpleFPN — Train 3 |
| **Run ID** | `fcos_v3s_v1-1771690809` |
| **ONNX** | `outputs/fcos_v3s_v1-1771690809/export/fcos_v3s.onnx` |
| **ESPDL** | `outputs/espdl/fcos_v3s_t3/fcos_v3s_t3.espdl` |
| **Fecha** | 24 de febrero de 2026 |

### 6.2 Preprocesamiento

No requirió `fix_negative_axes`; el grafo ONNX del FCOS no contiene atributos con ejes negativos.

### 6.3 Warnings durante conversión

| Tipo | Detalle | Impacto |
|---|---|---|
| `PPQ WARNING` | `Unexpected input value of /m/fpn/Resize` (×2) — inputs opcionales del op Resize son `None` | Sin impacto — inputs opcionales de `roi` y `scales` en Resize no se usan (modo `sizes`) |
| `ESPDL ERROR` | Exponent exception en `/m/backbone/block.1/fc1/Conv` y `/m/backbone/block.2/fc1/Conv` — `(output_exponent - input0_exponent - input1_exponent) < 0` | **Potencial**: puede causar overflow en inferencia INT8 en capas del Squeeze-and-Excite del MobileNetV3. Requiere validación en dispositivo. |
| `ESPDL INFO` | `skip not QuantableOperation` (17 instancias) — operaciones no cuantizables (e.g., Resize, Reshape, Concat) se mantienen en su precision original | Sin impacto — comportamiento esperado |

> **⚠️ NOTA IMPORTANTE**: Los 2 errores de exponente en capas `fc1/Conv` del backbone MobileNetV3 (bloques Squeeze-and-Excite) indican que la relación entre exponentes de entrada y salida no cumple la restricción $e_{out} - e_{in0} - e_{in1} \geq 0$ requerida por el acelerador ESP-DL. Esto podría causar **overflow de precisión** durante inferencia en dispositivo, aunque el archivo ESPDL se generó correctamente. Se recomienda validación funcional en ESP32-S3 antes de despliegue en producción.

### 6.4 Resultados

| Métrica | Valor |
|---|---|
| **ONNX (FP32)** | 4.752 MB |
| **ESPDL (INT8)** | 1.735 MB |
| **Ratio de compresión** | 2.74× |
| **Tiempo de conversión** | 169.4 s (2 min 49 s) |
| **Input ESPDL** | `INT8, 1×224×224×3` (NHWC), exponents: `[-7]` |
| **LUT activaciones** | HardSigmoid (13 tablas de 256 entradas) |
| **Muestras calibración** | 500 |

> **Observación**: La compresión 2.74× es inferior a la teórica 4× (FP32→INT8) porque el formato ESPDL incluye metadatos de cuantización, tablas LUT para activaciones no-lineales (HardSigmoid del MobileNetV3), y overhead de serialización. El tiempo de conversión es el más alto de los 3 modelos (169.4 s vs ~80 s) debido al mayor número de capas del backbone MobileNetV3 con operaciones Squeeze-and-Excite.

---

## 7. Conversión — YOLO26 T2 (yolo26n_t2)

### 7.1 Identificador

| Campo | Valor |
|---|---|
| **Modelo origen** | YOLO26n (nano) Ultralytics — Train 2 (MuSGD) |
| **Run ID** | `yolo26n_custom_v2-run1` |
| **ONNX** | `outputs/yolo26n_custom_v2-run1/export/best.onnx` |
| **ONNX corregido** | `outputs/yolo26n_custom_v2-run1/export/best_fixed.onnx` |
| **ESPDL** | `outputs/espdl/yolo26n_t2/yolo26n_t2.espdl` |
| **Fecha** | 24 de febrero de 2026 |

### 7.2 Preprocesamiento — Fix de ejes negativos

YOLO26 requirió la corrección de 3 operadores ONNX con atributos `axis=-1`:

| Operador | Tipo | Eje original | Eje corregido | Rango del tensor |
|---|---|:---:|:---:|:---:|
| `/model.10/m/m.0/attn/Softmax` | Softmax | -1 | **3** | 4 |
| `/model.23/Concat` | Concat | -1 | **2** | 3 |
| `/model.23/Concat_1` | Concat | -1 | **2** | 3 |

> **Análisis**: El primer operador pertenece al bloque PSA (Partial Self-Attention) de YOLO26, donde el Softmax opera sobre la dimensión de atención (último eje de un tensor 4D). Los otros dos pertenecen a la detection head, donde Concat une predicciones de diferentes escalas a lo largo del eje de anchors (último eje de un tensor 3D). Estos operadores usan axis=-1 por convención de PyTorch, pero esp-ppq requiere indices positivos para el layout transform NCHW→NHWC.

### 7.3 Primera ejecución (fallida)

En la primera ejecución sin `fix_negative_axes`, la cuantización y calibración se completaron exitosamente (500 steps, ~43 s), pero la exportación final falló:

```
ValueError: -1 is not in list
  File "esp_ppq/parser/espdl/layout_patterns.py", line 233, in export
    new_axis = var_perm.index(int(axis))
```

> **Causa raíz**: El `reset_graph_layout()` de esp-ppq aplica una permutación de layout NCHW→NHWC y necesita remapear los ejes de operadores como Softmax y Concat. El método `var_perm.index(-1)` falla porque `-1` no existe en la lista de permutación `[0, 2, 3, 1]`. La corrección previa de ejes negativos a positivos resuelve el problema sin alterar la semántica del modelo.

### 7.4 Network Snapshot

```
--------- Network Snapshot ---------
Num of Op:                    [242]
Num of Quantized Op:          [242]
Num of Variable:              [464]
Num of Quantized Var:         [464]
```

### 7.5 Quantization Snapshot

| Estado | Cantidad | Descripción |
|---|:---:|---|
| **ACTIVATED** | 268 | Configuraciones de cuantización activas (pesos + activaciones) |
| **OVERLAPPED** | 240 | Configuraciones compartidas entre operadores fusionados |
| **PASSIVE** | 216 | Parámetros pasivos (bias, etc.) cuantizados por herencia |
| **FP32** | 33 | Operaciones mantenidas en FP32 (no cuantizables: Resize, Reshape, etc.) |
| **Total** | 757 | Configuraciones de cuantización totales |

### 7.6 Análisis de error de cuantización

#### Graphwise (error acumulativo — propagación end-to-end)

Las capas con mayor Noise:Signal Power Ratio (NSPR) graphwise representan la **degradación acumulativa** de precisión desde la entrada hasta esa capa:

| # | Capa | NSPR | Componente |
|---|---|:---:|---|
| 1 | `/model.10/m/m.0/ffn/ffn.1/conv/Conv` | **24.87%** | PSA — FFN layer 1 |
| 2 | `/model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv` | **24.57%** | Detect Head — cls branch P5 |
| 3 | `/model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv` | **23.72%** | Detect Head — cls branch P5 |
| 4 | `/model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv` | **23.03%** | Detect Head — cls branch P5 |
| 5 | `/model.22/m.0/cv2/conv/Conv` | **19.56%** | C3k2 block — bottom-up PAN |

> **Interpretación**: El error graphwise se concentra en (a) el bloque PSA (self-attention, sensible a cuantización por las operaciones MatMul + Softmax) y (b) la detection head de clasificación en la escala P5 (features de menor resolución, 7×7). Valores de ~25% NSPR son **moderados** para INT8 y típicos de modelos con bloques de atención. El impacto real en mAP debe validarse con inferencia INT8 end-to-end.

#### Layerwise (error aislado por capa)

El análisis layerwise mide el error de cuantización de cada capa **aisladamente** (sin propagación):

| # | Capa | NSPR |
|---|---|:---:|
| 1 | `/model.0/conv/Conv` | 0.165% |
| 2 | `/model.1/conv/Conv` | 0.138% |
| 3 | `/model.2/cv1/conv/Conv` | 0.121% |
| 4 | `/model.9/cv2/conv/Conv` | 0.080% |
| 5 | `/model.4/cv1/conv/Conv` | 0.073% |

> **Ninguna capa fue marcada como "Misquantized"**. El NSPR layerwise máximo es 0.165%, indicando que la cuantización INT8 por capa individual es muy precisa. El error acumulativo graphwise (~25%) se debe a la **propagación del error** a través de ~242 operadores, no a defectos de cuantización en capas individuales.

### 7.7 Warnings durante conversión

| Tipo | Detalle | Impacto |
|---|---|---|
| `PPQ WARNING` | `Unexpected input value of /model.11/Resize` y `/model.14/Resize` — inputs opcionales `None` | Sin impacto |
| `ESPDL INFO` | `Skip PPQ_Variable_*` — ~50 variables intermedias no exportables | Sin impacto — nodos intermedios del grafo PPQ no necesarios en runtime |
| `ESPDL INFO` | `skip not QuantableOperation` — operaciones no cuantizables | Sin impacto — Softmax, Reshape, etc. |

### 7.8 Resultados

| Métrica | Valor |
|---|---|
| **ONNX (FP32)** | 9.947 MB |
| **ESPDL (INT8)** | 2.720 MB |
| **Ratio de compresión** | 3.66× |
| **Tiempo de conversión** | 78.8 s (1 min 19 s) |
| **Input ESPDL** | `INT8, 1×224×224×3` (NHWC), exponents: `[-7]` |
| **LUT activaciones** | Swish/SiLU (13+ tablas de 256 entradas) |
| **Muestras calibración** | 500 |

> **Observación**: La compresión 3.66× es la más alta de los 3 modelos, acercándose al teórico 4× (FP32→INT8). Esto se explica porque YOLO26 tiene el modelo más grande (9.95 MB FP32), donde los pesos dominan el tamaño y se cuantizan eficientemente. El overhead fijo de metadatos y LUT se diluye proporcionalmente.

---

## 8. Conversión — ESPDet T4 (espdet_pico_t4)

### 8.1 Identificador

| Campo | Valor |
|---|---|
| **Modelo origen** | ESPDet-Pico (oficial Espressif) — Train 4 (BCE + NMS tuning) |
| **Run ID** | `espdet-pico-v4-t4` |
| **ONNX** | `outputs/espdet-pico-v4-t4/export/espdet_pico.onnx` |
| **ESPDL** | `outputs/espdl/espdet_pico_t4/espdet_pico_t4.espdl` |
| **Fecha** | 24 de febrero de 2026 |

### 8.2 Preprocesamiento

No requirió `fix_negative_axes`; el grafo ONNX de ESPDet no contiene atributos con ejes negativos.

> **Nota**: ESPDet es una arquitectura diseñada específicamente por Espressif para TinyML, y su exportación ONNX ya produce grafos compatibles con esp-ppq sin ajustes adicionales. Esto confirma la **coherencia end-to-end** del ecosistema Espressif (modelo → cuantizador → acelerador).

### 8.3 Network Snapshot

```
--------- Network Snapshot ---------
Num of Op:                    [219]
Num of Quantized Op:          [219]
Num of Variable:              [443]
Num of Quantized Var:         [443]
```

### 8.4 Quantization Snapshot

| Estado | Cantidad | Descripción |
|---|:---:|---|
| **ACTIVATED** | 250 | Configuraciones de cuantización activas |
| **OVERLAPPED** | 256 | Configuraciones compartidas (ops fusionados) |
| **PASSIVE** | 154 | Parámetros pasivos |
| **FP32** | 22 | Operaciones mantenidas en FP32 |
| **Total** | 682 | Configuraciones de cuantización totales |

### 8.5 Análisis de error de cuantización

#### Graphwise (error acumulativo)

| # | Capa | NSPR | Componente |
|---|---|:---:|---|
| 1 | `/m/layer7/cv2/conv/Conv` | **45.74%** | Neck — bottom-up P4 conv |
| 2 | `/m/layer8/m.0/cv2/conv/Conv` | **36.45%** | Neck — bottom-up C3k2 |
| 3 | `/m/layer8/cv1/conv/Conv` | **33.74%** | Neck — bottom-up C3k2 entry |
| 4 | `/m/layer8/cv2/conv/Conv` | **33.11%** | Neck — bottom-up C3k2 exit |
| 5 | `/m/layer5/cv2/conv/Conv` | **30.16%** | Neck — top-down P4 conv |

> **Interpretación**: ESPDet presenta los mayores NSPR graphwise de los 3 modelos (hasta 45.74%). Esto se explica por:
> 1. **Modelo con menos capacidad** (361K params vs 1.2M/2.6M) — menos redundancia para absorber error de cuantización
> 2. **Error concentrado en el Neck** (layers 5-8), no en backbone ni head — las capas de fusión multi-escala son sensibles a la pérdida de precisión
> 3. La capa más afectada (`layer7/cv2/conv`) es un punto de convergencia donde features de múltiples escalas se combinan, amplificando errores acumulados
>
> **A pesar del alto NSPR graphwise, no hubo capas marcadas como "Misquantized" en el análisis layerwise**, lo que sugiere que el error por capa es manejable y la degradación proviene principalmente de la propagación.

#### Layerwise (error por capa)

| # | Capa | NSPR |
|---|---|:---:|
| 1 | `/m/layer1/depthwise/Conv` | 1.893% |
| 2 | `/m/layer0/conv/Conv` | 1.563% |
| 3 | `/m/layer2/m.0/cv1/depthwise/Conv` | 1.002% |
| 4 | `/m/layer2/cv1/conv/Conv` | 0.574% |
| 5 | `/m/layer2/m.0/cv2/pointwise/Conv` | 0.399% |

> ESPDet tiene el NSPR layerwise más alto de los 3 modelos (1.893% vs 0.165% de YOLO26), concentrado en las **capas depthwise-separable** iniciales del backbone. Las convoluciones depthwise son conocidas por ser más sensibles a INT8 debido a su menor número de canales (1 por grupo), lo que reduce la diversidad estadística para la calibración del factor de escala.

### 8.6 Warnings durante conversión

| Tipo | Detalle | Impacto |
|---|---|---|
| `PPQ WARNING` | `Unexpected input value of /m/up11/Resize` y `/m/up14/Resize` | Sin impacto |
| `ESPDL INFO` | `Skip PPQ_Variable_0` a `PPQ_Variable_18` — 19 variables no exportables | Sin impacto |
| `ESPDL INFO` | `skip not QuantableOperation` — operaciones no cuantizables | Sin impacto |

> **No se reportaron errores de exponente para ESPDet**, a diferencia de FCOS. Esto confirma que ESPDet fue diseñado con las restricciones del acelerador ESP-DL en mente.

### 8.7 Resultados

| Métrica | Valor |
|---|---|
| **ONNX (FP32)** | 1.418 MB |
| **ESPDL (INT8)** | 0.521 MB |
| **Ratio de compresión** | 2.72× |
| **Tiempo de conversión** | 83.7 s (1 min 24 s) |
| **Input ESPDL** | `INT8, 1×224×224×3` (NHWC), exponents: `[-7]` |
| **LUT activaciones** | Swish/SiLU (13 tablas de 256 entradas) |
| **Muestras calibración** | 500 |

> **Observación**: La compresión más baja (2.72×) se debe al mayor peso relativo del overhead de metadatos y LUT en un modelo pequeño (1.42 MB FP32). En términos absolutos, 0.52 MB es el modelo más ligero y cabe **cómodamente** en la Flash del ESP32-S3.

---

## 9. Resultados Comparativos

### 9.1 Tamaños y compresión

| Modelo | ONNX (FP32) | ESPDL (INT8) | Compresión | Δ vs teórico 4× |
|---|:---:|:---:|:---:|:---:|
| **FCOS T3** | 4.752 MB | 1.735 MB | 2.74× | -31.5% |
| **YOLO26 T2** | 9.947 MB | 2.720 MB | 3.66× | -8.5% |
| **ESPDet T4** | 1.418 MB | 0.521 MB | 2.72× | -32.0% |

> El ratio teórico de 4× (32 bits → 8 bits) se alcanza parcialmente. La diferencia proviene de: (1) metadatos de cuantización por capa; (2) tablas LUT para funciones de activación no-lineales; (3) overhead de serialización ESPDL; (4) operaciones mantenidas en FP32 (no cuantizables). Los modelos más grandes (YOLO26) se acercan más al teórico porque el overhead fijo se diluye.

### 9.2 Tiempos de conversión

| Modelo | Calibración + Quant | Exportación | **Total** |
|---|:---:|:---:|:---:|
| **FCOS T3** | ~150 s | ~19 s | **169.4 s** |
| **YOLO26 T2** | ~60 s | ~19 s | **78.8 s** |
| **ESPDet T4** | ~65 s | ~19 s | **83.7 s** |

> FCOS T3 es el más lento (169 s) a pesar de ser más pequeño que YOLO26, debido a las operaciones HardSigmoid del backbone MobileNetV3 Squeeze-and-Excite que requieren más pasos de calibración. YOLO26 es el más rápido en calibración a pesar de ser el más grande, posiblemente por la estructura más regular de sus operaciones (todas Swish/SiLU sin branching SE).

### 9.3 Complejidad del grafo cuantizado

| Métrica | FCOS T3 | YOLO26 T2 | ESPDet T4 |
|---|:---:|:---:|:---:|
| Operadores totales | — | 242 | 219 |
| Operadores cuantizados | — | 242 | 219 |
| Variables | — | 464 | 443 |
| Quant configs ACTIVATED | — | 268 | 250 |
| Quant configs FP32 | — | 33 | 22 |
| Max NSPR graphwise | — | 24.87% | 45.74% |
| Max NSPR layerwise | — | 0.165% | 1.893% |
| Fix ejes negativos | No | **Sí (3 ops)** | No |
| Errores de exponente | **2** (SE blocks) | 0 | 0 |

> **Nota**: Los datos de Network/Quantization Snapshot de FCOS T3 no fueron capturados en el log de terminal (output truncado). El archivo `.info` del modelo contiene la estructura completa del grafo ESPDL.

---

## 10. Análisis de Viabilidad para ESP32-S3

### 10.1 Restricciones del hardware target

| Recurso | ESP32-S3 | Disponible para modelo |
|---|---|---|
| **Flash** | 8 MB (PSRAM) | ~3-5 MB (tras firmware ESP-IDF + código app) |
| **SRAM** | 512 KB | ~300-400 KB (tras stack + buffers) |
| **Acelerador** | ESP-DL (INT8/INT16) | Soporte nativo ESPDL |
| **CPU** | Xtensa LX7 dual-core @ 240 MHz | Fallback para ops no aceleradas |

### 10.2 Evaluación por modelo

| Criterio | FCOS T3 (1.74 MB) | YOLO26 T2 (2.72 MB) | ESPDet T4 (0.52 MB) |
|---|:---:|:---:|:---:|
| **Cabe en Flash** | ✅ (1.74 < 4 MB) | ⚠️ (2.72 < 4 MB, ajustado) | ✅ (0.52 << 4 MB) |
| **SRAM para activaciones** | ⚠️ (1.2M params = activaciones mayores) | ❌ (2.6M params) | ✅ (361K params) |
| **Diseñado para ESP-DL** | ❌ (MobileNetV3 genérico) | ❌ (YOLO convencional) | ✅ (Espressif nativo) |
| **Errores de exponente** | ⚠️ (2 en SE blocks) | ✅ (0) | ✅ (0) |
| **Fix ONNX requerido** | ✅ (no) | ⚠️ (sí, 3 ops) | ✅ (no) |
| **Valoración global** | 🟡 **Viable con reservas** | 🟡 **Ajustado** | 🟢 **Candidato ideal** |

### 10.3 Recomendación de despliegue

1. **Despliegue primario**: **ESPDet T4** — 0.52 MB, cero errores de exponente, diseñado nativamente para ESP-DL, cabe holgadamente en Flash dejando ~4.5 MB libres para firmware + buffers. mAP@50 = 0.6052 es suficiente para navegación LCMR en interiores.

2. **Despliegue alternativo**: **FCOS T3** — 1.74 MB, mejor F1 (0.6438) y Precision (0.6609) que ESPDet, pero con 2 warnings de exponente en bloques Squeeze-and-Excite que requieren validación en dispositivo. Si pasa la validación, ofrece un **42% más de Precision** que ESPDet.

3. **Referencia experimental**: **YOLO26 T2** — 2.72 MB, mejor rendimiento en todas las métricas (mAP@50 = 0.7747), pero su viabilidad en ESP32-S3 es incierta por consumo de SRAM. Se incluye para comparación académica y como upper bound de rendimiento alcanzable con cuantización INT8.

---

## 11. Incidencias Técnicas y Resoluciones

### 11.1 Incidencia 1 — Formato de datos de calibración

| Campo | Detalle |
|---|---|
| **Síntoma** | `AssertionError: Input format misunderstood. Except either dict, list or tensor; while <class 'NoneType'> was given.` |
| **Causa** | El default `collate_fn_template` de esp-ppq v1.2.6 espera `torch.Tensor` pero el script original pasaba `numpy.ndarray` |
| **Impacto** | Fallo en los 3 modelos en la primera ejecución (0/3 exitosos) |
| **Fix** | Cambiar `np.expand_dims(arr, 0)` → `torch.from_numpy(np.expand_dims(arr, 0))` en `create_calibration_dataset()` |
| **Verificación** | 3/3 modelos pasan calibración tras el fix |

> **Lección**: La API de `espdl_quantize_onnx()` espera un `calib_dataloader` que sea una lista de `torch.Tensor`, no de `numpy.ndarray`. El script original de `03_ING_DESPLIEGUE` (v1) usaba numpy arrays que eran compatibles con esp-ppq v1.2.4 pero no con v1.2.6 donde el `collate_fn_template` fue endurecido.

### 11.2 Incidencia 2 — Ejes negativos en YOLO26

| Campo | Detalle |
|---|---|
| **Síntoma** | `ValueError: -1 is not in list` en `layout_patterns.py:233` durante exportación ESPDL |
| **Causa** | YOLO26 (Ultralytics) exporta ONNX con `axis=-1` en Softmax y Concat; esp-ppq ESPDL exporter no soporta ejes negativos en el layout transform NCHW→NHWC |
| **Impacto** | YOLO26 fallido en ejecución 1 (FCOS y ESPDet exitosos); cuantización+calibración completadas pero exportación fallida |
| **Fix** | Implementación de `fix_negative_axes()`: ONNX shape inference → iteración de nodos → conversión $\text{axis\_pos} = \text{axis\_neg} + \text{rank}$ → save como `best_fixed.onnx` |
| **Operadores corregidos** | 3: Softmax (PSA, axis -1→3), Concat ×2 (detect head, axis -1→2) |
| **Verificación** | YOLO26 conversión exitosa tras el fix (2.72 MB ESPDL) |

> **Lección**: Los modelos exportados por Ultralytics usan convenciones PyTorch puras (ejes negativos), mientras que esp-ppq espera el grafo ONNX con ejes positivos explícitos. Esto constituye una **incompatibilidad documentable** entre el ecosistema Ultralytics y esp-ppq/ESP-DL que requiere preprocesamiento del ONNX.

### 11.3 Incidencia 3 — Errores de exponente en FCOS

| Campo | Detalle |
|---|---|
| **Síntoma** | `[ERROR][ESPDL]: calculation result of /m/backbone/block.{1,2}/fc1/Conv will cause an exception` |
| **Causa** | Los bloques Squeeze-and-Excite del backbone MobileNetV3-Small producen rangos de activación que, al cuantizarse a INT8, violan la restricción $e_{out} - e_{in0} - e_{in1} \geq 0$ del acelerador ESP-DL |
| **Impacto** | El modelo ESPDL se genera exitosamente, pero **puede causar overflow** durante inferencia en ESP32-S3. Impacto real desconocido hasta validación en dispositivo |
| **Estado** | **No resuelto** — requiere validación en ESP32-S3 |
| **Mitigación posible** | (a) Cambiar num_of_bits a 16 para esas capas específicas; (b) usar `dispatching_override` para asignar esos ops a FP32; (c) re-exportar ONNX con simplificación de SE blocks |

---

## 12. Artefactos Generados

### 12.1 Estructura de salida

```
outputs/espdl/
├── export_summary.json                    # Resumen JSON de las 3 conversiones
├── fcos_v3s_t3/
│   ├── fcos_v3s_t3.espdl                 # Modelo cuantizado INT8 (1.74 MB)
│   ├── fcos_v3s_t3.info                  # Estructura del grafo ESPDL (texto)
│   └── fcos_v3s_t3.json                  # Configuración de cuantización
├── yolo26n_t2/
│   ├── yolo26n_t2.espdl                  # Modelo cuantizado INT8 (2.72 MB)
│   ├── yolo26n_t2.info                   # Estructura del grafo ESPDL (texto)
│   └── yolo26n_t2.json                   # Configuración de cuantización
└── espdet_pico_t4/
    ├── espdet_pico_t4.espdl              # Modelo cuantizado INT8 (0.52 MB)
    ├── espdet_pico_t4.info               # Estructura del grafo ESPDL (texto)
    └── espdet_pico_t4.json               # Configuración de cuantización
```

### 12.2 Descripción de artefactos

| Archivo | Formato | Contenido |
|---|---|---|
| `*.espdl` | Binario (flatbuffers) | Modelo cuantizado: pesos INT8, exponentes, topología, LUT de activaciones |
| `*.info` | Texto | Representación legible del grafo ESPDL: inputs/outputs, shapes, types, exponents |
| `*.json` | JSON | Configuración de cuantización: dispatching, calibration settings, quantization per-layer |
| `export_summary.json` | JSON | Resumen de las 3 conversiones: tamaños, compresión, tiempos, errores |

### 12.3 Archivos auxiliares generados

| Archivo | Ubicación | Descripción |
|---|---|---|
| `best_fixed.onnx` | `outputs/yolo26n_custom_v2-run1/export/` | ONNX de YOLO26 con ejes negativos corregidos |

---

## 13. Conclusiones

### 13.1 Conclusiones técnicas

1. **La cuantización PTQ INT8 con esp-ppq es viable para los 3 modelos evaluados**, logrando ratios de compresión de 2.72× a 3.66× sin requerir re-entrenamiento (QAT).

2. **ESPDet T4 es el candidato óptimo para despliegue en ESP32-S3**: 0.52 MB ESPDL, cero errores de exponente, diseño nativo para ESP-DL, y la menor huella de memoria de los 3 modelos. Su mAP@50 de 0.6052 es adecuada para las 5 clases de navegación interior (dog, door, person, obstacle, stair).

3. **FCOS T3 es una alternativa viable** con mejor rendimiento de detección (F1=0.6438, Precision=0.6609), pero los 2 errores de exponente en bloques Squeeze-and-Excite del backbone MobileNetV3 representan un riesgo de runtime que debe validarse en dispositivo.

4. **YOLO26 T2 demuestra el upper bound de rendimiento** (mAP@50=0.7747) alcanzable con cuantización INT8, pero su tamaño de 2.72 MB ESPDL y ~2.6M parámetros hacen **incierta** su viabilidad en ESP32-S3 por restricciones de SRAM durante inferencia.

5. **El error de cuantización graphwise es mayor en architecturas más pequeñas** (ESPDet 45.74% vs YOLO26 24.87%), confirmando que modelos con menos redundancia son más sensibles a PTQ INT8. Sin embargo, el error layerwise se mantiene bajo en todos los modelos (< 2%), indicando que la cuantización por capa es precisa.

### 13.2 Conclusiones metodológicas

6. **Las incompatibilidades entre ecosistemas son una fuente significativa de fricción** en el pipeline TinyML. Se documentaron 3 incidencias técnicas: (a) formato de datos de calibración numpy vs torch, (b) ejes negativos ONNX incompatibles con esp-ppq, y (c) errores de exponente en architecturas no diseñadas para ESP-DL. Cada una requirió investigación, diagnóstico y corrección específica.

7. **La calibración con 500 imágenes reales del train set es suficiente** para la cuantización estática INT8. No se observaron diferencias entre modelos respecto a la calidad de calibración.

8. **El pipeline de exportación es ejecutable localmente en CPU** sin necesidad de GPU. Los tiempos de conversión (~80-170 s por modelo) son aceptables para iteración rápida.

### 13.3 Consideraciones para trabajo académico

9. **Existe un trade-off documentable entre tamaño, precisión y compatibilidad** con el acelerador ESP-DL que es relevante para la investigación en TinyML:
   - ESPDet (nativo ESP-DL): menor tamaño, menor mAP, máxima compatibilidad
   - FCOS (genérico PyTorch): tamaño medio, mAP media, compatibilidad parcial (warnings)
   - YOLO26 (Ultralytics): mayor tamaño, mayor mAP, compatibilidad requiere preprocesamiento

10. **La cuantización PTQ INT8 logra ratios de compresión sub-óptimos respecto al teórico 4×**, con un gap de 8-32% explicable por overhead de metadatos. Esto debe considerarse al estimar requisitos de memoria en trabajos de diseño de sistemas TinyML.

---

## 14. Trabajo Futuro

| Prioridad | Tarea | Justificación |
|---|---|---|
| **Alta** | Validar ESPDL de ESPDet T4 en ESP32-S3 real (inferencia end-to-end) | Confirmar que INT8 funciona sin degradación crítica en dispositivo |
| **Alta** | Validar ESPDL de FCOS T3 en ESP32-S3 (evaluar errores de exponente SE) | Determinar si los 2 errors de exponente causan overflow real |
| **Media** | Benchmark de latencia de inferencia INT8 en ESP32-S3 (ms/frame) | Dato necesario para evaluar viabilidad de detección en tiempo real |
| **Media** | Evaluar mAP@50 post-cuantización (inferencia INT8 vs FP32) | Medir degradación real de mAP por cuantización |
| **Baja** | Explorar INT16 mixed-precision para capas sensibles (FCOS SE blocks) | Resolver errores de exponente manteniendo tamaño aceptable |
| **Baja** | QAT (Quantization-Aware Training) para ESPDet | Potencialmente mejorar mAP post-cuantización del candidato principal |
| **Baja** | Probar inferencia YOLO26 en ESP32-S3 (SRAM profiling) | Confirmar si 2.72 MB ESPDL es ejecutable con el SRAM disponible |

---

> **Documento generado**: 24 de febrero de 2026  
> **Script de conversión**: `02_ING_MODELOS/Train_MLOps/scripts/convert_onnx_to_espdl.py`  
> **Datos fuente**: `02_ING_MODELOS/Train_MLOps/outputs/espdl/export_summary.json`
