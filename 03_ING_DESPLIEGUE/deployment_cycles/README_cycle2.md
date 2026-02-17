# Ciclo 2 — Mejora de Cuantización y Diagnóstico YOLO

> **Inicio:** 13 de febrero de 2026  
> **Objetivo:** Resolver el problema de 0 detecciones YOLO on-device mediante mejoras en el pipeline de cuantización, validación mAP pre-flash, y análisis end-to-end de la cadena de preprocesamiento y postprocesamiento  
> **Referencia:** [esp-detection](https://github.com/espressif/esp-detection) — pipeline oficial de Espressif para detección en ESP32  
> **Estado:** Completado — la cuantización está validada; el problema se localiza en el runtime on-device

---

## 1. Contexto

El Ciclo 1 dejó la infraestructura completa (firmware, cámara, dashboard, post-procesamiento) pero con un problema crítico: **YOLO11n produce 0 detecciones on-device** (scores sigmoid < 0.002) a pesar de que la simulación PPQ en Python muestra sigmoid ≈ 0.82.

Tras analizar el repositorio `esp-detection`, se identificaron 3 diferencias clave con nuestro pipeline:

| Aspecto | Nuestro pipeline (Ciclo 1) | esp-detection |
|---------|---------------------------|---------------|
| Equalization | No activada (o invocada incorrectamente) | `setting.equalization = True`, 3 iteraciones |
| `model->minimize()` | No llamada | Llamada tras construir `dl::Model` |
| Validación mAP pre-flash | Inexistente | `eval_quantized_model.py` con COCO mAP |

---

## 2. Acciones implementadas

### Acción 1 — Equalization en cuantización ✅

**Archivo:** `models/quantize_models_esp.py`

Añadida configuración de equalization tras `QuantizationSettingFactory.espdl_setting()`:
```python
quant_setting.equalization = True
quant_setting.equalization_setting.iterations = 3
quant_setting.equalization_setting.value_threshold = 2.0
```

La equalization redistribuye los rangos de pesos entre capas Conv adyacentes, reduciendo la degradación de cuantización en tensores de pocas dimensiones (como los scores con solo 5 canales de clase).

**Resultado:** El algoritmo de equalization **encontró 0 pares elegibles** para YOLO11n. La estructura del grafo YOLO11n (con SiLU, Concat, C2f) no presenta pares Conv consecutivos que cumplan los criterios. Esto es un comportamiento esperado, no un error.

### Acción 2 — `model->minimize()` en firmware ✅

**Archivo:** `firmware/components/inference/inference_engine.cpp`

Añadido `s_model->minimize();` inmediatamente después del constructor `new dl::Model(...)`:
```cpp
s_model->minimize();
```

Libera buffers intermedios no necesarios tras la construcción del grafo de ejecución, reduciendo uso de PSRAM.

**Estado:** Código modificado. Se aplicará en el siguiente build+flash.

### Acción 3 — Script de evaluación mAP pre-flash ✅

**Archivo:** `models/eval_quantized.py` (~600 líneas)

Script de validación end-to-end que:
1. Cuantiza el ONNX con las settings del ciclo (equalization)
2. Evalúa mAP float (ONNX original con Ultralytics `DetectionValidator`)
3. Simula inferencia INT8 con `TorchExecutor` de esp-ppq
4. Decodifica las 6 salidas crudas (box0-2 + score0-2) con DFL integral + sigmoid + dist2bbox
5. Computa mAP50 y mAP50-95 INT8 sobre el dataset de validación (2,066 imágenes)
6. Ejecuta diagnóstico comparativo Float ONNX vs INT8 sobre una imagen de calibración
7. Compara resultados e imprime veredicto (gate de calidad)

Gate automático:
- **✅ PASS:** degradación mAP50 < 25%
- **⚠️ Marginal:** degradación 25-50%
- **⛔ FAIL:** degradación > 50% — NO flashear

**Uso:**
```bash
python models/eval_quantized.py --model yolo11n          # float + INT8
python models/eval_quantized.py --model yolo11n --skip-float  # solo INT8
python models/eval_quantized.py --model yolo26n
```

**Bugs corregidos durante desarrollo** (8 iteraciones de depuración):

| # | Error | Causa raíz | Fix |
|---|---|---|---|
| 1 | Dataset incorrecto | `data.yaml` apuntaba a `dataset_maestro_aug` (train=val=14,250 imgs) | Creado `models/data.yaml` con ruta absoluta a `datasets/yolo26/valid/` (2,066 imgs) |
| 2 | mAP ≈ 0 | Orden de clases: training `['dog','door','obstacle','person','stair']` vs eval `['obstacle','dog','person','stair','door']` | Corregido `CLASS_NAMES` y `data.yaml` al orden Roboflow (alfabético) |
| 3 | 5D tensor crash | `unsqueeze(0)` sobre datos de calibración que ya eran `[1,3,224,224]` | Añadido condicional `if x.dim() == 3` |
| 4 | `TypeError: __init__()` | Ultralytics 8.4.14 cambió API: `DetectionValidator(overrides=...)` → `DetectionValidator(args=...)` | Cambiado a `args=get_cfg(overrides={...})` |
| 5 | `AttributeError: 'str'.get` | `self.args.data` es un string path, no un dict | Añadido `check_det_dataset()` para parsear string→dict |
| 6 | `AttributeError: NoneType.type` | `self.device` era None en la subclase | Añadido `self.device = torch.device("cpu")` |
| 7 | `IndexError: too many indices` | Ultralytics 8.4.14 espera predicciones como dicts, no tensores `[M,6]` | Reescrito `postprocess_preds()` para devolver `{"bboxes", "conf", "cls", "extra"}` |

---

## 3. Organización previa

### Limpieza de `models/`
- Movidos **9 variantes experimentales** de `.espdl` + metadata (27 archivos) a `models/cycle1_archive/`
- Movidos **12 scripts de utilidad/diagnóstico** a `models/cycle1_archive/scripts/`
- Movidos **2 ONNX legacy** (`*_fixed.onnx`) a `models/cycle1_archive/`
- Documentación del Ciclo 1 en `deployment_cycles/README_cycle1.md`

### Estado limpio de `models/`
| Categoría | Archivos |
|-----------|----------|
| Modelos fuente | `yolo11n_v1_best.pt`, `yolo26n_v1_best.pt`, `MBNTv3S_*.keras` |
| ONNX ESP | `*_esp.onnx` (+ `.data`) |
| ESPDL primarios | `*.espdl` + `.info` + `.json` (3 modelos) |
| ONNX originales | `*.onnx` (3 modelos) |
| Calibración | `calib_set_nchw.pkl`, `calib_set_nhwc.pkl` |
| Pipeline scripts | `export_onnx_esp.py`, `quantize_models_esp.py`, `create_calib_set.py`, `validate_onnx.py`, `quantize_and_validate.py`, `eval_quantized.py` |

---

## 4. Resultados de evaluación mAP

### 4.1 YOLO11n — Float vs INT8

Validación sobre 2,066 imágenes del split `valid/` de `datasets/yolo26/`:

| Métrica | Float (ONNX) | INT8 (TorchExecutor) | Degradación |
|---------|-------------|---------------------|-------------|
| **mAP50** | **0.7925** | **0.7418** | **6.4%** |
| **mAP50-95** | **0.5424** | **0.4883** | **10.0%** |
| Precision | 0.834 | 0.792 | 5.0% |
| Recall | 0.710 | 0.679 | 4.4% |

**Gate de calidad: ✅ PASS** (degradación mAP50 = 6.4% < 25%)

### 4.2 Diagnóstico Float vs INT8 (imagen de calibración)

Comparación de sigmoid_max por nivel de score sobre una misma imagen:

| Nivel | Float ONNX sigmoid_max | INT8 sigmoid_max | Degradación |
|-------|----------------------|-----------------|-------------|
| score0 (P3, 28×28) | 0.34 | 0.29 | -15% |
| score1 (P4, 14×14) | 0.56 | 0.38 | -32% |
| score2 (P5, 7×7) | 0.11 | 0.005 | -95% |

La degradación en P5 (stride=32, solo 49 anchors) es severa pero tiene poco impacto en mAP global porque P5 contribuye pocos candidatos. P3 y P4 mantienen scores funcionales.

### 4.3 YOLO26n

**Pendiente.** Ejecución: `python models/eval_quantized.py --model yolo26n`

### 4.4 On-device (post-flash)

**Pendiente.** Requiere actualización de offsets (ver sección 7) y flash.

| Modelo | Detecciones | Latencia | FPS | PSRAM libre |
|--------|------------|----------|-----|-------------|
| YOLO11n | — | — | — | — |
| YOLO26n | — | — | — | — |
| MBNTv3S | 4/frame | 846 ms | 1.1 | — |

---

## 5. Conclusión principal del Ciclo 2

> **La cuantización INT8 NO es la causa del problema de 0 detecciones on-device.**

Con una degradación mAP50 de solo 6.4%, el modelo INT8 mantiene capacidad de detección funcional. Los scores post-sigmoid en el simulador INT8 alcanzan valores razonables (0.29-0.38 en P3/P4), muy por encima de cualquier threshold práctico.

El problema de 0 detecciones debe estar en el **runtime on-device** (firmware ESP32-S3), no en la calidad de la cuantización.

---

## 6. Análisis de normalización (cadena completa)

Se verificó la consistencia de la normalización de imágenes en toda la cadena:

| Etapa | Archivo | Rango | Método | Coincide |
|-------|---------|-------|--------|----------|
| **Entrenamiento** | `src_colab/utils_train.py` (Ultralytics) | [0, 1] | Interno Ultralytics `/255` | — (referencia) |
| **Calibración** | `models/create_calib_set.py` | [0, 1] | `img.astype(float32) / 255.0` | ✅ |
| **Cuantización** | esp-ppq (automático) | exp=-7 | int8/128 ≈ [0, 0.992] | ✅ |
| **Firmware** | `firmware/components/image_proc/image_proc.cpp` | [0, 127] | `pixel × 128 / 255` | ✅ |

**Conclusión:** La normalización es **consistente** en toda la cadena. No hay mismatch de preprocesamiento.

Notas de interés del código de entrenamiento (`02_ING_MODELOS/GoogleCloudAI/`):
- El entrenamiento usa la API estándar de Ultralytics (`model.train(**params)`), que internamente normaliza a [0,1]
- No se aplica normalización tipo ImageNet (mean/std subtraction)
- El dataset utiliza el orden de clases Roboflow (alfabético): `['dog', 'door', 'obstacle', 'person', 'stair']`
- Definido en `src_colab/config.py` → `DATASET_MASTER_CLASSES`

---

## 7. Análisis del postprocesamiento firmware

Se revisó todo el código de postprocesamiento en el firmware (`postprocess_yolo.cpp`, `postprocess_common.cpp`, `inference_engine.cpp`).

### 7.1 Verificaciones que PASAN

| Aspecto | Detalle | Estado |
|---------|---------|--------|
| Layout NHWC | ESP-DL convierte NCHW→NHWC; firmware accede `(y*W+x)*C+c` | ✅ |
| Nombres de tensores | ONNX: box0/score0…score2 → ESPDL: mismos nombres en binario (verificado con búsqueda en .espdl) | ✅ |
| DFL integral | Softmax numerically-stable + weighted sum — idéntico al Python | ✅ |
| dist2bbox | `x1=(cx-dist_l)×stride`, normalizado a [0,1] | ✅ |
| Sigmoid | `1/(1+exp(-x))` estándar | ✅ |
| NMS | Insertion sort + greedy suppress per-class | ✅ |
| Orden de clases en firmware | `["dog","door","obstacle","person","stair"]` — coincide con data.yaml de entrenamiento | ✅ |
| Score threshold | 0.10 — permisivo para INT8 | ✅ |
| Path INT8 vs FLOAT | Detecta `score_dtype` y bifurca correctamente | ✅ |
| Loop principal | capture→preprocess→inference→postprocess→broadcast | ✅ |

### 7.2 Bug encontrado: desajuste de tamaño YOLO11n

| Parámetro | `app_config.h` | Archivo `.espdl` real | Diferencia |
|-----------|----------------|----------------------|------------|
| **YOLO11n size** | **2,800,272** | **2,802,432** | **+2,160 bytes** |
| YOLO26n size | 2,639,168 | 2,639,168 | 0 ✅ |
| MBNTv3S size | 681,088 | 681,088 | 0 ✅ |

El `yolo11n_v1_best.espdl` fue re-cuantizado con equalization (13-Feb) y creció 2,160 bytes. `app_config.h` y `flash_models.sh` conservan los tamaños del Ciclo 1.

**Consecuencia en la partición flash:**
```
YOLO11n real end:   0x0A7000 + 2,802,432 = 0x353300
YOLO26n offset:     0x353000
OVERLAP:            768 bytes → los últimos 768 bytes de YOLO11n serían sobrescritos por YOLO26n
```

> **Nota:** Este bug es **nuevo del Ciclo 2** (la re-cuantización cambió el tamaño). No puede ser la causa original del problema de 0 detecciones del Ciclo 1, ya que el modelo de Ciclo 1 (`yolo11n_v1_best_C_percentile.espdl`) era exactamente 2,800,272 bytes.

---

## 8. Hipótesis pendientes y acciones para el Ciclo 3

El postprocesamiento es algorítmicamente correcto y la cuantización es funcional. Sin embargo, persiste el problema de 0 detecciones on-device. Las hipótesis deben verificarse con **logs seriales del ESP32-S3**.

### 8.1 Hipótesis ordenadas por probabilidad

| # | Hipótesis | Descripción | Cómo verificar |
|---|-----------|-------------|----------------|
| H1 | **Exponent inesperado** | Si el score exponent es muy negativo (ej. -14), la dequantización `raw × 2^(-14)` comprimiría logits a valores ínfimos → `sigmoid ≈ 0.5` para todo → ninguno pasa threshold | Log de `inference_init`: verificar exponent de cada output |
| H2 | **Camera byte order RGB565** | Si el byte order big/little endian está invertido por un cambio de configuración de la cámara, los colores se revuelven → el modelo ve "ruido" → scores bajos | Endpoint `/debug/image` del dashboard: comparar BMP visual. También log DIAG input: si min/max están en rango razonable |
| H3 | **Model loading parcial** | Si el mmap no cubre todo el flatbuffer, o hay corrupción, los pesos se leen parcialmente | Log de `inference_init`: verificar que todas las 6 salidas aparecen con shapes correctas |
| H4 | **Score dtype mismatch** | Si `DATA_TYPE_FLOAT` no es `0` en la versión de ESP-DL usada, el firmware tomaría el path INT8 para datos float o viceversa | Log DIAG output: verificar dtype reportado vs real |

### 8.2 Acciones recomendadas para Ciclo 3

**Prioridad 1 — Diagnóstico con logs seriales:**
1. Conectar ESP32-S3 y capturar boot + 3 primeros frames con `idf.py monitor`
2. Verificar las líneas `DIAG input`, `DIAG output` e `inference_init` Output logs
3. Con estos datos, confirmar o descartar H1-H4

**Prioridad 2 — Fix del size mismatch (para poder flashear el modelo de Ciclo 2):**
1. Actualizar `MODEL_YOLO11N_SIZE` en `app_config.h` a `2802432`
2. Recalcular `MODEL_YOLO26N_OFFSET` al siguiente múltiplo de 4KB: `0x354000`
3. Actualizar `flash_models.sh` con el nuevo offset
4. Actualizar comentarios de `partitions.csv`

**Prioridad 3 — Evaluaciones pendientes:**
1. Ejecutar `eval_quantized.py --model yolo26n`
2. Flash y test on-device con el modelo corregido

---

## 9. Entorno técnico

| Componente | Versión |
|------------|---------|
| ESP-DL | 3.2.4 |
| esp-ppq | 1.2.4 |
| Ultralytics | 8.4.14 |
| PyTorch | 2.10.0 |
| Python | 3.10.19 |
| ONNX Runtime | 1.23.2 |
| Hardware host | Apple M1 MacBook Pro (CPU-only) |
| Hardware target | ESP32-S3 Freenove CAM (OV5640, 8MB PSRAM, 16MB flash) |
| Modelo | YOLO11n 224×224, 2.6M params, reg_max=16, 5 clases |
| Dataset | 9,722 train + 2,066 valid (`datasets/yolo26/`) |
| Clases | `['dog', 'door', 'obstacle', 'person', 'stair']` (orden Roboflow alfabético) |
