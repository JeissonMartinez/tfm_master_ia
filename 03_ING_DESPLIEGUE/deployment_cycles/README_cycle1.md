# Ciclo 1 — Despliegue Base en ESP32-S3

> **Período:** Enero – Febrero 2026  
> **Plataforma:** ESP32-S3 (Freenove CAM, OV5640, 8 MB PSRAM, 16 MB flash)  
> **Framework:** ESP-IDF v5.4.3 + ESP-DL 3.2.4 + esp-ppq 1.2.4

---

## 1. Objetivo

Desplegar tres modelos de detección de objetos entrenados en el ciclo de entrenamiento previo (5 clases: `dog`, `door`, `obstacle`, `person`, `stair`) sobre un ESP32-S3, con firmware completo que incluya captura de cámara, inferencia INT8, post-procesamiento (NMS), dashboard WiFi y sistema de métricas.

### Modelos a desplegar

| Modelo | Arquitectura | Parámetros | Formato origen |
|--------|-------------|------------|----------------|
| YOLO11n | Ultralytics YOLOv11 nano (reg_max=16, DFL) | ~2.6 M | `.pt` |
| YOLO26n | Ultralytics YOLOv26 nano (reg_max=1) | ~2.3 M | `.pt` |
| MBNTv3S-SSDLite | MobileNetV3-Small + SSD head | ~1.5 M | `.keras` |

---

## 2. Infraestructura construida

### 2.1 Componentes de firmware (ESP-IDF)

Se crearon **8 componentes** con arquitectura modular:

| Componente | Función |
|-----------|---------|
| `camera_handler` | Inicialización OV5640, captura RGB565 320×240 |
| `image_proc` | Redimensionado bilineal a 224×224, normalización INT8 (pixel − 128, escalado por exponent) |
| `inference` | Carga de modelo desde partición flash vía `esp_partition_mmap`, ejecución con `dl::Model`, modo dual-core |
| `postprocess` | Decodificadores YOLO (DFL + direct) y SSD (anchor-based), NMS on-device |
| `metrics` | Temporización por frame, sensor de temperatura, contadores de memoria |
| `network` | WiFi AP (`ESP32_TFM`), servidor HTTP, WebSocket para streaming |
| `dashboard` | Página HTML/JS embebida para visualización en navegador |
| `app_config` | Configuración centralizada (modelo activo, umbrales, resolución) |

### 2.2 Layout de particiones flash

```
Offset     Tamaño    Contenido
0x010000   4 MB      Aplicación (factory)
0x410000   7 MB      Partición 'models' (datos binarios)
```

Dentro de la partición `models`:
- **Offset 0x000000:** MBNTv3S-SSDLite (681 KB)
- **Offset 0x0A7000:** YOLO11n (2.67 MB)
- **Offset 0x35F000:** YOLO26n (2.57 MB)
- Uso total: 85% de los 7 MB disponibles.

### 2.3 Pipeline de conversión

```
.pt / .keras  →  ONNX (opset 13)  →  onnxsim  →  esp-ppq (INT8)  →  .espdl
```

Para los modelos YOLO se desarrolló un export especializado (`export_onnx_esp.py`) que:
- Monkey-patchea el forward de `Detect` para generar **6 salidas crudas** (box0/score0 por cada nivel de feature map) en lugar de la salida única post-DFL/sigmoid
- Monkey-patchea `Attention.forward` para reemplazar `einsum` por `matmul` explícito (compatibilidad ESP-DL)
- Genera ONNX con input `[1, 3, 224, 224]` y 6 outputs

---

## 3. Optimizaciones de rendimiento

Se aplicaron varias optimizaciones durante el ciclo para reducir la latencia de YOLO11n:

| Optimización | Antes | Después | Impacto |
|-------------|-------|---------|---------|
| Dual-core (`RUNTIME_MODE_MULTI_CORE`) | 3,514 ms | ~2,000 ms | −43% |
| RAM interna para intermedios (100 KB) | PSRAM only | Mixto | −15% |
| Cache de datos 64 KB (vs 32 KB default) | Misses altos | Cache hits | −10% |
| Compilación `-O2` (vs `-Og`) | Debug | Release | −10-15% |
| Eliminación de `model->profile()` en init | +3.5s + WDT | Sin overhead | Estabilidad |
| **Acumulado** | **3,514 ms** | **~896 ms** | **−74%** |

**Referencia oficial Espressif:** YOLO11n a 640×640 = 26.2 s, a 320×320 = 6.2 s. Nuestros 896 ms a 224×224 son **consistentes** con el benchmark oficial.

---

## 4. Resultados por modelo

### 4.1 YOLO11n — 0 detecciones ❌

**Estado:** Inferencia funcional, post-procesamiento correcto, pero **todas las puntuaciones de clase son fuertemente negativas** tras cuantización INT8.

**Diagnóstico clave:**

| Salida | Rango INT8 | Exponente | Deq. máx | Sigmoid máx |
|--------|-----------|-----------|----------|-------------|
| score0 [1,28,28,5] | [-128, -60] | -3 | -7.50 | 0.0006 |
| score1 [1,14,14,5] | [-45, -26] | -2 | -6.50 | 0.0015 |
| score2 [1,7,7,5] | [-50, -32] | -2 | -8.00 | 0.0003 |
| box0-2 | Rango variado | -3 | ±9.8 | N/A |

**Validación Python:** El ONNX flotante produce `sigmoid_max = 0.83` con hasta 19 scores por encima de 0.3 en imágenes de calibración. La simulación PPQ con `TorchExecutor` también arroja `sigmoid_max ≈ 0.82`. **El problema ocurre solo en la ejecución on-device con ESP-DL.**

Métricas: **inf=896 ms, FPS=1.0, PSRAM libre=4,480 KB, RAM interna libre=122 KB**

### 4.2 MBNTv3S-SSDLite — Detecciones exitosas ✅

**Estado:** Primer modelo en producir detecciones reales on-device.

- **4 detecciones por frame** (threshold=0.10)
- **Inferencia: 846 ms, FPS=1.1**
- Input dtype diferente (Float16/cuantizado con exponent=0)
- Post-procesamiento anchor-based: 1,470 priors (2 feature maps: 14×14 y 7×7, 6 anchors/celda)
- Las puntuaciones son post-sigmoid (aplicadas dentro del modelo)

**Nota:** No se validó la precisión de las detecciones (correctitud de bounding boxes y clases).

### 4.3 YOLO26n — No probado ⏸️

Se re-exportó con el mismo patrón de 6 salidas y se cuantizó, pero **nunca se flasheó ni se probó on-device**. Arquitectura idéntica en pipeline pero con `reg_max=1` (sin DFL, decodificación directa de 4 canales de box).

---

## 5. Variantes de cuantización probadas (YOLO11n)

Se iteró extensamente sobre la cuantización intentando resolver el problema de scores:

| Variante | Calibrador | Opciones avanzadas | Resultado on-device |
|----------|-----------|-------------------|---------------------|
| Original (1 salida, head completo) | KL | — | 0 det (exponent=1, scale=2.0 destruye scores 0-1) |
| 6 salidas, KL | KL per-tensor | — | 0 det (scores negativos) |
| A: KL + Eq + BC | KL | equalization + bias_correct | 0 det (equalization encontró 0 pares) |
| B: MinMax | MinMax | — | 0 det |
| C: Percentile | Percentile | — | 0 det (seleccionado como "mejor") |
| FP32 scores | Mixto INT8/FP32 | Score convs en FP32 | Error ESP-DL: no soporta Conv FP32 |

**Hallazgo crítico:** La variante con equalization (A) reportó "0 equalization pairs found", lo que sugiere que la API de equalization no se invocó correctamente o que la configuración era insuficiente.

---

## 6. Problemas técnicos resueltos

| Problema | Causa | Solución |
|----------|-------|----------|
| 7 errores de compilación inicial | API ESP-DL 3.2.4 diferente a docs | Reescritura de `inference_engine.cpp`, includes, CMakeLists |
| WDT resets durante init | `model->profile()` bloqueaba >5 s | Eliminado; WDT deshabilitado para CPU0 |
| Latencia 3.5 s por frame | Single-core, PSRAM-only, -Og | Dual-core, RAM interna, -O2, cache 64 KB |
| ONNX con head de detección | Exponent=1 destruye scores | Re-export con 6 salidas crudas |
| Stack overflow en tareas | Tamaño default insuficiente | Aumentado a 8 KB+ |
| WiFi memory conflicts | PSRAM allocation | Reserva de pool de 32 KB para DMA |

---

## 7. Estado final del Ciclo 1

### Completado ✅
- Infraestructura completa de firmware (8 componentes, compilación, flash)
- Pipeline de conversión `.pt` → `.espdl` con 6 salidas
- Captura de cámara OV5640 + preprocesamiento INT8
- WiFi AP + Dashboard HTTP + WebSocket
- Sistema de métricas (FPS, latencia, temperatura, memoria)
- MBNTv3S produciendo detecciones on-device
- Optimización de latencia: 3,514 ms → 896 ms (−74%)

### No completado ❌
- YOLO11n: 0 detecciones on-device (problema de cuantización no resuelto)
- YOLO26n: nunca probado on-device
- Fase 4 (benchmarking 1,000 frames por modelo) no alcanzada
- Validación de precisión de detecciones (mAP) no implementada

---

## 8. Análisis de causa raíz pendiente

El problema principal del Ciclo 1 es la **discrepancia entre simulación PPQ (sigmoid ≈ 0.82) y ejecución ESP-DL on-device (sigmoid < 0.002)** para los tensores de scores de YOLO11n. Las hipótesis de trabajo para el Ciclo 2 son:

1. **Equalization insuficiente:** El pipeline de cuantización NO usa equalization correctamente. El repo `espressif/esp-detection` sí lo usa (`setting.equalization = True`, 3 iteraciones). La variante "eqbc" del Ciclo 1 falló porque posiblemente la API se invocó incorrectamente.

2. **Falta de `model->minimize()`:** El firmware no llama a `minimize()` tras construir el `dl::Model`. Espressif lo hace en su pipeline de referencia. Podría afectar la asignación de buffers intermedios.

3. **Ausencia de validación mAP pre-flash:** No existe un script que calcule mAP sobre el modelo cuantizado antes de flashearlo. Se perdieron horas flasheando modelos defectuosos.

---

## 9. Artefactos generados

### En producción (`models/`)
- 3 modelos fuente (`.pt`, `.keras`)
- 3 ONNX ESP-compatibles (`*_esp.onnx`)
- 3 `.espdl` primarios con `.info`/`.json`
- 2 archivos de calibración (`.pkl`)
- 5 scripts del pipeline

### Archivados (`models/cycle1_archive/`)
- 9 variantes experimentales de `.espdl` con metadata
- 12 scripts de utilidad/diagnóstico
- 2 ONNX legacy parcheados

---

## 10. Plan para Ciclo 2

Basado en el análisis del repositorio `espressif/esp-detection` y la comparación con nuestro pipeline:

1. **Acción 1:** Activar equalization correctamente en `quantize_models_esp.py`
2. **Acción 2:** Añadir `model->minimize()` en `inference_engine.cpp`
3. **Acción 3:** Crear `eval_quantized.py` con cálculo de mAP (basado en `yolo11n_eval.py` de Espressif) como gate de validación pre-flash
