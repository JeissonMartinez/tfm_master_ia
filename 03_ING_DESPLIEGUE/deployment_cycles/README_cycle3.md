# Ciclo 3 — Integración de Modelos ESPDL Finales (Partition-based)

> **Inicio:** julio 2026  
> **Objetivo:** Integrar los 2 modelos ESPDL seleccionados (ESPDet Pico T4 + YOLO26n T2 ESP) en el firmware, con carga desde particiones flash independientes, preprocesamiento corregido y postprocesamiento FCOS/DFL completo  
> **Referencia:** `02_ING_MODELOS/Train_MLOps/docs/Instructivo_Despliegue_ESPDL.md`  
> **Estado:** ✅ Despliegue 1 (baseline) completado — ambos modelos validados on-device

---

## 0. Resumen Ejecutivo — Despliegue 1 (Baseline)

**Fecha:** 24 de febrero de 2026  
**Hardware:** ESP32-S3 WROOM N16R8 (Freenove CAM Board) + OV5640 @ QVGA  
**Firmware:** ESP-IDF v5.4.3 · esp-dl v3.x · Build 8.5 MB (~85% de partición factory)  
**WiFi:** STA mode → SSID "JM" · IP 192.168.1.31

### Resultados On-Device

| Métrica | ESPDet Pico T4 | YOLO26n T2 ESP |
|---------|:--------------:|:--------------:|
| **Inferencia** | ~405 ms | ~2885 ms |
| **Preprocesamiento** | ~14–275 ms | ~15–270 ms |
| **Postprocesamiento** | ~0.7–1.4 ms | ~3.5–4.8 ms |
| **Total por frame** | ~550–680 ms | ~3140–3170 ms |
| **FPS** | **~1.5–1.8** | **~0.3** |
| **Detecciones/frame** | 1–3 típico | 0–1 típico |
| **Heap interno libre** | ~69 KB | ~43 KB |
| **PSRAM libre** | ~6288 KB | ~3947 KB |
| **Consumo modelo (PSRAM)** | ~1900 KB | ~4240 KB |
| **mAP@50 (entrenamiento)** | 0.5319 | 0.4343 |

### Diagnóstico

- **ESPDet Pico T4:** Modelo liviano (361K params, 545 KB), inferencia sub-segundo. Detecciones consistentes (1–3/frame). Relación FPS/precisión favorable para navegación de LCMR en tiempo real.
- **YOLO26n T2 ESP:** Modelo pesado (2.6M params, 2.71 MB), inferencia ~2.9s. Demasiado lento para uso en tiempo real (~0.3 FPS). Detecciones esporádicas (0–1/frame). Consume ~4.2 MB de PSRAM solo para activaciones.
- **Warnings `cam_hal`:** `FB-SIZE: 0 != 153600` y `EV-VSYNC-OVF` son esperados — DMA pierde sincronía cuando la inferencia bloquea >400 ms. Los frames corruptos se descartan y se recapturan automáticamente.
- **Memoria estable:** No se observan memory leaks en ninguno de los dos modelos (heap/PSRAM constantes entre frames).

### Conclusión Baseline

ESPDet Pico T4 es el modelo viable para el LCMR. YOLO26n T2 ESP requiere optimizaciones significativas (cuantización más agresiva, poda, o reducción de input) para ser utilizable en tiempo real.

---

## 1. Contexto

El Ciclo 2 completó la validación de cuantización y diagnosticó que el problema de 0 detecciones se localizaba en el runtime on-device. Se identificaron 3 causas raíz:

| Problema | Detalle | Solución Ciclo 3 |
|----------|---------|-------------------|
| **Normalización incorrecta** | Firmware usaba `pixel - 128` (TFLite ZP=128), pero ESPDL usa `round(pixel/255 × 128)` con exp=-7 | Nueva función `image_preprocess_espdl()` |
| **espdl_engine.cpp era un stub** | Todos los métodos eran TODO, API incorrecta (`dl_model_base.h` en vez de `.hpp`) | Reescritura completa con API real esp-dl v3.x |
| **Acceso a outputs por índice** | El orden de tensores en la ejecución no coincide con el orden lógico | Acceso por nombre (`get_output("score0")`) |

### Modelos seleccionados

| Modelo | Tipo | Tamaño | Params | mAP@50 | Outputs | Post-proceso |
|--------|------|--------|--------|--------|---------|--------------|
| ESPDet Pico T4 | FCOS anchor-free | 545 KB | 361K | 0.5319 | 6 (3 score + 3 box) | Direct distances → NMS |
| YOLO26n T2 ESP | DFL integral | 2.71 MB | 2.6M | 0.4343 | 6 (3 score + 3 box) | DFL bins → softmax → dist2bbox → NMS |

---

## 2. Diseño de Particiones Flash

Se desestimó `EMBED_FILES` por limitaciones de tamaño de `factory` y se optó por **2 particiones data independientes** que esp-dl mapea vía `esp_partition_mmap()` automáticamente.

### Layout (`partitions.csv`)

```
# Name,           Type, SubType, Offset,     Size
nvs,              data, nvs,     0x9000,     0x6000
phy_init,         data, phy,     0xF000,     0x1000
factory,          app,  factory, 0x10000,    0xA00000    # 10 MB firmware
model_espdet,     data, 0x40,    0xA10000,   0x100000    # 1 MB (ESPDet Pico T4: 545 KB)
model_yolo26,     data, 0x40,    0xB10000,   0x300000    # 3 MB (YOLO26n T2 ESP: 2.71 MB)
```

**Total:** ~14 MB de 16 MB disponibles. Subtipo `0x40` = custom data.

### Flash independiente de modelos

```bash
bash scripts/flash_models.sh /dev/tty.wchusbserial5B414963901
# Equivale a:
# esptool.py write_flash 0xA10000 models/espdl/espdet_pico_t4.espdl
# esptool.py write_flash 0xB10000 models/espdl/yolo26n_t2_esp.espdl
```

Ventaja: se puede re-flashear un modelo sin recompilar ni re-flashear el firmware completo.

---

## 3. Archivos Modificados

### 3.1. `partitions.csv` ✅
- Reconstruida con layout de 2 particiones (ver sección 2)

### 3.2. `scripts/flash_models.sh` 🆕
- Script para flashear modelos independientemente con `esptool.py`
- Autodetecta puerto serial, valida archivos, flash en un solo comando

### 3.3. `main/app_config.h` ✅
- **ModelType enum:** Añadidos `ESPDET_PICO` y `YOLO26N_ESP`
- **Umbrales específicos:** `ESPDET_CONF=0.35/IOU=0.40`, `YOLO26ESP_CONF=0.25/IOU=0.45`
- **Constantes FCOS/DFL:** `GRID_STRIDES[3]={8,16,32}`, `GRID_SIZES[3]={28,14,7}`, `DFL_REG_MAX=16`
- **WiFi STA:** Configuración actualizada a modo estación (`WIFI_STA_SSID`, `WIFI_STA_PASS`, `WIFI_MAX_RETRY=10`)

### 3.4. `main/inference_engine.h` ✅
- Añadidos 3 métodos virtuales con default implementations:
  - `get_output_by_name(const char*)` → acceso multi-output por nombre
  - `get_output_exponent(const char*)` → exponent power-of-2
  - `get_output_shape_by_name(const char*, int*, int*)` → shape por nombre

### 3.5. `main/espdl_engine.h` ✅
- `is_output_int8()` → retorna `true` (antes `false`)
- `get_output_scale()` → calcula `2^exponent` (antes retornaba `1.0f`)
- Añadidos overrides: `get_output_by_name()`, `get_output_exponent()`, `get_output_shape_by_name()`
- Documentación actualizada con API key references

### 3.6. `main/espdl_engine.cpp` ✅ (REESCRITURA COMPLETA)

**Antes:** 168 líneas de stubs/TODO, header incorrecto (`dl_model_base.h`), `void* model_handle`.  
**Ahora:** Implementación completa con API real esp-dl v3.x:

```cpp
// Construcción desde partición
m_impl->model = new dl::Model(
    partition_label,
    fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
    0,                          // max_internal_size → máximo PSRAM
    dl::MEMORY_MANAGER_GREEDY,
    nullptr,                    // sin encriptación
    true                        // param_copy para mejor rendimiento
);

// Alimentar input (exponent=-7, INT8)
TensorBase* input = model->get_input();
input->assign({1, 224, 224, 3}, data, -7, dl::DATA_TYPE_INT8);

// Inferencia dual-core
model->run(dl::RUNTIME_MODE_MULTI_CORE);

// Acceso por nombre (CRÍTICO — el orden por índice no es estable)
TensorBase* score0 = model->get_output("score0");
int8_t* values = score0->get_element_ptr<int8_t>();
int exp = score0->exponent;  // power-of-2 exponent
```

**Features clave:**
- Cacheo de nombres de outputs en `std::vector<std::string>` para acceso por índice
- Logging detallado de shapes/dtypes/exponents al cargar
- Manejo de excepciones en constructor (try/catch)
- Soporte `const`-safe vía `const_cast` para API de esp-dl que no provee accessors const

### 3.7. `main/postprocess.h` + `main/postprocess.cpp` ✅

**2 nuevas funciones:**

#### `postprocess_espdet_espdl()` — FCOS Anchor-Free

Pipeline por escala (s ∈ {0,1,2}):
1. Obtener `score{s}` y `box{s}` por nombre del engine
2. Para cada celda (gx, gy) del grid:
   - Dequantizar scores: `float = int8 * 2^exp`
   - Sigmoid → best class → filtrar por conf_thr
   - Dequantizar box `[l,t,r,b]` → ReLU (max 0)
   - dist2bbox: `x1 = (cx - l*stride)/224`, etc.
3. NMS per-class

**Tensor specs ESPDet (validadas on-device):**
| Tensor | Shape | Exponent | Stride |
|--------|-------|----------|--------|
| score0 | 1×28×28×5 | -3 | 8 |
| score1 | 1×14×14×5 | -3 | 16 |
| score2 | 1×7×7×5 | -3 | 32 |
| box0 | 1×28×28×4 | -3 | 8 |
| box1 | 1×14×14×4 | -3 | 16 |
| box2 | 1×7×7×4 | -4 | 32 |

#### `postprocess_yolo26_espdl()` — DFL Integral

Pipeline igual que ESPDet excepto el decode de boxes:
1. Box tensors tienen shape `[1,H,W,64]` = 4 direcciones × 16 bins
2. Para cada dirección d:
   - Dequantizar 16 bins
   - Softmax con estabilización numérica (resta max)
   - Weighted sum: `dist = Σ(softmax[k] × k)` para k=0..15
3. dist2bbox igual que ESPDet

**Tensor specs YOLO26n T2 ESP (validadas on-device):**
| Tensor | Shape | Exponent | Stride |
|--------|-------|----------|--------|
| score0 | 1×28×28×5 | -3 | 8 |
| score1 | 1×14×14×5 | -2 | 16 |
| score2 | 1×7×7×5 | -3 | 32 |
| box0 | 1×28×28×64 | -3 | 8 |
| box1 | 1×14×14×64 | -3 | 16 |
| box2 | 1×7×7×64 | -3 | 32 |

### 3.8. `main/image_proc.h` + `main/image_proc.cpp` ✅

**Nueva función:** `image_preprocess_espdl()`

La diferencia de normalización era la causa raíz principal de las 0 detecciones:

| Método | Rango | Fórmula | Uso |
|--------|-------|---------|-----|
| `image_preprocess()` | [-128, 127] | `pixel_u8 - 128` | TFLite (zero_point=128) |
| `image_preprocess_espdl()` | [0, 127] | `round(pixel_u8/255 × 128)` | ESP-DL (exp=-7) |

```cpp
// Aritmética entera equivalente a round(pixel/255.0 * 128.0), clamp a 127
int val = (pixel * 128 + 127) / 255;
return (int8_t)(val > 127 ? 127 : val);
```

### 3.9. `main/wifi_manager.h` + `main/wifi_manager.cpp` ✅ (REESCRITURA COMPLETA)

**Antes:** Modo AP (punto de acceso propio).  
**Ahora:** Modo STA (estación conectada a red doméstica):
- Event group para espera bloqueante de conexión
- Auto-reconnect hasta `WIFI_MAX_RETRY` intentos
- IP dinámica por DHCP, accesible vía `wifi_get_ip()`
- Log detallado de estado (SSID, IP, canal, RSSI)

### 3.10. `main/main.cpp` ✅

Cambios principales:
- **Default model:** Seleccionable en compile-time vía `#define ACTIVE_MODEL_TYPE`
- **make_model_config():** 2 nuevos cases con `espdl_partition` labels
- **run_postprocess():** 2 nuevos cases delegando a `postprocess_espdet_espdl()` y `postprocess_yolo26_espdl()`
- **inference_task():** Nuevo branch `needs_espdl_preprocess` que llama a `image_preprocess_espdl()` cuando `engine == ESP_DL`
- **app_main():** Llama `wifi_init_sta()`, log condicional por engine type
- **Task stack en PSRAM:** `xTaskCreatePinnedToCoreWithCaps(..., MALLOC_CAP_SPIRAM)` — heap interno insuficiente tras carga de modelo + WiFi

### 3.11. `CMakeLists.txt` (raíz del proyecto) 🆕
- Creado el archivo top-level requerido por ESP-IDF (faltaba, era bloqueante para build)
- `cmake_minimum_required(VERSION 3.16)`, `project(tfm_tinyml_detector)`

### 3.12. `main/CMakeLists.txt` ✅
- Eliminada sección comentada de `EMBED_FILES` (ya no aplica)
- Comentario actualizado indicando carga por particiones

---

## 4. API esp-dl v3.x — Referencia Rápida

```
Namespace dl:: → Model, TensorBase, MEMORY_MANAGER_GREEDY, RUNTIME_MODE_MULTI_CORE, DATA_TYPE_INT8
Namespace fbs:: → MODEL_LOCATION_IN_FLASH_PARTITION

dl::Model(partition_label, fbs::MODEL_LOCATION_IN_FLASH_PARTITION, max_internal=0, mm=GREEDY)
  .get_input() → TensorBase*
  .get_output("name") → TensorBase*
  .get_outputs() → std::map<string, TensorBase*>&
  .run(dl::RUNTIME_MODE_MULTI_CORE)

TensorBase
  .assign(vector<int> shape, const void* data, int exponent, dtype_t dtype)
  .get_element_ptr<int8_t>() → int8_t*
  .shape → std::vector<int>
  .exponent → int     // float_val = int8_val × 2^exponent
  .dtype → dtype_t

DL_SCALE(exp) = 2^exp = (exp >= 0) ? (1 << exp) : (1.0 / (1 << -exp))
```

---

## 5. Flujo de Despliegue

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Flashear firmware                                            │
│    idf.py -p /dev/tty.wchusbserial5B414963901 flash             │
│                                                                 │
│ 2. Flashear modelos (independiente del firmware)                │
│    bash scripts/flash_models.sh /dev/tty.wchusbserial5B414963901│
│                                                                 │
│ 3. Monitorear                                                   │
│    idf.py -p /dev/tty.wchusbserial5B414963901 monitor           │
│                                                                 │
│ Boot sequence:                                                  │
│    metrics_init → camera_init → image_proc_init                 │
│    → postprocess_init → EspDlEngine::init(partition_label)      │
│    → dl::Model(partition, PARTITION, 0, GREEDY)                 │
│    → wifi_init_sta → webserver_start                            │
│    → xTaskCreatePinnedToCoreWithCaps (Core 0, PSRAM stack)      │
│      loop: capture → preprocess_espdl → invoke → postprocess    │
│           → metrics → webserver_broadcast                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Cambiar Modelo Activo

Para cambiar entre ESPDet Pico y YOLO26n T2 ESP, editar `main/main.cpp` (~línea 56):

```cpp
// ESPDet Pico T4 (~1.5–1.8 FPS, mejor mAP, recomendado para LCMR)
#define ACTIVE_MODEL_TYPE   ModelType::ESPDET_PICO

// YOLO26n T2 ESP (~0.3 FPS, modelo pesado, requiere optimización)
#define ACTIVE_MODEL_TYPE   ModelType::YOLO26N_ESP
```

Ambos usan `EngineType::ESP_DL` automáticamente vía `make_model_config()`.  
Después de cambiar: `idf.py build && idf.py -p PORT flash` (no es necesario re-flashear modelos).

---

## 7. Validación On-Device — Despliegue 1 (Baseline)

### 7.1. ESPDet Pico T4

**Fecha de prueba:** 24 de febrero de 2026  
**Configuración:** `ACTIVE_MODEL_TYPE = ModelType::ESPDET_PICO`, conf=0.35, IoU=0.40

```
Boot:
  Heap interno libre: 254 KB → post-modelo: 174 KB (consumo: ~80 KB interno)
  PSRAM libre: 8189 KB → post-modelo: 6288 KB (consumo: ~1900 KB PSRAM)
  Modelo cargado desde partición model_espdet
  6 outputs: score0/1/2 + box0/1/2 (exponents -3/-3/-3/-3/-3/-4)
  Input: INT8 exp=-7 shape=[1,224,224,3]
  WiFi STA: IP 192.168.1.31

Inferencia (steady state):
  [#1]  550.1 ms (pre:14.3  inf:407.1 post:0.7) | 1.8 FPS | 2 dets
  [#2]  680.9 ms (pre:271.2 inf:408.3 post:1.4) | 1.5 FPS | 3 dets
  [#3]  673.5 ms (pre:262.3 inf:409.8 post:1.4) | 1.5 FPS | 2 dets
  ...
  heap:69K psram:6288K (estable entre frames)
```

**Observaciones:**
- Inferencia consistente ~405 ms
- Preprocesamiento variable: ~14 ms (frame disponible) vs ~270 ms (esperando captura DMA)
- Post-procesamiento FCOS insignificante (<1.5 ms)
- 1–3 detecciones por frame — modelo produce resultados funcionales
- Sin memory leaks

### 7.2. YOLO26n T2 ESP

**Fecha de prueba:** 24 de febrero de 2026  
**Configuración:** `ACTIVE_MODEL_TYPE = ModelType::YOLO26N_ESP`, conf=0.25, IoU=0.45

```
Boot:
  Heap interno libre: 254 KB → post-modelo: 124 KB (consumo: ~130 KB interno)
  PSRAM libre: 8189 KB → post-modelo: 4046 KB (consumo: ~4143 KB PSRAM)
  Modelo cargado desde partición model_yolo26
  6 outputs: box0/1/2 + score0/1/2 (exponents -3/-3/-3/-3/-2/-3)
  Input: INT8 exp=-7 shape=[1,224,224,3]
  WiFi STA: IP 192.168.1.31

Inferencia (steady state):
  [#1]  2911.1 ms (pre:15.0   inf:2892.6 post:3.5) | 0.3 FPS | 0 dets
  [#2]  3127.3 ms (pre:241.0  inf:2882.7 post:3.5) | 0.3 FPS | 0 dets
  [#3]  3157.1 ms (pre:262.4  inf:2890.3 post:4.4) | 0.3 FPS | 1 dets
  ...
  [#8]  3158.3 ms (pre:268.0  inf:2885.5 post:4.8) | 0.3 FPS | 1 dets
  ...
  [#15] 3140.0 ms (pre:248.8  inf:2887.6 post:3.5) | 0.3 FPS | 0 dets
  heap:43K psram:3947K (estable entre frames)
```

**Observaciones:**
- Inferencia ~2885 ms (~7× más lento que ESPDet Pico)
- Modelo consume ~4.1 MB de PSRAM para activaciones (vs ~1.9 MB de ESPDet)
- Solo 26 KB de heap interno libre post-carga (margen mínimo)
- 0–1 detecciones por frame con umbral de confianza 0.25 (vs 1–3 de ESPDet con 0.35)
- Post-procesamiento DFL más costoso (3.5–4.8 ms vs 0.7–1.4 ms FCOS)
- `cam_hal` warnings más frecuentes por bloqueo prolongado (~3s) del loop de captura
- Sin memory leaks

### 7.3. Comparativa Cuantitativa

| Fase | ESPDet Pico T4 | YOLO26n T2 ESP | Factor |
|------|:--------------:|:--------------:|:------:|
| Inferencia | 407 ms | 2885 ms | **7.1×** |
| Preprocesamiento | 14–275 ms | 15–270 ms | ~1× |
| Postprocesamiento | 0.7–1.4 ms | 3.5–4.8 ms | ~3.4× |
| FPS efectivo | 1.5–1.8 | 0.3 | **5–6×** |
| RAM PSRAM modelo | 1.9 MB | 4.1 MB | 2.2× |
| RAM interna post-init | 69 KB libre | 43 KB libre | — |
| Detecciones/frame | 1–3 | 0–1 | — |
| Parámetros modelo | 361K | 2.6M | 7.2× |
| Tamaño .espdl | 545 KB | 2.71 MB | 5.0× |

### 7.4. Warnings de Cámara — Análisis

| Warning | Causa | Impacto |
|---------|-------|---------|
| `cam_hal: FB-SIZE: 0 != 153600` | DMA no completó llenado del frame buffer antes de ser solicitado | Frame descartado, reintento automático (agrega ~250 ms al preprocess) |
| `cam_hal: EV-VSYNC-OVF` | Overflow de eventos VSYNC — múltiples frames perdidos mientras inferencia bloqueaba | Informativo, no funcional |
| `cam_hal: EV-EOF-OVF` | Overflow de end-of-frame events | Similar a VSYNC-OVF, esperado con inf >400 ms |

Estos warnings son inherentes al modelo de single-thread (captura→inferencia secuencial). No representan un error ni degradan la funcionalidad — solo incrementan la latencia aparente de preprocesamiento.

---

## 8. Bugs Encontrados y Corregidos

### 8.1. Falta de `CMakeLists.txt` raíz
- **Síntoma:** `idf.py build` fallaba inmediatamente
- **Causa:** Archivo top-level de CMake no existía en el proyecto
- **Fix:** Creado `CMakeLists.txt` estándar ESP-IDF

### 8.2. WiFi en modo AP en vez de STA
- **Síntoma:** El dispositivo creaba su propia red WiFi en vez de conectarse a la red doméstica
- **Causa:** `wifi_manager.cpp` implementaba `wifi_init_ap()`
- **Fix:** Reescritura completa a modo STA con event group, DHCP, auto-reconnect

### 8.3. Tarea de inferencia no se creaba
- **Síntoma:** `No se pudo crear la tarea de inferencia` en boot
- **Causa:** Tras carga del modelo ESP-DL (~130 KB interno) + WiFi (~80 KB), no quedaba bloque contiguo de 32 KB en heap interno para el stack de la tarea
- **Fix:** `xTaskCreatePinnedToCoreWithCaps(..., MALLOC_CAP_SPIRAM)` — stack alojado en PSRAM

---

## 9. Próximos Pasos (Post-Baseline)

- [ ] **Ajuste de umbrales:** Calibrar conf/IoU con imágenes reales de la cámara para ambos modelos
- [ ] **Streaming MJPEG:** Bloque B — stream de video con bounding boxes dibujados client-side
- [ ] **Controles WebSocket:** Bloque C — toggle de inferencia, cambio de modelo en runtime (sin recompilación)
- [ ] **Profiling detallado:** Desglose de latencia por capa/operador ESP-DL para identificar cuellos de botella
- [ ] **Optimización YOLO26n:** Evaluar si reducción de input (160×160), poda, o cuantización más agresiva mejora el FPS a niveles utilizables
- [ ] **Pipeline asíncrono:** Evaluar captura en Core 1 + inferencia en Core 0 para reducir latencia de preprocesamiento
