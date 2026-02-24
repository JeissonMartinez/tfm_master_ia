# Ciclo 3 — Integración de Modelos ESPDL Finales (Partition-based)

> **Inicio:** julio 2026  
> **Objetivo:** Integrar los 2 modelos ESPDL seleccionados (ESPDet Pico T4 + YOLO26n T2 ESP) en el firmware, con carga desde particiones flash independientes, preprocesamiento corregido y postprocesamiento FCOS/DFL completo  
> **Referencia:** `02_ING_MODELOS/Train_MLOps/docs/Instructivo_Despliegue_ESPDL.md`  
> **Estado:** Implementado — pendiente de compilación y validación on-device

---

## 1. Contexto

El Ciclo 2 completó la validación de cuantización y diagnosticó que el problema de 0 detecciones se localizaba en el runtime on-device.  Se identificaron 3 causas raíz:

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

**Total:** ~14 MB de 16 MB disponibles.  Subtipo `0x40` = custom data.

### Flash independiente de modelos

```bash
bash scripts/flash_models.sh [/dev/ttyXXX]
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

**Tensor specs ESPDet:**
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

**Tensor specs YOLO26n T2 ESP:**
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

### 3.9. `main/main.cpp` ✅

Cambios principales:
- **Default model:** Cambiado de `YOLO11N + TFLITE_MICRO` a `ESPDET_PICO + ESP_DL`
- **make_model_config():** 2 nuevos cases con `espdl_partition` labels
- **run_postprocess():** 2 nuevos cases delegando a `postprocess_espdet_espdl()` y `postprocess_yolo26_espdl()`
- **inference_task():** Nuevo branch `needs_espdl_preprocess` que llama a `image_preprocess_espdl()` cuando `engine == ESP_DL`
- **app_main():** Log condicional — muestra partición ESPDL en vez de tflite_size
- **Eliminados:** Stubs de `EMBED_FILES` extern symbols y `HAS_ESPDL_MODELS`

### 3.10. `main/CMakeLists.txt` ✅
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
│ 1. Flashear firmware (solo la primera vez o con cambios de FW)  │
│    idf.py -p PORT flash                                         │
│                                                                 │
│ 2. Flashear modelos (independiente del firmware)                │
│    bash scripts/flash_models.sh [PORT]                          │
│                                                                 │
│ 3. Monitorear                                                   │
│    idf.py -p PORT monitor                                       │
│                                                                 │
│ Boot sequence:                                                  │
│    camera_init → image_proc_init → postprocess_init             │
│    → EspDlEngine::init("model_espdet")                          │
│    → dl::Model(partition, PARTITION, 0, GREEDY)                 │
│    → mmap de partición flash                                    │
│    → inference_task en Core 0                                   │
│      loop: capture → preprocess_espdl → invoke → postprocess    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Cambiar Modelo Activo

Para cambiar entre ESPDet Pico y YOLO26n T2 ESP, editar `main/main.cpp`:

```cpp
// ESPDet Pico T4 (más rápido, mejor mAP)
#define ACTIVE_MODEL_TYPE   ModelType::ESPDET_PICO

// YOLO26n T2 ESP (más preciso en boxes, más lento)
#define ACTIVE_MODEL_TYPE   ModelType::YOLO26N_ESP
```

Ambos usan `EngineType::ESP_DL` automáticamente vía `make_model_config()`.

---

## 7. Próximos Pasos

- [ ] **Compilación y flash:** `idf.py build && idf.py flash && bash scripts/flash_models.sh`
- [ ] **Validación on-device:** Verificar detecciones con los 2 modelos
- [ ] **Ajuste de umbrales:** Calibrar conf/IoU con imágenes reales de la cámara
- [ ] **Streaming + bbox overlay:** Bloque B — MJPEG stream con bboxes dibujados client-side
- [ ] **Toggle/switch:** Bloque C — Controles WebSocket para activar/desactivar inferencia y cambiar modelo en runtime
- [ ] **Profiling:** Medir latencia por escala/fase y comparar ESPDet vs YOLO26n
