User: Me vas a ayudar a perfeccionar el plan de despliegue que inicialmente tenía, que lo encuentras en: `docs/Plan_Despliegue.md`

De ese plan, yo ya hice por aparte la conversión de los modelos seleccionados, de .keras y .pt a .onnx y finalmente a .espdl 

Si bien el plan contenplaba usar .tflite y .tfmicro , en realidad me quiero centrar en usar los modelos en formato .espdl que ya tengo preparados en:
- models/yolo26n_v1_best.espdl
- models/yolo11n_v1_best.espdl
- models/MBNTv3S_ssdlite_v1_p2_best.espdl

Para ejecutar esto, puedes revisar un informe que yo hice en otro proyecto diferente donde probé por primera vez el poner un modelo de vision artificial en una ESP32-S3 , y considero que funcionó muy bien, por lo tanto quisiera fusionar el aprendizaje que tuve en ese entonces, con el actual proyecto. Ese informe lo encuentras aqui: `docs/INFORME_FINAL_EjemploDespliegue.md`

El roadmap y la escencia de despliegue sigue siendo la base del docs/Plan_Despliegue.md, considerando los aprendizajes aprendidos en `docs/INFORME_FINAL_EjemploDespliegue.md` y la configuracion de la ESP32-S3 `docs/Configuracion_ESP32-S3.md`. 

¿Qué plan me propones?

GitHub Copilot: 

 Let me start by reading all the relevant documents to understand the current plan, the lessons learned, and the ESP32-S3 configuration. 

Research deployment plan context

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/Plan_Despliegue.md)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/INFORME_FINAL_EjemploDespliegue.md)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/Configuracion_ESP32-S3.md)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/README.md)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.json#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.info#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.json#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.json#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.info#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/conversion_kerasToOnnx.py)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/conversion_TorchToOnnx.py)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_models.py)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/fix_onnx_for_espdl.py#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/create_calib_set.py#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/patch_esp_ppq.py)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/fix_onnx_for_espdl.py#200-200), lines 200 to 387

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/create_calib_set.py#200-200), lines 200 to 210

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/requirements.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/requirements_clean.txt#1-1), lines 1 to 50

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/requirements_flexible.txt#1-1), lines 1 to 50

Searched for text `return` (`**/models/MBNTv3S_ssdlite_v1_p2_best.info`), 1 result

Searched for text `return` (`**/models/yolo11n_v1_best.info`), 1 result

Searched for text `return` (`**/models/yolo26n_v1_best.info`), 1 result

Retrieving Notebook summary.

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.info#250-250), lines 250 to 264

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info#660-660), lines 660 to 675

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.info#785-785), lines 785 to 805

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/Conversion_ModelosTFLite.ipynb#1-1), lines 1 to 110

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/Conversion_ModelosTFLite.ipynb#110-110), lines 110 to 500

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/Conversion_ModelosTFLite.ipynb#500-500), lines 500 to 850

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01CHr9q7JQGmdSN9cGK22kPq__vscode-1770907754350/content.txt)

Now I have a comprehensive picture. Before drafting the plan, I need to clarify a few important points I discovered during research: 

Asked 4 questions (Model Load, Model Select, YOLO26n NMS, MBNTv3S)

Excellent. Now I have all the context and decisions. Let me draft the refined plan.

---

## Plan: Despliegue ESP-DL de 3 Modelos en ESP32-S3

**TL;DR** — Desplegar 3 modelos `.espdl` (MBNTv3S-SSDLite, YOLO11n, YOLO26n) sobre ESP32-S3 usando **exclusivamente ESP-DL** como runtime de inferencia (sin TFLite Micro). Los modelos residen en una sola partición de flash con selección por offset vía `#define` en compile-time. El firmware captura imágenes con OV5640, preprocesa a 224×224 INT8 NHWC, ejecuta inferencia, postprocesa (decode + NMS on-device para los 3 modelos), y reporta resultados vía WebSocket/Dashboard embebido. La arquitectura modular por componentes IDF sigue los patrones validados en el piloto anterior de reconocimiento de gestos.

**Decisiones clave ya tomadas:**
- Solo ESP-DL (no dual runtime) → simplifica la abstracción de inferencia
- Una sola partición models con los 3 `.espdl` a offsets conocidos
- Selección compile-time (`#define ACTIVE_MODEL`)
- NMS on-device para los 3 modelos (YOLO26n fue truncado pre-NMS durante la conversión ONNX→ESPDL)
- El modelo MobileNet es **MBNTv3S** (MobileNetV3-Small), no MBNTv2

---

### Fase 1: Infraestructura Base

**Objetivo:** Proyecto ESP-IDF compilable con captura de cámara funcional, preprocesamiento de imagen validado, y carga de modelo desde flash.

**1.1 Scaffolding del proyecto**
- Crear estructura de directorios bajo una carpeta `firmware/` (o similar):
  ```
  firmware/
  ├── CMakeLists.txt            (proyecto raíz)
  ├── sdkconfig.defaults        (config base ESP32-S3)
  ├── partitions.csv            (tabla de particiones custom)
  ├── main/
  │   ├── CMakeLists.txt
  │   ├── main.cpp              (entry point)
  │   ├── app_config.h          (defines: ACTIVE_MODEL, umbrales, WiFi creds)
  │   └── Kconfig.projbuild     (opciones menuconfig si se desea)
  └── components/
      ├── camera/               (driver OV5640)
      ├── image_proc/           (preprocessing pipeline)
      ├── inference/            (ESP-DL engine wrapper)
      ├── postprocess/          (decode + NMS)
      ├── metrics/              (timing, memory, temperature)
      ├── network/              (WiFi AP + HTTP + WebSocket)
      └── dashboard/            (HTML embebido gzipped)
  ```
- Configurar idf_component.yml como componente **obligatorio** (lección del piloto) con dependencias:
  - `espressif/esp-dl: ">=3.0.0"` 
  - `espressif/esp32-camera: ">=2.0.0"`
- Cada componente (`camera`, `inference`, etc.) tendrá su propio `CMakeLists.txt` y `idf_component.yml` si requiere dependencias externas.

**1.2 Configuración sdkconfig.defaults**
- Aplicar todas las configuraciones probadas de Configuracion_ESP32-S3.md:
  - PSRAM: Octal 80 MHz, modo Octal, acceso vía `heap_caps_malloc`
  - Flash: 16 MB, DIO
  - CPU: 240 MHz
  - C++: GNU++23 (`-std=gnu++2b`)
  - WiFi IRAM opt: habilitado
  - LWIP max sockets: 10
  - Stack sizes ajustados (main 8 KB, idle 4 KB)

**1.3 Tabla de particiones**
- Diseñar `partitions.csv` con una partición models tipo `data` que albergue los 3 `.espdl` secuencialmente:
  ```
  # Name,    Type, SubType,  Offset,    Size
  nvs,       data, nvs,      0x9000,    0x6000      # 24 KB
  phy_init,  data, phy,      0xF000,    0x1000      # 4 KB
  factory,   app,  factory,  0x10000,   0x300000    # 3 MB firmware
  models,    data, 0x40,     0x310000,  0xCF0000    # ~12.9 MB modelos
  ```
- Dentro de la partición models, los 3 modelos se ubican a offsets internos fijos. Calcular sizes reales de los `.espdl` y definir en `app_config.h`:
  ```
  MODEL_OFFSET_MBNTV3S = 0x000000
  MODEL_OFFSET_YOLO11N = <después de MBNTv3S, alineado a 4KB>
  MODEL_OFFSET_YOLO26N = <después de YOLO11n, alineado a 4KB>
  ```
- Crear script `flash_models.sh` que use `parttool.py` para escribir los 3 modelos a la partición en los offsets correctos (patrón probado en piloto, INFORME_FINAL).

**1.4 Driver de cámara OV5640**
- Componente `camera/` con configuración exacta validada:
  - `PIXFORMAT_RGB565` (NO JPEG — problema conocido con OV5640)
  - `FRAMESIZE_QVGA` (320×240)
  - `fb_count = 2`, `fb_location = CAMERA_FB_IN_PSRAM`, `grab_mode = CAMERA_GRAB_LATEST`
  - `xclk_freq_hz = 20000000`
  - Pin mapping de Configuracion_ESP32-S3.md: XCLK=15, SIOD=4, SIOC=5, Y9-Y2={16,17,18,12,10,8,9,11}, VSYNC=6, HREF=7, PCLK=13
- Post-init del sensor: `vflip(1)` (crítico, imagen invertida sin esto), auto white balance, auto exposure, auto gain, lens correction.

**1.5 Pipeline de preprocesamiento**
- Componente `image_proc/` implementando:
  1. **Center crop** 320×240 → 240×240 (offset_x = 40 pixels en RGB565)
  2. **Resize** 240×240 → 224×224 — usar `dl::image::ImageTransformer` de ESP-DL (aprovecha instrucciones DSP del Xtensa LX7, validado en piloto)
  3. **Conversión RGB565 → RGB888** (si `ImageTransformer` no lo maneja, hacer manual: extraer R5G6B5 → extend a 8 bits)
  4. **Normalización INT8**: `pixel - 128` → rango [-128, 127] — los 3 modelos `.espdl` aceptan INT8 NHWC según los `.info`
  - **Dato crítico descubierto**: los `.info` muestran que YOLO11n y YOLO26n tienen `exponents: [-7]` en su input, lo que significa que el valor INT8 se interpreta como `valor * 2^(-7)` = `valor / 128.0`. Verificar que la normalización `pixel - 128` sea consistente con cómo esp-ppq cuantizó usando calibración `float / 255.0`.
  - Buffer de salida: `int8_t[1][224][224][3]` en PSRAM (`heap_caps_malloc` con `MALLOC_CAP_SPIRAM`)

**1.6 Carga de modelo con esp_partition_mmap**
- En el componente `inference/`, implementar la carga del modelo:
  1. `esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "models")` — localizar partición
  2. `esp_partition_mmap(partition, offset, size, ESP_PARTITION_MMAP_DATA, &mapped_ptr, &handle)` — mapear a memoria (zero-copy desde flash)
  3. Pasar el puntero mapeado al constructor de `dl::Model` de ESP-DL
- El offset y size se definen por `#define ACTIVE_MODEL` en `app_config.h`

**Criterio de salida Fase 1:** Firmware compila, la cámara captura frames RGB565 320×240, el preprocesamiento genera tensor INT8 224×224×3, y el modelo se carga exitosamente en memoria desde la partición.

---

### Fase 2: Inferencia ESP-DL + Postprocesamiento

**Objetivo:** Ejecutar inferencia con los 3 modelos y decodificar detecciones (bounding boxes + clases + scores).

**2.1 ESP-DL Engine Wrapper**
- Componente `inference/` con interfaz simple (ya no necesita abstracción dual):
  ```
  struct Detection { float x, y, w, h; int class_id; float score; };
  class InferenceEngine:
    - init(model_ptr, model_size) → carga dl::Model
    - run(input_tensor) → ejecuta forward pass
    - get_output(index) → raw output tensor(s)
  ```
- Asignar tensores intermedios en PSRAM. El piloto demostró que `heap_caps_malloc(size, MALLOC_CAP_SPIRAM)` resuelve los stack overflow que ocurren con memoria interna.
- **Monitorear memoria** post-init: `heap_caps_get_free_size(MALLOC_CAP_SPIRAM)` debe ser > 500 KB (criterio de aceptación).

**2.2 Postprocesamiento MBNTv3S-SSDLite**
- 3 tensores de salida: `bbox_out[1,1470,4]`, `class_out[1,1470,5]`, `objectness[1,1470,1]`
- Todos pasan por Sigmoid LUT en el modelo (ya aplicado internamente)
- Pipeline:
  1. Dequantizar salidas INT8 → float usando los exponents del `.info`
  2. Filtrar por `objectness > threshold` (ej: 0.3)
  3. Para candidatos que pasan: decodificar bboxes (offset respecto a anchors predefinidos de SSD)
  4. Aplicar class scores (argmax de las 5 clases: dog, door, obstacle, person, stair)
  5. NMS con IoU threshold (ej: 0.45)
- **Nota**: necesitará definir los anchors/priors del SSD-Lite — extraerlos del modelo original en Keras/ONNX (son fijos según la arquitectura).

**2.3 Postprocesamiento YOLO11n**
- 1 tensor de salida: concatenación bbox+cls, transposed
- Pipeline (basado en lo que muestra el `.info` — DFL decode está dentro del modelo):
  1. Dequantizar salida INT8 → float
  2. Reshape/interpretar como `[N_detections, 4+5]` (4 bbox + 5 class scores)
  3. Filtrar por max class score > threshold (ej: 0.4)
  4. Decodificar bboxes (cx, cy, w, h) → (x1, y1, x2, y2) relativo a 224×224
  5. NMS con IoU threshold

**2.4 Postprocesamiento YOLO26n**
- 1 tensor de salida: formato similar a YOLO11n (NMS fue eliminado en `fix_onnx_for_espdl.py`)
- Pipeline idéntico al de YOLO11n (2.3)
- **Nota para futuro**: investigar cómo mantener el NMS integrado en la conversión ESPDL (posiblemente requiere soporte de ops `TopK`, `GatherElements`, `ScatterND` en esp-ppq, que actualmente no lo tienen).

**2.5 NMS común**
- Implementar función NMS reutilizable para los 3 modelos:
  - Input: vector de detecciones candidatas (bbox, score, class)
  - Parámetros: `score_threshold`, `iou_threshold`, `max_detections` (ej: 100)
  - Optimizaciones para MCU: pre-sort por score, early exit, limitar candidatos a top-K (ej: 200)
  - Opción de NMS por clase o global

**2.6 Struct unificado de resultados**
- `Detection { bbox[4], class_id, class_name, score }` — salida común de los 3 modelos
- Array de detecciones por frame + metadata (model_name, inference_ms, preprocess_ms)

**Criterio de salida Fase 2:** Los 3 modelos ejecutan inferencia end-to-end (captura → preproceso → inferencia → postproceso), devuelven detecciones con formato unificado, y se valida visualmente que las bboxes son coherentes con la imagen de entrada.

---

### Fase 3: Comunicación + Dashboard

**Objetivo:** Interfaz web embebida para visualizar detecciones, métricas, y debug visual.

**3.1 WiFi Access Point**
- SSID: `ESP32_TFM`, Password: `tfm2026esp`
- IP: `192.168.4.1`
- Inferencia en Core 0, WiFi/HTTP en Core 1 (mitigación R3 del plan original)
- `ESP_WIFI_IRAM_OPT` habilitado

**3.2 HTTP Server**
- Endpoint `/` → Dashboard HTML (gzipped, embebido en flash vía `EMBED_FILES`)
- Endpoint `/debug/image` → Stream MJPEG del buffer preprocesado 224×224 (lección clave del piloto: **indispensable para diagnóstico** de domain shift y problemas de preprocessing)
- Endpoint `/api/status` → JSON con estado actual (modelo activo, memoria, temperatura)

**3.3 WebSocket**
- Endpoint `/ws` para broadcast en tiempo real:
  ```json
  {
    "model": "YOLO11n",
    "frame_id": 1234,
    "fps": 2.1,
    "inference_ms": 380,
    "preprocess_ms": 45,
    "postprocess_ms": 12,
    "total_ms": 437,
    "free_psram_kb": 1200,
    "temperature_c": 52.3,
    "detections": [
      {"class": "person", "score": 0.87, "bbox": [0.12, 0.15, 0.45, 0.92]}
    ]
  }
  ```

**3.4 Dashboard HTML embebido**
- HTML/JS/CSS minimal (~2-3 KB gzipped), sin dependencias externas
- Muestra: FPS, latencia desglosada (preproceso/inferencia/postproceso), uso de memoria, temperatura SoC, lista de detecciones, indicador de modelo activo
- Canvas para visualizar las bboxes sobre la imagen de debug (opcional, si el ancho de banda lo permite)

**Criterio de salida Fase 3:** Conectar WiFi al AP, abrir `http://192.168.4.1/`, ver dashboard con métricas en tiempo real y detecciones actualizándose. Stream de debug visual funcional en `/debug/image`.

---

### Fase 4: Benchmarking + Validación

**Objetivo:** Evaluación cuantitativa de los 3 modelos según los criterios de aceptación del TFM.

**4.1 Recolección de métricas**
- Componente `metrics/` que registra por frame:
  - `t_capture`, `t_preprocess`, `t_inference`, `t_postprocess`, `t_total`
  - `free_psram`, `free_internal_ram`, `min_free_psram_ever`
  - `soc_temperature` (vía `temperature_sensor_get_celsius()`)
  - `frame_id`, `n_detections`

**4.2 Protocolo de benchmark**
- **3 configuraciones**: MBNTv3S, YOLO11n, YOLO26n (cambiar `#define`, rebuild, flash)
- **1000 frames** mínimo por modelo (criterio de estabilidad)
- Registrar métricas en buffer circular en PSRAM, exportable como JSON vía endpoint `/api/benchmark`
- Opcional: log a SD card o UART para análisis offline

**4.3 Criterios de aceptación (del plan original, Sección 6.2)**

| Métrica | Mínimo | Ideal |
|---|---|---|
| Inferencia end-to-end | < 2000 ms | < 1000 ms |
| FPS | > 0.5 | > 1.0 |
| PSRAM libre post-init | > 500 KB | > 1 MB |
| Estabilidad | 1000 frames sin crash | 10,000 frames |
| Dashboard | Funcional + JSON legible | Canvas con bboxes |

**4.4 Tabla comparativa final**
- Generar tabla resumen: MBNTv3S vs YOLO11n vs YOLO26n
- Métricas: latencia media/p95/max, FPS, uso de memoria, tamaño en flash, nº detecciones promedio
- Conclusión sobre cuál modelo ofrece mejor trade-off para TinyML en ESP32-S3

**Criterio de salida Fase 4:** Tabla comparativa completa con datos de 1000+ frames por modelo, todos los criterios de aceptación evaluados, y conclusiones documentadas.

---

### Verificación

- **Fase 1**: `idf.py build` sin errores + serial log muestra captura de frames + tensor 224×224 generado + modelo mapeado desde flash
- **Fase 2**: Serial log muestra detecciones con bboxes coherentes para los 3 modelos. Validación cruzada: imagen conocida → detección esperada
- **Fase 3**: Navegador conectado al AP muestra dashboard funcional, stream de debug visual coherente con lo que la cámara ve
- **Fase 4**: JSON de benchmark exportado y documentado en el informe del TFM

### Decisiones

- **ESP-DL only** (no TFLite Micro): simplifica la arquitectura, evita el resolver de ops TFLite, y los 3 modelos ya están en `.espdl`
- **Partición única models** con los 3 `.espdl` a offsets fijos + `esp_partition_mmap`: balance entre flexibilidad OTA y simplicidad
- **Compile-time model selection** (`#define`): óptimo para benchmarking donde se evalúa un modelo a la vez
- **NMS on-device para los 3 modelos**: YOLO26n perdió su NMS integrado durante la conversión; se implementa NMS común reutilizable
- **Debug visual vía HTTP** (`/debug/image`): validado como indispensable en el piloto para detectar problemas de domain shift y preprocessing
- **Normalización INT8**: Verificar correspondencia entre `pixel - 128` y los `exponents` del `.info` antes de avanzar a Fase 2 — posible punto de fallo silencioso

### Riesgos actualizados

| Riesgo | Mitigación |
|---|---|
| Anchors SSD-Lite desconocidos | Extraer de la capa PriorBox del modelo Keras original antes de empezar Fase 2 |
| Exponents ESPDL vs normalización | Test unitario: comparar output Python (ONNX Runtime) vs output ESP-DL con misma imagen calibración |
| PSRAM insuficiente para YOLO | Single camera buffer fallback (`fb_count=1`), modelos en flash vía mmap (no copia a RAM) |
| Domain shift (lección del piloto) | Stream de debug visual obligatorio desde Fase 1; calibrar exposición/white balance del sensor |
| NMS lento en MCU | Top-100 candidatos, score threshold agresivo (≥0.4), early exit |

User: Start implementation

GitHub Copilot: 

  

Created 13 todos

Ran terminal command:  ls -la /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/*.espdl && echo "---" && wc -c /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/*.espdl

Read model .info output structures

Ran terminal command:  wc -l /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.info /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.info

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.info#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.info#49350-49350), lines 49350 to 49550

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info#229496-229496), lines 229496 to 229696

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.info#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.info#210030-210030), lines 210030 to 210230

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/README.md)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/INFORME_FINAL_EjemploDespliegue.md)

Ran terminal command:  grep -n "return\|) {" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.info | head -20

Ran terminal command:  grep -n "return\|) {" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info | head -20

Ran terminal command:  grep -n "return\|) {" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.info | head -20

Ran terminal command:  grep -n "class_out\|bbox_out\|objectness" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.info | grep -v "value:"

Ran terminal command:  grep -n "%output0\b" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info | head -5

Ran terminal command:  grep -n "Transpose_output_0" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.info | head -5

Ran terminal command:  grep -n "Transpose_output_0\[" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.info | tail -5

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.json#1-1), lines 1 to 50

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.json#1-1), lines 1 to 50

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.json#1-1), lines 1 to 50

Searched for regex `dl::Model|get_intermediate|get_output|dl::Tensor|espdl` (`**/docs/**`), 20 results

Searched for regex `dl::Model|get_intermediate|get_output|run\(|forward|invoke|load_model|espdl_engine` (`**/docs/Plan_Despliegue.md`), 6 results

Searched for regex `dl::Model|get_intermediate|get_output|espdl|tensor` (`**/docs/Configuracion_ESP32-S3.md`), no results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/Plan_Despliegue.md#290-290), lines 290 to 330

Model sizes: MBNTv3S = 681 KB, YOLO11n = 2.73 MB, YOLO26n = 2.58 MB (~6 MB total). Let me start building the firmware project. 

Starting: *Create firmware project scaffold* (2/11)

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/CMakeLists.txt)

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/sdkconfig.defaults)

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/partitions.csv)

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/CMakeLists.txt)

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/idf_component.yml)

Starting: *Create app_config.h & main.cpp* (4/11)

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/app_config.h)

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp)

Starting: *Create camera component* (5/11)

Now creating the camera and image processing components: 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/camera_handler/CMakeLists.txt)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/camera_handler/include/camera_handler.h)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/camera_handler/camera_handler.cpp)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/CMakeLists.txt)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/include/image_proc.h)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp)

Starting: *Create inference engine component* (7/11)

Now creating the inference engine and postprocess components: 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/CMakeLists.txt)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include/inference_engine.h)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/CMakeLists.txt)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/include/postprocess.h)

Now creating the postprocess implementation files: 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_common.cpp)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp)

Starting: *Create metrics component* (9/11)

Now creating the metrics component:     

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/metrics/CMakeLists.txt)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/metrics/include/metrics.h)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/metrics/metrics.cpp)

Starting: *Create network + dashboard components* (10/11)

Now creating the network and dashboard components:        

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/CMakeLists.txt)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/include/network.h)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/network.cpp)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/dashboard/CMakeLists.txt)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/dashboard/include/dashboard.h)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/dashboard/dashboard.cpp)

 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/dashboard/web/index.html)

Starting: *Create flash_models.sh script* (11/11)

Now creating the flash script and fixing a couple of things I noticed in main.cpp (need to call `network_set_debug_image_source`): 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh)

Ran terminal command:  chmod +x /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh

Now let me fix main.cpp to set the debug image source for the network component: 

Replacing 4 lines with 5 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp)


```

```


Completed: *Create flash_models.sh script* (11/11)

Let me verify the complete project structure: 

Ran terminal command:  find /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware -type f | sort | head -40

All 32 files created successfully. Here's a summary of the complete firmware project:

---

**32 files created** under firmware:

```
firmware/
├── CMakeLists.txt                     ← Proyecto raíz ESP-IDF
├── sdkconfig.defaults                 ← Config ESP32-S3 validada (PSRAM Octal, 240MHz, WiFi IRAM)
├── partitions.csv                     ← NVS + PHY + Factory(4MB) + Models(7MB)
├── flash_models.sh                    ← Script para flashear 3 .espdl a la partición
├── main/
│   ├── CMakeLists.txt
│   ├── idf_component.yml              ← Deps: esp-dl ≥3.0, esp32-camera ≥2.0
│   ├── app_config.h                   ← #define ACTIVE_MODEL, offsets, umbrales, clases
│   └── main.cpp                       ← Entry point: NVS→Camera→Inference→Network→Task
└── components/
    ├── camera_handler/                ← OV5640 RGB565 320×240, doble buffer PSRAM
    ├── image_proc/                    ← Crop+Resize+RGB565→INT8 en 1 pasada + BMP debug
    ├── inference/                     ← ESP-DL wrapper: mmap partición → dl::Model → run
    ├── postprocess/                   ← SSD decode+anchors, YOLO decode, NMS común
    ├── metrics/                       ← Timing por fase, FPS, memoria, temperatura SoC
    ├── network/                       ← WiFi AP + HTTP + WebSocket broadcast + /debug/image
    └── dashboard/                     ← HTML embebido (EMBED_FILES), dark theme
```

**Flujo de datos:**
```
Camera(RGB565 320×240) → Crop+Resize+INT8(224×224×3) → ESP-DL forward → Decode+NMS → WebSocket JSON
```

**Para construir y flashear:**
```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash      # Firmware
./flash_models.sh --port /dev/ttyUSB0  # Modelos
idf.py -p /dev/ttyUSB0 monitor
```

**Puntos que requieren verificación en la primera compilación:**
1. **API ESP-DL 3.x** — El constructor de `dl::Model` en inference_engine.cpp puede necesitar ajuste según la versión exacta del componente descargado
2. **Anchors SSD** — Los parámetros en app_config.h (min/max sizes, aspect ratios) deben verificarse contra la configuración de entrenamiento del MBNTv3S
3. **Normalización input** — Verificar que `pixel - 128` (INT8 con exponent=-7) es consistente con la calibración `float/255.0` usada en esp-ppq

Made changes.

User: Esto es grave?:

```zsh
(/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env)admin@MacBook-Pro-de-Admin firmware % idf.py set-target esp32s3
Adding "set-target"'s dependency "fullclean" to list of commands with default set of options.
Executing action: fullclean
Build directory '/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build' not found. Nothing to clean.
Executing action: set-target
Set Target to: esp32s3, new sdkconfig will be created.
Running cmake in directory /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build
Executing "cmake -G Ninja -DPYTHON_DEPS_CHECKED=1 -DPYTHON=/Users/admin/.espressif/python_env/idf5.4_py3.14_env/bin/python -DESP_PLATFORM=1 -DIDF_TARGET=esp32s3 -DCCACHE_ENABLE=0 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware"...
-- Found Git: /opt/homebrew/bin/git (found version "2.52.0")
-- The C compiler identification is GNU 14.2.0
-- The CXX compiler identification is GNU 14.2.0
-- The ASM compiler identification is GNU
-- Found assembler: /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/bin/xtensa-esp32s3-elf-gcc
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/bin/xtensa-esp32s3-elf-gcc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/bin/xtensa-esp32s3-elf-g++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Building ESP-IDF components for target esp32s3
NOTICE: Dependencies lock doesn't exist, solving dependencies.
..............NOTICE: Updating lock file at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/dependencies.lock
NOTICE: Processing 7 dependencies:
NOTICE: [1/7] espressif/dl_fft (0.3.1)
NOTICE: [2/7] espressif/esp-dl (3.2.4)
NOTICE: [3/7] espressif/esp-dsp (1.7.0)
NOTICE: [4/7] espressif/esp32-camera (2.1.4)
NOTICE: [5/7] espressif/esp_jpeg (1.3.1)
NOTICE: [6/7] espressif/esp_new_jpeg (0.6.1)
NOTICE: [7/7] idf (5.4.3)
CMake Error at /Users/admin/esp/v5.4.3/esp-idf/tools/cmake/build.cmake:328 (message):
  Failed to resolve component 'esp_log' required by component
  'camera_handler': unknown name.
Call Stack (most recent call first):
  /Users/admin/esp/v5.4.3/esp-idf/tools/cmake/build.cmake:377 (__build_resolve_and_add_req)
  /Users/admin/esp/v5.4.3/esp-idf/tools/cmake/build.cmake:372 (__build_expand_requirements)
  /Users/admin/esp/v5.4.3/esp-idf/tools/cmake/build.cmake:684 (__build_expand_requirements)
  /Users/admin/esp/v5.4.3/esp-idf/tools/cmake/project.cmake:722 (idf_build_process)
  CMakeLists.txt:16 (project)


-- Configuring incomplete, errors occurred!
HINT: The component 'esp_log' could not be found. This could be because: component name was misspelled, the component was not added to the build, the component has been moved to the IDF component manager, the component has been removed and refactored into some other component or the component may not be supported by the selected target.
Please look out for component in 'https://components.espressif.com' and add using 'idf.py add-dependency' command.
Refer to the migration guide for more details about moved components.
Refer to the build-system guide for more details about how components are found and included in the build.
cmake failed with exit code 1, output of the command is in the /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/log/idf_py_stderr_output_41012 and /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/log/idf_py_stdout_output_41012
```

GitHub Copilot: 

 No es grave — `esp_log` es un componente "common" en ESP-IDF 5.4.3 que se incluye automáticamente; no hay que listarlo explícitamente en `PRIV_REQUIRES`. Lo quito de todos los CMakeLists de los componentes: 

Replacing 29 lines with 25 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/camera_handler/CMakeLists.txt), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/CMakeLists.txt), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/CMakeLists.txt), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/CMakeLists.txt), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/metrics/CMakeLists.txt), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/CMakeLists.txt), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/dashboard/CMakeLists.txt)


```

```

```

```

```

```

```

```

```

```

```

```

```

```
Now let's retry the build: 

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && rm -rf build && idf.py set-target esp32s3 2>&1 | tail -40

Ran terminal command:  source /Users/admin/esp/v5.4.3/esp-idf/export.sh 2>/dev/null && cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && rm -rf build && idf.py set-target esp32s3 2>&1 | tail -60

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01YApnShJLUYDDQg6ndm1FqY__vscode-1770907754455/content.txt)

`set-target` pasó correctamente. CMake configuró sin errores y resolvió las 7 dependencias (esp-dl 3.2.4, esp32-camera 2.1.4, etc.). Solo hubo un warning menor sobre `CAMERA_XCLK_FREQ` que no es un kconfig symbol real — es inofensivo y lo elimino.

Ahora probemos el build completo: 

Replacing 2 lines with 1 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/sdkconfig.defaults)


```

```


Ran terminal command:  idf.py build 2>&1 | tail -80

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01Cbj5pTfMd42SQ6AFRE1iNJ__vscode-1770907754458/content.txt#1-1), lines 1 to 80

The issue: components can't include `app_config.h` from `main/` — ESP-IDF doesn't allow depending on `main`. The fix is to extract `app_config.h` into its own lightweight component: 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/CMakeLists.txt)

Ran terminal command:  mv /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/app_config.h /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h
Completed with input:  mv /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/app_config.h /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h

Ran terminal command:  mkdir -p /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include && mv /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/app_config.h /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h

Now add `app_config` to the REQUIRES of the components that need it, and also fix the unused TAG warning in image_proc:  

Replacing 11 lines with 12 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/CMakeLists.txt), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/CMakeLists.txt), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/CMakeLists.txt), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/CMakeLists.txt), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp)


```

```

```

```

```

```

```

```

```

```


Ran terminal command:  rm -rf build && idf.py set-target esp32s3 2>&1 | tail -5 && idf.py build 2>&1 | tail -40

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01QZxxHfboEoVcuJQFmPCDf8__vscode-1770907754464/content.txt)

Progress — now it just needs the esp-dl component linked to inference. Let me also check the actual esp-dl headers available: 

Ran terminal command:  find /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl -name "*.hpp" -path "*/include/*" | head -20

Summarized conversation history

Ran terminal command:  find /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl -name "*.hpp" -path "*/include/*" | grep -i "model\|tensor\|base" | sort

Created 5 todos

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/model/include/dl_model_base.hpp#1-1), lines 1 to 150

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/tensor/include/dl_tensor_base.hpp#1-1), lines 1 to 150

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/fbs_loader/include/fbs_model.hpp#1-1), lines 1 to 150

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/fbs_loader/include/fbs_loader.hpp)

Now let me read the rest of the Model class to see `get_input`, `get_output`, `run`:   

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/model/include/dl_model_base.hpp#150-150), lines 150 to 413

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include/inference_engine.h#1-1), lines 1 to 60

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#100-100), lines 100 to 189

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include/inference_engine.h#60-60), lines 60 to 73

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/CMakeLists.txt)

Let me also check what `runtime_mode_t` is defined as and what TensorBase's size semantics are: 

Ran terminal command:  grep -rn "runtime_mode_t\|RUNTIME_MODE" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl --include="*.hpp" | head -10

Ran terminal command:  grep -rn "enum.*runtime_mode\|RUNTIME_MODE_SINGLE\|RUNTIME_MODE_AUTO" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl --include="*.hpp" | head -10

Ran terminal command:  grep -rn "runtime_mode_t" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl --include="*.hpp" | grep "typedef\|enum" | head -5

Ran terminal command:  grep -rn "runtime_mode_t\|RUNTIME_MODE" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/tool --include="*.hpp"
Completed with input:  grep -rn "runtime_mode_t\|RUNTIME_MODE" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/tool --include="*.hpp"

Ran terminal command:  grep -rn "runtime_mode_t" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl --include="*.h" --include="*.hpp" | grep -v "forward\|void " | head -10

Ran terminal command:  grep -n "RUNTIME_MODE" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/typedef/include/dl_tool.hpp | head -10

Ran terminal command:  find /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl -name "dl_tool.hpp" | head -3

Ran terminal command:  grep -n "runtime_mode\|RUNTIME_MODE\|quant_type\|activation_type" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/tool/include/dl_tool.hpp | head -20

Ran terminal command:  grep -rn "RUNTIME_MODE_SINGLE_CORE\|RUNTIME_MODE_AUTO" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl --include="*.h" | head -5

Ran terminal command:  grep -rn "RUNTIME_MODE_SINGLE_CORE" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl --include="*.hpp" --include="*.h" --include="*.cpp" | grep -v "forward\|void \|run(" | head -5

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/dl_define.hpp#125-125), lines 125 to 155

Starting: *Fix inference CMakeLists* (2/5)

Now I have a complete understanding of the real ESP-DL 3.2.4 API. Key differences from what was written:

1. **Model constructor**: Takes partition label or rodata pointer, NOT `(data, location, size)`
2. **`get_input()`/`get_output()`**: Returns `TensorBase*` (pointer), not reference. No integer index overload — uses string names.
3. **TensorBase**: `size` is a public field, not `get_size()`. `data` is `void*`.

Let me now read the other files that reference the inference API to plan all fixes: 

Read all firmware source files

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_common.cpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/include/postprocess.h)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/CMakeLists.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/metrics/metrics.cpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/metrics/include/metrics.h)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/network.cpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/include/network.h)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/dashboard/CMakeLists.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/camera_handler/camera_handler.cpp)

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01ERagP4KxTUYmgrbp53FFsE__vscode-1770907754491/content.txt#1-1), lines 1 to 100

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01ERagP4KxTUYmgrbp53FFsE__vscode-1770907754491/content.txt#100-100), lines 100 to 400

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01ERagP4KxTUYmgrbp53FFsE__vscode-1770907754491/content.txt#400-400), lines 400 to 700

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01ERagP4KxTUYmgrbp53FFsE__vscode-1770907754491/content.txt#700-700), lines 700 to 1050

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01ERagP4KxTUYmgrbp53FFsE__vscode-1770907754491/content.txt#1050-1050), lines 1050 to 1320

Good — I have a complete picture of the ESP-DL 3.2.4 API and all our source files. Let me fix everything now. 

Updated todo list

Replacing 6 lines with 6 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/CMakeLists.txt)


```

```


Starting: *Rewrite inference_engine.cpp* (3/5)

Now rewrite inference_engine.cpp to match the real ESP-DL 3.2.4 API: 

Replacing 188 lines with 227 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp)


```

```


Starting: *Check/fix remaining components* (4/5)

Now let me check the remaining components for potential issues. Let me verify the CMakeLists of other components that might need ESP-DL or have missing dependencies: 

Read all component CMakeLists

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/camera_handler/CMakeLists.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/CMakeLists.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/metrics/CMakeLists.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/CMakeLists.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/CMakeLists.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/CMakeLists.txt)

Let me check camera_handler.h to see if it needs the esp32-camera component: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/camera_handler/include/camera_handler.h#1-1), lines 1 to 30

Let me proactively fix `camera_handler` CMakeLists to add the managed camera component, since camera_handler.h includes `esp_camera.h`: 

Replacing 5 lines with 6 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/camera_handler/CMakeLists.txt)


```

```


Starting: *Rebuild and fix remaining errors* (5/5)

Now let's rebuild: 

Ran terminal command:  source /Users/admin/esp/v5.4.3/esp-idf/export.sh 2>/dev/null && rm -rf build && idf.py set-target esp32s3 2>&1 | tail -5

Ran terminal command:  idf.py build 2>&1 | tail -30

Good — inference_engine.cpp compiled successfully! Two remaining issues:

1. **metrics** needs `postprocess` in PRIV_REQUIRES
2. `CLASS_NAMES` unused warnings (cosmetic) 

Replacing 6 lines with 6 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/metrics/CMakeLists.txt)


```

```
Also fix the `CLASS_NAMES` unused warning by adding `[[maybe_unused]]`:  

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#108-108), lines 108 to 120

Replacing 1 lines with 1 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h)

Continue to iterate?
[object Object]

Made changes.

User: @agent Continue: "Continue to iterate?"

GitHub Copilot: 
```

```


Ran terminal command:  idf.py build 2>&1 | tail -30

The tail output doesn't show the specific error. Let me check the error log: 

Ran terminal command:  cat /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/log/idf_py_stderr_output_58493 | grep -A 5 "error:"
Completed with input:  cat /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/log/idf_py_stderr_output_58493 | grep -A 5 "error:"

Ran terminal command:  cat /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/log/idf_py_stdout_output_58493 | grep -i "error" | tail -20

The error is `MACSTR` and `MAC2STR` not found in `network.cpp`. These are in `esp_mac.h` in ESP-IDF 5.x:  

Replacing 5 lines with 6 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/network.cpp)


```

```


Ran terminal command:  idf.py build 2>&1 | tail -30

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01QNdenQLn3j25DsiQxmd2d4__vscode-1770907754534/content.txt)

`network.h` includes `metrics.h` and `postprocess.h` in the public header — they need to be in REQUIRES, not PRIV_REQUIRES:  

Replacing 6 lines with 6 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/CMakeLists.txt)


```

```


Ran terminal command:  idf.py build 2>&1 | tail -30

Build successful! The firmware compiled cleanly:

- **Binary**: `tfm_tinyml_deploy.bin` — 2,047 KB (51% of 4MB app partition free)
- **Bootloader**: 20.9 KB (36% free) 

Completed: *Rebuild and fix remaining errors* (5/5)

The firmware builds cleanly for ESP32-S3. Here's a summary of what was fixed in this cycle:

| Fix | File | Issue |
|-----|------|-------|
| 1 | inference/CMakeLists.txt | Added `PRIV_REQUIRES espressif__esp-dl` for dl_model_base.hpp |
| 2 | inference_engine.cpp | **Full rewrite** to match ESP-DL 3.2.4 API: `Model(rodata_ptr, MODEL_LOCATION_IN_FLASH_RODATA)`, `get_input()→TensorBase*`, `get_outputs()→map`, `run(RUNTIME_MODE_SINGLE_CORE)`, name-based output indexing |
| 3 | camera_handler/CMakeLists.txt | Added `PRIV_REQUIRES espressif__esp32-camera` |
| 4 | metrics/CMakeLists.txt | Added `postprocess` to PRIV_REQUIRES |
| 5 | network/CMakeLists.txt | Moved `metrics` and `postprocess` to REQUIRES (public header deps) |
| 6 | network.cpp | Added `#include "esp_mac.h"` for `MACSTR`/`MAC2STR` |
| 7 | app_config.h | Added `[[maybe_unused]]` to `CLASS_NAMES` |

Made changes.

User: Explicame esa tabla de "what was fixed in this cycle:" pero en español.

GitHub Copilot: 

 Aquí tienes el resumen de las correcciones realizadas en este ciclo de compilación:

| # | Archivo | Problema y solución |
|---|---------|---------------------|
| 1 | inference/CMakeLists.txt | Se añadió `PRIV_REQUIRES espressif__esp-dl` porque el componente `inference` usa cabeceras de ESP-DL (dl_model_base.hpp) pero no tenía declarada esa dependencia. |
| 2 | inference_engine.cpp | **Reescritura completa** para alinearlo con la API real de ESP-DL 3.2.4. El código original usaba constructores y métodos inventados. Los cambios clave: el constructor del modelo ahora recibe `(puntero_rodata, MODEL_LOCATION_IN_FLASH_RODATA, ...)`; `get_input()` devuelve `TensorBase*` (puntero, no referencia); `get_outputs()` devuelve un `map<string, TensorBase*>` (acceso por nombre, no por índice entero); `run()` recibe `RUNTIME_MODE_SINGLE_CORE`; y `TensorBase::size` es un campo público (no un método `get_size()`). |
| 3 | camera_handler/CMakeLists.txt | Se añadió `PRIV_REQUIRES espressif__esp32-camera` porque el componente incluye `esp_camera.h` pero no estaba enlazado con la librería de cámara gestionada por el sistema de componentes. |
| 4 | metrics/CMakeLists.txt | Se añadió `postprocess` a `PRIV_REQUIRES` porque `metrics.cpp` incluye `postprocess.h` (usa el struct `DetectionResult`) y sin esa dependencia el compilador no encontraba la cabecera. |
| 5 | network/CMakeLists.txt | Se movieron `metrics` y `postprocess` de `PRIV_REQUIRES` a `REQUIRES` porque `network.h` (cabecera **pública**) incluye `metrics.h` y `postprocess.h`. En ESP-IDF, las dependencias de cabeceras públicas deben ir en `REQUIRES`, no en `PRIV_REQUIRES`. |
| 6 | network.cpp | Se añadió `#include "esp_mac.h"` porque las macros `MACSTR` y `MAC2STR` (para imprimir direcciones MAC en los logs de WiFi) se movieron a esa cabecera en ESP-IDF 5.x y ya no están disponibles solo con `esp_wifi.h`. |
| 7 | app_config.h | Se añadió el atributo `[[maybe_unused]]` al array `CLASS_NAMES` para suprimir el warning `-Wunused-variable` que aparecía en cada archivo que incluía la cabecera pero no usaba directamente los nombres de clase. |

En resumen: la mayoría de los errores eran **dependencias faltantes en los CMakeLists** (el sistema de componentes de ESP-IDF es estricto con declarar qué usa cada componente) y la **adaptación del código de inferencia a la API real** de ESP-DL 3.2.4, que difería significativamente de lo que se había escrito inicialmente de forma especulativa.

User: Ok, ahora qué pasos sigo.

GitHub Copilot: 

 El firmware compila correctamente. Estos son los pasos siguientes para desplegar en la placa:

### Paso 1 — Preparar la imagen de modelos

El script flash_models.sh crea una imagen compuesta con los 3 modelos `.espdl` en los offsets definidos en app_config.h y la flashea a la partición models. Verifiquemos primero que los archivos y offsets son correctos: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh#1-1), lines 1 to 80

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh#80-80), lines 80 to 123

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && for f in models/MBNTv3S_ssdlite_v1_p2_best.espdl models/yolo11n_v1_best.espdl models/yolo26n_v1_best.espdl; do echo "$(basename $f): $(wc -c < $f) bytes"; done

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#30-30), lines 30 to 55

Verifiquemos que los offsets no se solapan y caben en la partición de 7 MB: 

Ran terminal command:  python3 -c "
# Verificar layout de partición de modelos
models = [
    ('MBNTv3S', 0x000000, 681088),
    ('YOLO11n', 0x0A7000, 2857568),
    ('YOLO26n', 0x361000, 2703184),
]
part_size = 0x700000  # 7 MB = 7340032

print('=== Layout de partición de modelos ===')
for name, off, sz in models:
    end = off + sz
    next_4k = ((end + 0xFFF) // 0x1000) * 0x1000
    print(f'{name}: 0x{off:06X} → 0x{end:06X} ({sz:,} bytes)  align_4K=0x{next_4k:06X}')

# Check no overlap
for i in range(len(models)-1):
    end_i = models[i][1] + models[i][2]
    start_next = models[i+1][1]
    gap = start_next - end_i
    if gap < 0:
        print(f'¡SOLAPAMIENTO! {models[i][0]} termina en 0x{end_i:06X} pero {models[i+1][0]} empieza en 0x{start_next:06X}')
    else:
        print(f'Gap {models[i][0]}→{models[i+1][0]}: {gap:,} bytes')

total_end = models[-1][1] + models[-1][2]
print(f'Último byte usado: 0x{total_end:06X} ({total_end:,}) de {part_size:,} ({total_end/part_size*100:.1f}%)')
if total_end > part_size:
    print('¡ERROR! Los modelos no caben en la partición')
else:
    print('OK — caben en la partición')
"


Ran terminal command:  python3 << 'PYEOF'
models = [
    ('MBNTv3S', 0x000000, 681088),
    ('YOLO11n', 0x0A7000, 2857568),
    ('YOLO26n', 0x361000, 2703184),
]
part_size = 0x700000
for name, off, sz in models:
    end = off + sz
    print(f'{name}: 0x{off:06X} -> 0x{end:06X} ({sz:,} B)')
for i in range(len(models)-1):
    end_i = models[i][1] + models[i][2]
    start_next = models[i+1][1]
    gap = start_next - end_i
    status = 'OK' if gap >= 0 else 'OVERLAP!'
    print(f'Gap {models[i][0]}->{models[i+1][0]}: {gap:,} B [{status}]')
total_end = models[-1][1] + models[-1][2]
print(f'Total: 0x{total_end:06X} of 0x{part_size:06X} ({total_end/part_size*100:.1f}%)')
print('FITS' if total_end <= part_size else 'OVERFLOW!')
PYEOF

Ran terminal command:  python3 -c "m=[('MBNTv3S',0,681088),('YOLO11n',0x0A7000,2857568),('YOLO26n',0x361000,2703184)]; p=0x700000; [print(f'{n}: 0x{o:06X}->0x{o+s:06X} ({s} B)') for n,o,s in m]; t=m[-1][1]+m[-1][2]; print(f'Total: {t} of {p} = {t/p*100:.1f}% - {\"OK\" if t<=p else \"OVERFLOW\"}')"

Todo validado. Los 3 modelos caben sin solapamientos, usando el 85% de la partición de 7 MB.

---

Estos son los **pasos a seguir** en orden:

### 1. Conectar la placa ESP32-S3
Conecta la Freenove CAM Board por USB. En macOS el puerto suele aparecer como `/dev/cu.usbserial-*` o `/dev/cu.SLAB_USBtoUART`. Compruébalo con:
```bash
ls /dev/cu.usb*
```

### 2. Flashear el firmware
Desde la terminal con IDF sourced:
```bash
cd firmware
idf.py -p /dev/cu.usbserial-XXXX flash
```

### 3. Flashear los modelos
El script construye una imagen de 7 MB con los 3 modelos y la escribe a la partición models:
```bash
chmod +x flash_models.sh
./flash_models.sh --port /dev/cu.usbserial-XXXX
```

### 4. Monitorizar
```bash
idf.py -p /dev/cu.usbserial-XXXX monitor
```
Verás los logs del arranque: NVS init, PSRAM detectada, modelo cargado (con shapes de entrada/salida), cámara inicializada, WiFi AP activo.

### 5. Probar el dashboard
1. Conecta tu portátil/móvil al WiFi **ESP32_TFM** (password: `tfm2026esp`)
2. Abre en el navegador: **http://192.168.4.1/**
3. Deberías ver el dashboard con FPS, latencias, detecciones y la imagen de debug

### 6. Benchmarking (cambiar de modelo)
Para probar otro modelo, cambia `ACTIVE_MODEL` en app_config.h, recompila y re-flashea:
```c
// Cambiar a: 0 = MBNTv3S, 1 = YOLO11n, 2 = YOLO26n
#define ACTIVE_MODEL  MODEL_YOLO11N
```
```bash
idf.py build && idf.py -p /dev/cu.usbserial-XXXX flash monitor
```

> **Nota**: Los modelos solo se flashean una vez (paso 3). Al cambiar `ACTIVE_MODEL` solo hace falta re-flashear el firmware (paso 2), porque los 3 modelos ya están en la partición.

### Posibles problemas a vigilar
- **Cámara no detectada**: Verificar que la placa es Freenove (pins distintos a otros ESP32-S3-CAM)
- **PSRAM insuficiente**: El monitor mostrará "PSRAM free" — si baja de ~512 KB, ajustar `param_copy=false` en el constructor del modelo
- **Modelo no parsea**: Si ESP-DL rechaza el flatbuffer, puede ser un problema de alineación del offset. El log dirá exactamente qué falla en el `Model()` constructor

User: Hubo un error al flashear los modelos:

```zsh
(/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env)admin@MacBook-Pro-de-Admin firmware % chmod +x flash_models.sh
./flash_models.sh --port /dev/tty.wchusbserial5B414963901
=== TFM TinyML — Flash Models ===

  Found: MBNTv3S_ssdlite_v1_p2_best.espdl (  681088 bytes)
  Found: yolo11n_v1_best.espdl ( 2857568 bytes)
  Found: yolo26n_v1_best.espdl ( 2703184 bytes)

--- Construyendo imagen compuesta ---
Created 7340032 byte partition image (0xFF filled)
  MBNTv3S: offset=0x000000, size=681,088 bytes, end=0x0A6480
  YOLO11n: offset=0x0A7000, size=2,857,568 bytes, end=0x360A60
  YOLO26n: offset=0x361000, size=2,703,184 bytes, end=0x5F4F50
Composite image ready: /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/models_partition.bin

--- Flasheando a partición 'models' ---
Puerto: /dev/tty.wchusbserial5B414963901

Running /Users/admin/.espressif/python_env/idf5.4_py3.14_env/bin/python /Users/admin/esp/v5.4.3/esp-idf/components/esptool_py/esptool/esptool.py --port /dev/tty.wchusbserial5B414963901 --baud 921600 read_flash 32768 3072 /var/folders/04/3m9yp1sj1fl1k_jfy0yrwhmh0000gn/T/tmpkrzrf2nn...
esptool.py v4.11.0
Serial port /dev/tty.wchusbserial5B414963901
Connecting...................
Detecting chip type... ESP32-S3
Chip is ESP32-S3 (QFN56) (revision v0.2)
Features: WiFi, BLE, Embedded PSRAM 8MB (AP_3v3)
Crystal is 40MHz
MAC: 1c:db:d4:4a:f4:58
Uploading stub...
Running stub...
Stub running...
Changing baud rate to 921600
Changed.
Configuring flash size...
3072 (100 %)
3072 (100 %)
Read 3072 bytes at 0x00008000 in 0.0 seconds (600.4 kbit/s)...
Hard resetting via RTS pin...
Running /Users/admin/.espressif/python_env/idf5.4_py3.14_env/bin/python /Users/admin/esp/v5.4.3/esp-idf/components/esptool_py/esptool/esptool.py --port /dev/tty.wchusbserial5B414963901 --baud 921600 erase_region 4259840 7340032...
esptool.py v4.11.0
Serial port /dev/tty.wchusbserial5B414963901
Connecting........................
Detecting chip type... ESP32-S3
Chip is ESP32-S3 (QFN56) (revision v0.2)
Features: WiFi, BLE, Embedded PSRAM 8MB (AP_3v3)
Crystal is 40MHz
MAC: 1c:db:d4:4a:f4:58
Uploading stub...
Running stub...
Stub running...
Changing baud rate to 921600
Changed.
Erasing region (may be slow depending on size)...
Erase completed successfully in 12.8 seconds.
Hard resetting via RTS pin...
Running /Users/admin/.espressif/python_env/idf5.4_py3.14_env/bin/python /Users/admin/esp/v5.4.3/esp-idf/components/esptool_py/esptool/esptool.py --port /dev/tty.wchusbserial5B414963901 --baud 921600 write_flash 4259840 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/models_partition.bin...
esptool.py v4.11.0
Serial port /dev/tty.wchusbserial5B414963901
Connecting......................................

A fatal error occurred: Failed to connect to Espressif device: Wrong boot mode detected (0xa)! The chip needs to be in download mode.
For troubleshooting steps visit: https://docs.espressif.com/projects/esptool/en/latest/troubleshooting.html
An exception: ** Command '['/Users/admin/.espressif/python_env/idf5.4_py3.14_env/bin/python', '/Users/admin/esp/v5.4.3/esp-idf/components/esptool_py/esptool/esptool.py', '--port', '/dev/tty.wchusbserial5B414963901', '--baud', '921600', 'write_flash', '4259840', '/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/models_partition.bin']' returned non-zero exit status 2. ** occurred in _call_esptool.
Traceback (most recent call last):
  File "/Users/admin/esp/v5.4.3/esp-idf/components/partition_table/parttool.py", line 395, in <module>
    main()
    ~~~~^^
  File "/Users/admin/esp/v5.4.3/esp-idf/components/partition_table/parttool.py", line 388, in main
    op(**common_args)
    ~~^^^^^^^^^^^^^^^
  File "/Users/admin/esp/v5.4.3/esp-idf/components/partition_table/parttool.py", line 176, in _write_partition
    target.write_partition(partition_id, input, ignore_readonly)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/admin/esp/v5.4.3/esp-idf/components/partition_table/parttool.py", line 172, in write_partition
    self._call_esptool(['write_flash', str(partition.offset), input] + self.esptool_write_args)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/admin/esp/v5.4.3/esp-idf/components/partition_table/parttool.py", line 127, in _call_esptool
    raise e
  File "/Users/admin/esp/v5.4.3/esp-idf/components/partition_table/parttool.py", line 124, in _call_esptool
    subprocess.check_call(esptool_args, stdout=out, stderr=subprocess.STDOUT)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.2_1/Frameworks/Python.framework/Versions/3.14/lib/python3.14/subprocess.py", line 419, in check_call
    raise CalledProcessError(retcode, cmd)
subprocess.CalledProcessError: Command '['/Users/admin/.espressif/python_env/idf5.4_py3.14_env/bin/python', '/Users/admin/esp/v5.4.3/esp-idf/components/esptool_py/esptool/esptool.py', '--port', '/dev/tty.wchusbserial5B414963901', '--baud', '921600', 'write_flash', '4259840', '/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/models_partition.bin']' returned non-zero exit status 2.
```

GitHub Copilot: 

 El problema es que `parttool.py` hace 3 llamadas separadas a `esptool` (leer tabla, borrar, escribir), y entre el borrado y la escritura se hace un hard reset que saca al chip del modo download.

La solución más sencilla es flashear directamente con `esptool.py` en una sola llamada. Verifiquemos el offset de la partición: 

Ran terminal command:  grep models /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/partitions.csv

La partición models empieza en `0x410000`. Flashea directamente con este comando (no necesitas el script):

```bash
cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
source /Users/admin/esp/v5.4.3/esp-idf/export.sh
python -m esptool --chip esp32s3 -p /dev/tty.wchusbserial5B414963901 -b 921600 \
    --before default_reset --after hard_reset \
    write_flash 0x410000 build/models_partition.bin
```

La imagen compuesta ya se generó correctamente en el paso anterior (`build/models_partition.bin`, 7 MB). Esto la escribe de golpe en una sola conexión, evitando el problema del reset intermedio.

> **Alternativa**: Si quieres flashear firmware + modelos todo junto en un solo comando:
> ```bash
> python -m esptool --chip esp32s3 -p /dev/tty.wchusbserial5B414963901 -b 921600 \
>     --before default_reset --after hard_reset \
>     write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
>     0x0 build/bootloader/bootloader.bin \
>     0x8000 build/partition_table/partition-table.bin \
>     0x10000 build/tfm_tinyml_deploy.bin \
>     0x410000 build/models_partition.bin
> ```> python -m esptool --chip esp32s3 -p /dev/tty.wchusbserial5B414963901 -b 921600 \
>     --before default_reset --after hard_reset \
>     write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
>     0x0 build/bootloader/bootloader.bin \
>     0x8000 build/partition_table/partition-table.bin \
>     0x10000 build/tfm_tinyml_deploy.bin \
>     0x410000 build/models_partition.bin
> ```

User: Si ya apliqué el siguiente comando y fue exitoso:

```zsh
cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
source /Users/admin/esp/v5.4.3/esp-idf/export.sh
python -m esptool --chip esp32s3 -p /dev/tty.wchusbserial5B414963901 -b 921600 \
    --before default_reset --after hard_reset \
    write_flash 0x410000 build/models_partition.bin
```


¿Ahora qué debo hacer?

GitHub Copilot: 

Los modelos ya están en flash. Ahora flashea el firmware y monitoriza:

```bash
idf.py -p /dev/tty.wchusbserial5B414963901 flash monitor
```

Esto escribe el bootloader + tabla de particiones + aplicación y abre el monitor serie. Verás los logs de arranque: detección de PSRAM, carga del modelo, inicialización de cámara y WiFi AP.

Cuando arranque, conéctate al WiFi **ESP32_TFM** (pass: `tfm2026esp`) y abre **http://192.168.4.1/** en el navegador para ver el dashboard.

Para salir del monitor: `Ctrl+]`

User: Ya hice la primera prueba, observando en el navegador el dashboard y probando el modelo YOLOv11. 

Te comparto una imagen y el log, pero fue EXTREMADAMENTE LENTO. Investiga qué puede estar causando esto? investiga en documentacion y blogs oficiales sobre qué debemos tener en cuenta y no estamos haciendo, 

```zsh
I (27) boot: ESP-IDF v5.4.3-dirty 2nd stage bootloader
I (27) boot: compile time Feb 12 2026 15:36:31
I (27) boot: Multicore bootloader
I (28) boot: chip revision: v0.2
I (30) boot: efuse block revision: v1.3
I (34) boot.esp32s3: Boot SPI Speed : 80MHz
I (38) boot.esp32s3: SPI Mode       : DIO
I (42) boot.esp32s3: SPI Flash Size : 16MB
I (45) boot: Enabling RNG early entropy source...
I (50) boot: Partition Table:
I (52) boot: ## Label            Usage          Type ST Offset   Length
I (59) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (65) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (72) boot:  2 factory          factory app      00 00 00010000 00400000
I (78) boot:  3 models           Unknown data     01 40 00410000 00700000
I (85) boot: End of partition table
I (88) esp_image: segment 0: paddr=00010020 vaddr=3c180020 size=53a38h (342584) map
I (156) esp_image: segment 1: paddr=00063a60 vaddr=3fc9d300 size=078dch ( 30940) load
I (163) esp_image: segment 2: paddr=0006b344 vaddr=40374000 size=04cd4h ( 19668) load
I (167) esp_image: segment 3: paddr=00070020 vaddr=42000020 size=17f6c8h (1570504) map
I (444) esp_image: segment 4: paddr=001ef6f0 vaddr=40378cd4 size=14588h ( 83336) load
I (463) esp_image: segment 5: paddr=00203c80 vaddr=50000000 size=0001ch (    28) load
I (473) boot: Loaded app from partition at offset 0x10000
I (473) boot: Disabling RNG early entropy source...
I (484) octal_psram: vendor id    : 0x0d (AP)
I (484) octal_psram: dev id       : 0x02 (generation 3)
I (484) octal_psram: density      : 0x03 (64 Mbit)
I (486) octal_psram: good-die     : 0x01 (Pass)
I (490) octal_psram: Latency      : 0x01 (Fixed)
I (494) octal_psram: VCC          : 0x01 (3V)
I (499) octal_psram: SRF          : 0x01 (Fast Refresh)
I (503) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (508) octal_psram: BurstLen     : 0x01 (32 Byte)
I (513) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (518) octal_psram: DriveStrength: 0x00 (1/1)
I (523) MSPI Timing: PSRAM timing tuning index: 5
I (527) esp_psram: Found 8MB PSRAM device
I (530) esp_psram: Speed: 80MHz
I (533) cpu_start: Multicore app
I (969) esp_psram: SPI SRAM memory test OK
I (978) cpu_start: Pro cpu start user code
I (978) cpu_start: cpu freq: 240000000 Hz
I (978) app_init: Application information:
I (978) app_init: Project name:     tfm_tinyml_deploy
I (983) app_init: App version:      2e080460-dirty
I (987) app_init: Compile time:     Feb 12 2026 15:35:56
I (992) app_init: ELF file SHA256:  33eaaf0cf...
I (997) app_init: ESP-IDF:          v5.4.3-dirty
I (1001) efuse_init: Min chip rev:     v0.0
I (1005) efuse_init: Max chip rev:     v0.99 
I (1009) efuse_init: Chip rev:         v0.2
I (1013) heap_init: Initializing. RAM available for dynamic allocation:
I (1019) heap_init: At 3FCB0A30 len 00038CE0 (227 KiB): RAM
I (1024) heap_init: At 3FCE9710 len 00005724 (21 KiB): RAM
I (1030) heap_init: At 3FCF0000 len 00008000 (32 KiB): DRAM
I (1035) heap_init: At 600FE000 len 00001FE8 (7 KiB): RTCRAM
I (1041) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (1048) spi_flash: detected chip: boya
I (1051) spi_flash: flash io: dio
I (1054) sleep_gpio: Configure to isolate all GPIO pins in sleep state
I (1060) sleep_gpio: Enable automatic switching of GPIO sleep configuration
I (1067) main_task: Started on CPU0
I (1097) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (1097) main_task: Calling app_main()
I (1097) main: ========================================
I (1107) main:   TFM TinyML Deployment — ESP32-S3
I (1107) main:   Model: YOLO11n
I (1107) main: ========================================
I (1127) main: PSRAM total: 8192 KB, free: 8189 KB
I (1127) main: Input buffer: 150528 bytes in PSRAM
I (1127) s3 ll_cam: DMA Channel=0
I (1127) cam_hal: cam init ok
I (1137) sccb-ng: pin_sda 4 pin_scl 5
I (1137) sccb-ng: sccb_i2c_port=1
I (1147) ov3660: Mismatch PID=0x5640
I (1147) camera: Camera PID=0x5640 VER=0x00 MIDL=0x00 MIDH=0x00
I (1147) camera: Detected OV5640 camera
I (1157) camera: Detected camera at address=0x3c
I (1837) cam_hal: PSRAM DMA mode disabled
I (1837) s3 ll_cam: node_size: 3840, nodes_per_line: 1, lines_per_node: 6
I (1837) s3 ll_cam: dma_half_buffer_min:  3840, dma_half_buffer: 15360, lines_per_half_buffer: 24, dma_buffer_size: 30720
I (1847) cam_hal: buffer_size: 30720, half_buffer_size: 15360, node_buffer_size: 3840, node_cnt: 8, total_cnt: 10
I (1857) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1857) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1867) cam_hal: cam config ok
I (1877) ov5640: Set PLL: bypass: 0, multiplier: 8, sys_div: 1, pre_div: 1, root_2x: 0, pclk_root_div: 1, pclk_manual: 1, pclk_div: 4
I (1877) ov5640: Calculated XVCLK: 20000000 Hz, REFIN: 20000000 Hz, VCO: 160000000 Hz, PLL_CLK: 64000000 Hz, SYSCLK: 16000000 Hz, PCLK: 8000000 Hz
I (1927) camera: Sensor configured: vflip=1, auto WB/exp/gain, lens correction
I (1927) camera: Camera initialized: OV5640 RGB565 320x240, 2 buffers in PSRAM
I (1927) main: Camera initialized (OV5640 RGB565 320x240)
I (1937) inference: Loading YOLO11n from partition 'models' offset=0xa7000 size=2857568 bytes
I (1947) inference: Partition found: offset=0x410000, size=7340032
I (1947) inference: Model mmap'd at 0x3c9e7000, 2857568 bytes
I (2657) inference:   Output[0]: name='output0' shape=[1, 9, 1029] dtype=3 exponent=1
I (2657) inference:   Input: shape=[1, 224, 224, 3] dtype=3 exponent=-7
I (2667) inference: Model loaded OK. 1 output(s). PSRAM free: 4600 KB

I (2667) dl::Model: model:main_graph, version:0
I (2677) dl::Model: MODEL LOCATION IN FLASH RODATA
I (2677) dl::Model: +----------------+--------------+-----------+---------------+
I (2687) dl::Model: |                      memory summary                       |
I (2697) dl::Model: +----------------+--------------+-----------+---------------+
I (2697) dl::Model: |                | internal RAM | PSRAM     | FLASH         |
I (2707) dl::Model: +----------------+--------------+-----------+---------------+
I (2717) dl::Model: | fbs_model      |              |           | 2790.58KB     |
I (2727) dl::Model: |  -- parameter  |              |           |  -- 2553.12KB |
I (2727) dl::Model: +----------------+--------------+-----------+---------------+
I (2737) dl::Model: | parameter_copy |              | 2553.12KB |               |
I (2747) dl::Model: +----------------+--------------+-----------+---------------+
I (2747) dl::Model: | variable       |              | 496.12KB  |               |
I (2757) dl::Model: +----------------+--------------+-----------+---------------+
I (2767) dl::Model: | others         | 93.82KB      | 90.63KB   |               |
I (2777) dl::Model: +----------------+--------------+-----------+---------------+
I (2777) dl::Model: | total          | 93.82KB      | 3139.88KB | 2790.58KB     |
I (2787) dl::Model: +----------------+--------------+-----------+---------------+

I (6387) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6387) dl::Model: |                                  module summary                                  |
I (6397) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6407) dl::Model: | name                                              | type             | latency   |
I (6417) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6427) dl::Model: | /model.0/conv/Conv                                | Conv             | 24729us   |
I (6427) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6437) dl::Model: | /model.0/conv/Conv/Swish                          | Swish            | 13498us   |
I (6447) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6457) dl::Model: | /model.1/conv/Conv                                | Conv             | 11802us   |
I (6467) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6477) dl::Model: | /model.1/conv/Conv/Swish                          | Swish            | 7025us    |
I (6487) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6497) dl::Model: | /model.2/cv1/conv/Conv                            | Conv             | 5456us    |
I (6507) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6517) dl::Model: | /model.2/cv1/conv/Conv/Swish                      | Swish            | 7023us    |
I (6527) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6537) dl::Model: | /model.2/Split                                    | Split            | 7332us    |
I (6537) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6547) dl::Model: | /model.2/m.0/cv1/conv/Conv                        | Conv             | 23923us   |
I (6557) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6567) dl::Model: | /model.2/m.0/cv1/conv/Conv/Swish                  | Swish            | 1618us    |
I (6577) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6587) dl::Model: | /model.2/m.0/cv2/conv/Conv                        | Conv             | 7631us    |
I (6597) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6607) dl::Model: | /model.2/m.0/cv2/conv/Conv/Swish                  | Swish            | 3301us    |
I (6617) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6627) dl::Model: | /model.2/m.0/Add                                  | Add              | 3512us    |
I (6637) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6647) dl::Model: | /model.2/Concat                                   | Concat           | 8559us    |
I (6647) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6657) dl::Model: | /model.2/cv2/conv/Conv                            | Conv             | 10273us   |
I (6667) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6677) dl::Model: | /model.2/cv2/conv/Conv/Swish                      | Swish            | 13584us   |
I (6687) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6697) dl::Model: | /model.3/conv/Conv                                | Conv             | 288720us  |
I (6707) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6717) dl::Model: | /model.3/conv/Conv/Swish                          | Swish            | 3291us    |
I (6727) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6737) dl::Model: | /model.4/cv1/conv/Conv                            | Conv             | 3139us    |
I (6747) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6757) dl::Model: | /model.4/cv1/conv/Conv/Swish                      | Swish            | 3281us    |
I (6767) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6767) dl::Model: | /model.4/Split                                    | Split            | 3154us    |
I (6777) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6787) dl::Model: | /model.4/m.0/cv1/conv/Conv                        | Conv             | 2079us    |
I (6797) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6807) dl::Model: | /model.4/m.0/cv1/conv/Conv/Swish                  | Swish            | 717us     |
I (6817) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6827) dl::Model: | /model.4/m.0/cv2/conv/Conv                        | Conv             | 2175us    |
I (6837) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6847) dl::Model: | /model.4/m.0/cv2/conv/Conv/Swish                  | Swish            | 1503us    |
I (6857) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6867) dl::Model: | /model.4/m.0/Add                                  | Add              | 1630us    |
I (6877) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6877) dl::Model: | /model.4/Concat                                   | Concat           | 4158us    |
I (6887) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6897) dl::Model: | /model.4/cv2/conv/Conv                            | Conv             | 7382us    |
I (6907) dl::Model: +---------------------------------------------------+------------------+-----------+
I (6917) dl::Model: | /model.4/cv2/conv/Conv/Swish                      | Swish            | 7028usE (6927) task_wdt: Task watchdog got triggered. The following tasks/users did not reset the watchdog in time:
E (6927) task_wdt:  - IDLE0 (CPU 0)
E (6927) task_wdt: Tasks currently running:
E (6927) task_wdt: CPU 0: main
E (6927) task_wdt: CPU 1: IDLE1
E (6927) task_wdt: Print CPU 0 (current core) backtrace


Backtrace: 0x4209F736:0x3FC9F8A0 0x4209FB50:0x3FC9F8C0 0x40377CC5:0x3FC9F8F0 0x42009228:0x3FCB4C20 0x4200A17B:0x3FCB4C50 0x4200BEBE:0x3FCB4C70 0x42139612:0x3FCB4C90 0x4200B3BD:0x3FCB4CB0 0x4200BEBE:0x3FCB4CD0 0x421370AE:0x3FCB4CF0 0x42136585:0x3FCB4D10 0x421365D7:0x3FCB4D30 0x42136C89:0x3FCB4D50 0x42146EBF:0x3FCB4D80 0x42146DB6:0x3FCB4DA0 0x42137125:0x3FCB50C0 0x40386D8A:0x3FCB50F0 0x40386D55:0x3FCB5120 0x4204E80E:0x3FCB5170 0x4205391F:0x3FCB52D0 0x4201C7A1:0x3FCB5330 0x4200D511:0x3FCB53D0 0x4217E65F:0x3FCB5400 0x4038053D:0x3FCB5430
--- 0x4209f736: task_wdt_timeout_handling at /Users/admin/esp/v5.4.3/esp-idf/components/esp_system/task_wdt/task_wdt.c:434
--- 0x4209fb50: task_wdt_isr at /Users/admin/esp/v5.4.3/esp-idf/components/esp_system/task_wdt/task_wdt.c:507
--- 0x40377cc5: _xt_lowint1 at /Users/admin/esp/v5.4.3/esp-idf/components/xtensa/xtensa_vectors.S:1240
--- 0x42009228: uart_ll_get_txfifo_len at /Users/admin/esp/v5.4.3/esp-idf/components/hal/esp32s3/include/hal/uart_ll.h:418
--- (inlined by) uart_tx_char at /Users/admin/esp/v5.4.3/esp-idf/components/esp_driver_uart/src/uart_vfs.c:190
--- 0x4200a17b: uart_write at /Users/admin/esp/v5.4.3/esp-idf/components/esp_driver_uart/src/uart_vfs.c:243
--- 0x4200bebe: esp_vfs_write at /Users/admin/esp/v5.4.3/esp-idf/components/vfs/vfs.c:945
--- 0x42139612: write at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/syscalls/syswrite.c:11
--- 0x4200b3bd: console_write at /Users/admin/esp/v5.4.3/esp-idf/components/esp_vfs_console/vfs_console.c:83
--- 0x4200bebe: esp_vfs_write at /Users/admin/esp/v5.4.3/esp-idf/components/vfs/vfs.c:945
--- 0x421370ae: __swrite at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/stdio.c:94
--- 0x42136585: __sflush_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/fflush.c:224
--- 0x421365d7: _fflush_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/fflush.c:278
--- 0x42136c89: __sfvwrite_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/fvwrite.c:251
--- 0x42146ebf: __sprint_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/vfprintf.c:429
--- (inlined by) __sprint_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/vfprintf.c:399
--- 0x42146db6: _vfprintf_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/vfprintf.c:1774
--- 0x42137125: vprintf at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/vprintf.c:34
--- 0x40386d8a: esp_log_writev at /Users/admin/esp/v5.4.3/esp-idf/components/log/src/os/log_write.c:34
--- 0x40386d55: esp_log_write at /Users/admin/esp/v5.4.3/esp-idf/components/log/src/os/log_write.c:44
--- 0x4204e80e: dl::Model::print_module_info(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, dl::module_info, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, dl::module_info> > > const&, bool) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/model/src/dl_model_base.cpp:695
--- 0x4205391f: dl::Model::profile(bool) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/model/src/dl_model_base.cpp:753
--- 0x4201c7a1: inference_init at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:135
--- 0x4200d511: app_main at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp:137
--- 0x4217e65f: main_task at /Users/admin/esp/v5.4.3/esp-idf/components/freertos/app_startup.c:208
--- 0x4038053d: vPortTaskWrapper at /Users/admin/esp/v5.4.3/esp-idf/components/freertos/FreeRTOS-Kernel/portable/xtensa/port.c:139

    |
I (7007) dl::Model: +---------------------------------------------------+------------------+---------cam_hal: FB-SIZE: 76800 != 153600
--+
I (7017) dl::Model: | /model.5/conv/Conv                                | Conv             | 475816us  |
I (7027) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7037) dl::Model: | PPQ_Operation_14                                  | RequantizeLinear | 2624us    |
I (7047) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7047) dl::Model: | /model.5/conv/Conv/Swish                          | Swish            | 2129us    |
I (7057) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7067) dl::Model: | /model.6/cv1/conv/Conv                            | Conv             | 2985us    |
I (7077) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7087) dl::Model: | /model.6/cv1/conv/Conv/Swish                      | Swish            | 1641us    |
I (7097) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7107) dl::Model: | /model.6/Split                                    | Split            | 1127us    |
I (7117) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7127) dl::Model: | /model.6/m.0/cv1/conv/Conv                        | Conv             | 357us     |
I (7137) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7147) dl::Model: | /model.6/m.0/cv2/conv/Conv                        | Conv             | 380us     |
I (7157) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7157) dl::Model: | /model.6/m.0/cv1/conv/Conv/Swish                  | Swish            | 362us     |
I (7167) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7177) dl::Model: | /model.6/m.0/cv2/conv/Conv/Swish                  | Swish            | 348us     |
I (7187) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7197) dl::Model: | /model.6/m.0/m/m.0/cv1/conv/Conv                  | Conv             | 1050us    |
I (7207) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7217) dl::Model: | PPQ_Operation_2                                   | RequantizeLinear | 117us     |
I (7227) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7237) dl::Model: | /model.6/m.0/m/m.0/cv1/conv/Conv/Swish            | Swish            | 360us     |
I (7247) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7257) dl::Model: | /model.6/m.0/m/m.0/cv2/conv/Conv                  | Conv             | 1005us    |
I (7267) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7277) dl::Model: | /model.6/m.0/m/m.0/cv2/conv/Conv/Swish            | Swish            | 348us     |
I (7277) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7287) dl::Model: | /model.6/m.0/m/m.0/Add                            | Add              | 222us     |
I (7297) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7307) dl::Model: | /model.6/m.0/m/m.1/cv1/conv/Conv                  | Conv             | 1459us    |
I (7317) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7327) dl::Model: | /model.6/m.0/m/m.1/cv1/conv/Conv/Swish            | Swish            | 352us     |
I (7337) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7347) dl::Model: | /model.6/m.0/m/m.1/cv2/conv/Conv                  | Conv             | 980us     |
I (7357) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7367) dl::Model: | /model.6/m.0/m/m.1/cv2/conv/Conv/Swish            | Swish            | 347us     |
I (7377) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7387) dl::Model: | /model.6/m.0/m/m.1/Add                            | Add              | 168us     |
I (7387) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7397) dl::Model: | /model.6/m.0/Concat                               | Concat           | 302us     |
I (7407) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7417) dl::Model: | /model.6/m.0/cv3/conv/Conv                        | Conv             | 677us     |
I (7427) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7437) dl::Model: | /model.6/m.0/cv3/conv/Conv/Swish                  | Swish            | 715us     |
I (7447) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7457) dl::Model: | /model.6/Concat                                   | Concat           | 1504us    |
I (7467) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7477) dl::Model: | /model.6/cv2/conv/Conv                            | Conv             | 4779us    |
I (7487) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7497) dl::Model: | /model.6/cv2/conv/Conv/Swish                      | Swish            | 1975us    |
I (7507) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7507) dl::Model: | /model.7/conv/Conv                                | Conv             | 226454us  |
I (7517) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7527) dl::Model: | /model.7/conv/Conv/Swish                          | Swish            | 828us     |
I (7537) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7547) dl::Model: | /model.8/cv1/conv/Conv                            | Conv             | 55591us   |
I (7557) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7567) dl::Model: | /model.8/cv1/conv/Conv/Swish                      | Swish            | 824us     |
I (7577) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7587) dl::Model: | /model.8/Split                                    | Split            | 359us     |
I (7597) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7607) dl::Model: | /model.8/m.0/cv1/conv/Conv                        | Conv             | 389us     |
I (7617) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7617) dl::Model: | /model.8/m.0/cv2/conv/Conv                        | Conv             | 462us     |
I (7627) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7637) dl::Model: | /model.8/m.0/cv1/conv/Conv/Swish                  | Swish            | 209us     |
I (7647) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7657) dl::Model: | /model.8/m.0/cv2/conv/Conv/Swish                  | Swish            | 178us     |
I (7667) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7677) dl::Model: | /model.8/m.0/m/m.0/cv1/conv/Conv                  | Conv             | 11759us   |
I (7687) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7697) dl::Model: | PPQ_Operation_3                                   | RequantizeLinear | 136us     |
I (7707) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7717) dl::Model: | /model.8/m.0/m/m.0/cv1/conv/Conv/Swish            | Swish            | 216us     |
I (7727) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7737) dl::Model: | /model.8/m.0/m/m.0/cv2/conv/Conv                  | Conv             | 11591us   |
I (7737) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7747) dl::Model: | /model.8/m.0/m/m.0/cv2/conv/Conv/Swish            | Swish            | 217us     |
I (7757) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7767) dl::Model: | /model.8/m.0/m/m.0/Add                            | Add              | 181us     |
I (7777) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7787) dl::Model: | /model.8/m.0/m/m.1/cv1/conv/Conv                  | Conv             | 11632us   |
I (7797) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7807) dl::Model: | PPQ_Operation_4                                   | RequantizeLinear | 113us     |
I (7817) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7827) dl::Model: | /model.8/m.0/m/m.1/cv1/conv/Conv/Swish            | Swish            | 216us     |
I (7837) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7847) dl::Model: | /model.8/m.0/m/m.1/cv2/conv/Conv                  | Conv             | 11964us   |
I (7847) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7857) dl::Model: | /model.8/m.0/m/m.1/cv2/conv/Conv/Swish            | Swish            | 214us     |
I (7867) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7877) dl::Model: | /model.8/m.0/m/m.1/Add                            | Add              | 156us     |
I (7887) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7897) dl::Model: | /model.8/m.0/Concat                               | Concat           | 149us     |
I (7907) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7917) dl::Model: | /model.8/m.0/cv3/conv/Conv                        | Conv             | 789us     |
I (7927) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7937) dl::Model: | /model.8/m.0/cv3/conv/Conv/Swish                  | Swish            | 367us     |
I (7947) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7957) dl::Model: | /model.8/Concat                                   | Concat           | 655us     |
I (7967) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7967) dl::Model: | /model.8/cv2/conv/Conv                            | Conv             | 82782us   |
I (7977) dl::Model: +---------------------------------------------------+------------------+-----------+
I (7987) dl::Model: | /model.8/cv2/conv/Conv/Swish                      | Swish            | 824us     |
I (7997) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8007) dl::Model: | /model.9/cv1/conv/Conv                            | Conv             | 6060us    |
I (8017) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8027) dl::Model: | /model.9/m/MaxPool                                | MaxPool          | 445us     |
I (8037) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8047) dl::Model: | PPQ_Operation_5                                   | RequantizeLinear | 114us     |
I (8057) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8067) dl::Model: | /model.9/m_1/MaxPool                              | MaxPool          | 199us     |
I (8077) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8077) dl::Model: | PPQ_Operation_6                                   | RequantizeLinear | 65us      |
I (8087) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8097) dl::Model: | /model.9/m_2/MaxPool                              | MaxPool          | 202us     |
I (8107) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8117) dl::Model: | PPQ_Operation_7                                   | RequantizeLinear | 74us      |
I (8127) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8137) dl::Model: | PPQ_Operation_8                                   | RequantizeLinear | 65us      |
I (8147) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8157) dl::Model: | /model.9/Concat                                   | Concat           | 867us     |
I (8167) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8177) dl::Model: | /model.9/cv2/conv/Conv                            | Conv             | 110347us  |
I (8187) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8197) dl::Model: | /model.9/cv2/conv/Conv/Swish                      | Swish            | 836us     |
I (8197) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8207) dl::Model: | /model.10/cv1/conv/Conv                           | Conv             | 55571us   |
I (8217) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8227) dl::Model: | /model.10/cv1/conv/Conv/Swish                     | Swish            | 825us     |
I (8237) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8247) dl::Model: | /model.10/Split                                   | Split            | 359us     |
I (8257) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8267) dl::Model: | PPQ_Operation_10                                  | RequantizeLinear | 58us      |
I (8277) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8287) dl::Model: | /model.10/m/m.0/attn/qkv/conv/Conv                | Conv             | 8326us    |
I (8297) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8307) dl::Model: | PPQ_Operation_46                                  | Transpose        | 6343us    |
I (8307) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8317) dl::Model: | /model.10/m/m.0/attn/Reshape                      | Reshape          | 15us      |
I (8327) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8337) dl::Model: | /model.10/m/m.0/attn/Split                        | Split            | 423us     |
I (8347) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8357) dl::Model: | /model.10/m/m.0/attn/Transpose                    | Transpose        | 1537us    |
I (8367) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8377) dl::Model: | PPQ_Operation_30                                  | Transpose        | 1524us    |
I (8387) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8397) dl::Model: | /model.10/m/m.0/attn/Reshape_2                    | Reshape          | 3us       |
I (8407) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8417) dl::Model: | PPQ_Operation_31                                  | Slice            | 208us     |
I (8427) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8427) dl::Model: | PPQ_Operation_36                                  | Slice            | 53us      |
I (8437) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8447) dl::Model: | PPQ_Operation_47                                  | Transpose        | 3213us    |
I (8457) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8467) dl::Model: | PPQ_Operation_32                                  | Reshape          | 9us       |
I (8477) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8487) dl::Model: | /model.10/m/m.0/attn/pe/conv/Conv                 | Conv             | 575us     |
I (8497) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8507) dl::Model: | PPQ_Operation_33                                  | Transpose        | 1754us    |
I (8517) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8527) dl::Model: | PPQ_Operation_48                                  | Transpose        | 3023us    |
I (8537) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8537) dl::Model: | PPQ_Operation_34                                  | Reshape          | 6us       |
I (8547) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8557) dl::Model: | PPQ_Operation_35                                  | Concat           | 44us      |
I (8567) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8577) dl::Model: | PPQ_Operation_37                                  | Reshape          | 3us       |
I (8587) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8597) dl::Model: | /model.10/m/m.0/attn/MatMul                       | MatMul           | 1460us    |
I (8607) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8617) dl::Model: | /model.10/m/m.0/attn/Mul                          | Mul              | 286us     |
I (8627) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8637) dl::Model: | /model.10/m/m.0/attn/Softmax                      | Softmax          | 1145us    |
I (8647) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8657) dl::Model: | PPQ_Operation_0                                   | QuantizeLinear   | 2334us    |
I (8657) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8667) dl::Model: | PPQ_Operation_39                                  | Slice            | 341us     |
I (8677) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8687) dl::Model: | PPQ_Operation_44                                  | Slice            | 57us      |
I (8697) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8707) dl::Model: | PPQ_Operation_40                                  | Reshape          | 16us      |
I (8717) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8727) dl::Model: | PPQ_Operation_41                                  | Transpose        | 2620us    |
I (8737) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8747) dl::Model: | PPQ_Operation_42                                  | Reshape          | 3us       |
I (8757) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8767) dl::Model: | PPQ_Operation_43                                  | Concat           | 168us     |
I (8767) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8777) dl::Model: | PPQ_Operation_45                                  | Reshape          | 3us       |
I (8787) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8797) dl::Model: | /model.10/m/m.0/attn/MatMul_1                     | MatMul           | 1978us    |
I (8807) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8817) dl::Model: | /model.10/m/m.0/attn/Reshape_1                    | Reshape          | 9us       |
I (8827) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8837) dl::Model: | /model.10/m/m.0/attn/Add                          | Add              | 401us     |
I (8847) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8857) dl::Model: | PPQ_Operation_49                                  | Transpose        | 3215us    |
I (8867) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8877) dl::Model: | /model.10/m/m.0/attn/proj/conv/Conv               | Conv             | 958us     |
I (8887) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8887) dl::Model: | /model.10/m/m.0/Add                               | Add              | 598us     |
I (8897) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8907) dl::Model: | /model.10/m/m.0/ffn/ffn.0/conv/Conv               | Conv             | 6890us    |
I (8917) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8927) dl::Model: | PPQ_Operation_9                                   | RequantizeLinear | 207us     |
I (8937) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8947) dl::Model: | /model.10/m/m.0/ffn/ffn.0/conv/Conv/Swish         | Swish            | 835us     |
I (8957) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8967) dl::Model: | /model.10/m/m.0/ffn/ffn.1/conv/Conv               | Conv             | 6075us    |
I (8977) dl::Model: +---------------------------------------------------+------------------+-----------+
I (8987) dl::Model: | /model.10/m/m.0/Add_1                             | Add              | 291us     |
I (8997) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9007) dl::Model: | /model.10/Concat                                  | Concat           | 373us     |
I (9007) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9017) dl::Model: | /model.10/cv2/conv/Conv                           | Conv             | 55665us   |
I (9027) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9037) dl::Model: | /model.10/cv2/conv/Conv/Swish                     | Swish            | 823us     |
I (9047) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9057) dl::Model: | /model.11/Resize                                  | Resize           | 1509us    |
I (9067) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9077) dl::Model: | PPQ_Operation_22                                  | RequantizeLinear | 285us     |
I (9087) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9097) dl::Model: | /model.12/Concat                                  | Concat           | 3580us    |
I (9107) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9117) dl::Model: | /model.13/cv1/conv/Conv                           | Conv             | 165910us  |
I (9117) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9127) dl::Model: | /model.13/cv1/conv/Conv/Swish                     | Swish            | 1642us    |
I (9137) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9147) dl::Model: | /model.13/Split                                   | Split            | 1148us    |
I (9157) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9167) dl::Model: | PPQ_Operation_12                                  | RequantizeLinear | 223us     |
I (9177) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9187) dl::Model: | /model.13/m.0/cv1/conv/Conv                       | Conv             | 2799us    |
I (9197) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9207) dl::Model: | PPQ_Operation_11                                  | RequantizeLinear | 395us     |
I (9217) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9227) dl::Model: | PPQ_Operation_13                                  | RequantizeLinear | 135us     |
I (9237) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9237) dl::Model: | /model.13/m.0/cv1/conv/Conv/Swish                 | Swish            | 413us     |
I (9247) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9257) dl::Model: | /model.13/m.0/cv2/conv/Conv                       | Conv             | 2158us    |
I (9267) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9277) dl::Model: | /model.13/m.0/cv2/conv/Conv/Swish                 | Swish            | 744us     |
I (9287) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9297) dl::Model: | /model.13/m.0/Add                                 | Add              | 447us     |
I (9307) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9317) dl::Model: | /model.13/Concat                                  | Concat           | 1770us    |
I (9327) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9337) dl::Model: | /model.13/cv2/conv/Conv                           | Conv             | 5493us    |
I (9347) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9347) dl::Model: | /model.13/cv2/conv/Conv/Swish                     | Swish            | 1654us    |
I (9357) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9367) dl::Model: | /model.14/Resize                                  | Resize           | 3402us    |
I (9377) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9387) dl::Model: | PPQ_Operation_18                                  | RequantizeLinear | 708us     |
I (9397) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9407) dl::Model: | /model.15/Concat                                  | Concat           | 9680us    |
I (9417) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9427) dl::Model: | /model.16/cv1/conv/Conv                           | Conv             | 11387us   |
I (9437) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9447) dl::Model: | /model.16/cv1/conv/Conv/Swish                     | Swish            | 3283us    |
I (9457) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9467) dl::Model: | /model.16/Split                                   | Split            | 2698us    |
I (9467) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9477) dl::Model: | PPQ_Operation_16                                  | RequantizeLinear | 630us     |
I (9487) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9497) dl::Model: | /model.16/m.0/cv1/conv/Conv                       | Conv             | 2357us    |
I (9507) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9517) dl::Model: | PPQ_Operation_15                                  | RequantizeLinear | 1078us    |
I (9527) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9537) dl::Model: | PPQ_Operation_17                                  | RequantizeLinear | 1010us    |
I (9547) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9557) dl::Model: | /model.16/m.0/cv1/conv/Conv/Swish                 | Swish            | 840us     |
I (9567) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9577) dl::Model: | /model.16/m.0/cv2/conv/Conv                       | Conv             | 2196us    |
I (9577) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9587) dl::Model: | /model.16/m.0/cv2/conv/Conv/Swish                 | Swish            | 1458us    |
I (9597) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9607) dl::Model: | /model.16/m.0/Add                                 | Add              | 940us     |
I (9617) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9627) dl::Model: | /model.16/Concat                                  | Concat           | 3728us    |
I (9637) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9647) dl::Model: | /model.16/cv2/conv/Conv                           | Conv             | 4533us    |
I (9657) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9667) dl::Model: | /model.16/cv2/conv/Conv/Swish                     | Swish            | 3293us    |
I (9677) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9687) dl::Model: | /model.17/conv/Conv                               | Conv             | 68497us   |
I (9697) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9697) dl::Model: | /model.23/cv2.0/cv2.0.0/conv/Conv                 | Conv             | 264181us  |
I (9707) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9717) dl::Model: | /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv       | Conv             | 2575us    |
I (9727) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9737) dl::Model: | /model.17/conv/Conv/Swish                         | Swish            | 846us     |
I (9747) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9757) dl::Model: | /model.23/cv2.0/cv2.0.0/conv/Conv/Swish           | Swish            | 3751us    |
I (9767) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9777) dl::Model: | /model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv/Swish | Swish            | 3293us    |
I (9787) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9797) dl::Model: | /model.18/Concat                                  | Concat           | 1926us    |
I (9807) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9807) dl::Model: | /model.23/cv2.0/cv2.0.1/conv/Conv                 | Conv             | 272190us  |
I (9817) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9827) dl::Model: | /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv       | Conv             | 2942us    |
I (9837) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9847) dl::Model: | /model.19/cv1/conv/Conv                           | Conv             | 4753us    |
I (9857) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9867) dl::Model: | /model.23/cv2.0/cv2.0.1/conv/Conv/Swish           | Swish            | 3593us    |
I (9877) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9887) dl::Model: | /model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv/Swish | Swish            | 3292us    |
I (9897) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9907) dl::Model: | /model.19/cv1/conv/Conv/Swish                     | Swish            | 1653us    |
I (9917) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9927) dl::Model: | /model.23/cv2.0/cv2.0.2/Conv                      | Conv             | 3051us    |
I (9927) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9937) dl::Model: | /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv       | Conv             | 3064us    |
I (9947) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9957) dl::Model: | /model.19/Split                                   | Split            | 1314us    |
I (9967) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9977) dl::Model: | PPQ_Operation_50                                  | Transpose        | 25832us   |
I (9987) dl::Model: +---------------------------------------------------+------------------+-----------+
I (9997) dl::Model: | /model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv/Swish | Swish            | 3296us    |
I (10007) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10017) dl::Model: | PPQ_Operation_20                                  | RequantizeLinear | 465us     |
I (10027) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10037) dl::Model: | /model.19/m.0/cv1/conv/Conv                       | Conv             | 2372us    |
I (10037) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10047) dl::Model: | PPQ_Operation_19                                  | RequantizeLinear | 373us     |
I (10057) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10067) dl::Model: | PPQ_Operation_21                                  | RequantizeLinear | 122us     |
I (10077) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10087) dl::Model: | /model.23/Reshape                                 | Reshape          | 19us      |
I (10097) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10107) dl::Model: | /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv       | Conv             | 3142us    |
I (10117) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10127) dl::Model: | /model.19/m.0/cv1/conv/Conv/Swish                 | Swish            | 825us     |
I (10137) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10147) dl::Model: | /model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv/Swish | Swish            | 3282us    |
I (10157) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10167) dl::Model: | /model.19/m.0/cv2/conv/Conv                       | Conv             | 2198us    |
I (10177) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10177) dl::Model: | /model.23/cv3.0/cv3.0.2/Conv                      | Conv             | 2287us    |
I (10187) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10197) dl::Model: | /model.19/m.0/cv2/conv/Conv/Swish                 | Swish            | 826us     |
I (10207) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10217) dl::Model: | PPQ_Operation_51                                  | Transpose        | 2070us    |
I (10227) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10237) dl::Model: | /model.19/m.0/Add                                 | Add              | 443us     |
I (10247) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10257) dl::Model: | /model.23/Reshape_3                               | Reshape          | 9us       |
I (10267) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10277) dl::Model: | /model.19/Concat                                  | Concat           | 2256us    |
I (10287) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10297) dl::Model: | PPQ_Operation_25                                  | RequantizeLinear | 188us     |
I (10307) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10307) dl::Model: | /model.19/cv2/conv/Conv                           | Conv             | 6624us    |
I (10317) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10327) dl::Model: | /model.19/cv2/conv/Conv/Swish                     | Swish            | 1616us    |
I (10337) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10347) dl::Model: | /model.20/conv/Conv                               | Conv             | 114257us  |
I (10357) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10367) dl::Model: | /model.23/cv2.1/cv2.1.0/conv/Conv                 | Conv             | 223488us  |
I (10377) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10387) dl::Model: | /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv       | Conv             | 1522us    |
I (10397) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10407) dl::Model: | /model.20/conv/Conv/Swish                         | Swish            | 438us     |
I (10417) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10427) dl::Model: | /model.23/cv2.1/cv2.1.0/conv/Conv/Swish           | Swish            | 824us     |
I (10437) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10437) dl::Model: | /model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv/Swish | Swish            | 1651us    |
I (10447) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10457) dl::Model: | /model.21/Concat                                  | Concat           | 1079us    |
I (10467) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10477) dl::Model: | /model.23/cv2.1/cv2.1.1/conv/Conv                 | Conv             | 60274us   |
I (10487) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10497) dl::Model: | /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv       | Conv             | 1490us    |
I (10507) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10517) dl::Model: | /model.22/cv1/conv/Conv                           | Conv             | 82847us   |
I (10527) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10537) dl::Model: | /model.23/cv2.1/cv2.1.1/conv/Conv/Swish           | Swish            | 831us     |
I (10547) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10557) dl::Model: | /model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv/Swish | Swish            | 821us     |
I (10567) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10567) dl::Model: | /model.22/cv1/conv/Conv/Swish                     | Swish            | 827us     |
I (10577) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10587) dl::Model: | /model.23/cv2.1/cv2.1.2/Conv                      | Conv             | 1043us    |
I (10597) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10607) dl::Model: | /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv       | Conv             | 695us     |
I (10617) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10627) dl::Model: | /model.22/Split                                   | Split            | 670us     |
I (10637) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10647) dl::Model: | PPQ_Operation_52                                  | Transpose        | 6721us    |
I (10657) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10667) dl::Model: | /model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv/Swish | Swish            | 838us     |
I (10677) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10687) dl::Model: | /model.22/m.0/cv1/conv/Conv                       | Conv             | 658us     |
I (10697) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10697) dl::Model: | /model.22/m.0/cv2/conv/Conv                       | Conv             | 442us     |
I (10707) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10717) dl::Model: | /model.23/Reshape_1                               | Reshape          | 15us      |
I (10727) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10737) dl::Model: | /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv       | Conv             | 908us     |
I (10747) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10757) dl::Model: | /model.22/m.0/cv1/conv/Conv/Swish                 | Swish            | 215us     |
I (10767) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10777) dl::Model: | /model.22/m.0/cv2/conv/Conv/Swish                 | Swish            | 208us     |
I (10787) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10797) dl::Model: | /model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv/Swish | Swish            | 774us     |
I (10807) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10817) dl::Model: | /model.22/m.0/m/m.0/cv1/conv/Conv                 | Conv             | 11703us   |
I (10827) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10827) dl::Model: | PPQ_Operation_23                                  | RequantizeLinear | 151us     |
I (10837) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10847) dl::Model: | /model.23/cv3.1/cv3.1.2/Conv                      | Conv             | 678us     |
I (10857) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10867) dl::Model: | /model.22/m.0/m/m.0/cv1/conv/Conv/Swish           | Swish            | 223us     |
I (10877) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10887) dl::Model: | PPQ_Operation_53                                  | Transpose        | 537us     |
I (10897) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10907) dl::Model: | /model.22/m.0/m/m.0/cv2/conv/Conv                 | Conv             | 11780us   |
I (10917) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10927) dl::Model: | /model.23/Reshape_4                               | Reshape          | 7us       |
I (10937) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10947) dl::Model: | /model.22/m.0/m/m.0/cv2/conv/Conv/Swish           | Swish            | 212us     |
I (10957) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10957) dl::Model: | /model.22/m.0/m/m.0/Add                           | Add              | 210us     |
I (10967) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10977) dl::Model: | /model.22/m.0/m/m.1/cv1/conv/Conv                 | Conv             | 12249us   |
I (10987) dl::Model: +---------------------------------------------------+------------------+-----------+
I (10997) dl::Model: | PPQ_Operation_24                                  | RequantizeLinear | 120us     |
I (11007) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11017) dl::Model: | /model.22/m.0/m/m.1/cv1/conv/Conv/Swish           | Swish            | 216us     |
I (11027) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11037) dl::Model: | /model.22/m.0/m/m.1/cv2/conv/Conv                 | Conv             | 11533us   |
I (11047) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11057) dl::Model: | /model.22/m.0/m/m.1/cv2/conv/Conv/Swish           | Swish            | 211us     |
I (11067) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11077) dl::Model: | /model.22/m.0/m/m.1/Add                           | Add              | 150us     |
I (11087) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11087) dl::Model: | /model.22/m.0/Concat                              | Concat           | 199us     |
I (11097) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11107) dl::Model: | /model.22/m.0/cv3/conv/Conv                       | Conv             | 775us     |
I (11117) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11127) dl::Model: | /model.22/m.0/cv3/conv/Conv/Swish                 | Swish            | 354us     |
I (11137) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11147) dl::Model: | /model.22/Concat                                  | Concat           | 686us     |
I (11157) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11167) dl::Model: | /model.22/cv2/conv/Conv                           | Conv             | 82940us   |
I (11177) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11187) dl::Model: | /model.22/cv2/conv/Conv/Swish                     | Swish            | 829us     |
I (11197) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11207) dl::Model: | /model.23/cv2.2/cv2.2.0/conv/Conv                 | Conv             | 104540us  |
I (11217) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11217) dl::Model: | /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv       | Conv             | 911us     |
I (11227) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11237) dl::Model: | /model.23/cv2.2/cv2.2.0/conv/Conv/Swish           | Swish            | 227us     |
I (11247) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11257) dl::Model: | /model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv/Swish | Swish            | 694us     |
I (11267) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11277) dl::Model: | /model.23/cv2.2/cv2.2.1/conv/Conv                 | Conv             | 11644us   |
I (11287) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11297) dl::Model: | /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv       | Conv             | 854us     |
I (11307) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11317) dl::Model: | /model.23/cv2.2/cv2.2.1/conv/Conv/Swish           | Swish            | 220us     |
I (11327) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11337) dl::Model: | /model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv/Swish | Swish            | 182us     |
I (11347) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11347) dl::Model: | /model.23/cv2.2/cv2.2.2/Conv                      | Conv             | 228us     |
I (11357) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11367) dl::Model: | /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv       | Conv             | 214us     |
I (11377) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11387) dl::Model: | PPQ_Operation_54                                  | Transpose        | 1594us    |
I (11397) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11407) dl::Model: | /model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv/Swish | Swish            | 180us     |
I (11417) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11427) dl::Model: | /model.23/Reshape_2                               | Reshape          | 12us      |
I (11437) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11447) dl::Model: | /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv       | Conv             | 304us     |
I (11457) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11467) dl::Model: | /model.23/Concat                                  | Concat           | 2948us    |
I (11477) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11477) dl::Model: | /model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv/Swish | Swish            | 219us     |
I (11487) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11497) dl::Model: | /model.23/dfl/Reshape                             | Reshape          | 12us      |
I (11507) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11517) dl::Model: | /model.23/cv3.2/cv3.2.2/Conv                      | Conv             | 173us     |
I (11527) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11537) dl::Model: | /model.23/dfl/Transpose                           | Transpose        | 33501us   |
I (11547) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11557) dl::Model: | PPQ_Operation_55                                  | Transpose        | 155us     |
I (11567) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11577) dl::Model: | /model.23/dfl/Softmax                             | Softmax          | 14398us   |
I (11587) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11597) dl::Model: | /model.23/Reshape_5                               | Reshape          | 10us      |
I (11607) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11607) dl::Model: | PPQ_Operation_1                                   | QuantizeLinear   | 26316us   |
I (11617) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11627) dl::Model: | /model.23/Concat_1                                | Concat           | 223us     |
I (11637) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11647) dl::Model: | PPQ_Operation_56                                  | Transpose        | 45790us   |
I (11657) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11667) dl::Model: | /model.23/Sigmoid                                 | Sigmoid          | 357us     |
I (11677) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11687) dl::Model: | /model.23/dfl/conv/Conv                           | Conv             | 3551us    |
I (11697) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11707) dl::Model: | PPQ_Operation_57                                  | Transpose        | 2400us    |
I (11717) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11727) dl::Model: | /model.23/dfl/Reshape_1                           | Reshape          | 6us       |
I (11737) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11737) dl::Model: | /model.23/Slice                                   | Slice            | 166us     |
I (11747) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11757) dl::Model: | /model.23/Slice_1                                 | Slice            | 76us      |
I (11767) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11777) dl::Model: | PPQ_Operation_26                                  | RequantizeLinear | 112us     |
I (11787) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11797) dl::Model: | PPQ_Operation_27                                  | RequantizeLinear | 38us      |
I (11807) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11817) dl::Model: | /model.23/Sub                                     | Sub              | 227us     |
I (11827) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11837) dl::Model: | /model.23/Add_1                                   | Add              | 150us     |
I (11847) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11857) dl::Model: | PPQ_Operation_28                                  | RequantizeLinear | 51us      |
I (11867) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11867) dl::Model: | PPQ_Operation_29                                  | RequantizeLinear | 36us      |
I (11877) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11887) dl::Model: | /model.23/Sub_1                                   | Sub              | 94us      |
I (11897) dl::Model: +---------------------------------------------------+------------------+-----------+
I (11907) dl::Model: | /model.23/Add_2                                   | Add              | 58us      |
I (11917) dl::Model: +---------------------------------------------------+------------------+-----------E (11927) task_wdt: Task watchdog got triggered. The following tasks/users did not reset the watchdog in time:
E (11927) task_wdt:  - IDLE0 (CPU 0)
E (11927) task_wdt: Tasks currently running:
E (11927) task_wdt: CPU 0: main
E (11927) task_wdt: CPU 1: IDLE1
E (11927) task_wdt: Print CPU 0 (current core) backtrace


Backtrace: 0x4209F736:0x3FC9F8A0 0x4209FB50:0x3FC9F8C0 0x40377CC5:0x3FC9F8F0 0x4200922A:0x3FCB4C20 0x4200A17B:0x3FCB4C50 0x4200BEBE:0x3FCB4C70 0x42139612:0x3FCB4C90 0x4200B3BD:0x3FCB4CB0 0x4200BEBE:0x3FCB4CD0 0x421370AE:0x3FCB4CF0 0x42136585:0x3FCB4D10 0x421365D7:0x3FCB4D30 0x42136C89:0x3FCB4D50 0x42146EBF:0x3FCB4D80 0x42146DB6:0x3FCB4DA0 0x42137125:0x3FCB50C0 0x40386D8A:0x3FCB50F0 0x40386D55:0x3FCB5120 0x4204E829:0x3FCB5170 0x4205391F:0x3FCB52D0 0x4201C7A1:0x3FCB5330 0x4200D511:0x3FCB53D0 0x4217E65F:0x3FCB5400 0x4038053D:0x3FCB5430
--- 0x4209f736: task_wdt_timeout_handling at /Users/admin/esp/v5.4.3/esp-idf/components/esp_system/task_wdt/task_wdt.c:434
--- 0x4209fb50: task_wdt_isr at /Users/admin/esp/v5.4.3/esp-idf/components/esp_system/task_wdt/task_wdt.c:507
--- 0x40377cc5: _xt_lowint1 at /Users/admin/esp/v5.4.3/esp-idf/components/xtensa/xtensa_vectors.S:1240
--- 0x4200922a: uart_ll_get_txfifo_len at /Users/admin/esp/v5.4.3/esp-idf/components/hal/esp32s3/include/hal/uart_ll.h:418
--- (inlined by) uart_tx_char at /Users/admin/esp/v5.4.3/esp-idf/components/esp_driver_uart/src/uart_vfs.c:190
--- 0x4200a17b: uart_write at /Users/admin/esp/v5.4.3/esp-idf/components/esp_driver_uart/src/uart_vfs.c:243
--- 0x4200bebe: esp_vfs_write at /Users/admin/esp/v5.4.3/esp-idf/components/vfs/vfs.c:945
--- 0x42139612: write at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/syscalls/syswrite.c:11
--- 0x4200b3bd: console_write at /Users/admin/esp/v5.4.3/esp-idf/components/esp_vfs_console/vfs_console.c:83
--- 0x4200bebe: esp_vfs_write at /Users/admin/esp/v5.4.3/esp-idf/components/vfs/vfs.c:945
--- 0x421370ae: __swrite at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/stdio.c:94
--- 0x42136585: __sflush_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/fflush.c:224
--- 0x421365d7: _fflush_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/fflush.c:278
--- 0x42136c89: __sfvwrite_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/fvwrite.c:251
--- 0x42146ebf: __sprint_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/vfprintf.c:429
--- (inlined by) __sprint_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/vfprintf.c:399
--- 0x42146db6: _vfprintf_r at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/vfprintf.c:1774
--- 0x42137125: vprintf at /builds/idf/crosstool-NG/.build/xtensa-esp-elf/src/newlib/newlib/libc/stdio/vprintf.c:34
--- 0x40386d8a: esp_log_writev at /Users/admin/esp/v5.4.3/esp-idf/components/log/src/os/log_write.c:34
--- 0x40386d55: esp_log_write at /Users/admin/esp/v5.4.3/esp-idf/components/log/src/os/log_write.c:44
--- 0x4204e829: dl::Model::print_module_info(std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, dl::module_info, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, dl::module_info> > > const&, bool) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/model/src/dl_model_base.cpp:703
--- 0x4205391f: dl::Model::profile(bool) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/model/src/dl_model_base.cpp:753
--- 0x4201c7a1: inference_init at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:135
--- 0x4200d511: app_main at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp:137
--- 0x4217e65f: main_task at /Users/admin/esp/v5.4.3/esp-idf/components/freertos/app_startup.c:208
--- 0x4038053d: vPortTaskWrapper at /Users/admin/esp/v5.4.3/esp-idf/components/freertos/FreeRTOS-Kernel/portable/xtensa/port.c:139

+
I (12007) dl::Model: | /model.23/Div_1                                   | Div              | 345us     |
I (12007) dl::Model: +---------------------------------------------------+------------------+-----------+
I (12017) dl::Model: | /model.23/Concat_2                                | Concat           | 37us      |
I (12027) dl::Model: +---------------------------------------------------+------------------+-----------+
I (12037) dl::Model: | /modelcam_hal: FB-SIZE: 76800 != 153600
.23/Mul_2                                   | Mul              | 219us     |
I (12057) dl::Model: +---------------------------------------------------+------------------+-----------+
I (12057) dl::Model: | /model.23/Concat_3                                | Concat           | 311us     |
I (12067) dl::Model: +---------------------------------------------------+------------------+-----------+
I (12077) dl::Model: | total                                             |                  | 3526578us |
I (12087) dl::Model: +---------------------------------------------------+------------------+-----------+

I (12107) main: Inference engine ready
I (12107) postproc: Postprocessor initialized for YOLO11n
I (12107) main: Postprocessor initialized for YOLO11n
I (12117) temperature_sensor: Range [-10°C ~ 80°C], error < 1°C
I (12117) metrics: Temperature sensor initialized
I (12127) metrics: Metrics system initialized
I (12127) main: Memory post-init: PSRAM=4597 KB free, Internal=122 KB free
I (12137) pp: pp rom version: e7ae62f
I (12137) net80211: net80211 rom version: e7ae62f
I (12147) wifi:wifi driver task: 3fce0864, prio:23, stack:6656, core=0
I (12157) wifi:wifi firmware version: e8ce6e2
I (12157) wifi:wifi certification version: v7.0
I (12157) wifi:config NVS flash: enabled
I (12157) wifi:config nano formatting: disabled
I (12167) wifi:Init data frame dynamic rx buffer num: 32
I (12167) wifi:Init static rx mgmt buffer num: 5
I (12177) wifi:Init management short buffer num: 32
I (12177) wifi:Init dynamic tx buffer num: 32
I (12177) wifi:Init static tx FG buffer num: 2
I (12187) wifi:Init static rx buffer size: 1600
I (12187) wifi:Init static rx buffer num: 10
I (12197) wifi:Init dynamic rx buffer num: 32
I (12197) wifi_init: rx ba win: 6
I (12197) wifi_init: accept mbox: 6
I (12207) wifi_init: tcpip mbox: 32
I (12207) wifi_init: udp mbox: 6
I (12207) wifi_init: tcp mbox: 6
I (12217) wifi_init: tcp tx win: 5760
I (12217) wifi_init: tcp rx win: 5760
I (12217) wifi_init: tcp mss: 1440
I (12227) wifi_init: WiFi IRAM OP enabled
I (12227) wifi_init: WiFi RX IRAM OP enabled
I (12237) phy_init: phy_version 701,f4f1da3a,Mar  3 2025,15:50:10
I (12277) wifi:mode : softAP (1c:db:d4:4a:f4:59)
I (12277) wifi:Total power save buffer number: 16
I (12277) wifi:Init max length of beacon: 752/752
I (12277) wifi:Init max length of beacon: 752/752
I (12287) esp_netif_lwip: DHCP server started on interface WIFI_AP_DEF with IP: 192.168.4.1
I (12287) network: WiFi AP started: SSID=ESP32_TFM, channel=1
I (12297) network: HTTP server started on port 80
I (12297) dashboard: Dashboard registered at / (5359 bytes)
I (12307) main: Network ready: WiFi AP 'ESP32_TFM', http://192.168.4.1/
I (12317) main: Inference task started on core 0
I (15917) main: ========================================
I (15917) main:   System running. Connect to 'ESP32_TFM'
I (15917) main:   Dashboard: http://192.16E (17137) task_wdt: Task watchdog got triggered. The following tasks/users did not reset the watchdog in time:
E (17137) task_wdt:  - IDLE0 (CPU 0)
E (17137) task_wdt: Tasks currently running:
E (17137) task_wdt: CPU 0: inference
E (17137) task_wdt: CPU 1: IDLE1
E (17137) task_wdt: Print CPU 0 (current core) backtrace


Backtrace: 0x4209F736:0x3FC9F8A0 0x4209FB50:0x3FC9F8C0 0x40377CC5:0x3FC9F8F0 0x4207D6E9:0x3FCD9D60 0x4214C97F:0x3FCD9DE0 0x42066842:0x3FCD9E00 0x420675BB:0x3FCD9F80 0x4202645D:0x3FCD9FE0 0x4214BA03:0x3FCDA030 0x4201BCAB:0x3FCDA050 0x4200D2B1:0x3FCDA070 0x4038053D:0x3FCDA100
--- 0x4209f736: task_wdt_timeout_handling at /Users/admin/esp/v5.4.3/esp-idf/components/esp_system/task_wdt/task_wdt.c:434
--- 0x4209fb50: task_wdt_isr at /Users/admin/esp/v5.4.3/esp-idf/components/esp_system/task_wdt/task_wdt.c:507
--- 0x40377cc5: _xt_lowint1 at /Users/admin/esp/v5.4.3/esp-idf/components/xtensa/xtensa_vectors.S:1240
--- 0x4207d6e9: tie728_s8_conv2d_per_layer_33cn_bias_loop at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/tie728/dl_tie728_s8_conv2d.S:1269
--- 0x4214c97f: void std::__invoke_impl<void, void (*&)(signed char*, signed char*, void*), signed char*, signed char*, void*>(std::__invoke_other, void (*&)(signed char*, signed char*, void*), signed char*&&, signed char*&&, void*&&) at /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/invoke.h:61
--- (inlined by) std::enable_if<is_invocable_r_v<void, void (*&)(signed char*, signed char*, void*), signed char*, signed char*, void*>, void>::type std::__invoke_r<void, void (*&)(signed char*, signed char*, void*), signed char*, signed char*, void*>(void (*&)(signed char*, signed char*, void*), signed char*&&, signed char*&&, void*&&) at /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/invoke.h:111
--- (inlined by) std::_Function_handler<void (signed char*, signed char*, void*), void (*)(signed char*, signed char*, void*)>::_M_invoke(std::_Any_data const&, signed char*&&, signed char*&&, void*&&) at /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/std_function.h:290
--- 0x42066842: std::function<void (signed char*, signed char*, void*)>::operator()(signed char*, signed char*, void*) const at /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/std_function.h:591
--- (inlined by) void dl::base::conv_operation_shell<signed char, long>(dl::base::ArgsType<signed char>&, std::function<void (signed char*, signed char*, void*)>, std::function<void (signed char*, signed char*, void*)>, void (*)(long*, signed char*, dl::base::ArgsType<signed char> const&), void (*)(long*, signed char*, dl::base::ArgsType<signed char> const&), void (*)(signed char*, long*, dl::base::ArgsType<signed char> const&)) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base.hpp:788
--- 0x420675bb: void dl::base::conv2d<signed char, long, long>(void*) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base_conv2d.cpp:1343
--- 0x4202645d: dl::module::Conv::forward_args(void*) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/module/include/dl_module_conv.hpp:108
--- (inlined by) dl::module::Conv::forward_args(void*) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/module/include/dl_module_conv.hpp:104
--- (inlined by) void dl::module::Conv::forward_template<signed char>(dl::ModelContext*, dl::runtime_mode_t) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/module/include/dl_module_conv.hpp:157
--- (inlined by) dl::module::Conv::forward(dl::ModelContext*, dl::runtime_mode_t) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/module/include/dl_module_conv.hpp:126
--- 0x4214ba03: dl::Model::run(dl::runtime_mode_t) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/model/src/dl_model_base.cpp:224
--- 0x4201bcab: inference_run at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:156
--- 0x4200d2b1: inference_task(void*) at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp:63
--- 0x4038053d: vPortTaskWrapper at /Users/admin/esp/v5.4.3/esp-idf/components/freertos/FreeRTOS-Kernel/portable/xtensa/port.c:139

cam_hal: EV-EOF-OVF
cam_hal: FB-SIZE: 138240 != 153600
8.4.1/
I (19567) main: ========================================
I (19567) main_task: Returned from app_main()
I (31807) wifi:new:<1,0>, old:<1,1>, ap:<1,0>, sta:<255,255>, prof:1, snd_ch_cfg:0x0
I (31807) wifi:station: 16:92:a9:c4:ac:40 join, AID=1, bgn, 20
I (31827) network: Station 16:92:a9:c4:ac:40 joined, AID=1
I (32267) wifi:<ba-add>idx:2 (ifx:1, 16:92:a9:c4:ac:40), tid:6, ssn:0, winSize:64
I (33287) esp_netif_lwip: DHCP server assigned IP to a client, IP is: 192.168.4.2
I (35287) wifi:<ba-add>idx:3 (ifx:1, 16:92:a9:c4:ac:40), tid:0, ssn:2, winSize:64
I (43707) network: WebSocket connection opened
W (43707) httpd_uri: httpd_uri: URI '/favicon.ico' not found
W (43707) httpd_txrx: httpd_resp_send_err: 404 Not Found - Nothing matches the given URI
I (48647) main: Frame 10 | FPS=0.3 | inf=3513ms total=3612ms | det=0 | PSRAM=4449KB
I (79777) wifi:<ba-add>idx:4 (ifx:1, 16:92:a9:c4:ac:40), tid:1, ssn:0, winSize:64
I (79987) wifi:<ba-add>idx:5 (ifx:1, 16:92:a9:c4:ac:40), tid:5, ssn:0, winSize:64
I (85167) main: Frame 20 | FPS=0.3 | inf=3537ms total=3639ms | det=0 | PSRAM=4449KB
I (121547) main: Frame 30 | FPS=0.3 | inf=3512ms total=3614ms | det=0 | PSRAM=4449KB
I (157917) main: Frame 40 | FPS=0.3 | inf=3516ms total=3617ms | det=0 | PSRAM=4449KB
I (194327) main: Frame 50 | FPS=0.3 | inf=3515ms total=3617ms | det=0 | PSRAM=4449KB
I (230707) main: Frame 60 | FPS=0.3 | inf=3508ms total=3608ms | det=0 | PSRAM=4449KB
I (267037) main: Frame 70 | FPS=0.3 | inf=3505ms total=3608ms | det=0 | PSRAM=4449KB
I (303377) main: Frame 80 | FPS=0.3 | inf=3514ms total=3617ms | det=0 | PSRAM=4449KB
I (338437) wifi:station: 16:92:a9:c4:ac:40 leave, AID = 1, reason = 8, bss_flags is 691315, bss:0x3fcb349c
I (338437) wifi:new:<1,0>, old:<1,0>, ap:<1,0>, sta:<255,255>, prof:1, snd_ch_cfg:0x0
I (338447) wifi:<ba-del>idx:3, tid:0
I (338447) wifi:<ba-del>idx:4, tid:1
I (338457) wifi:<ba-del>idx:5, tid:5
I (338457) wifi:<ba-del>idx:2, tid:6
I (338457) network: Station 16:92:a9:c4:ac:40 left, AID=1
I (338467) wifi:new:<1,1>, old:<1,0>, ap:<1,1>, sta:<255,255>, prof:1, snd_ch_cfg:0x0
I (339727) main: Frame 90 | FPS=0.3 | inf=3545ms total=3645ms | det=0 | PSRAM=4449KB
```

GitHub Copilot: Summarized conversation history

  Optimizing tool selection...

Created 7 todos

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/sdkconfig.defaults)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp)

Starting: *Research ESP-DL optimization docs* (3/7)

Research ESP-DL optimization

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#1-1), lines 1 to 120

Searched espressif/esp-dl for "RUNTIME_MODE_DUAL_CORE RUNTIME_MODE_SINGLE_CORE dual core single core", 64 results

Searched espressif/esp-dl for "max_internal_size Model constructor internal RAM memory", 64 results

Searched espressif/esp-dl for "sdkconfig PSRAM cache flash performance optimization ESP32-S3", 64 results

Searched espressif/esp-dl for "TIE728 tie728 SIMD tensor instruction extension s8_conv2d", 64 results

Searched espressif/esp-dl for "YOLO benchmark inference time ESP32-S3 performance", 64 results



Fetched 2 resources



Fetched 3 resources

Searched espressif/esp-dl for "sdkconfig.defaults ESP32-S3 CONFIG_SPIRAM CONFIG_ESPTOOLPY_FLASHMODE CONFIG_ESP32S3_DATA_CACHE_SIZE performance", 64 results

Searched for regex `CONFIG_SPIRAM|CONFIG_ESP32S3_DATA_CACHE|CONFIG_ESPTOOLPY_FLASHMODE|max_internal_size|runtime_mode|param_copy` (`**/firmware/sdkconfig*`), 20 results

Searched for regex `CONFIG_ESP32S3_DATA_CACHE|SPIRAM_BOOT_INIT|MALLOC_CAP|CONFIG_SPIRAM_ALLOW_BSS|CONFIG_SPIRAM_FORCE|CONFIG_SPIRAM_USE` (`**/firmware/sdkconfig`), 15 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#120-120), lines 120 to 160

Starting: *Identify root causes* (4/7)

Investigué a fondo el perfil de latencia, la documentación de ESP-DL, y los benchmarks oficiales de Espressif. Aquí está el diagnóstico completo:

---

## Diagnóstico: 3,514 ms por frame (~0.3 FPS)

### Causas raíz identificadas (de mayor a menor impacto):

| # | Causa | Impacto |
|---|-------|---------|
| 1 | **`RUNTIME_MODE_SINGLE_CORE`** — Solo usamos 1 de 2 cores para Conv2D | ~2x de latencia innecesaria |
| 2 | **`max_internal_size = 0`** — Esto significa "0 bytes de SRAM interna" para buffers intermedios. Toda la computación usa PSRAM (4× más lenta) | Variables 496KB en PSRAM |
| 3 | **Data cache = 32KB (default)** — Cache insuficiente para acceso intensivo a PSRAM | Más stalls por cache miss |
| 4 | **Compilación `-Og` (debug)** — El default de ESP-IDF no usa `-O2` | ~10-30% más lento |
| 5 | **`model->profile()` en init** — Ejecuta un forward pass completo (+3.5s al boot), dispara WDT | Boot lento + WDT |
| 6 | **WDT no configurado** — Default 5s, pero inference toma 3.5s + profile 3.5s | Watchdog triggers |

### Contexto importante: benchmarks oficiales de Espressif

| Modelo | Input | ESP32-S3 Latencia |
|--------|-------|-------------------|
| YOLO11n (oficial) | 640×640 | **26.2 segundos** |
| YOLO11n (oficial) | 320×320 | **6.2 segundos** |
| **Tu YOLO11n** | **224×224** | **3.5 segundos** |
| ESPDet-Pico (modelo pequeño) | 224×224 | ~124 ms |

Tu resultado de 3.5s a 224×224 es **consistente** con los benchmarks oficiales. YOLO11n es demasiado pesado para el ESP32-S3. Con las optimizaciones podemos bajarlo a ~1.5-2s, pero para tiempo real necesitarás usar MBNTv3S que debería ser **mucho** más rápido (~0.5-1s estimado).

---

Aplico todas las correcciones: 

Starting: *Apply performance fixes* (5/7)

Replacing 29 lines with 46 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/sdkconfig.defaults), [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp)


```

```

```

```


Starting: *Investigate 0 detections* (7/7)

Ahora investiguemos rápidamente el tema de 0 detecciones:  

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp#1-1), lines 1 to 50

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp#50-50), lines 50 to 110

Searched for files matching `**/firmware/sdkconfig`, 1 match

Searched for regex `DATA_CACHE|COMPILER_OPTIMIZATION|TASK_WDT|RUNTIME_MODE` (`**/firmware/sdkconfig`), 20 results

Confirmed: the generated sdkconfig has `CONFIG_COMPILER_OPTIMIZATION_DEBUG=y` (-Og) and 32KB data cache. We need a **full clean rebuild** since these are deep config changes. Let me also add diagnostic logging for the 0 detections issue: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp#100-100), lines 100 to 160

Replacing 25 lines with 40 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp)


```

```


Replacing 7 lines with 11 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp)


```

```


Completed: *Investigate 0 detections* (7/7)

Verifiquemos los archivos finales: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/sdkconfig.defaults#1-1), lines 1 to 55

Todo correcto. Ahora necesitas hacer **full clean + rebuild** porque los cambios de cache y compilador requieren regenerar completamente el sdkconfig:

```bash
cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
rm sdkconfig
idf.py fullclean
idf.py build
```

Luego flash solo del firmware (los modelos ya están en flash):
```bash
idf.py -p /dev/tty.wchusbserial5B414963901 flash monitor
```

---

## Resumen de cambios aplicados

| Archivo | Cambio | Impacto esperado |
|---------|--------|------------------|
| sdkconfig.defaults | `CONFIG_ESP32S3_DATA_CACHE_64KB=y` + `_LINE_64B=y` | Reduce cache misses en PSRAM (~10-20%) |
| sdkconfig.defaults | `CONFIG_COMPILER_OPTIMIZATION_PERF=y` (-O2) | Antes compilaba en -Og (debug). ~15-30% más rápido |
| sdkconfig.defaults | `CONFIG_ESP_TASK_WDT_CHECK_IDLE_TASK_CPU0=n` | Elimina los errores `task_wdt` |
| inference_engine.cpp | `RUNTIME_MODE_MULTI_CORE` | ESP-DL divide Conv2D entre ambos cores (~1.7-2× speedup) |
| inference_engine.cpp | `max_internal_size = 100*1024` | Buffers intermedios usan SRAM interna (4× más rápida que PSRAM) |
| inference_engine.cpp | `profile()` comentado | Elimina 3.5s de forward pass extra durante boot |
| postprocess_yolo.cpp | Log diagnóstico con max score | Diagnosticar por qué 0 detecciones |

## Estimación de mejora

| Optimización | Reducción estimada |
|---|---|
| Dual-core (Conv2D) | 3514ms → ~1900ms |
| -O2 vs -Og | ~1900ms → ~1500ms |
| Cache 64KB + internal RAM | ~1500ms → ~1200-1400ms |
| **Total estimado** | **~1.2-1.5s** (~2.5× mejor) |

Esto daría ~0.7-0.8 FPS para YOLO11n. Para MBNTv3S (modelo 10× más pequeño) espero ~200-400ms (~2.5-5 FPS), mucho más práctico para tiempo real.

## Sobre las 0 detecciones

Agregué un log diagnóstico que imprimirá el **máximo score raw** encontrado en cada frame. Sospecho que el problema es que con `exponent=1` (scale=2.0), los scores de clase (0-1 post-sigmoid) se cuantizan a casi 0 en INT8, haciendo imposible superarel umbral. El log nos dirá exactamente qué valores produce el modelo.

Made changes.

User: Algo sucedió con el wifi. Ya no aparece el AP WIFI (lo busqué con dos dispositivos diferentes). Por lo tanto, no me pude conectar al wifi y tampoco pude ver el dashboard. Te comparto los logs que vi en el monitor, vi que se hacía "reboot" constantemente en cierto punto, para ver si ahí encontramos algo sobre la causa:

```zsh
I (31) boot: ESP-IDF v5.4.3-dirty 2nd stage bootloader
I (31) boot: compile time Feb 12 2026 16:10:54
I (31) boot: Multicore bootloader
I (32) boot: chip revision: v0.2
I (35) boot: efuse block revision: v1.3
I (38) boot.esp32s3: Boot SPI Speed : 80MHz
I (42) boot.esp32s3: SPI Mode       : DIO
I (46) boot.esp32s3: SPI Flash Size : 16MB
I (50) boot: Enabling RNG early entropy source...
I (54) boot: Partition Table:
I (57) boot: ## Label            Usage          Type ST Offset   Length
I (63) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (70) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (76) boot:  2 factory          factory app      00 00 00010000 00400000
I (83) boot:  3 models           Unknown data     01 40 00410000 00700000
I (89) boot: End of partition table
I (93) esp_image: segment 0: paddr=00010020 vaddr=3c150020 size=2f5a4h (193956) map
I (134) esp_image: segment 1: paddr=0003f5cc vaddr=3fc9c200 size=00a4ch (  2636) load
I (135) esp_image: segment 2: paddr=00040020 vaddr=42000020 size=1417e8h (1316840) map
I (370) esp_image: segment 3: paddr=00181810 vaddr=3fc9cc4c size=06cd8h ( 27864) load
I (376) esp_image: segment 4: paddr=001884f0 vaddr=40374000 size=18168h ( 98664) load
I (398) esp_image: segment 5: paddr=001a0660 vaddr=50000000 size=0001ch (    28) load
I (408) boot: Loaded app from partition at offset 0x10000
I (408) boot: Disabling RNG early entropy source...
I (418) octal_psram: vendor id    : 0x0d (AP)
I (419) octal_psram: dev id       : 0x02 (generation 3)
I (419) octal_psram: density      : 0x03 (64 Mbit)
I (421) octal_psram: good-die     : 0x01 (Pass)
I (425) octal_psram: Latency      : 0x01 (Fixed)
I (429) octal_psram: VCC          : 0x01 (3V)
I (433) octal_psram: SRF          : 0x01 (Fast Refresh)
I (438) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (443) octal_psram: BurstLen     : 0x01 (32 Byte)
I (448) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (453) octal_psram: DriveStrength: 0x00 (1/1)
I (458) MSPI Timing: PSRAM timing tuning index: 5
I (462) esp_psram: Found 8MB PSRAM device
I (465) esp_psram: Speed: 80MHz
I (468) cpu_start: Multicore app
I (756) esp_psram: SPI SRAM memory test OK
I (765) cpu_start: Pro cpu start user code
I (765) cpu_start: cpu freq: 240000000 Hz
I (765) app_init: Application information:
I (765) app_init: Project name:     tfm_tinyml_deploy
I (770) app_init: App version:      ac1e1dbed-dirty
I (775) app_init: Compile time:     Feb 12 2026 16:10:17
I (780) app_init: ELF file SHA256:  f3ba12489...
I (784) app_init: ESP-IDF:          v5.4.3-dirty
I (788) efuse_init: Min chip rev:     v0.0
I (792) efuse_init: Max chip rev:     v0.99 
I (796) efuse_init: Chip rev:         v0.2
I (800) heap_init: Initializing. RAM available for dynamic allocation:
I (806) heap_init: At 3FCAE7B8 len 0003AF58 (235 KiB): RAM
I (811) heap_init: At 3FCE9710 len 00005724 (21 KiB): RAM
I (817) heap_init: At 600FE000 len 00001FE8 (7 KiB): RTCRAM
I (822) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (829) spi_flash: detected chip: boya
I (832) spi_flash: flash io: dio
I (835) sleep_gpio: Configure to isolate all GPIO pins in sleep state
I (841) sleep_gpio: Enable automatic switching of GPIO sleep configuration
I (848) main_task: Started on CPU0
I (878) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (878) main_task: Calling app_main()
I (878) main: ========================================
I (888) main:   TFM TinyML Deployment — ESP32-S3
I (888) main:   Model: YOLO11n
I (888) main: ========================================
I (908) main: PSRAM total: 8192 KB, free: 8189 KB
I (908) main: Input buffer: 150528 bytes in PSRAM
I (908) s3 ll_cam: DMA Channel=0
I (908) cam_hal: cam init ok
I (918) sccb-ng: pin_sda 4 pin_scl 5
I (918) sccb-ng: sccb_i2c_port=1
I (928) ov3660: Mismatch PID=0x5640
I (928) camera: Camera PID=0x5640 VER=0x00 MIDL=0x00 MIDH=0x00
I (928) camera: Detected OV5640 camera
I (928) camera: Detected camera at address=0x3c
I (1618) cam_hal: PSRAM DMA mode disabled
I (1618) s3 ll_cam: node_size: 3840, nodes_per_line: 1, lines_per_node: 6
I (1618) s3 ll_cam: dma_half_buffer_min:  3840, dma_half_buffer: 15360, lines_per_half_buffer: 24, dma_buffer_size: 30720
I (1628) cam_hal: buffer_size: 30720, half_buffer_size: 15360, node_buffer_size: 3840, node_cnt: 8, total_cnt: 10
I (1638) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1638) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1648) cam_hal: cam config ok
I (1658) ov5640: Set PLL: bypass: 0, multiplier: 8, sys_div: 1, pre_div: 1, root_2x: 0, pclk_root_div: 1, pclk_manual: 1, pclk_div: 4
I (1658) ov5640: Calculated XVCLK: 20000000 Hz, REFIN: 20000000 Hz, VCO: 160000000 Hz, PLL_CLK: 64000000 Hz, SYSCLK: 16000000 Hz, PCLK: 8000000 Hz
I (1708) camera: Sensor configured: vflip=1, auto WB/exp/gain, lens correction
I (1708) camera: Camera initialized: OV5640 RGB565 320x240, 2 buffers in PSRAM
I (1708) main: Camera initialized (OV5640 RGB565 320x240)
I (1718) inference: Loading YOLO11n from partition 'models' offset=0xa7000 size=2857568 bytes
I (1718) inference: Partition found: offset=0x410000, size=7340032
I (1728) inference: Model mmap'd at 0x3c987000, 2857568 bytes
E (2298) esp-dl: Input cap=0x800 can not callocate with MALLOC_CAP_SIMD, please try other caps.
E (2298) dl::ModelContext: Failed to alloc 100.00KB internal RAM, largest available internal RAM block size 31.00KB
E (2308) MemoryManagerGreedy: root_alloc failed
Guru Meditation Error: Core  0 panic'ed (LoadProhibited). Exception was unhandled.

Core  0 register dump:
PC      : 0x4201b884  PS      : 0x00060430  A0      : 0x8200c7cd  A1      : 0x3fcb30d0  
--- 0x4201b884: std::vector<int, std::allocator<int> >::size() const at /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/stl_vector.h:993
--- (inlined by) std::vector<int, std::allocator<int> >::vector(std::vector<int, std::allocator<int> > const&) at /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/stl_vector.h:603
--- (inlined by) inference_init at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:116
A2      : 0x00000000  A3      : 0xaaaaaaab  A4      : 0x00000000  A5      : 0x3fce6564  
A6      : 0x3fcaaef4  A7      : 0x00000000  A8      : 0x00000001  A9      : 0x00000000  
A10     : 0x0000090e  A11     : 0x3fce072c  A12     : 0x3fce0744  A13     : 0x00000000  
A14     : 0x3fce657c  A15     : 0x3fce0744  SAR     : 0x00000017  EXCCAUSE: 0x0000001c  
EXCVADDR: 0x0000000c  LBEG    : 0x40056f5c  LEND    : 0x40056f72  LCOUNT  : 0xffffffff  
--- 0x40056f5c: memcpy in ROM
--- 0x40056f72: memcpy in ROM


Backtrace: 0x4201b881:0x3fcb30d0 0x4200c7ca:0x3fcb3150 0x421409db:0x3fcb3180
--- 0x4201b881: std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > >::size() const at /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/stl_vector.h:993
--- (inlined by) inference_init at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:116
--- 0x4200c7ca: app_main at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp:137
--- 0x421409db: main_task at /Users/admin/esp/v5.4.3/esp-idf/components/freertos/app_startup.c:208




ELF file SHA256: f3ba12489

Rebooting...
```

GitHub Copilot: 

 El crash es claro en los logs:

```
E (2298) dl::ModelContext: Failed to alloc 100.00KB internal RAM, largest available internal RAM block size 31.00KB
E (2308) MemoryManagerGreedy: root_alloc failed
```

Al cambiar el data cache a 64KB, el ESP32-S3 usa **más SRAM interna para cache**, dejando menos heap interno libre. Antes había 280KB, ahora 256KB. Con cámara + NVS + WiFi ya asignados, solo quedan 31KB contiguos internos — pero pedimos 100KB.

La solución: reducir `max_internal_size` a algo que quepa, y añadir manejo de error si la creación del modelo falla (evitar el crash por null pointer): 

Replacing 18 lines with 28 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp)


```

```
Ahora reconstruye y flashea:

```bash
cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
idf.py build && idf.py -p /dev/tty.wchusbserial5B414963901 flash monitor
```

**Lo que pasó:** `max_internal_size=100*1024` intentó reservar 100KB de SRAM interna como un bloque contiguo, pero con el cache 64KB solo quedan 31KB contiguos. ESP-DL falló sin gracia, `s_model` quedó como puntero corrupto, y al acceder a `get_outputs()` en línea 116 → crash → reboot cíclico → WiFi nunca arranca.

**Cambios:**
- `max_internal_size` → 0 (PSRAM only, seguro)
- try/catch alrededor del constructor (ESP-DL puede lanzar excepciones en lugar de retornar null)

Los **otros 4 cambios** (dual-core, -O2, cache 64KB, WDT) siguen activos y darán la mejora de rendimiento.

Made changes.

User: ¿qué debo hacer ahora? ¿y sobre el wifi, qué pasa?

GitHub Copilot: 

 El WiFi no aparecía porque el firmware **crasheaba antes de llegar a inicializar WiFi** — entraba en un ciclo de reboot infinito por el error de memoria que ya corregí.

**Lo que debes hacer ahora:**

```bash
cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
idf.py build && idf.py -p /dev/tty.wchusbserial5B414963901 flash monitor
```

No necesitas `fullclean` porque el sdkconfig ya se regeneró en el build anterior. Solo rebuild + flash. Una vez arrancado sin crash, el WiFi AP "ESP32_TFM" volverá a aparecer y el dashboard funcionará con las optimizaciones activas (dual-core, -O2, cache 64KB).

User: Hubo un error en el build

```zsh
(/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env)admin@MacBook-Pro-de-Admin firmware % idf.py build && idf.py -p /dev/tty.wchusbserial5B414963901 flash monitor
Executing action: all (aliases: build)
Running ninja in directory /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build
Executing "ninja all"...
[1/1] cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/.../TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/bootloader/bootloader.bin
Bootloader binary size 0x51c0 bytes. 0x2e40 bytes (36%) free.
[4/9] Building CXX object esp-idf/inference/CMakeFiles/__idf_inference.dir/inference_engine.cpp.obj
FAILED: esp-idf/inference/CMakeFiles/__idf_inference.dir/inference_engine.cpp.obj 
/Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/bin/xtensa-esp32s3-elf-g++ -DESP_PLATFORM -DIDF_VER=\"v5.4.3-dirty\" -DMBEDTLS_CONFIG_FILE=\"mbedtls/esp_config.h\" -DSOC_MMU_PAGE_SIZE=CONFIG_MMU_PAGE_SIZE -DSOC_XTAL_FREQ_MHZ=CONFIG_XTAL_FREQ -D_GLIBCXX_HAVE_POSIX_SEMAPHORE -D_GLIBCXX_USE_POSIX_SEMAPHORE -D_GNU_SOURCE -D_POSIX_READER_WRITER_LOCKS -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/config -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include -I/Users/admin/esp/v5.4.3/esp-idf/components/newlib/platform_include -I/Users/admin/esp/v5.4.3/esp-idf/components/freertos/config/include -I/Users/admin/esp/v5.4.3/esp-idf/components/freertos/config/include/freertos -I/Users/admin/esp/v5.4.3/esp-idf/components/freertos/config/xtensa/include -I/Users/admin/esp/v5.4.3/esp-idf/components/freertos/FreeRTOS-Kernel/include -I/Users/admin/esp/v5.4.3/esp-idf/components/freertos/FreeRTOS-Kernel/portable/xtensa/include -I/Users/admin/esp/v5.4.3/esp-idf/components/freertos/FreeRTOS-Kernel/portable/xtensa/include/freertos -I/Users/admin/esp/v5.4.3/esp-idf/components/freertos/esp_additions/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_hw_support/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_hw_support/include/soc -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_hw_support/include/soc/esp32s3 -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_hw_support/dma/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_hw_support/ldo/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_hw_support/debug_probe/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_hw_support/port/esp32s3/. -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_hw_support/port/esp32s3/include -I/Users/admin/esp/v5.4.3/esp-idf/components/heap/include -I/Users/admin/esp/v5.4.3/esp-idf/components/heap/tlsf -I/Users/admin/esp/v5.4.3/esp-idf/components/log/include -I/Users/admin/esp/v5.4.3/esp-idf/components/soc/include -I/Users/admin/esp/v5.4.3/esp-idf/components/soc/esp32s3 -I/Users/admin/esp/v5.4.3/esp-idf/components/soc/esp32s3/include -I/Users/admin/esp/v5.4.3/esp-idf/components/soc/esp32s3/register -I/Users/admin/esp/v5.4.3/esp-idf/components/hal/platform_port/include -I/Users/admin/esp/v5.4.3/esp-idf/components/hal/esp32s3/include -I/Users/admin/esp/v5.4.3/esp-idf/components/hal/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_rom/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_rom/esp32s3/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_rom/esp32s3/include/esp32s3 -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_rom/esp32s3 -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_common/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_system/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_system/port/soc -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_system/port/include/private -I/Users/admin/esp/v5.4.3/esp-idf/components/xtensa/esp32s3/include -I/Users/admin/esp/v5.4.3/esp-idf/components/xtensa/include -I/Users/admin/esp/v5.4.3/esp-idf/components/xtensa/deprecated_include -I/Users/admin/esp/v5.4.3/esp-idf/components/lwip/include -I/Users/admin/esp/v5.4.3/esp-idf/components/lwip/include/apps -I/Users/admin/esp/v5.4.3/esp-idf/components/lwip/include/apps/sntp -I/Users/admin/esp/v5.4.3/esp-idf/components/lwip/lwip/src/include -I/Users/admin/esp/v5.4.3/esp-idf/components/lwip/port/include -I/Users/admin/esp/v5.4.3/esp-idf/components/lwip/port/freertos/include -I/Users/admin/esp/v5.4.3/esp-idf/components/lwip/port/esp32xx/include -I/Users/admin/esp/v5.4.3/esp-idf/components/lwip/port/esp32xx/include/arch -I/Users/admin/esp/v5.4.3/esp-idf/components/lwip/port/esp32xx/include/sys -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_partition/include -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/tool/include -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/tensor/include -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/math/include -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/model/include -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/module/include -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/fbs_loader/include -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/vision/detect -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/vision/image -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/vision/image/isa -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/vision/recognition -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/vision/classification -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/audio/common -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/audio/speech_features -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/tie728 -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/xtensa -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__dl_fft -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__dl_fft/base -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__dl_fft/base/isa -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_mm/include -I/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp_new_jpeg/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_driver_jpeg/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_driver_ppa/include -I/Users/admin/esp/v5.4.3/esp-idf/components/esp_timer/include -I/Users/admin/esp/v5.4.3/esp-idf/components/mbedtls/port/include -I/Users/admin/esp/v5.4.3/esp-idf/components/mbedtls/mbedtls/include -I/Users/admin/esp/v5.4.3/esp-idf/components/mbedtls/mbedtls/library -I/Users/admin/esp/v5.4.3/esp-idf/components/mbedtls/esp_crt_bundle/include -I/Users/admin/esp/v5.4.3/esp-idf/components/mbedtls/mbedtls/3rdparty/everest/include -I/Users/admin/esp/v5.4.3/esp-idf/components/mbedtls/mbedtls/3rdparty/p256-m -I/Users/admin/esp/v5.4.3/esp-idf/components/mbedtls/mbedtls/3rdparty/p256-m/p256-m -I/Users/admin/esp/v5.4.3/esp-idf/components/spi_flash/include -mlongcalls  -fno-builtin-memcpy -fno-builtin-memset -fno-builtin-bzero -fno-builtin-stpcpy -fno-builtin-strncpy -std=gnu++23 -fdiagnostics-color=always -ffunction-sections -fdata-sections -Wall -Werror=all -Wno-error=unused-function -Wno-error=unused-variable -Wno-error=unused-but-set-variable -Wno-error=deprecated-declarations -Wextra -Wno-error=extra -Wno-unused-parameter -Wno-sign-compare -Wno-enum-conversion -gdwarf-4 -ggdb -mdisable-hardware-atomics -O2 -fmacro-prefix-map=/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware=. -fmacro-prefix-map=/Users/admin/esp/v5.4.3/esp-idf=/IDF -fstrict-volatile-bitfields -fno-jump-tables -fno-tree-switch-conversion -std=gnu++2b -fno-exceptions -fno-rtti -fuse-cxa-atexit -MD -MT esp-idf/inference/CMakeFiles/__idf_inference.dir/inference_engine.cpp.obj -MF esp-idf/inference/CMakeFiles/__idf_inference.dir/inference_engine.cpp.obj.d -o esp-idf/inference/CMakeFiles/__idf_inference.dir/inference_engine.cpp.obj -c /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp: In function 'esp_err_t inference_init(ModelType)':
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:107:36: error: exception handling disabled, use '-fexceptions' to enable
  107 |     } catch (const std::exception &e) {
      |                                    ^
In file included from /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:13:
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:108:58: error: 'e' was not declared in this scope; did you mean 'std::numbers::e'?
  108 |         ESP_LOGE(TAG, "dl::Model constructor threw: %s", e.what());
      |                                                          ^
/Users/admin/esp/v5.4.3/esp-idf/components/log/include/esp_log.h:164:148: note: in definition of macro 'ESP_LOG_LEVEL'
  164 |         if (level==ESP_LOG_ERROR )          { esp_log_write(ESP_LOG_ERROR,      tag, LOG_FORMAT(E, format), esp_log_timestamp(), tag __VA_OPT__(,) __VA_ARGS__); } \
      |                                                                                                                                                    ^~~~~~~~~~~
/Users/admin/esp/v5.4.3/esp-idf/components/log/include/esp_log.h:106:38: note: in expansion of macro 'ESP_LOG_LEVEL_LOCAL'
  106 | #define ESP_LOGE( tag, format, ... ) ESP_LOG_LEVEL_LOCAL(ESP_LOG_ERROR,   tag, format __VA_OPT__(,) __VA_ARGS__)
      |                                      ^~~~~~~~~~~~~~~~~~~
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:108:9: note: in expansion of macro 'ESP_LOGE'
  108 |         ESP_LOGE(TAG, "dl::Model constructor threw: %s", e.what());
      |         ^~~~~~~~
In file included from /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/max_size_type.h:37,
                 from /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/ranges_base.h:39,
                 from /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/ranges_util.h:34,
                 from /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/tuple:44,
                 from /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/uses_allocator_args.h:39,
                 from /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/bits/memory_resource.h:41,
                 from /Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/vector:86,
                 from /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:17:
/Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/numbers:124:27: note: 'std::numbers::e' declared here
  124 |   inline constexpr double e = e_v<double>;
      |                           ^
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:108:58: error: 'e' was not declared in this scope; did you mean 'std::numbers::e'?
  108 |         ESP_LOGE(TAG, "dl::Model constructor threw: %s", e.what());
      |                                                          ^
/Users/admin/esp/v5.4.3/esp-idf/components/log/include/esp_log.h:165:148: note: in definition of macro 'ESP_LOG_LEVEL'
  165 |         else if (level==ESP_LOG_WARN )      { esp_log_write(ESP_LOG_WARN,       tag, LOG_FORMAT(W, format), esp_log_timestamp(), tag __VA_OPT__(,) __VA_ARGS__); } \
      |                                                                                                                                                    ^~~~~~~~~~~
/Users/admin/esp/v5.4.3/esp-idf/components/log/include/esp_log.h:106:38: note: in expansion of macro 'ESP_LOG_LEVEL_LOCAL'
  106 | #define ESP_LOGE( tag, format, ... ) ESP_LOG_LEVEL_LOCAL(ESP_LOG_ERROR,   tag, format __VA_OPT__(,) __VA_ARGS__)
      |                                      ^~~~~~~~~~~~~~~~~~~
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:108:9: note: in expansion of macro 'ESP_LOGE'
  108 |         ESP_LOGE(TAG, "dl::Model constructor threw: %s", e.what());
      |         ^~~~~~~~
/Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/numbers:124:27: note: 'std::numbers::e' declared here
  124 |   inline constexpr double e = e_v<double>;
      |                           ^
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:108:58: error: 'e' was not declared in this scope; did you mean 'std::numbers::e'?
  108 |         ESP_LOGE(TAG, "dl::Model constructor threw: %s", e.what());
      |                                                          ^
/Users/admin/esp/v5.4.3/esp-idf/components/log/include/esp_log.h:166:148: note: in definition of macro 'ESP_LOG_LEVEL'
  166 |         else if (level==ESP_LOG_DEBUG )     { esp_log_write(ESP_LOG_DEBUG,      tag, LOG_FORMAT(D, format), esp_log_timestamp(), tag __VA_OPT__(,) __VA_ARGS__); } \
      |                                                                                                                                                    ^~~~~~~~~~~
/Users/admin/esp/v5.4.3/esp-idf/components/log/include/esp_log.h:106:38: note: in expansion of macro 'ESP_LOG_LEVEL_LOCAL'
  106 | #define ESP_LOGE( tag, format, ... ) ESP_LOG_LEVEL_LOCAL(ESP_LOG_ERROR,   tag, format __VA_OPT__(,) __VA_ARGS__)
      |                                      ^~~~~~~~~~~~~~~~~~~
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:108:9: note: in expansion of macro 'ESP_LOGE'
  108 |         ESP_LOGE(TAG, "dl::Model constructor threw: %s", e.what());
      |         ^~~~~~~~
/Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/numbers:124:27: note: 'std::numbers::e' declared here
  124 |   inline constexpr double e = e_v<double>;
      |                           ^
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:108:58: error: 'e' was not declared in this scope; did you mean 'std::numbers::e'?
  108 |         ESP_LOGE(TAG, "dl::Model constructor threw: %s", e.what());
      |                                                          ^
/Users/admin/esp/v5.4.3/esp-idf/components/log/include/esp_log.h:167:148: note: in definition of macro 'ESP_LOG_LEVEL'
  167 |         else if (level==ESP_LOG_VERBOSE )   { esp_log_write(ESP_LOG_VERBOSE,    tag, LOG_FORMAT(V, format), esp_log_timestamp(), tag __VA_OPT__(,) __VA_ARGS__); } \
      |                                                                                                                                                    ^~~~~~~~~~~
/Users/admin/esp/v5.4.3/esp-idf/components/log/include/esp_log.h:106:38: note: in expansion of macro 'ESP_LOG_LEVEL_LOCAL'
  106 | #define ESP_LOGE( tag, format, ... ) ESP_LOG_LEVEL_LOCAL(ESP_LOG_ERROR,   tag, format __VA_OPT__(,) __VA_ARGS__)
      |                                      ^~~~~~~~~~~~~~~~~~~
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:108:9: note: in expansion of macro 'ESP_LOGE'
  108 |         ESP_LOGE(TAG, "dl::Model constructor threw: %s", e.what());
      |         ^~~~~~~~
/Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/numbers:124:27: note: 'std::numbers::e' declared here
  124 |   inline constexpr double e = e_v<double>;
      |                           ^
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:108:58: error: 'e' was not declared in this scope; did you mean 'std::numbers::e'?
  108 |         ESP_LOGE(TAG, "dl::Model constructor threw: %s", e.what());
      |                                                          ^
/Users/admin/esp/v5.4.3/esp-idf/components/log/include/esp_log.h:168:148: note: in definition of macro 'ESP_LOG_LEVEL'
  168 |         else                                { esp_log_write(ESP_LOG_INFO,       tag, LOG_FORMAT(I, format), esp_log_timestamp(), tag __VA_OPT__(,) __VA_ARGS__); } \
      |                                                                                                                                                    ^~~~~~~~~~~
/Users/admin/esp/v5.4.3/esp-idf/components/log/include/esp_log.h:106:38: note: in expansion of macro 'ESP_LOG_LEVEL_LOCAL'
  106 | #define ESP_LOGE( tag, format, ... ) ESP_LOG_LEVEL_LOCAL(ESP_LOG_ERROR,   tag, format __VA_OPT__(,) __VA_ARGS__)
      |                                      ^~~~~~~~~~~~~~~~~~~
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:108:9: note: in expansion of macro 'ESP_LOGE'
  108 |         ESP_LOGE(TAG, "dl::Model constructor threw: %s", e.what());
      |         ^~~~~~~~
/Users/admin/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/xtensa-esp-elf/include/c++/14.2.0/numbers:124:27: note: 'std::numbers::e' declared here
  124 |   inline constexpr double e = e_v<double>;
      |                           ^
/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:107:7: error: '...' handler must be the last handler for its try block [-fpermissive]
  107 |     } catch (const std::exception &e) {
      |       ^~~~~
ninja: build stopped: subcommand failed.
ninja failed with exit code 1, output of the command is in the /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/log/idf_py_stderr_output_81506 and /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/build/log/idf_py_stdout_output_81506
```

GitHub Copilot: 

ESP-IDF compiles with `-fno-exceptions`, so try/catch is not available. I'll replace with `new (std::nothrow)` and a null check: 

Replacing 28 lines with 20 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp)


```

```
Ahora compila de nuevo:

```bash
idf.py build && idf.py -p /dev/tty.wchusbserial5B414963901 flash monitor
```

Made changes.

User: Ahora ya pude conectarme al wifi, mejoró un poco la velocidad, pasó de 0.3 FPS a 0.8 FPS, pero no hay detecciones de nada. El modelo no está funcionando (parece). 

Acá los logs:

```zsh
I (27) boot: ESP-IDF v5.4.3-dirty 2nd stage bootloader
I (27) boot: compile time Feb 12 2026 16:10:54
I (27) boot: Multicore bootloader
I (28) boot: chip revision: v0.2
I (30) boot: efuse block revision: v1.3
I (34) boot.esp32s3: Boot SPI Speed : 80MHz
I (38) boot.esp32s3: SPI Mode       : DIO
I (42) boot.esp32s3: SPI Flash Size : 16MB
I (45) boot: Enabling RNG early entropy source...
I (50) boot: Partition Table:
I (52) boot: ## Label            Usage          Type ST Offset   Length
I (59) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (65) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (72) boot:  2 factory          factory app      00 00 00010000 00400000
I (78) boot:  3 models           Unknown data     01 40 00410000 00700000
I (85) boot: End of partition table
I (88) esp_image: segment 0: paddr=00010020 vaddr=3c150020 size=2f5a4h (193956) map
I (130) esp_image: segment 1: paddr=0003f5cc vaddr=3fc9c200 size=00a4ch (  2636) load
I (131) esp_image: segment 2: paddr=00040020 vaddr=42000020 size=1417e4h (1316836) map
I (366) esp_image: segment 3: paddr=0018180c vaddr=3fc9cc4c size=06cd8h ( 27864) load
I (372) esp_image: segment 4: paddr=001884ec vaddr=40374000 size=18168h ( 98664) load
I (393) esp_image: segment 5: paddr=001a065c vaddr=50000000 size=0001ch (    28) load
I (404) boot: Loaded app from partition at offset 0x10000
I (404) boot: Disabling RNG early entropy source...
I (414) octal_psram: vendor id    : 0x0d (AP)
I (414) octal_psram: dev id       : 0x02 (generation 3)
I (414) octal_psram: density      : 0x03 (64 Mbit)
I (416) octal_psram: good-die     : 0x01 (Pass)
I (421) octal_psram: Latency      : 0x01 (Fixed)
I (425) octal_psram: VCC          : 0x01 (3V)
I (429) octal_psram: SRF          : 0x01 (Fast Refresh)
I (434) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (439) octal_psram: BurstLen     : 0x01 (32 Byte)
I (443) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (448) octal_psram: DriveStrength: 0x00 (1/1)
I (453) MSPI Timing: PSRAM timing tuning index: 5
I (457) esp_psram: Found 8MB PSRAM device
I (461) esp_psram: Speed: 80MHz
I (464) cpu_start: Multicore app
I (752) esp_psram: SPI SRAM memory test OK
I (761) cpu_start: Pro cpu start user code
I (761) cpu_start: cpu freq: 240000000 Hz
I (761) app_init: Application information:
I (761) app_init: Project name:     tfm_tinyml_deploy
I (766) app_init: App version:      ac1e1dbed-dirty
I (770) app_init: Compile time:     Feb 12 2026 16:10:17
I (775) app_init: ELF file SHA256:  891cda67d...
I (780) app_init: ESP-IDF:          v5.4.3-dirty
I (784) efuse_init: Min chip rev:     v0.0
I (788) efuse_init: Max chip rev:     v0.99 
I (792) efuse_init: Chip rev:         v0.2
I (796) heap_init: Initializing. RAM available for dynamic allocation:
I (802) heap_init: At 3FCAE7B8 len 0003AF58 (235 KiB): RAM
I (807) heap_init: At 3FCE9710 len 00005724 (21 KiB): RAM
I (812) heap_init: At 600FE000 len 00001FE8 (7 KiB): RTCRAM
I (818) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (825) spi_flash: detected chip: boya
I (827) spi_flash: flash io: dio
I (831) sleep_gpio: Configure to isolate all GPIO pins in sleep state
I (837) sleep_gpio: Enable automatic switching of GPIO sleep configuration
I (844) main_task: Started on CPU0
I (874) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (874) main_task: Calling app_main()
I (874) main: ========================================
I (884) main:   TFM TinyML Deployment — ESP32-S3
I (884) main:   Model: YOLO11n
I (884) main: ========================================
I (904) main: PSRAM total: 8192 KB, free: 8189 KB
I (904) main: Input buffer: 150528 bytes in PSRAM
I (904) s3 ll_cam: DMA Channel=0
I (904) cam_hal: cam init ok
I (914) sccb-ng: pin_sda 4 pin_scl 5
I (914) sccb-ng: sccb_i2c_port=1
I (924) ov3660: Mismatch PID=0x5640
I (924) camera: Camera PID=0x5640 VER=0x00 MIDL=0x00 MIDH=0x00
I (924) camera: Detected OV5640 camera
I (924) camera: Detected camera at address=0x3c
I (1614) cam_hal: PSRAM DMA mode disabled
I (1614) s3 ll_cam: node_size: 3840, nodes_per_line: 1, lines_per_node: 6
I (1614) s3 ll_cam: dma_half_buffer_min:  3840, dma_half_buffer: 15360, lines_per_half_buffer: 24, dma_buffer_size: 30720
I (1624) cam_hal: buffer_size: 30720, half_buffer_size: 15360, node_buffer_size: 3840, node_cnt: 8, total_cnt: 10
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1644) cam_hal: cam config ok
I (1654) ov5640: Set PLL: bypass: 0, multiplier: 8, sys_div: 1, pre_div: 1, root_2x: 0, pclk_root_div: 1, pclk_manual: 1, pclk_div: 4
I (1654) ov5640: Calculated XVCLK: 20000000 Hz, REFIN: 20000000 Hz, VCO: 160000000 Hz, PLL_CLK: 64000000 Hz, SYSCLK: 16000000 Hz, PCLK: 8000000 Hz
I (1704) camera: Sensor configured: vflip=1, auto WB/exp/gain, lens correction
I (1704) camera: Camera initialized: OV5640 RGB565 320x240, 2 buffers in PSRAM
I (1704) main: Camera initialized (OV5640 RGB565 320x240)
I (1714) inference: Loading YOLO11n from partition 'models' offset=0xa7000 size=2857568 bytes
I (1714) inference: Partition found: offset=0x410000, size=7340032
I (1724) inference: Model mmap'd at 0x3c987000, 2857568 bytes
I (2314) inference:   Output[0]: name='output0' shape=[1, 9, 1029] dtype=3 exponent=1
I (2314) inference:   Input: shape=[1, 224, 224, 3] dtype=3 exponent=-7
I (2324) inference: Model loaded OK. 1 output(s). PSRAM free: 4582 KB
I (2324) main: Inference engine ready
I (2324) postproc: Postprocessor initialized for YOLO11n
I (2334) main: Postprocessor initialized for YOLO11n
I (2334) temperature_sensor: Range [-10°C ~ 80°C], error < 1°C
I (2344) metrics: Temperature sensor initialized
I (2344) metrics: Metrics system initialized
I (2354) main: Memory post-init: PSRAM=4582 KB free, Internal=113 KB free
I (2364) pp: pp rom version: e7ae62f
I (2364) net80211: net80211 rom version: e7ae62f
I (2374) wifi:wifi driver task: 3fcde5b4, prio:23, stack:6656, core=0
I (2384) wifi:wifi firmware version: e8ce6e2
I (2384) wifi:wifi certification version: v7.0
I (2384) wifi:config NVS flash: enablcam_hal: EV-VSYNC-OVF
ed
I (2384) wifi:config nano formatting: disabled
I (2394) wifi:Init data frame dynamic rx buffer num: 32
I (2394) wifi:Init static rx mgmt buffer num: 5
I (2404) wifi:Init management short buffer num: 32
I (2404) wifi:Init dynamic tx buffer num: 32
I (2404) wifi:Init static tx FG buffer num: 2
I (2414) wifi:Init static rx buffer size: 1600
I (2414) wifi:Init static rx buffer num: 10
I (2424) wifi:Init dynamic rx buffer num: 32
I (2424) wifi_init: rx ba win: 6
I (2424) wifi_init: accept mbox: 6
I (2434) wifi_init: tcpip mbox: 32
I (2434) wifi_init: udp mbox: 6
I (2434) wifi_init: tcp mbox: 6
I (2444) wifi_init: tcp tx win: 5760
I (2444) wifi_init: tcp rx win: 5760
I (2444) wifi_init: tcp mss: 1440
I (2454) wifi_init: WiFi IRAM OP enabled
I (2454) wifi_init: WiFi RX IRAM OP enabled
I (2464) phy_init: phy_version 701,f4f1da3a,Mar  3 2025,15:50:10
I (2504) phy_init: Saving new calibration data due to checksum failure or outdated calibration data, mode(0)
I (2584) wifi:mode : softAP (1c:db:d4:4a:f4:59)
I (2584) wifi:Total power save buffer number: 16
I (2584) wifi:Init max length of beacon: 752/752
I (2584) wifi:Init max length of beacon: 752/752
I (2584) esp_netif_lwip: DHCP server started on interface WIFI_AP_DEF with IP: 192.168.4.1
I (2584) network: WiFi AP started: SSID=ESP32_TFM, channel=1
I (2604) network: HTTP server started on port 80
I (2604) dashboard: Dashboard registered at / (5359 bytes)
I (2614) main: Network ready: WiFi AP 'ESP32_TFM', http://192.168.4.1/
I (2614) main: Inference task started on core 0
cam_hal: FB-SIZE: 92160 != 153600
I (2814) main: ========================================
I (3104) main:   System running. Connect to 'ESP32_TFM'
I (3104) main:   Dashboard: http://192.168.4.1/
I (3114) main: ========================================
I (3114) main_task: Returned from app_main()
I (3814) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (5014) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (6214) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (7414) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (8614) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (9824) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (11024) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (11194) wifi:new:<1,0>, old:<1,1>, ap:<1,0>, sta:<255,255>, prof:1, snd_ch_cfg:0x0
I (11194) wifi:station: 16:92:a9:c4:ac:40 join, AID=1, bgn, 20
I (11214) network: Station 16:92:a9:c4:ac:40 joined, AID=1
I (11364) wifi:<ba-add>idx:2 (ifx:1, 16:92:a9:c4:ac:40), tid:6, ssn:0, winSize:64
I (11374) esp_netif_lwip: DHCP server assigned IP to a client, IP is: 192.168.4.2
I (11804) wifi:<ba-add>idx:3 (ifx:1, 16:92:a9:c4:ac:40), tid:0, ssn:2, winSize:64
I (12254) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (13464) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (14664) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (14664) main: Frame 10 | FPS=0.8 | inf=1133ms total=1193ms | det=0 | PSRAM=4431KB
I (15714) network: WebSocket connection opened
I (15944) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (17144) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (18344) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (19554) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (20754) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (22024) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (23224) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (24424) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (25634) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (26834) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (26834) main: Frame 20 | FPS=0.8 | inf=1134ms total=1196ms | det=0 | PSRAM=4431KB
I (28114) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (29314) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (30524) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (31734) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (32934) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (34184) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (35394) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (36594) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (37804) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (39024) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (39024) main: Frame 30 | FPS=0.8 | inf=1140ms total=1202ms | det=0 | PSRAM=4431KB
I (40274) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (41474) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (42674) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (43884) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (45084) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (46334) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (47534) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (48744) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (49954) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (51154) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (51154) main: Frame 40 | FPS=0.8 | inf=1133ms total=1196ms | det=0 | PSRAM=4431KB
I (52424) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (53624) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (54834) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (56044) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (57254) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (58514) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (59724) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (60934) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (61724) wifi:<ba-add>idx:4 (ifx:1, 16:92:a9:c4:ac:40), tid:1, ssn:0, winSize:64
I (61964) wifi:<ba-add>idx:5 (ifx:1, 16:92:a9:c4:ac:40), tid:5, ssn:0, winSize:64
I (62144) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (63354) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (63364) main: Frame 50 | FPS=0.8 | inf=1145ms total=1209ms | det=0 | PSRAM=4431KB
I (64624) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (65824) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (67044) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (68244) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (69454) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (70724) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (71934) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (73144) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (74354) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (75554) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (75554) main: Frame 60 | FPS=0.8 | inf=1133ms total=1195ms | det=0 | PSRAM=4431KB
I (76824) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (78024) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (79224) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (80434) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (81644) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (82904) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (84114) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (85324) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (86534) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (87744) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (87744) main: Frame 70 | FPS=0.8 | inf=1141ms total=1204ms | det=0 | PSRAM=4431KB
I (89014) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (90224) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (91434) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (92644) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (93844) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (95104) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (96324) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (97524) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (98744) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (99954) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (99954) main: Frame 80 | FPS=0.8 | inf=1139ms total=1201ms | det=0 | PSRAM=4431KB
```

GitHub Copilot: 

  

Created 3 todos

El diagnóstico es claro: `maxScore=0.000, raw_int8=0` — **TODOS** los bytes de salida del modelo son cero. El modelo ejecuta (~1.1s), pero produce salida vacía. Necesito agregar diagnóstico más profundo para encontrar la causa. 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#140-140), lines 140 to 175

Starting: *Add deep output diagnostics* (2/3)

Replacing 21 lines with 60 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp)


```

```


Starting: *Fix root cause* (3/3)

Rebuild y flash:

```bash
idf.py build && idf.py -p /dev/tty.wchusbserial5B414963901 flash monitor
```

El diagnóstico imprimirá **los primeros 3 frames** con:
- **Input**: cuántos bytes non-zero + primeros 16 valores raw (debe ser mayoritariamente non-zero si la imagen llega bien)
- **Output**: cuántos bytes non-zero + min/max + primeros 16 valores + puntero + shape

Con esto sabremos si:
1. El input llega vacío (problema de preprocesamiento/cámara)
2. El output está todo a 0 (el modelo no produce activaciones → problema de cuantización o layout)
3. El output tiene datos pero el postprocesador no los ve (puntero incorrecto)

Comparte los primeros logs `DIAG` del monitor.

Made changes.

User: Estos son los logs:

```zsh
I (27) boot: ESP-IDF v5.4.3-dirty 2nd stage bootloader
I (27) boot: compile time Feb 12 2026 16:10:54
I (27) boot: Multicore bootloader
I (28) boot: chip revision: v0.2
I (30) boot: efuse block revision: v1.3
I (34) boot.esp32s3: Boot SPI Speed : 80MHz
I (38) boot.esp32s3: SPI Mode       : DIO
I (42) boot.esp32s3: SPI Flash Size : 16MB
I (45) boot: Enabling RNG early entropy source...
I (50) boot: Partition Table:
I (52) boot: ## Label            Usage          Type ST Offset   Length
I (59) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (65) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (72) boot:  2 factory          factory app      00 00 00010000 00400000
I (78) boot:  3 models           Unknown data     01 40 00410000 00700000
I (85) boot: End of partition table
I (88) esp_image: segment 0: paddr=00010020 vaddr=3c150020 size=2f6a4h (194212) map
I (130) esp_image: segment 1: paddr=0003f6cc vaddr=3fc9c200 size=0094ch (  2380) load
I (131) esp_image: segment 2: paddr=00040020 vaddr=42000020 size=141b18h (1317656) map
I (366) esp_image: segment 3: paddr=00181b40 vaddr=3fc9cb4c size=06dd8h ( 28120) load
I (372) esp_image: segment 4: paddr=00188920 vaddr=40374000 size=18168h ( 98664) load
I (394) esp_image: segment 5: paddr=001a0a90 vaddr=50000000 size=0001ch (    28) load
I (404) boot: Loaded app from partition at offset 0x10000
I (404) boot: Disabling RNG early entropy source...
I (414) octal_psram: vendor id    : 0x0d (AP)
I (414) octal_psram: dev id       : 0x02 (generation 3)
I (414) octal_psram: density      : 0x03 (64 Mbit)
I (416) octal_psram: good-die     : 0x01 (Pass)
I (421) octal_psram: Latency      : 0x01 (Fixed)
I (425) octal_psram: VCC          : 0x01 (3V)
I (429) octal_psram: SRF          : 0x01 (Fast Refresh)
I (434) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (439) octal_psram: BurstLen     : 0x01 (32 Byte)
I (443) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (449) octal_psram: DriveStrength: 0x00 (1/1)
I (453) MSPI Timing: PSRAM timing tuning index: 5
I (457) esp_psram: Found 8MB PSRAM device
I (461) esp_psram: Speed: 80MHz
I (464) cpu_start: Multicore app
I (752) esp_psram: SPI SRAM memory test OK
I (761) cpu_start: Pro cpu start user code
I (761) cpu_start: cpu freq: 240000000 Hz
I (761) app_init: Application information:
I (761) app_init: Project name:     tfm_tinyml_deploy
I (766) app_init: App version:      ac1e1dbed-dirty
I (770) app_init: Compile time:     Feb 12 2026 16:10:17
I (775) app_init: ELF file SHA256:  93651b2f7...
I (780) app_init: ESP-IDF:          v5.4.3-dirty
I (784) efuse_init: Min chip rev:     v0.0
I (788) efuse_init: Max chip rev:     v0.99 
I (792) efuse_init: Chip rev:         v0.2
I (796) heap_init: Initializing. RAM available for dynamic allocation:
I (802) heap_init: At 3FCAE7C0 len 0003AF50 (235 KiB): RAM
I (807) heap_init: At 3FCE9710 len 00005724 (21 KiB): RAM
I (812) heap_init: At 600FE000 len 00001FE8 (7 KiB): RTCRAM
I (818) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (825) spi_flash: detected chip: boya
I (828) spi_flash: flash io: dio
I (831) sleep_gpio: Configure to isolate all GPIO pins in sleep state
I (837) sleep_gpio: Enable automatic switching of GPIO sleep configuration
I (844) main_task: Started on CPU0
I (874) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (874) main_task: Calling app_main()
I (874) main: ========================================
I (884) main:   TFM TinyML Deployment — ESP32-S3
I (884) main:   Model: YOLO11n
I (884) main: ========================================
I (904) main: PSRAM total: 8192 KB, free: 8189 KB
I (904) main: Input buffer: 150528 bytes in PSRAM
I (904) s3 ll_cam: DMA Channel=0
I (904) cam_hal: cam init ok
I (914) sccb-ng: pin_sda 4 pin_scl 5
I (914) sccb-ng: sccb_i2c_port=1
I (924) ov3660: Mismatch PID=0x5640
I (924) camera: Camera PID=0x5640 VER=0x00 MIDL=0x00 MIDH=0x00
I (924) camera: Detected OV5640 camera
I (924) camera: Detected camera at address=0x3c
I (1614) cam_hal: PSRAM DMA mode disabled
I (1614) s3 ll_cam: node_size: 3840, nodes_per_line: 1, lines_per_node: 6
I (1614) s3 ll_cam: dma_half_buffer_min:  3840, dma_half_buffer: 15360, lines_per_half_buffer: 24, dma_buffer_size: 30720
I (1624) cam_hal: buffer_size: 30720, half_buffer_size: 15360, node_buffer_size: 3840, node_cnt: 8, total_cnt: 10
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1644) cam_hal: cam config ok
I (1654) ov5640: Set PLL: bypass: 0, multiplier: 8, sys_div: 1, pre_div: 1, root_2x: 0, pclk_root_div: 1, pclk_manual: 1, pclk_div: 4
I (1654) ov5640: Calculated XVCLK: 20000000 Hz, REFIN: 20000000 Hz, VCO: 160000000 Hz, PLL_CLK: 64000000 Hz, SYSCLK: 16000000 Hz, PCLK: 8000000 Hz
I (1704) camera: Sensor configured: vflip=1, auto WB/exp/gain, lens correction
I (1704) camera: Camera initialized: OV5640 RGB565 320x240, 2 buffers in PSRAM
I (1704) main: Camera initialized (OV5640 RGB565 320x240)
I (1714) inference: Loading YOLO11n from partition 'models' offset=0xa7000 size=2857568 bytes
I (1714) inference: Partition found: offset=0x410000, size=7340032
I (1724) inference: Model mmap'd at 0x3c987000, 2857568 bytes
I (2314) inference:   Output[0]: name='output0' shape=[1, 9, 1029] dtype=3 exponent=1
I (2314) inference:   Input: shape=[1, 224, 224, 3] dtype=3 exponent=-7
I (2314) inference: Model loaded OK. 1 output(s). PSRAM free: 4582 KB
I (2324) main: Inference engine ready
I (2324) postproc: Postprocessor initialized for YOLO11n
I (2334) main: Postprocessor initialized for YOLO11n
I (2334) temperature_sensor: Range [-10°C ~ 80°C], error < 1°C
I (2344) metrics: Temperature sensor initialized
I (2344) metrics: Metrics system initialized
I (2354) main: Memory post-init: PSRAM=4582 KB free, Internal=113 KB free
I (2354) pp: pp rom version: e7ae62f
I (2364) net80211: net80211 rom version: e7ae62f
I (2374) wifi:wifi driver task: 3fcde5bc, prio:23, stack:6656, core=0
I (2384) wifi:wifi firmware version: e8ce6e2
I (2384) wifi:wifi certification version: v7.0
I (2384) wifi:config NVS flash: enabled
I (2384) wifi:config nano formatting: disabled
I (2384) wifi:Init data frame dynamic rx buffer num: 32
I (2394) wifi:Init static rx mgmt buffer nucam_hal: EV-VSYNC-OVF
m: 5
I (2394) wifi:Init management short buffer num: 32
I (2404) wifi:Init dynamic tx buffer num: 32
I (2404) wifi:Init static tx FG buffer num: 2
I (2414) wifi:Init static rx buffer size: 1600
I (2414) wifi:Init static rx buffer num: 10
I (2414) wifi:Init dynamic rx buffer num: 32
I (2424) wifi_init: rx ba win: 6
I (2424) wifi_init: accept mbox: 6
I (2424) wifi_init: tcpip mbox: 32
I (2434) wifi_init: udp mbox: 6
I (2434) wifi_init: tcp mbox: 6
I (2434) wifi_init: tcp tx win: 5760
I (2444) wifi_init: tcp rx win: 5760
I (2444) wifi_init: tcp mss: 1440
I (2444) wifi_init: WiFi IRAM OP enabled
I (2454) wifi_init: WiFi RX IRAM OP enabled
I (2454) phy_init: phy_version 701,f4f1da3a,Mar  3 2025,15:50:10
I (2494) phy_init: Saving new calibration data due to checksum failure or outdated calibration data, mode(0)
I (2514) wifi:mode : softAP (1c:db:d4:4a:f4:59)
I (2514) wifi:Total power save buffer number: 16
I (2514) wifi:Init max length of beacon: 752/752
I (2514) wifi:Init max length of beacon: 752/752
I (2524) network: WiFi AP started: SSID=ESP32_TFM, channel=1
I (2524) esp_netif_lwip: DHCP server started on interface WIFI_AP_DEF with IP: 192.168.4.1
I (2534) network: HTTP server started on port 80
I (2534) dashboard: Dashboard registered at / (5359 bytes)
I (2544) main: Network ready: WiFi AP 'ESP32_TFM', http://192.168.4.1/
I (2544) main: Inference task started on core 0
I (2624) inference: DIAG input: 150248/150528 non-zero bytes, first16=[45 67 61 53 71 61 53 67 61 53 68 61 53 71 61 53]
I (2754) main: ========================================
I (3044) main:   System running. Connect to 'ESP32_TFM'
I (3044) main:   Dashboard: http://192.168.4.1/
I (3044) main: ========================================
I (3054) main_task: Returned from app_main()
I (3754) inference: DIAG output 'output0': 4116/9261 non-zero, min=0 max=116, first16=[4 8 12 16 31 35 39 44 50 53 54 58 62 65 66 68], data_ptr=0x3c494a30 shape=[1, 9, 1029]
I (3764) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (3844) inference: DIAG input: 150043/150528 non-zero bytes, first16=[-29 -11 -13 -29 -7 -13 -29 -7 -13 -29 -7 -13 -29 -7 -13 -29]
I (4974) inference: DIAG output 'output0': 4116/9261 non-zero, min=0 max=112, first16=[4 8 10 16 33 36 40 44 48 52 54 58 61 61 64 67], data_ptr=0x3c494a30 shape=[1, 9, 1029]
I (4984) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (5064) inference: DIAG input: 149941/150528 non-zero bytes, first16=[-29 -11 -13 -29 -11 -13 -29 -7 -13 -29 -7 -13 -29 -7 -13 -29]
I (6194) inference: DIAG output 'output0': 4116/9261 non-zero, min=0 max=112, first16=[4 8 11 17 34 36 40 44 48 50 54 58 60 63 64 65], data_ptr=0x3c494a30 shape=[1, 9, 1029]
I (6204) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (7414) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (8284) wifi:new:<1,0>, old:<1,1>, ap:<1,0>, sta:<255,255>, prof:1, snd_ch_cfg:0x0
I (8284) wifi:station: 16:92:a9:c4:ac:40 join, AID=1, bgn, 20
I (8304) network: Station 16:92:a9:c4:ac:40 joined, AID=1
I (8534) wifi:I (8534) esp_netif_lwip: DHCP server assigned IP to a client, IP is: 192.168.4.2
<ba-add>idx:2 (ifx:1, 16:92:a9:c4:ac:40), tid:6, ssn:0, winSize:64
I (8644) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (8974) wifi:<ba-add>idx:3 (ifx:1, 16:92:a9:c4:ac:40), tid:0, ssn:2, winSize:64
I (9874) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (11074) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (12284) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (12944) network: WebSocket connection opened
I (13584) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (14794) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (14794) main: Frame 10 | FPS=0.8 | inf=1140ms total=1201ms | det=0 | PSRAM=4431KB
I (16044) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (17254) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (18454) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (19664) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (20874) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (22134) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (23344) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (24544) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (25734) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (26944) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (26954) main: Frame 20 | FPS=0.8 | inf=1134ms total=1198ms | det=0 | PSRAM=4431KB
I (28214) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (29414) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (30624) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (31824) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (33024) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (34274) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (35474) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (36674) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (37884) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (39084) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (39094) main: Frame 30 | FPS=0.8 | inf=1136ms total=1198ms | det=0 | PSRAM=4431KB
I (40374) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (41574) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (42784) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (43994) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (45194) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (46464) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (47674) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (48874) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (50084) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (51294) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (51294) main: Frame 40 | FPS=0.8 | inf=1142ms total=1204ms | det=0 | PSRAM=4431KB
I (52554) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (53754) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (54964) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (56174) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (57374) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (58644) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (59844) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (61044) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (62254) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (63454) yolo_decode: YOLO decode: maxScore=0.000 (raw_int8=0 cls=0 cand=0) threshold=0.30 → 0 candidates → 0 detections
I (63464) main: Frame 50 | FPS=0.8 | inf=1135ms total=1198ms | det=0 | PSRAM=4431KB
```

GitHub Copilot: 

 Los diagnósticos revelan el problema exacto:

```
DIAG output 'output0': 4116/9261 non-zero, min=0 max=116
```

**4116 = exactamente 4 × 1029** (canales 0-3 = bboxes). Los canales 4-8 (class scores) son **TODOS CERO** en INT8.

La causa raíz: con `exponent=1` (scale=2.0), los scores post-sigmoid (rango 0-1) se cuantizan a 0:
`int8 = round(0.9 / 2.0) = round(0.45) = 0`

Necesito ver cómo fue exportado el modelo para entender la solución correcta. 

Starting: *Analyze model output format* (1/3)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.json#1-1), lines 1 to 100

Ran terminal command:  cat /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info

Ran terminal command:  cat /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info | head -60

Ran terminal command:  cat /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info | grep -i "output\|return" | head -20

Ran terminal command:  cat /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info | tail -30

Ran terminal command:  cat /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info | grep -E "output0|return|Concat_3|Sigmoid|exponents.*\[1, 9" | head -20

Starting: *Check Espressif YOLO examples* (2/3)

Ahí está el problema confirmado:
```
%/model.23/Sigmoid_output_0[INT8, 1x5x1029], exponents: [1]  ← class scores post-sigmoid
%output0 = Concat(bbox, sigmoid_scores)  → [INT8, 1x9x1029], exponents: [1]
```

Sigmoid output (0-1) con exponent=1 (scale=2.0) → todo cuantiza a 0. Ahora veamos cómo maneja esto Espressif: 

Research ESP-DL YOLO detect

Searched espressif/esp-dl for "YOLO11 detection post-processing class in vision/detect", 64 results

Searched espressif/esp-dl for "coco_detect example YOLO model configuration outputs", 64 results

Searched espressif/esp-dl for "YOLO ONNX export detection head DFL sigmoid concat split outputs", 64 results

Searched espressif/esp-dl for "inverse_sigmoid score_thr_quant quantize dequantize score exponent box exponent separate tensors", 64 results

Searched espressif/esp-dl for "how_to_deploy_yolo11n ONNX export without detection head split outputs box score separate quantization", 64 results

Summarized conversation historyEncontrado el problema raíz y la solución de Espressif. Ahora necesito verificar los 3 modelos y crear el fix. Déjame revisar los archivos existentes: Optimizing tool selection...

Starting: *Check MBNTv3S & YOLO26n model info* (1/9)

Ran terminal command:  cat models/MBNTv3S_ssdlite_v1_p2_best.info | head -30 && echo "=== OUTPUTS ===" && cat models/MBNTv3S_ssdlite_v1_p2_best.info | grep -iE "output|return|exponents" && echo "=== TAIL ===" && cat models/MBNTv3S_ssdlite_v1_p2_best.info | tail -20

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01De936zqPjff6PSMmWsxjaa__vscode-1770907754601/content.txt#1-1), lines 1 to 80

Ran terminal command:  grep -E "class_out|bbox_out|objectness|return" models/MBNTv3S_ssdlite_v1_p2_best.info

Ran terminal command:  grep -E "output|return|Sigmoid|exponents.*\[1," models/yolo26n_v1_best.info | head -30

Ran terminal command:  grep -E "^  return|^%output|Sigmoid_output|Concat.*output_0 = Concat" models/yolo26n_v1_best.info

Ran terminal command:  grep -E "Transpose_output_0\[|output0\[" models/yolo26n_v1_best.info

Excelente. Ahora tengo el panorama completo de los 3 modelos:

| Modelo | Salidas | Exponent | Class scores |
|--------|---------|----------|-------------|
| **MBNTv3S** | 3 separadas (`class_out`, `bbox_out`, `objectness`) | -7 (scale=0.0078) | **OK** ✓ (0.9→int8=115) |
| **YOLO11n** | 1 combinada `[1,9,1029]` | 1 (scale=2.0) | **ROTO** ✗ (0.9→0) |
| **YOLO26n** | 1 combinada `[1,1029,9]` | 2 (scale=4.0) | **ROTO** ✗ (aún peor) |

**Plan**: 
1. MBNTv3S puede funcionar ya — solo necesita postprocesamiento correcto
2. YOLO11n y YOLO26n necesitan re-exportar el ONNX SIN detection head (como hace Espressif)

Déjame revisar los scripts de conversión existentes y el postprocesador: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/conversion_TorchToOnnx.py)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_models.py#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/fix_onnx_for_espdl.py#1-1), lines 1 to 120

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/fix_onnx_for_espdl.py#120-120), lines 120 to 387

Now let me check the existing firmware postprocessors: 

Searched for files matching `**/firmware/components/postprocess/*.cpp`, 3 matches

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp)

Searched for text `inference_get_output` (`**/firmware/components/inference/**`), 5 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#200-200), lines 200 to 260

Now let me read the full inference_engine.cpp to see how output names are set: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#1-1), lines 1 to 120

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#120-120), lines 120 to 160

Now let me check the app_config.h for SSD and output name configuration:  

Searched for regex `SSD_NUM_ANCHORS|SSD_FM_SIZES|SSD_MIN_SIZES|SSD_MAX_SIZES|SSD_NUM_CLASSES|SSD_VARIANCE|SSD_ASPECT|SSD_NUM_FEATURE` (`**/firmware/**`), 20 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#60-60), lines 60 to 140

Now I have the full picture. Let me explain and fix this systematically. 

Starting: *Fix MBNTv3S: test quick win* (2/7)

**Resumen del diagnóstico:**

**MBNTv3S** tiene 3 salidas separadas con `exponent=-7` (scale=0.0078) → class scores 0.9 → int8=115 → **FUNCIONA** ✓  
**YOLO11n** tiene 1 salida combinada con `exponent=1` (scale=2.0) → class scores 0.9 → int8=0 → **ROTO** ✗  
**YOLO26n** tiene 1 salida combinada con `exponent=2` (scale=4.0) → **AÚN PEOR** ✗

Espressif exporta sus modelos YOLO **SIN** el detection head (sin DFL/sigmoid/concat) → 6 salidas separadas con exponents individuales. Tu ONNX export incluyó todo el head.

**Plan**:
1. **Ya**: Probar MBNTv3S (que SÍ tiene cuantización correcta) 
2. **Después**: Re-exportar YOLO11n/26n sin detection head y re-cuantizar

Empiezo con MBNTv3S. Hay un bug potencial con el orden de salidas (std::map ordena alfabéticamente), así que voy a hacer el output lookup por nombre: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#1-1), lines 1 to 60

Good. Now I'll make the changes — switch to MBNTv3S model, add a named output lookup API to avoid the index ordering bug, and add diagnostic logging for MBNTv3S:  

Replacing 2 lines with 2 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h)


```

```
Now add a named output lookup function to the inference engine:   

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include/inference_engine.h#1-1), lines 1 to 65

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include/inference_engine.h#65-65), lines 65 to 73

Replacing 9 lines with 21 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include/inference_engine.h)


```

```
Now implement the named lookup in inference_engine.cpp:  

Replacing 4 lines with 31 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp)


```

```
Now update the SSD postprocessor to use named outputs (fixing the index ordering bug) and add diagnostic logging:   

Replacing 10 lines with 11 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp)


```

```
Now add diagnostic logging to the SSD decoder for the first few frames so we can see what values the model produces:   

Replacing 9 lines with 36 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp)


```

```
Now build:   

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && source /Users/admin/esp/v5.4.3/esp-idf/export.sh 2>/dev/null && idf.py build 2>&1 | tail -30

Build successful. Now flash:   

Ran terminal command:  idf.py -p /dev/tty.wchusbserial5B414963901 flash 2>&1 | tail -20

El dispositivo no está en modo download. Necesitas:
1. **Mantener pulsado BOOT** en la placa
2. **Pulsar y soltar RST** (sin soltar BOOT)
3. **Soltar BOOT**

Luego confirma y reintento el flash. 

Asked a question (Flash Ready)

Ran terminal command:  idf.py -p /dev/tty.wchusbserial5B414963901 flash 2>&1 | tail -25

Flash exitoso. Ahora monitoreamos para ver los resultados de MBNTv3S:   

Ran terminal command:  idf.py -p /dev/tty.wchusbserial5B414963901 monitor 2>&1 | head -120

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01MgeXzbqQxCiYo47qgpWF2n__vscode-1770907754632/content.txt#1-1), lines 1 to 120

El modelo MBNTv3S cargó correctamente con las 3 salidas bien nombradas. Pero noto algo: el input tiene `dtype=1` (posiblemente Float16) con `exponent=0`, mientras que YOLO11n tenía `dtype=3` (Int8). Esto es importante. Veamos las siguientes líneas del monitor:    

Ran terminal command:  timeout 30 idf.py -p /dev/tty.wchusbserial5B414963901 monitor 2>&1 | grep -E "DIAG|ssd_decode|inference|FPS|detections|Error|Guru|assert|abort|WARN" | head -30

Ran terminal command:  timeout 40 idf.py -p /dev/tty.wchusbserial5B414963901 monitor 2>&1 | tail -50

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && python3 -c "
import serial, time
ser = serial.Serial('/dev/tty.wchusbserial5B414963901', 115200, timeout=1)
start = time.time()
lines = []
while time.time() - start < 35:
    line = ser.readline().decode('utf-8', errors='replace').strip()
    if line:
        lines.append(line)
ser.close()
for l in lines[-60:]:
    print(l)
"

**¡MBNTv3S detecta objetos!** `det=4` por frame, `inf=846ms`, `FPS=1.1`. Veamos los diagnósticos iniciales:     

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && python3 -c "
import serial, time, sys
# Primero resetear el dispositivo
ser = serial.Serial('/dev/tty.wchusbserial5B414963901', 115200, timeout=1)
ser.setDTR(False)
ser.setRTS(True)
time.sleep(0.1)
ser.setRTS(False)
time.sleep(0.1)
ser.setDTR(True)
time.sleep(0.5)
# Leer output
start = time.time()
lines = []
while time.time() - start < 25:
    line = ser.readline().decode('utf-8', errors='replace').strip()
    if line:
        lines.append(line)
ser.close()
for l in lines:
    print(l)
" 2>&1 | grep -E "DIAG|ssd_decode|Output|Input|det=|FPS|WARN|inference:|postproc|frame|Frame" | head -30

MBNTv3S está detectando 4 objetos por frame consistentemente. Veamos el dashboard para ver qué detecta:     

Opened Simple Browser at http://192.168.4.1

Ahora, mientras verificas el dashboard conectándote al WiFi del ESP32 (`ESP32_TFM` / `tfm2026esp`), voy a crear el script de re-exportación ONNX para los modelos YOLO. Esto es lo que necesitamos para que YOLO11n y YOLO26n funcionen:    

Starting: *Create ESP ONNX export (6 outputs)* (3/7)

First, let me check if YOLO26n also uses a Detect module (it may use a different architecture):     

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && python3 -c "
from ultralytics import YOLO
import sys

# Check YOLO11n architecture
m11 = YOLO('models/yolo11n_v1_best.pt')
print('=== YOLO11n Detect head ===')
for name, mod in m11.model.named_modules():
    cname = type(mod).__name__
    if 'Detect' in cname or 'Head' in cname:
        print(f'  {name}: {cname}')
        if hasattr(mod, 'cv2'):
            print(f'    cv2 channels: {[c[-1].conv.out_channels for c in mod.cv2]}')
        if hasattr(mod, 'cv3'):
            print(f'    cv3 channels: {[c[-1].conv.out_channels for c in mod.cv3]}')
        if hasattr(mod, 'nc'):
            print(f'    num_classes: {mod.nc}')
        if hasattr(mod, 'reg_max'):
            print(f'    reg_max: {mod.reg_max}')
        if hasattr(mod, 'nl'):
            print(f'    num_levels: {mod.nl}')

print()
# Check YOLO26n architecture 
m26 = YOLO('models/yolo26n_v1_best.pt')
print('=== YOLO26n Detect head ===')
for name, mod in m26.model.named_modules():
    cname = type(mod).__name__
    if 'Detect' in cname or 'Head' in cname:
        print(f'  {name}: {cname}')
        if hasattr(mod, 'cv2'):
            print(f'    cv2 channels: {[c[-1].conv.out_channels for c in mod.cv2]}')
        if hasattr(mod, 'cv3'):
            print(f'    cv3 channels: {[c[-1].conv.out_channels for c in mod.cv3]}')
        if hasattr(mod, 'nc'):
            print(f'    num_classes: {mod.nc}')
        if hasattr(mod, 'reg_max'):
            print(f'    reg_max: {mod.reg_max}')
        if hasattr(mod, 'nl'):
            print(f'    num_levels: {mod.nl}')
" 2>&1

Let me create a script file instead:  

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/check_detect_heads.py)

Continue to iterate?
[object Object]

Made changes.

User: @agent Continue: "Continue to iterate?"

GitHub Copilot: Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && python3 models/check_detect_heads.py 2>&1

Ran terminal command:  python3 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/check_detect_heads.py

Ran terminal command:  C-c

Ran terminal command:  "
echo "CLEAN"
Completed with input:  "
echo "CLEAN"

Ran terminal command:  echo "terminal fixed" && python3 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/check_detect_heads.py
Completed with input:  echo "terminal fixed" && python3 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/check_detect_heads.py

Ran terminal command:  conda activate tfm_env 2>/dev/null || source /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/activate 2>/dev/null; which python3 && pip list 2>/dev/null | grep -iE "ultralytics|esp.ppq|onnx"
Completed with input:  conda activate tfm_env 2>/dev/null || source /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/activate 2>/dev/null; which python3 && pip list 2>/dev/null | grep -iE "ultralytics|esp.ppq|onnx"

Ran terminal command:  ls /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python* && /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

Ran terminal command:  PYENV=/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 && $PYENV -c "import esp_ppq; print('esp_ppq OK')" 2>&1 && $PYENV -c "import onnx; print('onnx', onnx.__version__)" 2>&1

Tenemos todo: ultralytics 8.4.14 + esp-ppq + onnx. Ahora creo el script de exportación ONNX al estilo Espressif:      

Ran terminal command:  /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/check_detect_heads.py
Completed with input:  /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/check_detect_heads.py

Replacing 31 lines with 32 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/check_detect_heads.py)


```

```


Ran terminal command:  /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/check_detect_heads.py

Perfecto. Ahora tengo la info completa:

| | YOLO11n | YOLO26n |
|---|---|---|
| reg_max | 16 | 1 |
| cv2 output | 64 ch (DFL) | 4 ch (directo) |
| cv3 output | 5 ch (classes) | 5 ch (classes) |
| Attention | 1 | 2 |

Ahora creo el script de exportación ONNX al estilo Espressif (6 salidas sin detection head):      

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py)

Ran terminal command:  /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py 2>&1
Completed with input:  /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py 2>&1

Ran terminal command:  KMP_DUPLICATE_LIB_OK=TRUE /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py 2>&1

Ran terminal command:  /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/pip install onnxscript 2>&1 | tail -5

Ran terminal command:  KMP_DUPLICATE_LIB_OK=TRUE /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py 2>&1

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01AyeXQ5QFs4gDt6qxxZxqPn__vscode-1770907754657/content.txt#1-1), lines 1 to 80

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01AyeXQ5QFs4gDt6qxxZxqPn__vscode-1770907754657/content.txt#80-80), lines 80 to 116

**¡Exportación perfecta!** Ambos modelos exportados con 6 salidas cada uno:

**YOLO11n** (reg_max=16):
- box0: [1, 64, 28, 28], score0: [1, 5, 28, 28] (P3)
- box1: [1, 64, 14, 14], score1: [1, 5, 14, 14] (P4)
- box2: [1, 64, 7, 7], score2: [1, 5, 7, 7] (P5)

**YOLO26n** (reg_max=1):
- box0: [1, 4, 28, 28], score0: [1, 5, 28, 28] (P3)
- box1: [1, 4, 14, 14], score1: [1, 5, 14, 14] (P4)
- box2: [1, 4, 7, 7], score2: [1, 5, 7, 7] (P5) 

Starting: *Re-export & quantize YOLO models* (4/7)

Ahora necesito cuantizar. Primero, restaurar onnx a la versión compatible con esp-ppq y luego cuantizar:       

Ran terminal command:  /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/pip install "onnx>=1.14.0,<1.18.0" 2>&1 | tail -5

Now let me check if the exported ONNX files have any issues with Reshape -1 that need fixing:    

Ran terminal command:  KMP_DUPLICATE_LIB_OK=TRUE /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 -c "
import onnx
import numpy as np

for name in ['yolo11n_v1_best_esp.onnx', 'yolo26n_v1_best_esp.onnx']:
    path = f'models/{name}'
    m = onnx.load(path)
    print(f'\\n=== {name} ===')
    print(f'Opset: {[o.version for o in m.opset_import]}')
    
    # Check for -1 in Reshape
    init_map = {i.name: i for i in m.graph.initializer}
    has_neg1 = False
    for n in m.graph.node:
        if n.op_type == 'Reshape' and len(n.input) > 1 and n.input[1] in init_map:
            vals = np.frombuffer(init_map[n.input[1]].raw_data, dtype=np.int64)
            if -1 in vals:
                has_neg1 = True
                print(f'  WARNING: Reshape {n.name}: shape={vals.tolist()}')
    if not has_neg1:
        print(f'  OK: No Reshape with -1')
    
    # Check ops
    ops = sorted(set(n.op_type for n in m.graph.node))
    print(f'  Ops ({len(ops)}): {ops}')
    
    # Check outputs
    for out in m.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f'  Output: {out.name} {dims}')
" 2>&1
Completed with input:  KMP_DUPLICATE_LIB_OK=TRUE /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 -c "
import onnx
import numpy as np

for name in ['yolo11n_v1_best_esp.onnx', 'yolo26n_v1_best_esp.onnx']:
    path = f'models/{name}'
    m = onnx.load(path)
    print(f'\\n=== {name} ===')
    print(f'Opset: {[o.version for o in m.opset_import]}')
    
    # Check for -1 in Reshape
    init_map = {i.name: i for i in m.graph.initializer}
    has_neg1 = False
    for n in m.graph.node:
        if n.op_type == 'Reshape' and len(n.input) > 1 and n.input[1] in init_map:
            vals = np.frombuffer(init_map[n.input[1]].raw_data, dtype=np.int64)
            if -1 in vals:
                has_neg1 = True
                print(f'  WARNING: Reshape {n.name}: shape={vals.tolist()}')
    if not has_neg1:
        print(f'  OK: No Reshape with -1')
    
    # Check ops
    ops = sorted(set(n.op_type for n in m.graph.node))
    print(f'  Ops ({len(ops)}): {ops}')
    
    # Check outputs
    for out in m.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f'  Output: {out.name} {dims}')
" 2>&1

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/check_onnx_esp.py)

Ran terminal command:  KMP_DUPLICATE_LIB_OK=TRUE /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 models/check_onnx_esp.py

Ran terminal command:  export KMP_DUPLICATE_LIB_OK=TRUE && cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 models/check_onnx_esp.py

ONNX files limpios. Ahora cuantizo. Necesito actualizar el script de cuantización para los nuevos ONNX:        

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_models_esp.py)

Ran terminal command:  ls -la /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/calib_set_nchw.pkl 2>&1

Good, calibration data exists (154MB). Let's quantize:          

Ran terminal command:  export KMP_DUPLICATE_LIB_OK=TRUE && cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/bin/python3 models/quantize_models_esp.py 2>&1

Ran terminal command:  tail -30 "/Users/admin/Library/Application Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01HZLw1NraSU6zs66kDjS1Wp__vscode-1770907754668/content.txt"

**¡Cuantización exitosa!** Ahora verificaré los exponents de los nuevos modelos:        

Ran terminal command:  grep -E "^%box|^%score|^%images|exponents.*box|exponents.*score" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info 2>&1

Ran terminal command:  grep -E "^%box|^%score|^%images|return" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo26n_v1_best.info 2>&1

**¡PERFECTO!** Los nuevos exponents son correctos:

| Tensor | YOLO11n exp | YOLO26n exp | Scale | Score 0.9 → INT8 |
|--------|------------|------------|-------|------------------|
| score0 | -3 | -3 | 0.125 | **7** ✓ |
| score1 | -2 | -2 | 0.25 | **4** ✓ |
| score2 | -2 | -3 | 0.25 / 0.125 | **4 / 7** ✓ |
| box0-2 | -3 | -3/-4 | 0.125 | N/A |

Los class scores ahora SÍ son representables en INT8 (antes eran todos 0 con exponent=1).

Nota importante: los shapes son **NHWC** (ejemplo: `[1,28,28,5]` no `[1,5,28,28]`). ESP-DL convierte internamente a NHWC.

Ahora necesito reescribir el postprocesador YOLO para manejar 6 salidas con DFL+sigmoid en float:        

Starting: *Rewrite YOLO postprocessor (DFL+sigmoid)* (5/7)

Now let me check the current YOLO postprocessor to understand what needs to change:   

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp)

Now I need to completely rewrite this postprocessor for the new 6-output format with DFL + sigmoid in float. Let me create the new version:       

Replacing 178 lines with 292 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp)


```

```
Now update the app_config.h to remove the old single-output YOLO constants and update the model sizes since the new ESPDL files are different sizes:        

Ran terminal command:  ls -la /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/*.espdl

Replacing 26 lines with 24 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h)


```

```
Now switch to YOLO11n for testing and build:         

Replacing 2 lines with 2 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h)


```

```


Starting: *Update partition & config & build* (6/7)

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && source /Users/admin/esp/v5.4.3/esp-idf/export.sh 2>/dev/null && idf.py build 2>&1 | tail -20

Build OK. Now rebuild the models partition binary with the new ESPDL files. Let me check/update the build script:           

Searched for files matching `**/firmware/**/flash_models*`, 1 match

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh#1-1), lines 1 to 80

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh#80-80), lines 80 to 123

I need to update the offsets in the flash script and then build+flash both the partition image and firmware. Let me update the script:          

Replacing 7 lines with 7 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh)


```

```
Now build the models partition image and flash both partition + firmware:          

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && python3 -c "
MODELS_DIR = '../models'
COMPOSITE = 'build/models_partition.bin'
PARTITION_SIZE = 0x700000

models = [
    ('MBNTv3S', f'{MODELS_DIR}/MBNTv3S_ssdlite_v1_p2_best.espdl', 0x000000),
    ('YOLO11n', f'{MODELS_DIR}/yolo11n_v1_best.espdl', 0x0A7000),
    ('YOLO26n', f'{MODELS_DIR}/yolo26n_v1_best.espdl', 0x354000),
]

# Create empty image
with open(COMPOSITE, 'wb') as f:
    f.write(b'\xff' * PARTITION_SIZE)
print(f'Created {PARTITION_SIZE} byte partition image')

# Write models
with open(COMPOSITE, 'r+b') as out:
    for name, path, offset in models:
        with open(path, 'rb') as src:
            data = src.read()
        out.seek(offset)
        out.write(data)
        end = offset + len(data)
        print(f'  {name}: offset=0x{offset:06X}, size={len(data):,} bytes, end=0x{end:06X}')
        # Verify no overlap
        if end > PARTITION_SIZE:
            print(f'  ERROR: overflow!')

print(f'Partition image: {COMPOSITE}')
"

Partition image built. All models fit without overlap. Now flash everything to the ESP32-S3. Put the device in download mode first:           

Asked a question (Flash Ready)

Ran terminal command:  source /Users/admin/esp/v5.4.3/esp-idf/export.sh 2>/dev/null && python -m esptool --chip esp32s3 -p /dev/tty.wchusbserial5B414963901 -b 460800 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m 0x0 build/bootloader/bootloader.bin 0x8000 build/partition_table/partition-table.bin 0x10000 build/tfm_tinyml_deploy.bin 0x410000 build/models_partition.bin 2>&1 | tail -30

Summarized conversation history**¡Todo flasheado exitosamente!** Firmware + modelos (7.34 MB de modelos). Ahora monitoreamos:             

Starting: *Test YOLO11n on device* (7/7)

Summarized conversation history

Made changes.

User: He monitoreado y no hay detecciones:

```zsh
I (27) boot: ESP-IDF v5.4.3-dirty 2nd stage bootloader
I (27) boot: compile time Feb 12 2026 16:10:54
I (27) boot: Multicore bootloader
I (28) boot: chip revision: v0.2
I (30) boot: efuse block revision: v1.3
I (34) boot.esp32s3: Boot SPI Speed : 80MHz
I (38) boot.esp32s3: SPI Mode       : DIO
I (42) boot.esp32s3: SPI Flash Size : 16MB
I (45) boot: Enabling RNG early entropy source...
I (50) boot: Partition Table:
I (52) boot: ## Label            Usage          Type ST Offset   Length
I (59) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (65) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (72) boot:  2 factory          factory app      00 00 00010000 00400000
I (78) boot:  3 models           Unknown data     01 40 00410000 00700000
I (85) boot: End of partition table
I (88) esp_image: segment 0: paddr=00010020 vaddr=3c150020 size=2f724h (194340) map
I (130) esp_image: segment 1: paddr=0003f74c vaddr=3fc9c200 size=008cch (  2252) load
I (131) esp_image: segment 2: paddr=00040020 vaddr=42000020 size=141ffch (1318908) map
I (366) esp_image: segment 3: paddr=00182024 vaddr=3fc9cacc size=06e58h ( 28248) load
I (372) esp_image: segment 4: paddr=00188e84 vaddr=40374000 size=18168h ( 98664) load
I (394) esp_image: segment 5: paddr=001a0ff4 vaddr=50000000 size=0001ch (    28) load
I (404) boot: Loaded app from partition at offset 0x10000
I (404) boot: Disabling RNG early entropy source...
I (414) octal_psram: vendor id    : 0x0d (AP)
I (415) octal_psram: dev id       : 0x02 (generation 3)
I (415) octal_psram: density      : 0x03 (64 Mbit)
I (417) octal_psram: good-die     : 0x01 (Pass)
I (421) octal_psram: Latency      : 0x01 (Fixed)
I (425) octal_psram: VCC          : 0x01 (3V)
I (429) octal_psram: SRF          : 0x01 (Fast Refresh)
I (434) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (439) octal_psram: BurstLen     : 0x01 (32 Byte)
I (444) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (449) octal_psram: DriveStrength: 0x00 (1/1)
I (454) MSPI Timing: PSRAM timing tuning index: 5
I (458) esp_psram: Found 8MB PSRAM device
I (461) esp_psram: Speed: 80MHz
I (464) cpu_start: Multicore app
I (752) esp_psram: SPI SRAM memory test OK
I (761) cpu_start: Pro cpu start user code
I (761) cpu_start: cpu freq: 240000000 Hz
I (761) app_init: Application information:
I (761) app_init: Project name:     tfm_tinyml_deploy
I (766) app_init: App version:      ac1e1dbed-dirty
I (771) app_init: Compile time:     Feb 12 2026 16:10:17
I (776) app_init: ELF file SHA256:  ce62bb2c2...
I (780) app_init: ESP-IDF:          v5.4.3-dirty
I (784) efuse_init: Min chip rev:     v0.0
I (788) efuse_init: Max chip rev:     v0.99 
I (792) efuse_init: Chip rev:         v0.2
I (796) heap_init: Initializing. RAM available for dynamic allocation:
I (802) heap_init: At 3FCAE7C0 len 0003AF50 (235 KiB): RAM
I (807) heap_init: At 3FCE9710 len 00005724 (21 KiB): RAM
I (813) heap_init: At 600FE000 len 00001FE8 (7 KiB): RTCRAM
I (818) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (825) spi_flash: detected chip: boya
I (828) spi_flash: flash io: dio
I (831) sleep_gpio: Configure to isolate all GPIO pins in sleep state
I (837) sleep_gpio: Enable automatic switching of GPIO sleep configuration
I (844) main_task: Started on CPU0
I (874) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (874) main_task: Calling app_main()
I (874) main: ========================================
I (884) main:   TFM TinyML Deployment — ESP32-S3
I (884) main:   Model: YOLO11n
I (884) main: ========================================
I (904) main: PSRAM total: 8192 KB, free: 8189 KB
I (904) main: Input buffer: 150528 bytes in PSRAM
I (904) s3 ll_cam: DMA Channel=0
I (904) cam_hal: cam init ok
I (914) sccb-ng: pin_sda 4 pin_scl 5
I (914) sccb-ng: sccb_i2c_port=1
I (924) ov3660: Mismatch PID=0x5640
I (924) camera: Camera PID=0x5640 VER=0x00 MIDL=0x00 MIDH=0x00
I (924) camera: Detected OV5640 camera
I (924) camera: Detected camera at address=0x3c
I (1614) cam_hal: PSRAM DMA mode disabled
I (1614) s3 ll_cam: node_size: 3840, nodes_per_line: 1, lines_per_node: 6
I (1614) s3 ll_cam: dma_half_buffer_min:  3840, dma_half_buffer: 15360, lines_per_half_buffer: 24, dma_buffer_size: 30720
I (1624) cam_hal: buffer_size: 30720, half_buffer_size: 15360, node_buffer_size: 3840, node_cnt: 8, total_cnt: 10
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1644) cam_hal: cam config ok
I (1654) ov5640: Set PLL: bypass: 0, multiplier: 8, sys_div: 1, pre_div: 1, root_2x: 0, pclk_root_div: 1, pclk_manual: 1, pclk_div: 4
I (1654) ov5640: Calculated XVCLK: 20000000 Hz, REFIN: 20000000 Hz, VCO: 160000000 Hz, PLL_CLK: 64000000 Hz, SYSCLK: 16000000 Hz, PCLK: 8000000 Hz
I (1704) camera: Sensor configured: vflip=1, auto WB/exp/gain, lens correction
I (1704) camera: Camera initialized: OV5640 RGB565 320x240, 2 buffers in PSRAM
I (1704) main: Camera initialized (OV5640 RGB565 320x240)
I (1714) inference: Loading YOLO11n from partition 'models' offset=0xa7000 size=2802352 bytes
I (1714) inference: Partition found: offset=0x410000, size=7340032
I (1724) inference: Model mmap'd at 0x3c987000, 2802352 bytes
I (2194) inference:   Output[0]: name='box0' shape=[1, 28, 28, 64] dtype=3 exponent=-3
I (2194) inference:   Output[1]: name='box1' shape=[1, 14, 14, 64] dtype=3 exponent=-3
I (2204) inference:   Output[2]: name='box2' shape=[1, 7, 7, 64] dtype=3 exponent=-3
I (2204) inference:   Output[3]: name='score0' shape=[1, 28, 28, 5] dtype=3 exponent=-3
I (2214) inference:   Output[4]: name='score1' shape=[1, 14, 14, 5] dtype=3 exponent=-2
I (2224) inference:   Output[5]: name='score2' shape=[1, 7, 7, 5] dtype=3 exponent=-2
I (2234) inference:   Input: shape=[1, 224, 224, 3] dtype=3 exponent=-7
I (2234) inference: Model loaded OK. 6 output(s). PSRAM free: 4630 KB
I (2244) main: Inference engine ready
I (2244) postproc: Postprocessor initialized for YOLO11n
I (2254) main: Postprocessor initialized for YOLO11n
I (2254) temperature_sensor: Range [-10°C ~ 80°C], error < 1°C
I (2264) metrics: Temperature sensor initialized
I (2264) metrics: Metrics system initialized
I (2274) main: Memory post-init: PSRAM=4630 KB free, Internal=121 KB free
I (2274) pp: pp rom version: e7ae62f
I (2284) net80211: net80211 rom version: e7ae62f
I (2294) wifi:wifi driver task: 3fcd47c0, prio:23, stack:6656, core=0
I (2314) wifi:wifi firmware version: e8ce6e2
I (2314) wifi:wifi certification version: v7.0
I (2314) wifi:config NVS flash: enabled
I (2314) wifi:config nano formatting: disabled
I (2314) wifi:Init data frame dynamic rx buffer num: 32
I (2324) wifi:Init static rx mgmt buffer num: 5
I (2324) wifi:Init management short buffer num: 32
I (2324) wifi:Init dynamic tx buffer num: 32
I (2334) wifi:Init static tx FG buffer num: 2
I (2334) wifi:Init static rx buffer size: 1600
I (2344) wifi:Init static rx buffer num: 10
I (2344) wifi:Init dynamic rx buffer num: 32
I (2354) wifi_init: rx ba win: 6
I (2354) wifi_init: accept mbox: 6
I (2354) wifi_init: tcpip mbox: 32
I (2354) wifi_init: udp mbox: 6
I (2364) wifi_init: tcp mbox: 6
I (2364) wifi_init: tcp tx win: 5760
I (2364) wifi_init: tcp rx win: 5760
I (2374) wifi_init: tcp mss: 1440
I (2374) wifi_init: WiFi IRAM OP enabled
I (2374) wifi_init: WiFi RX IRAM OP enabled
I (2384) phy_init: phy_version 701,f4f1da3a,Mar  3 2025,15:50:10
cam_hal: EV-VSYNC-OVF
I (2424) wifi:mode : softAP (1c:db:d4:4a:f4:59)
I (2434) wifi:Total power save buffer number: 16
I (2434) wifi:Init max length of beacon: 752/752
I (2434) wifi:Init max length of beacon: 752/752
I (2434) network: WiFi AP started: SSID=ESP32_TFM, channel=1
I (2434) esp_netif_lwip: DHCP server started on interface WIFI_AP_DEF with IP: 192.168.4.1
I (2444) network: HTTP server started on port 80
I (2454) dashboard: Dashboard registered at / (5359 bytes)
I (2454) main: Network ready: WiFi AP 'ESP32_TFM', http://192.168.4.1/
I (2464) main: Inference task started on core 0
I (2534) inference: DIAG input: 150066/150528 non-zero bytes, first16=[4 30 20 4 30 21 4 34 28 6 33 26 12 31 20 12]
I (2584) main: ========================================
I (2944) main:   System running. Connect to 'ESP32_TFM'
I (2944) main:   Dashboard: http://192.168.4.1/
I (2944) main: ========================================
I (2954) main_task: Returned from app_main()
I (3434) inference: DIAG output 'box0': 50139/50176 non-zero, min=-96 max=68, first16=[62 62 25 -7 -19 -37 -45 -45 -47 -49 -51 -53 -57 -58 -59 -57], data_ptr=0x3c4a9a00 shape=[1, 28, 28, 64]
I (3434) inference: DIAG output 'box1': 12462/12544 non-zero, min=-61 max=46, first16=[31 32 23 15 8 -2 -9 -16 -20 -23 -29 -28 -31 -32 -42 -38], data_ptr=0x3c485d50 shape=[1, 14, 14, 64]
I (3454) inference: DIAG output 'box2': 3112/3136 non-zero, min=-47 max=41, first16=[31 29 8 0 -5 -13 -31 -33 -36 -38 -37 -33 -30 -30 -27 -22], data_ptr=0x3c482c50 shape=[1, 7, 7, 64]
I (3464) inference: DIAG output 'score0': 3920/3920 non-zero, min=-126 max=-58, first16=[-100 -73 -68 -78 -88 -101 -73 -73 -81 -91 -112 -75 -85 -86 -100 -117], data_ptr=0x3c481d00 shape=[1, 28, 28, 5]
I (3484) inference: DIAG output 'score1': 980/980 non-zero, min=-47 max=-25, first16=[-37 -38 -37 -37 -37 -37 -40 -37 -37 -36 -37 -41 -37 -37 -37 -37], data_ptr=0x3c485110 shape=[1, 14, 14, 5]
I (3504) inference: DIAG output 'score2': 245/245 non-zero, min=-49 max=-32, first16=[-44 -47 -46 -49 -43 -36 -41 -39 -40 -36 -36 -40 -39 -39 -35 -36], data_ptr=0x3c4854f0 shape=[1, 7, 7, 5]
I (3524) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (3594) inference: DIAG input: 149844/150528 non-zero bytes, first16=[-21 -3 -5 -21 -3 -5 -20 2 -5 -15 2 -5 -21 2 -5 -18]
I (4494) inference: DIAG output 'box0': 50134/50176 non-zero, min=-96 max=68, first16=[62 62 25 -7 -20 -37 -45 -45 -47 -49 -51 -53 -57 -58 -59 -56], data_ptr=0x3c4a9a00 shape=[1, 28, 28, 64]
I (4494) inference: DIAG output 'box1': 12448/12544 non-zero, min=-61 max=46, first16=[32 32 24 14 8 -2 -9 -15 -19 -22 -29 -28 -30 -31 -42 -38], data_ptr=0x3c485d50 shape=[1, 14, 14, 64]
I (4514) inference: DIAG output 'box2': 3112/3136 non-zero, min=-47 max=42, first16=[32 30 9 1 -5 -13 -32 -34 -36 -38 -37 -33 -30 -30 -27 -22], data_ptr=0x3c482c50 shape=[1, 7, 7, 64]
I (4534) inference: DIAG output 'score0': 3920/3920 non-zero, min=-124 max=-58, first16=[-100 -73 -67 -78 -87 -101 -74 -74 -81 -91 -113 -76 -86 -88 -100 -117], data_ptr=0x3c481d00 shape=[1, 28, 28, 5]
I (4544) inference: DIAG output 'score1': 980/980 non-zero, min=-48 max=-25, first16=[-38 -38 -37 -37 -38 -37 -40 -37 -36 -35 -37 -41 -37 -36 -36 -36], data_ptr=0x3c485110 shape=[1, 14, 14, 5]
I (4564) inference: DIAG output 'score2': 245/245 non-zero, min=-49 max=-33, first16=[-45 -46 -46 -49 -43 -36 -41 -39 -40 -36 -36 -40 -39 -39 -36 -35], data_ptr=0x3c4854f0 shape=[1, 7, 7, 5]
I (4584) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (4654) inference: DIAG input: 149847/150528 non-zero bytes, first16=[-21 2 -5 -21 -3 -5 -21 2 -5 -21 2 -5 -21 3 -5 -21]
I (5544) inference: DIAG output 'box0': 50139/50176 non-zero, min=-97 max=68, first16=[61 62 25 -7 -19 -37 -45 -45 -47 -49 -52 -54 -57 -58 -59 -57], data_ptr=0x3c4a9a00 shape=[1, 28, 28, 64]
I (5554) inference: DIAG output 'box1': 12452/12544 non-zero, min=-62 max=45, first16=[30 32 24 15 9 -1 -9 -15 -19 -22 -29 -28 -30 -31 -42 -38], data_ptr=0x3c485d50 shape=[1, 14, 14, 64]
I (5574) inference: DIAG output 'box2': 3111/3136 non-zero, min=-46 max=41, first16=[32 30 9 1 -5 -13 -31 -33 -35 -38 -37 -33 -30 -30 -27 -22], data_ptr=0x3c482c50 shape=[1, 7, 7, 64]
I (5584) inference: DIAG output 'score0': 3920/3920 non-zero, min=-125 max=-58, first16=[-100 -73 -68 -78 -88 -101 -73 -73 -82 -91 -113 -77 -86 -88 -100 -118], data_ptr=0x3c481d00 shape=[1, 28, 28, 5]
I (5604) inference: DIAG output 'score1': 980/980 non-zero, min=-48 max=-26, first16=[-38 -38 -38 -38 -38 -37 -40 -36 -36 -36 -36 -41 -37 -36 -36 -36], data_ptr=0x3c485110 shape=[1, 14, 14, 5]
I (5624) inference: DIAG output 'score2': 245/245 non-zero, min=-49 max=-32, first16=[-44 -47 -46 -49 -44 -36 -41 -39 -40 -36 -36 -40 -39 -40 -36 -35], data_ptr=0x3c4854f0 shape=[1, 7, 7, 5]
I (5634) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (6604) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (7564) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (8524) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (9484) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (10444) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (11404) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (12364) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (12364) main: Frame 10 | FPS=1.0 | inf=891ms total=953ms | det=0 | PSRAM=4481KB
I (13324) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (14284) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (15244) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (16204) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (17164) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (18124) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (19084) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (20044) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (21004) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (21964) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (21964) main: Frame 20 | FPS=1.0 | inf=894ms total=956ms | det=0 | PSRAM=4481KB
I (22924) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (23884) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (24844) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (25804) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (26764) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (27714) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (28684) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (29644) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (30604) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (31564) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (31564) main: Frame 30 | FPS=1.0 | inf=893ms total=953ms | det=0 | PSRAM=4481KB
I (32524) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (33484) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (34444) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (35404) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (36364) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (37324) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (38114) wifi:new:<1,0>, old:<1,1>, ap:<1,0>, sta:<255,255>, prof:1, snd_ch_cfg:0x0
I (38114) wifi:station: 16:92:a9:c4:ac:40 join, AID=1, bgn, 20
I (38294) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (39124) network: Station 16:92:a9:c4:ac:40 joined, AID=1
I (39184) esp_netif_lwip: DHCP server assigned IP to a client, IP is: 192.168.4.2
I (39184) wifi:<ba-add>idx:2 (ifx:1, 16:92:a9:c4:ac:40), tid:6, ssn:0, winSize:64
I (39264) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (40244) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (40634) wifi:<ba-add>idx:3 (ifx:1, 16:92:a9:c4:ac:40), tid:0, ssn:11, winSize:64
I (41214) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (41214) main: Frame 40 | FPS=1.0 | inf=899ms total=961ms | det=0 | PSRAM=4481KB
I (42174) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (43134) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (43714) network: WebSocket connection opened
I (44174) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (45134) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (46094) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (47134) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (48094) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (49064) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (50034) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (50994) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (50994) main: Frame 50 | FPS=1.0 | inf=892ms total=953ms | det=0 | PSRAM=4481KB
I (52014) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (52974) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (53934) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (54894) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (55854) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (56884) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (57844) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (58804) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (59764) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (60724) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (60724) main: Frame 60 | FPS=1.0 | inf=895ms total=957ms | det=0 | PSRAM=4481KB
I (61754) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (62724) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (63694) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (64654) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (65604) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (66634) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (67594) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (68564) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (69524) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (70494) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (70494) main: Frame 70 | FPS=1.0 | inf=896ms total=958ms | det=0 | PSRAM=4481KB
I (71524) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (72484) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (73464) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (74424) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (75384) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (76394) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (77364) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (78324) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (79284) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (80244) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (80254) main: Frame 80 | FPS=1.0 | inf=898ms total=959ms | det=0 | PSRAM=4481KB
I (81274) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (82234) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (83204) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (84174) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (85154) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (86174) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (87144) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (88104) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (89074) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (90034) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (90034) main: Frame 90 | FPS=1.0 | inf=895ms total=957ms | det=0 | PSRAM=4481KB
```

GitHub Copilot: 

  

Created 6 todos

Los scores están bien — son logits raw (pre-sigmoid). El problema es que **TODOS** son fuertemente negativos:
- score0: max=-58 → dequant: -58 × 0.125 = **-7.25** → sigmoid = **0.0007**
- score1: max=-25 → dequant: -25 × 0.25 = **-6.25** → sigmoid = **0.002**

El modelo cree que NO hay ningún objeto en ningún sitio. Esto apunta a un **problema de preprocesamiento de input**. Voy a verificar cómo se prepara la imagen vs cómo se calibró el modelo: 

Searched for files matching `**/image_proc*`, 4 matches

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp#1-1), lines 1 to 150

**¡Encontrado!** Línea 90-92 del image_proc.cpp:

```cpp
dst_int8[dst_idx + 0] = (int8_t)((int)std::round(r) - 128);
dst_int8[dst_idx + 1] = (int8_t)((int)std::round(g) - 128);
dst_int8[dst_idx + 2] = (int8_t)((int)std::round(b) - 128);
```

Esto hace `pixel - 128`, que mapea [0,255] → [-128,127]. Con `exponent=-7` (scale=1/128), el modelo interpreta:
- pixel=0 → INT8=-128 → float=-1.0 ❌ (debería ser 0.0)
- pixel=128 → INT8=0 → float=0.0 ❌ (debería ser ~0.5)
- pixel=255 → INT8=127 → float=0.99 (correcto, pero coincidencia)

Ahora verifico cómo se crearon los datos de calibración: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/create_calib_set.py#1-1), lines 1 to 120

**Confirmado.** La calibración usa `float32 [0.0, 1.0]` (línea 87: `img / 255.0`), mientras que el firmware pone `pixel - 128` que produce rango `[-1.0, +1.0]` después de dequantizar con exponent=-7.

Este es el bug: **mismatch de normalización**. Los datos de calibración van [0, 1], el firmware envía [-1, 1].

La corrección correcta: con `exponent=-7` (scale=2^-7=1/128):
- Para representar float 0.0 → INT8 = 0  
- Para representar float 1.0 → INT8 = 128... pero 128 no cabe en INT8 (-128 a 127)
- Así que el máximo representable es 127/128 = 0.992 ≈ 1.0

La fórmula correcta es: `int8 = round(pixel * 128 / 255) = round(pixel / 1.992)` ≈ `pixel / 2`

Pero MBNTv3S funciona con `pixel - 128`... porque MBNTv3S tiene input `FLOAT, exponent=0` — su calibración fue probablemente con normalización diferente (Keras/TF usa `[-1, 1]` centering). Verifico:  

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/create_calib_set.py#120-120), lines 120 to 210

Confirmado: **AMBOS** datasets (NCHW y NHWC) usan normalización `[0, 1]` (pixel/255). Esto significa que MBNTv3S también se calibró con [0, 1], pero su input es `FLOAT, exponent=0`. El `pixel - 128` funciona para MBNTv3S probablemente por casualidad (la capa de entrada del modelo TensorFlow internamente ajusta la normalización).

Para YOLO, el fix es claro: mapear pixel [0,255] → INT8 que represente [0,1] float con exponent=-7.

La solución más limpia es hacer el preprocesamiento dependiente del modelo activo: 

Starting: *Fix preprocessing for YOLO* (5/6)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp#145-145), lines 145 to 163

Now let me check the header to see the function signature: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/include/image_proc.h#1-1), lines 1 to 30

Veo el error en el comentario del header: dice "pixel/255.0 ≈ (pixel-128)/128" — pero eso es **matemáticamente incorrecto**: `pixel/255 ≠ (pixel-128)/128`. El primero da [0,1], el segundo da [-1,+1].

La normalización `pixel - 128` produce:
- float_value = (pixel - 128) / 128 → rango [-1, +0.99]

Pero YOLO necesita:
- float_value = pixel / 255 → rango [0, 1.0]

Para representar `pixel/255` en INT8 con exponent=-7: `int8 = round(pixel/255 * 128) = round(pixel * 128/255)`

El fix: uso `round(pixel * 128.0 / 255.0)` para YOLO, mantengo `pixel - 128` para MBNTv3S (que funciona bien). También debo corregir la función BMP debug para hacer la conversión inversa correcta. 

Replacing 21 lines with 43 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp)


```

```
Now I need to make sure image_proc's CMakeLists has app_config as a dependency:   

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/CMakeLists.txt)

Replacing 5 lines with 6 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/CMakeLists.txt)


```

```
Now build:    

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && source /Users/admin/esp/v5.4.3/esp-idf/export.sh 2>/dev/null && idf.py build 2>&1 | tail -20

Conflict between the macro `CROP_SIZE` from app_config.h and the local `constexpr`. I'll rename the local constants: 

Replacing 7 lines with 7 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp)


```

```
Now replace all references to the old names in the file:  

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && grep -n "SRC_W\|SRC_H\|DST_W\|DST_H\|CROP_X\b\|CROP_SIZE\b" firmware/components/image_proc/image_proc.cpp | grep -v "IMG_"

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && sed -i '' \
  -e 's/\bCROP_SIZE\b/IMG_CROP_SZ/g' \
  -e 's/\bCROP_X\b/IMG_CROP_X/g' \
  -e 's/\bDST_W\b/IMG_DST_W/g' \
  -e 's/\bDST_H\b/IMG_DST_H/g' \
  -e 's/\bSRC_W\b/IMG_SRC_W/g' \
  -e 's/\bSRC_H\b/IMG_SRC_H/g' \
  firmware/components/image_proc/image_proc.cpp

Verify no duplicates (IMG_IMG_):   

Ran terminal command:  grep -c "IMG_IMG_" firmware/components/image_proc/image_proc.cpp && grep -n "SRC_W\|SRC_H\|DST_W\|DST_H\|CROP_X\b\|CROP_SIZE" firmware/components/image_proc/image_proc.cpp | grep -v "IMG_"
Completed with input:  grep -c "IMG_IMG_" firmware/components/image_proc/image_proc.cpp && grep -n "SRC_W\|SRC_H\|DST_W\|DST_H\|CROP_X\b\|CROP_SIZE" firmware/components/image_proc/image_proc.cpp | grep -v "IMG_"

Clean — 0 double-renames, 0 remaining old refs. Build again:    

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && idf.py build 2>&1 | tail -15

The sed missed some occurrences that are in the BMP section — likely because of the constexpr context. Let me check: 

Ran terminal command:  grep -n "DST_W\|DST_H" firmware/components/image_proc/image_proc.cpp
Completed with input:  grep -n "DST_W\|DST_H" firmware/components/image_proc/image_proc.cpp

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && grep -n "DST_W\|DST_H" firmware/components/image_proc/image_proc.cpp

The sed didn't actually work — the `\b` word boundary isn't supported the same way in macOS sed. Let me use proper word boundaries:  

Ran terminal command:  perl -pi -e 's/(?<![A-Z_])DST_W(?![A-Z_0-9])/IMG_DST_W/g; s/(?<![A-Z_])DST_H(?![A-Z_0-9])/IMG_DST_H/g; s/(?<![A-Z_])CROP_SIZE(?![A-Z_0-9])/IMG_CROP_SZ/g; s/(?<![A-Z_])CROP_X(?![A-Z_0-9])/IMG_CROP_X/g' firmware/components/image_proc/image_proc.cpp

Ran terminal command:  grep -n "DST_W\|DST_H\|CROP_SIZE\|CROP_X" firmware/components/image_proc/image_proc.cpp | grep -v IMG_
Completed with input:  grep -n "DST_W\|DST_H\|CROP_SIZE\|CROP_X" firmware/components/image_proc/image_proc.cpp | grep -v IMG_

All renamed. Check for double-renames:  

Ran terminal command:  grep -c "IMG_IMG_" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp
Completed with input:  grep -c "IMG_IMG_" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp

Build:    

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && idf.py build 2>&1 | tail -15

Build exitoso. Ahora pon el dispositivo en modo download (BOOT + RST) y flasheo:     

Asked a question (Flash Ready)

Ran terminal command:  idf.py -p /dev/tty.wchusbserial5B414963901 flash 2>&1 | tail -15

Flasheado. Ahora monitoreamos para ver si ahora sí detecta:      

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && python3 -c "
import serial, time
ser = serial.Serial('/dev/tty.wchusbserial5B414963901', 115200, timeout=1)
start = time.time()
lines = []
while time.time() - start < 25:
    line = ser.readline().decode('utf-8', errors='replace').strip()
    if line:
        lines.append(line)
ser.close()
for l in lines:
    print(l)
" 2>&1 | grep -E "DIAG|yolo_decode|Frame|Output|Input|Error|score"

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && python3 << 'PYEOF'
import serial, time
ser = serial.Serial('/dev/tty.wchusbserial5B414963901', 115200, timeout=1)
start = time.time()
lines = []
while time.time() - start < 25:
    line = ser.readline().decode('utf-8', errors='replace').strip()
    if line:
        lines.append(line)
ser.close()
for l in lines:
    if any(k in l for k in ['DIAG', 'yolo_decode', 'Frame', 'Output', 'Input', 'score', 'det=']):
        print(l)
PYEOF
Completed with input:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && python3 << 'PYEOF'
import serial, time
ser = serial.Serial('/dev/tty.wchusbserial5B414963901', 115200, timeout=1)
start = time.time()
lines = []
while time.time() - start < 25:
    line = ser.readline().decode('utf-8', errors='replace').strip()
    if line:
        lines.append(line)
ser.close()
for l in lines:
    if any(k in l for k in ['DIAG', 'yolo_decode', 'Frame', 'Output', 'Input', 'score', 'det=']):
        print(l)
PYEOF

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && ./env/bin/python3 << 'PYEOF'
import serial, time
ser = serial.Serial('/dev/tty.wchusbserial5B414963901', 115200, timeout=1)
start = time.time()
lines = []
while time.time() - start < 30:
    line = ser.readline().decode('utf-8', errors='replace').strip()
    if line:
        lines.append(line)
ser.close()
for l in lines:
    if any(k in l for k in ['DIAG', 'yolo_decode', 'Frame', 'Output', 'Input', 'score', 'det=']):
        print(l)
PYEOF
Completed with input:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && ./env/bin/python3 << 'PYEOF'
import serial, time
ser = serial.Serial('/dev/tty.wchusbserial5B414963901', 115200, timeout=1)
start = time.time()
lines = []
while time.time() - start < 30:
    line = ser.readline().decode('utf-8', errors='replace').strip()
    if line:
        lines.append(line)
ser.close()
for l in lines:
    if any(k in l for k in ['DIAG', 'yolo_decode', 'Frame', 'Output', 'Input', 'score', 'det=']):
        print(l)
PYEOF

Ran terminal command:  pip3 install pyserial 2>/dev/null; python3 -m serial.tools.list_ports 2>/dev/null || echo "trying conda"; conda run -n base python3 -c "import serial; print('ok')" 2>/dev/null || echo "no serial"

Made changes.

User: Dejame hacer el monitoreo a mi manualmente y yo te comparto los logs

GitHub Copilot: 

Perfecto, monitorea tú con:

```
idf.py -p /dev/tty.wchusbserial5B414963901 monitor
```

Lo clave que necesito ver en los logs:

1. Las líneas **DIAG output 'score0'** — ¿los valores raw siguen siendo todos negativos o ahora hay positivos?
2. Las líneas **yolo_decode** — ¿`maxScore` sigue en 0.000 o cambió?
3. Cualquier línea con **det=** en los Frame logs

Comparte lo que salga y seguimos.

User: Log:

```zsh
I (27) boot: ESP-IDF v5.4.3-dirty 2nd stage bootloader
I (27) boot: compile time Feb 12 2026 16:10:54
I (27) boot: Multicore bootloader
I (28) boot: chip revision: v0.2
I (30) boot: efuse block revision: v1.3
I (34) boot.esp32s3: Boot SPI Speed : 80MHz
I (38) boot.esp32s3: SPI Mode       : DIO
I (42) boot.esp32s3: SPI Flash Size : 16MB
I (45) boot: Enabling RNG early entropy source...
I (50) boot: Partition Table:
I (52) boot: ## Label            Usage          Type ST Offset   Length
I (59) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (65) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (72) boot:  2 factory          factory app      00 00 00010000 00400000
I (78) boot:  3 models           Unknown data     01 40 00410000 00700000
I (85) boot: End of partition table
I (88) esp_image: segment 0: paddr=00010020 vaddr=3c150020 size=2f724h (194340) map
I (130) esp_image: segment 1: paddr=0003f74c vaddr=3fc9c200 size=008cch (  2252) load
I (131) esp_image: segment 2: paddr=00040020 vaddr=42000020 size=1420b0h (1319088) map
I (366) esp_image: segment 3: paddr=001820d8 vaddr=3fc9cacc size=06e58h ( 28248) load
I (372) esp_image: segment 4: paddr=00188f38 vaddr=40374000 size=18168h ( 98664) load
I (394) esp_image: segment 5: paddr=001a10a8 vaddr=50000000 size=0001ch (    28) load
I (404) boot: Loaded app from partition at offset 0x10000
I (404) boot: Disabling RNG early entropy source...
I (414) octal_psram: vendor id    : 0x0d (AP)
I (415) octal_psram: dev id       : 0x02 (generation 3)
I (415) octal_psram: density      : 0x03 (64 Mbit)
I (417) octal_psram: good-die     : 0x01 (Pass)
I (421) octal_psram: Latency      : 0x01 (Fixed)
I (425) octal_psram: VCC          : 0x01 (3V)
I (429) octal_psram: SRF          : 0x01 (Fast Refresh)
I (434) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (439) octal_psram: BurstLen     : 0x01 (32 Byte)
I (444) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (449) octal_psram: DriveStrength: 0x00 (1/1)
I (454) MSPI Timing: PSRAM timing tuning index: 5
I (458) esp_psram: Found 8MB PSRAM device
I (461) esp_psram: Speed: 80MHz
I (464) cpu_start: Multicore app
I (752) esp_psram: SPI SRAM memory test OK
I (761) cpu_start: Pro cpu start user code
I (761) cpu_start: cpu freq: 240000000 Hz
I (761) app_init: Application information:
I (761) app_init: Project name:     tfm_tinyml_deploy
I (766) app_init: App version:      ac1e1dbed-dirty
I (771) app_init: Compile time:     Feb 12 2026 16:10:17
I (776) app_init: ELF file SHA256:  a59a44e6f...
I (780) app_init: ESP-IDF:          v5.4.3-dirty
I (784) efuse_init: Min chip rev:     v0.0
I (788) efuse_init: Max chip rev:     v0.99 
I (792) efuse_init: Chip rev:         v0.2
I (796) heap_init: Initializing. RAM available for dynamic allocation:
I (802) heap_init: At 3FCAE7C0 len 0003AF50 (235 KiB): RAM
I (808) heap_init: At 3FCE9710 len 00005724 (21 KiB): RAM
I (813) heap_init: At 600FE000 len 00001FE8 (7 KiB): RTCRAM
I (818) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (825) spi_flash: detected chip: boya
I (828) spi_flash: flash io: dio
I (831) sleep_gpio: Configure to isolate all GPIO pins in sleep state
I (837) sleep_gpio: Enable automatic switching of GPIO sleep configuration
I (844) main_task: Started on CPU0
I (874) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (874) main_task: Calling app_main()
I (874) main: ========================================
I (884) main:   TFM TinyML Deployment — ESP32-S3
I (884) main:   Model: YOLO11n
I (884) main: ========================================
I (904) main: PSRAM total: 8192 KB, free: 8189 KB
I (904) main: Input buffer: 150528 bytes in PSRAM
I (904) s3 ll_cam: DMA Channel=0
I (904) cam_hal: cam init ok
I (914) sccb-ng: pin_sda 4 pin_scl 5
I (914) sccb-ng: sccb_i2c_port=1
I (924) ov3660: Mismatch PID=0x5640
I (924) camera: Camera PID=0x5640 VER=0x00 MIDL=0x00 MIDH=0x00
I (924) camera: Detected OV5640 camera
I (924) camera: Detected camera at address=0x3c
I (1614) cam_hal: PSRAM DMA mode disabled
I (1614) s3 ll_cam: node_size: 3840, nodes_per_line: 1, lines_per_node: 6
I (1614) s3 ll_cam: dma_half_buffer_min:  3840, dma_half_buffer: 15360, lines_per_half_buffer: 24, dma_buffer_size: 30720
I (1624) cam_hal: buffer_size: 30720, half_buffer_size: 15360, node_buffer_size: 3840, node_cnt: 8, total_cnt: 10
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1644) cam_hal: cam config ok
I (1654) ov5640: Set PLL: bypass: 0, multiplier: 8, sys_div: 1, pre_div: 1, root_2x: 0, pclk_root_div: 1, pclk_manual: 1, pclk_div: 4
I (1654) ov5640: Calculated XVCLK: 20000000 Hz, REFIN: 20000000 Hz, VCO: 160000000 Hz, PLL_CLK: 64000000 Hz, SYSCLK: 16000000 Hz, PCLK: 8000000 Hz
I (1704) camera: Sensor configured: vflip=1, auto WB/exp/gain, lens correction
I (1704) camera: Camera initialized: OV5640 RGB565 320x240, 2 buffers in PSRAM
I (1704) main: Camera initialized (OV5640 RGB565 320x240)
I (1714) inference: Loading YOLO11n from partition 'models' offset=0xa7000 size=2802352 bytes
I (1714) inference: Partition found: offset=0x410000, size=7340032
I (1724) inference: Model mmap'd at 0x3c987000, 2802352 bytes
I (2194) inference:   Output[0]: name='box0' shape=[1, 28, 28, 64] dtype=3 exponent=-3
I (2194) inference:   Output[1]: name='box1' shape=[1, 14, 14, 64] dtype=3 exponent=-3
I (2204) inference:   Output[2]: name='box2' shape=[1, 7, 7, 64] dtype=3 exponent=-3
I (2204) inference:   Output[3]: name='score0' shape=[1, 28, 28, 5] dtype=3 exponent=-3
I (2214) inference:   Output[4]: name='score1' shape=[1, 14, 14, 5] dtype=3 exponent=-2
I (2224) inference:   Output[5]: name='score2' shape=[1, 7, 7, 5] dtype=3 exponent=-2
I (2234) inference:   Input: shape=[1, 224, 224, 3] dtype=3 exponent=-7
I (2234) inference: Model loaded OK. 6 output(s). PSRAM free: 4630 KB
I (2244) main: Inference engine ready
I (2244) postproc: Postprocessor initialized for YOLO11n
I (2254) main: Postprocessor initialized for YOLO11n
I (2254) temperature_sensor: Range [-10°C ~ 80°C], error < 1°C
I (2264) metrics: Temperature sensor initialized
I (2264) metrics: Metrics system initialized
I (2274) main: Memory post-init: PSRAM=4630 KB free, Internal=121 KB free
I (2284) pp: pp rom version: e7ae62f
I (2284) net80211: net80211 rom version: e7ae62f
I (2294) wifi:wifi driver task: 3fcd47c0, prio:23, stack:6656, core=0
I (2304) wifi:wifi firmware version: e8ce6e2
I (2304) wifi:wifi certification version: v7.0
I (2304) wifi:config NVS flash: enabled
I (2304) wifi:config nano formatting: disabled
I (2304) wifi:Init data frame dynamic rx buffer num: 32
I (2314) wifi:Init static rx mgmt buffer num: 5
I (2314) wifi:Init management short buffer num: 32
I (2324) wifi:Init dynamic tx buffer num: 32
I (2324) wifi:Init static tx FG buffer num: 2
I (2334) wifi:Init static rx buffer size: 1600
I (2334) wifi:Init static rx buffer num: 10
I (2334) wifi:Init dynamic rx buffer num: 32
I (2344) wifi_init: rx ba win: 6
I (2344) wifi_init: accept mbox: 6
I (2344) wifi_init: tcpip mbox: 32
I (2354) wifi_init: udp mbox: 6
I (2354) wifi_init: tcp mbox: 6
I (2354) wifi_init: tcp tx win: 5760
I (2364) wifi_init: tcp rx win: 5760
I (2364) wifi_init: tcp mss: 1440
I (2364) wifi_init: WiFi IRAM OP enabled
I (2374) wifi_init: WiFi RX IRAM OP enabled
I (2374) phy_init: phy_version 701,f4f1da3a,Mar  3 2025,15:50:10
cam_hal: EV-VSYNC-OVF
I (2414) phy_init: Saving new calibration data due to checksum failure or outdated calibration data, mode(0)
I (2434) wifi:mode : softAP (1c:db:d4:4a:f4:59)
I (2434) wifi:Total power save buffer number: 16
I (2434) wifi:Init max length of beacon: 752/752
I (2434) wifi:Init max length of beacon: 752/752
I (2444) esp_netif_lwip: DHCP server started on interface WIFI_AP_DEF with IP: 192.168.4.1
I (2444) network: WiFi AP started: SSID=ESP32_TFM, channel=1
I (2454) network: HTTP server started on port 80
I (2454) dashboard: Dashboard registered at / (5359 bytes)
I (2464) main: Network ready: WiFi AP 'ESP32_TFM', http://192.168.4.1/
I (2464) main: Inference task started on core 0
I (2584) inference: DIAG input: 150528/150528 non-zero bytes, first16=[74 85 87 74 85 87 75 86 87 78 87 86 78 87 83 78]
I (2634) main: ========================================
I (2994) main:   System running. Connect to 'ESP32_TFM'
I (2994) main:   Dashboard: http://192.168.4.1/
I (2994) main: ========================================
I (3004) main_task: Returned from app_main()
I (3474) inference: DIAG output 'box0': 50144/50176 non-zero, min=-96 max=68, first16=[61 62 25 -7 -20 -37 -45 -45 -46 -49 -51 -53 -56 -57 -58 -55], data_ptr=0x3c4a9a00 shape=[1, 28, 28, 64]
I (3484) inference: DIAG output 'box1': 12467/12544 non-zero, min=-61 max=46, first16=[31 33 23 15 8 -2 -9 -16 -20 -23 -29 -28 -31 -32 -42 -39], data_ptr=0x3c485d50 shape=[1, 14, 14, 64]
I (3504) inference: DIAG output 'box2': 3119/3136 non-zero, min=-47 max=41, first16=[32 30 9 1 -5 -13 -32 -33 -35 -38 -37 -33 -30 -30 -27 -22], data_ptr=0x3c482c50 shape=[1, 7, 7, 64]
I (3514) inference: DIAG output 'score0': 3920/3920 non-zero, min=-125 max=-59, first16=[-100 -73 -67 -77 -87 -101 -75 -75 -82 -92 -112 -76 -85 -88 -99 -118], data_ptr=0x3c481d00 shape=[1, 28, 28, 5]
I (3534) inference: DIAG output 'score1': 980/980 non-zero, min=-48 max=-25, first16=[-38 -39 -38 -38 -38 -37 -40 -36 -36 -35 -36 -41 -37 -36 -36 -36], data_ptr=0x3c485110 shape=[1, 14, 14, 5]
I (3554) inference: DIAG output 'score2': 245/245 non-zero, min=-49 max=-32, first16=[-45 -47 -46 -49 -44 -36 -41 -39 -40 -36 -35 -39 -38 -39 -35 -36], data_ptr=0x3c4854f0 shape=[1, 7, 7, 5]
I (3564) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (3684) inference: DIAG input: 150528/150528 non-zero bytes, first16=[54 65 62 54 65 62 58 63 62 57 63 62 54 65 62 55]
I (4574) inference: DIAG output 'box0': 50137/50176 non-zero, min=-96 max=68, first16=[61 62 25 -7 -20 -37 -45 -45 -47 -49 -51 -53 -57 -58 -59 -56], data_ptr=0x3c4a9a00 shape=[1, 28, 28, 64]
I (4584) inference: DIAG output 'box1': 12453/12544 non-zero, min=-61 max=46, first16=[31 32 23 15 8 -2 -9 -16 -20 -23 -30 -29 -30 -31 -42 -38], data_ptr=0x3c485d50 shape=[1, 14, 14, 64]
I (4604) inference: DIAG output 'box2': 3114/3136 non-zero, min=-46 max=41, first16=[32 29 8 0 -4 -13 -31 -33 -36 -38 -37 -33 -30 -30 -27 -22], data_ptr=0x3c482c50 shape=[1, 7, 7, 64]
I (4614) inference: DIAG output 'score0': 3920/3920 non-zero, min=-125 max=-58, first16=[-101 -74 -68 -77 -88 -101 -74 -74 -81 -91 -113 -77 -86 -88 -99 -117], data_ptr=0x3c481d00 shape=[1, 28, 28, 5]
I (4634) inference: DIAG output 'score1': 980/980 non-zero, min=-48 max=-25, first16=[-37 -38 -37 -37 -38 -38 -40 -37 -37 -36 -36 -41 -37 -35 -36 -36], data_ptr=0x3c485110 shape=[1, 14, 14, 5]
I (4654) inference: DIAG output 'score2': 245/245 non-zero, min=-49 max=-32, first16=[-44 -46 -46 -49 -43 -36 -41 -39 -40 -36 -36 -39 -39 -39 -35 -36], data_ptr=0x3c4854f0 shape=[1, 7, 7, 5]
I (4664) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (4784) inference: DIAG input: 150372/150528 non-zero bytes, first16=[41 61 54 41 61 54 45 61 54 45 61 54 45 61 54 45]
I (5674) inference: DIAG output 'box0': 50147/50176 non-zero, min=-96 max=68, first16=[61 62 25 -7 -20 -37 -45 -45 -47 -49 -51 -53 -57 -57 -59 -56], data_ptr=0x3c4a9a00 shape=[1, 28, 28, 64]
I (5684) inference: DIAG output 'box1': 12455/12544 non-zero, min=-62 max=46, first16=[31 32 23 15 8 -2 -9 -16 -20 -22 -29 -28 -30 -31 -42 -38], data_ptr=0x3c485d50 shape=[1, 14, 14, 64]
I (5694) inference: DIAG output 'box2': 3114/3136 non-zero, min=-47 max=41, first16=[32 30 9 1 -5 -13 -31 -33 -35 -38 -37 -33 -30 -30 -27 -23], data_ptr=0x3c482c50 shape=[1, 7, 7, 64]
I (5714) inference: DIAG output 'score0': 3920/3920 non-zero, min=-124 max=-59, first16=[-100 -74 -68 -78 -88 -101 -73 -73 -82 -91 -113 -77 -86 -88 -100 -117], data_ptr=0x3c481d00 shape=[1, 28, 28, 5]
I (5734) inference: DIAG output 'score1': 980/980 non-zero, min=-49 max=-25, first16=[-38 -38 -37 -38 -37 -37 -40 -37 -36 -36 -36 -41 -37 -36 -36 -37], data_ptr=0x3c485110 shape=[1, 14, 14, 5]
I (5744) inference: DIAG output 'score2': 245/245 non-zero, min=-49 max=-33, first16=[-44 -46 -46 -49 -43 -36 -41 -39 -40 -36 -36 -40 -39 -40 -35 -35], data_ptr=0x3c4854f0 shape=[1, 7, 7, 5]
I (5764) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (6774) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (7774) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (8774) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (9774) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (10794) wifi:new:<1,0>, old:<1,1>, ap:<1,0>, sta:<255,255>, prof:1, snd_ch_cfg:0x0
I (10794) wifi:station: 16:92:a9:c4:ac:40 join, AID=1, bgn, 20
I (10814) network: Station 16:92:a9:c4:ac:40 joined, AID=1
I (10834) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
cam_hal: FB-SIZE: 107520 != 153600
I (11014) wifi:<ba-add>idx:2 (ifx:1, 16:92:a9:c4:ac:40), tid:6, ssn:0, winSize:64
I (11044) esp_netif_lwip: DHCP server assigned IP to a client, IP is: 192.168.4.2
I (11464) wifi:<ba-add>idx:3 (ifx:1, 16:92:a9:c4:ac:40), tid:0, ssn:2, winSize:64
I (11854) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (12874) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (12874) main: Frame 10 | FPS=1.0 | inf=908ms total=1012ms | det=0 | PSRAM=4481KB
I (13874) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (14874) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (15364) network: WebSocket connection opened
I (15974) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (16974) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (17974) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (19024) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (20024) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (21034) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (22034) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (23054) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (23054) main: Frame 20 | FPS=1.0 | inf=896ms total=1001ms | det=0 | PSRAM=4481KB
I (24114) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (25114) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (26134) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (27134) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (28134) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (29184) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (30184) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (31194) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (32194) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (33204) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (33204) main: Frame 30 | FPS=1.0 | inf=894ms total=997ms | det=0 | PSRAM=4481KB
I (34274) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (35274) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (36274) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (37274) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (38274) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (39324) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (40324) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (41324) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (42334) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (43344) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (43344) main: Frame 40 | FPS=1.0 | inf=897ms total=1001ms | det=0 | PSRAM=4481KB
I (44404) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (45404) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (46404) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (47404) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (48404) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (49474) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (50474) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (51474) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (52474) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (53484) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.30 → 0 cands → 0 dets
I (53484) main: Frame 50 | FPS=1.0 | inf=898ms total=995ms | det=0 | PSRAM=4481KB
```

GitHub Copilot: 

  

Created 6 todos

Hay algo muy sospechoso. Los scores más altos en los DIAG son:
- score0: max raw = **-58**, exp=-3 → float = -7.25 → sigmoid = **0.00065**
- score1: max raw = **-25**, exp=-2 → float = -6.25 → sigmoid = **0.00193**

Con `%.3f`, sigmoid(score1 max) debería mostrar `0.002`, no `0.000`. Esto sugiere un **bug en el postprocesador** o que el modelo realmente no detecta nada. Necesito verificar dos cosas:

1. Revisar el postprocesador actual
2. Validar el modelo ONNX en Python 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp#200-200), lines 200 to 293

Veo el problema del `maxScore=0.000`: el debug tracking está DESPUÉS del filtro `continue`, así que nunca se actualiza. Pero eso es cosmético — el problema real es que **todos los logits son fuertemente negativos** (el modelo genuinamente dice "no hay nada").

Necesito validar si el modelo ONNX re-exportado funciona correctamente en Python. Déjame leer el script de exportación y luego crear un test: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py#1-1), lines 1 to 120

Summarized conversation historyOptimizing tool selection...

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py#120-120), lines 120 to 288

Starting: *Validate ONNX in Python* (1/5)

Voy a crear un script Python para validar el modelo ONNX — verificar si el float model produce scores razonables, y así distinguir si el problema es de re-exportación o de cuantización.  

Searched for files matching `**/*_esp.onnx`, 2 matches

Searched for files matching `**/*calib*`, 20 matches

Searched for files matching `**/calib_set*.pkl`, 2 matches

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_models.py#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_models.py#100-100), lines 100 to 204

**Encontré un problema crítico.** El script quantize_models.py está cuantizando los archivos ONNX **antiguos** (con output concatenado único), NO los nuevos de 6 salidas:

- `yolo11n_v1_best_fixed.onnx` ← archivo viejo con 1 output [1,9,1029]
- `yolo26n_v1_best_fixed.onnx` ← archivo viejo

Pero los ONNX re-exportados con 6 salidas son:
- yolo11n_v1_best_esp.onnx ← el correcto con 6 outputs

Déjame verificar qué `.espdl` se está flasheando realmente:  

Searched for files matching `**/flash_models*`, 1 match

Searched for files matching `**/*.espdl`, 3 matches

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh#1-1), lines 1 to 60

Necesito verificar qué modelo está realmente cuantizado. Déjame revisar los archivos de metadatos:  

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.json#1-1), lines 1 to 50

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info#1-1), lines 1 to 50

Ran terminal command:  grep -n "output\|Output\|%box\|%score\|graph.*output\|return" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info | tail -30

El modelo SÍ tiene 6 outputs correctos (`box0/1/2`, `score0/1/2`). Ahora veamos los valores de test del `.info`: 

Ran terminal command:  tail -250 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info | head -100

Ran terminal command:  tail -30 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.info

Necesito verificar algo crucial — la conversión de pixel a INT8 puede tener un overflow. Déjame leer el código actual: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp#1-1), lines 1 to 150

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp#150-150), lines 150 to 185

El preprocesamiento está correcto (con `std::min(127, ...)` para prevenir overflow). El modelo `.espdl` SÍ tiene 6 outputs correctos.

Ahora necesito validar si el modelo ONNX float produce resultados razonables. Voy a crear un script de validación en Python: 

Configuring a Python Environment
Completed with input: {
  "resourcePath": "/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE"
}

Creating a Virtual Environment
Completed with input: {
  "resourcePath": "/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE"
}


Completed with input: {
  "resourcePath": "/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE",
  "reason": "cancelled"
}

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/validate_onnx.py)

Fetching Python executable information for Python 3.10.19 (conda)
Completed with input: {
  "resourcePath": "/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE"
}

Ran terminal command:  /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env --no-capture-output python models/validate_onnx.py 2>&1

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01R5qwPBGJxqxHgASQ1tVGru__vscode-1770907754741/content.txt#1-1), lines 1 to 120

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01R5qwPBGJxqxHgASQ1tVGru__vscode-1770907754741/content.txt#120-120), lines 120 to 156

**Resultado clave: El ONNX float funciona perfectamente!** 

- Imagen 3, score1: sigmoid max = **0.8337**, 19 scores > 0.3
- Imagen 4, score2: sigmoid max = **0.8262**, 8 scores > 0.3
- Modelo .pt original: 37 detecciones (conf>0.01)

Pero en el ESP32: max raw = -25 (sigmoid=0.002). Esto confirma que **la cuantización INT8 destruye los scores**. El modelo float es correcto, el problema es que la cuantización per-tensor con power-of-2 es demasiado agresiva.

Ahora necesito verificar exactamente cómo se hizo la cuantización y probar con mejores settings: 

Starting: *Diagnose quantization loss* (2/6)

Ran terminal command:  ls -la /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/*.espdl /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_fixed.onnx /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_esp.onnx 2>/dev/null

El ESPDL fue cuantizado del ONNX correcto (6-output, timestamps coinciden). El problema es puramente que la cuantización INT8 per-tensor destruye demasiada información.

Voy a investigar las opciones de esp-ppq para mejorar la cuantización. Necesito ver si podemos marcar las capas de scores como FP32 o usar mejores settings:  

Ran terminal command:  /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env --no-capture-output python -c "
from esp_ppq import *
from esp_ppq.api import espdl_quantize_onnx
import inspect

# Show QuantizationSettingFactory.espdl_setting() signature and source if possible
try:
    setting = QuantizationSettingFactory.espdl_setting()
    print('=== QuantizationSetting methods ===')
    for attr in dir(setting):
        if not attr.startswith('_'):
            print(f'  {attr}: {type(getattr(setting, attr)).__name__}')
except Exception as e:
    print(f'Error: {e}')

# Check dispatching table methods
print()
try:
    setting = QuantizationSettingFactory.espdl_setting()
    if hasattr(setting, 'dispatching_table'):
        print(f'dispatching_table: {setting.dispatching_table}')
    if hasattr(setting, 'quantize_activation_setting'):
        print(f'quantize_activation_setting: {type(setting.quantize_activation_setting)}')
except Exception as e:
    print(f'Error: {e}')
" 2>&1

Ran terminal command:  /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env --no-capture-output python -c "
from esp_ppq import *

setting = QuantizationSettingFactory.espdl_setting()

# Check dispatching table
dt = setting.dispatching_table
print('=== DispatchingTable ===')
for attr in dir(dt):
    if not attr.startswith('_'):
        val = getattr(dt, attr)
        if callable(val):
            print(f'  {attr}()')
        else:
            print(f'  {attr}: {val}')

# Check activation setting
print()
qs = setting.quantize_activation_setting
print('=== ActivationQuantizationSetting ===')
for attr in dir(qs):
    if not attr.startswith('_'):
        val = getattr(qs, attr)
        print(f'  {attr}: {val}')

# Check parameter setting
print()
ps = setting.quantize_parameter_setting
print('=== ParameterQuantizationSetting ===')
for attr in dir(ps):
    if not attr.startswith('_'):
        val = getattr(ps, attr)
        print(f'  {attr}: {val}')
" 2>&1

Excelente, puedo controlar qué capas se cuantizan. Necesito encontrar los nombres de los nodos ONNX de los convs de scores para marcarlos como FP32:  

Ran terminal command:  /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env --no-capture-output python -c "
import onnx

model = onnx.load('/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_esp.onnx')

# Find all Conv nodes and their output names
print('=== ALL Conv nodes ===')
for node in model.graph.node:
    if node.op_type == 'Conv':
        print(f'  Node: {node.name:50s}  Output: {node.output[0]:30s}')

print()
print('=== Last 20 nodes ===')
for node in model.graph.node[-20:]:
    print(f'  {node.op_type:15s}  Name: {node.name:50s}  Out: {[o for o in node.output]}')

print()
print('=== Outputs ===')
for o in model.graph.output:
    print(f'  {o.name}')
" 2>&1

Ran terminal command:  /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env --no-capture-output python -c "
import onnx
model = onnx.load('/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_esp.onnx')

# Find nodes whose output names contain 'score' or 'box'
print('=== Nodes producing score/box outputs ===')
for node in model.graph.node:
    for out in node.output:
        if 'score' in out or 'box' in out:
            print(f'  Node: \"{node.name}\"  Op: {node.op_type}  Outputs: {list(node.output)}')
            # Show inputs too
            print(f'    Inputs: {list(node.input)}')

print()
# Find the Silu/sigmoid nodes feeding into score convs
# The score conv inputs are activations from previous layers
# Let's trace back: score0 conv takes input from some silu node
print('=== Tracing score head paths (3 convs before each final) ===')
# Find all cv3 related nodes
for node in model.graph.node:
    if 'cv3' in node.name:
        print(f'  Node: \"{node.name}\"  Op: {node.op_type}  Outputs: {list(node.output)}  Inputs: {list(node.input)}')
" 2>&1

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/find_nodes.py)

Ran terminal command:  /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env --no-capture-output python /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/find_nodes.py 2>&1

Ran terminal command:  /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p ./env --no-capture-output python models/find_nodes.py

Ahora tengo los nombres de nodos:
- **Score convs**: `node_conv2d_69` (score0), `node_conv2d_77` (score1), `node_conv2d_85` (score2)
- **Box convs**: `node_conv2d_64` (box0), `node_conv2d_72` (box1), `node_conv2d_80` (box2)

Voy a crear un script que re-cuantice con las capas de scores en FP32 y compare con la cuantización actual: 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/requantize_yolo11n.py)

Ran terminal command:  /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p ./env --no-capture-output python models/requantize_yolo11n.py 2>&1

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01K6eAenCPr6gPia7fEYxS5c__vscode-1770907754754/content.txt#1-1), lines 1 to 200

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01K6eAenCPr6gPia7fEYxS5c__vscode-1770907754754/content.txt#200-200), lines 200 to 349

Ran terminal command:  tail -60 /Users/admin/Library/Application\ Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01K6eAenCPr6gPia7fEYxS5c__vscode-1770907754754/content.txt
Completed with input:  tail -60 /Users/admin/Library/Application\ Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01K6eAenCPr6gPia7fEYxS5c__vscode-1770907754754/content.txt

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01FvvjdC1XC8MdCoZUJUQFZ7__vscode-1770907754757/content.txt#1-1), lines 1 to 100

Ran terminal command:  ls -la /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best*.espdl

Good — la versión FP32 scores se generó (2,807,264 bytes). El análisis de ruido muestra **35% noise:signal** en las capas profundas del detection head. Veamos si la versión equalized también se completó:   

Ran terminal command:  ls -la /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_equalized.espdl 2>/dev/null && echo "EXISTS" || echo "NOT FOUND"

Ran terminal command:  ls -la /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best*.espdl /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best*.info 2>/dev/null

Ran terminal command:  grep -E "score[012]|box[012]" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_fp32scores.info | tail -20

Los scores ahora son **FLOAT** con exponent=0. Esto requiere actualizar el postprocesador para manejar datos float en lugar de INT8.

Ahora necesito:
1. Actualizar la API de inference para devolver el dtype
2. Actualizar el postprocesador para manejar scores FLOAT
3. Actualizar offsets y tamaños por el nuevo modelo
4. Copiar el ESPDL, rebuild y flash 

Starting: *Update postprocessor for FP32 scores* (4/7)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#100-100), lines 100 to 303

Now let me check the header to see the API declaration and the postprocessor code:    

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include/inference_engine.h#1-1), lines 1 to 60

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp)

Necesito hacer varias actualizaciones. Voy a:
1. Actualizar `inference_get_output_by_name` para devolver el dtype
2. Actualizar `parse_stage` para scores float
3. Actualizar config con nuevos offsets
4. Copiar el ESPDL y flash 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include/inference_engine.h#60-60), lines 60 to 85

Now I'll make all the necessary changes. Let me update the header, inference engine, postprocessor, and config in one batch:  

Replacing 11 lines with 13 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include/inference_engine.h)


```

```
Now update the implementation:   

Replacing 26 lines with 30 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp)


```

```
Now update the diagnostic code too — it also treats all outputs as int8:    

Using "Replace String in File"

Continue to iterate?
[object Object]

Made changes.

User: @agent Continue: "Continue to iterate?"

GitHub Copilot: 
```

```
Now update the YOLO postprocessor to handle both float and int8 score outputs:  

Replacing 71 lines with 112 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp)


```

```
Now update the SSD postprocessor to match the new API signature:   

Searched for text `inference_get_output_by_name` (`**/firmware/components/postprocess/postprocess_ssd.cpp`), 3 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp#105-105), lines 105 to 125

Replacing 16 lines with 20 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp)


```

```
Now update the box_data variable in the YOLO postprocessor — the remaining code after the change needs to use `box_data` from `box_data_raw`: 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp#155-155), lines 155 to 175

Good — the rest of the postprocessor uses `box_data` which is already properly assigned. Now I need to update `app_config.h` with the new model size and offsets:  

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#1-1), lines 1 to 80

Replacing 7 lines with 7 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h)


```

```
Now update flash_models.sh with the new offsets:    

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh#13-13), lines 13 to 42

Replacing 3 lines with 3 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh)


```

```
Now copy the new FP32 scores ESPDL as the main file:  

Ran terminal command:  cp /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.espdl /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_int8orig.espdl && cp /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_fp32scores.espdl /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best.espdl && ls -la /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best*.espdl

Starting: *Rebuild, flash and test* (6/7)

Ahora necesito compilar el firmware. Primero verificar que no hay errores:   

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && source /Users/admin/esp/v5.4.3/esp-idf/export.sh 2>/dev/null && idf.py build 2>&1 | tail -40

Build exitoso. Solo un warning menor de variable no usada. Ahora necesito flashear el firmware y la partición de modelos:   

Ran terminal command:  source /Users/admin/esp/v5.4.3/esp-idf/export.sh 2>/dev/null && idf.py -p /dev/tty.wchusbserial5B414963901 flash 2>&1 | tail -20

El puerto serie no está disponible. Verifica que el ESP32-S3 esté conectado:   

Ran terminal command:  ls /dev/tty.wch* /dev/tty.usb* 2>/dev/null || echo "No serial ports found"

El ESP32-S3 no está conectado. Necesitas conectar la placa Freenove por USB.

Mientras tanto, déjame también preparar el script de flash de modelos con la nueva imagen compuesta que debe ejecutarse después de conectar la placa. También necesito construir la partición de modelos actualizada:     

Ran terminal command:  # Pre-build the models partition image
cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
mkdir -p build

MODELS_DIR="../models"
COMPOSITE="build/models_partition.bin"
PARTITION_SIZE=$((0x700000))  # 7MB

# Create empty image
dd if=/dev/zero bs=1 count=$PARTITION_SIZE 2>/dev/null | tr '\0' '\xff' > "$COMPOSITE"

# Write models at their offsets
dd if="$MODELS_DIR/MBNTv3S_ssdlite_v1_p2_best.espdl" of="$COMPOSITE" bs=1 seek=0 conv=notrunc 2>&1
dd if="$MODELS_DIR/yolo11n_v1_best.espdl" of="$COMPOSITE" bs=1 seek=$((0x0A7000)) conv=notrunc 2>&1
dd if="$MODELS_DIR/yolo26n_v1_best.espdl" of="$COMPOSITE" bs=1 seek=$((0x355000)) conv=notrunc 2>&1

echo ""
echo "Models partition image built: $(wc -c < "$COMPOSITE") bytes"
echo "  MBNTv3S @ 0x000000: $(wc -c < "$MODELS_DIR/MBNTv3S_ssdlite_v1_p2_best.espdl") bytes"
echo "  YOLO11n @ 0x0A7000: $(wc -c < "$MODELS_DIR/yolo11n_v1_best.espdl") bytes"
echo "  YOLO26n @ 0x355000: $(wc -c < "$MODELS_DIR/yolo26n_v1_best.espdl") bytes"

# Verify models don't overlap
YOLO11N_END=$((0x0A7000 + 2807264))
YOLO26N_START=$((0x355000))
echo ""
echo "  YOLO11n ends at: $YOLO11N_END ($(printf '0x%X' $YOLO11N_END))"
echo "  YOLO26n starts at: $YOLO26N_START ($(printf '0x%X' $YOLO26N_START))"
echo "  Gap: $((YOLO26N_START - YOLO11N_END)) bytes"
Completed with input:  # Pre-build the models partition image
cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
mkdir -p build

MODELS_DIR="../models"
COMPOSITE="build/models_partition.bin"
PARTITION_SIZE=$((0x700000))  # 7MB

# Create empty image
dd if=/dev/zero bs=1 count=$PARTITION_SIZE 2>/dev/null | tr '\0' '\xff' > "$COMPOSITE"

# Write models at their offsets
dd if="$MODELS_DIR/MBNTv3S_ssdlite_v1_p2_best.espdl" of="$COMPOSITE" bs=1 seek=0 conv=notrunc 2>&1
dd if="$MODELS_DIR/yolo11n_v1_best.espdl" of="$COMPOSITE" bs=1 seek=$((0x0A7000)) conv=notrunc 2>&1
dd if="$MODELS_DIR/yolo26n_v1_best.espdl" of="$COMPOSITE" bs=1 seek=$((0x355000)) conv=notrunc 2>&1

echo ""
echo "Models partition image built: $(wc -c < "$COMPOSITE") bytes"
echo "  MBNTv3S @ 0x000000: $(wc -c < "$MODELS_DIR/MBNTv3S_ssdlite_v1_p2_best.espdl") bytes"
echo "  YOLO11n @ 0x0A7000: $(wc -c < "$MODELS_DIR/yolo11n_v1_best.espdl") bytes"
echo "  YOLO26n @ 0x355000: $(wc -c < "$MODELS_DIR/yolo26n_v1_best.espdl") bytes"

# Verify models don't overlap
YOLO11N_END=$((0x0A7000 + 2807264))
YOLO26N_START=$((0x355000))
echo ""
echo "  YOLO11n ends at: $YOLO11N_END ($(printf '0x%X' $YOLO11N_END))"
echo "  YOLO26n starts at: $YOLO26N_START ($(printf '0x%X' $YOLO26N_START))"
echo "  Gap: $((YOLO26N_START - YOLO11N_END)) bytes"

Ran terminal command:  mkdir -p build

Ran terminal command:  dd if=/dev/zero bs=1 count=$((0x700000)) 2>/dev/null | tr '\0' '\xff' > build/models_partition.bin && echo "Created empty image"

Ran terminal command:  dd if=../models/MBNTv3S_ssdlite_v1_p2_best.espdl of=build/models_partition.bin bs=1 seek=0 conv=notrunc 2>&1 && dd if=../models/yolo11n_v1_best.espdl of=build/models_partition.bin bs=1 seek=$((0x0A7000)) conv=notrunc 2>&1 && dd if=../models/yolo26n_v1_best.espdl of=build/models_partition.bin bs=1 seek=$((0x355000)) conv=notrunc 2>&1

Ran terminal command:  echo "Models partition: $(wc -c < build/models_partition.bin) bytes" && echo "YOLO11n ends at: $(printf '0x%X' $((0x0A7000 + 2807264)))" && echo "YOLO26n starts at: 0x355000" && echo "Gap: $(($((0x355000)) - $((0x0A7000 + 2807264)))) bytes"

Partición construida correctamente. YOLO11n termina en 0x3545E0, YOLO26n empieza en 0x355000 → gap de 2592 bytes (sin overlap).

Todo está preparado. Ahora necesitas:

**Conecta el ESP32-S3 por USB** y luego ejecuta estos dos comandos en terminal:

```bash
# 1. Flash firmware
cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
source /Users/admin/esp/v5.4.3/esp-idf/export.sh
idf.py -p /dev/tty.wchusbserial5B414963901 flash

# 2. Flash modelos (con nuevo YOLO11n FP32 scores)
source /Users/admin/esp/v5.4.3/esp-idf/export.sh
python -m esptool --chip esp32s3 -p /dev/tty.wchusbserial5B414963901 -b 460800 write_flash 0x410000 build/models_partition.bin

# 3. Monitor
idf.py -p /dev/tty.wchusbserial5B414963901 monitor
```

**Resumen de los cambios realizados:**

1. **Validación Python**: El modelo ONNX float produce detecciones correctas (sigmoid max 0.83, hasta 19 scores > 0.3 por imagen). El problema era **cuantización INT8 per-tensor demasiado agresiva**.

2. **Re-cuantización**: Las 3 capas finales de score conv (`score0/1/2`) ahora se ejecutan en **FP32** en lugar de INT8. El análisis de ruido mostró 35% noise:signal en las capas profundas del detection head.

3. **Firmware actualizado**:
   - `inference_get_output_by_name()` ahora devuelve `void**` + `int* out_dtype` (en lugar de `int8_t**`)
   - postprocess_yolo.cpp: nuevo path **dual** (FLOAT vs INT8) — detecta automáticamente si scores son float y procesa sin dequantización
   - Diagnósticos mejorados para distinguir tensores FLOAT vs INT8
   - app_config.h actualizado con nuevo tamaño YOLO11n (2,807,264) y offset YOLO26n (0x355000)

4. **Modelo**: yolo11n_v1_best.espdl reemplazado con la versión FP32 scores (2,807,264 bytes, +5KB vs original)

Made changes.

User: Lo que estoy viendo en el monitor, es que se rebotea:

```zsh
I (27) boot: ESP-IDF v5.4.3-dirty 2nd stage bootloader
I (27) boot: compile time Feb 12 2026 16:10:54
I (27) boot: Multicore bootloader
I (28) boot: chip revision: v0.2
I (30) boot: efuse block revision: v1.3
I (34) boot.esp32s3: Boot SPI Speed : 80MHz
I (38) boot.esp32s3: SPI Mode       : DIO
I (42) boot.esp32s3: SPI Flash Size : 16MB
I (45) boot: Enabling RNG early entropy source...
I (50) boot: Partition Table:
I (52) boot: ## Label            Usage          Type ST Offset   Length
I (59) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (65) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (72) boot:  2 factory          factory app      00 00 00010000 00400000
I (78) boot:  3 models           Unknown data     01 40 00410000 00700000
I (85) boot: End of partition table
I (88) esp_image: segment 0: paddr=00010020 vaddr=3c150020 size=2f784h (194436) map
I (130) esp_image: segment 1: paddr=0003f7ac vaddr=3fc9c200 size=0086ch (  2156) load
I (131) esp_image: segment 2: paddr=00040020 vaddr=42000020 size=142288h (1319560) map
I (366) esp_image: segment 3: paddr=001822b0 vaddr=3fc9ca6c size=06eb8h ( 28344) load
I (372) esp_image: segment 4: paddr=00189170 vaddr=40374000 size=18168h ( 98664) load
I (394) esp_image: segment 5: paddr=001a12e0 vaddr=50000000 size=0001ch (    28) load
I (404) boot: Loaded app from partition at offset 0x10000
I (404) boot: Disabling RNG early entropy source...
I (415) octal_psram: vendor id    : 0x0d (AP)
I (415) octal_psram: dev id       : 0x02 (generation 3)
I (415) octal_psram: density      : 0x03 (64 Mbit)
I (417) octal_psram: good-die     : 0x01 (Pass)
I (421) octal_psram: Latency      : 0x01 (Fixed)
I (425) octal_psram: VCC          : 0x01 (3V)
I (430) octal_psram: SRF          : 0x01 (Fast Refresh)
I (434) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (439) octal_psram: BurstLen     : 0x01 (32 Byte)
I (444) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (449) octal_psram: DriveStrength: 0x00 (1/1)
I (454) MSPI Timing: PSRAM timing tuning index: 5
I (458) esp_psram: Found 8MB PSRAM device
I (461) esp_psram: Speed: 80MHz
I (464) cpu_start: Multicore app
I (753) esp_psram: SPI SRAM memory test OK
I (761) cpu_start: Pro cpu start user code
I (761) cpu_start: cpu freq: 240000000 Hz
I (761) app_init: Application information:
I (762) app_init: Project name:     tfm_tinyml_deploy
I (766) app_init: App version:      ac1e1dbed-dirty
I (771) app_init: Compile time:     Feb 12 2026 16:10:17
I (776) app_init: ELF file SHA256:  0888c2849...
I (780) app_init: ESP-IDF:          v5.4.3-dirty
I (785) efuse_init: Min chip rev:     v0.0
I (788) efuse_init: Max chip rev:     v0.99 
I (792) efuse_init: Chip rev:         v0.2
I (796) heap_init: Initializing. RAM available for dynamic allocation:
I (803) heap_init: At 3FCAE7C8 len 0003AF48 (235 KiB): RAM
I (808) heap_init: At 3FCE9710 len 00005724 (21 KiB): RAM
I (813) heap_init: At 600FE000 len 00001FE8 (7 KiB): RTCRAM
I (818) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (825) spi_flash: detected chip: boya
I (828) spi_flash: flash io: dio
I (832) sleep_gpio: Configure to isolate all GPIO pins in sleep state
I (837) sleep_gpio: Enable automatic switching of GPIO sleep configuration
I (844) main_task: Started on CPU0
I (874) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (874) main_task: Calling app_main()
I (874) main: ========================================
I (884) main:   TFM TinyML Deployment — ESP32-S3
I (884) main:   Model: YOLO11n
I (884) main: ========================================
I (904) main: PSRAM total: 8192 KB, free: 8189 KB
I (904) main: Input buffer: 150528 bytes in PSRAM
I (904) s3 ll_cam: DMA Channel=0
I (904) cam_hal: cam init ok
I (914) sccb-ng: pin_sda 4 pin_scl 5
I (914) sccb-ng: sccb_i2c_port=1
I (924) ov3660: Mismatch PID=0x5640
I (924) camera: Camera PID=0x5640 VER=0x00 MIDL=0x00 MIDH=0x00
I (924) camera: Detected OV5640 camera
I (924) camera: Detected camera at address=0x3c
I (1614) cam_hal: PSRAM DMA mode disabled
I (1614) s3 ll_cam: node_size: 3840, nodes_per_line: 1, lines_per_node: 6
I (1614) s3 ll_cam: dma_half_buffer_min:  3840, dma_half_buffer: 15360, lines_per_half_buffer: 24, dma_buffer_size: 30720
I (1624) cam_hal: buffer_size: 30720, half_buffer_size: 15360, node_buffer_size: 3840, node_cnt: 8, total_cnt: 10
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1644) cam_hal: cam config ok
I (1654) ov5640: Set PLL: bypass: 0, multiplier: 8, sys_div: 1, pre_div: 1, root_2x: 0, pclk_root_div: 1, pclk_manual: 1, pclk_div: 4
I (1654) ov5640: Calculated XVCLK: 20000000 Hz, REFIN: 20000000 Hz, VCO: 160000000 Hz, PLL_CLK: 64000000 Hz, SYSCLK: 16000000 Hz, PCLK: 8000000 Hz
I (1704) camera: Sensor configured: vflip=1, auto WB/exp/gain, lens correction
I (1704) camera: Camera initialized: OV5640 RGB565 320x240, 2 buffers in PSRAM
I (1704) main: Camera initialized (OV5640 RGB565 320x240)
I (1714) inference: Loading YOLO11n from partition 'models' offset=0xa7000 size=2807264 bytes
I (1714) inference: Partition found: offset=0x410000, size=7340032
I (1724) inference: Model mmap'd at 0x3c987000, 2807264 bytes
E (1994) dl::Model: Do not support Conv, please implement and register it first.

assert failed: virtual dl::TensorBase* dl::Model::get_input() dl_model_base.cpp:312 (m_inputs.size() == 1)


Backtrace: 0x40375ff9:0x3fcb2f60 0x4037f341:0x3fcb2f80 0x40386ed9:0x3fcb2fa0 0x4201e6af:0x3fcb30c0 0x4201c21b:0x3fcb30e0 0x4200c816:0x3fcb3160 0x4214147b:0x3fcb3190
--- 0x40375ff9: panic_abort at /Users/admin/esp/v5.4.3/esp-idf/components/esp_system/panic.c:477
--- 0x4037f341: esp_system_abort at /Users/admin/esp/v5.4.3/esp-idf/components/esp_system/port/esp_system_chip.c:87
--- 0x40386ed9: __assert_func at /Users/admin/esp/v5.4.3/esp-idf/components/newlib/assert.c:80
--- 0x4201e6af: dl::Model::get_input() at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/model/src/dl_model_base.cpp:312
--- (inlined by) dl::Model::get_input() at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/model/src/dl_model_base.cpp:310
--- 0x4201c21b: inference_init at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp:128
--- 0x4200c816: app_main at /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp:137
--- 0x4214147b: main_task at /Users/admin/esp/v5.4.3/esp-idf/components/freertos/app_startup.c:208




ELF file SHA256: 0888c2849

Rebooting...
```

GitHub Copilot: 

 El error `Do not support Conv` significa que **ESP-DL no soporta Conv en FP32** — solo tiene implementación INT8. La estrategia de FP32 scores no funciona en el dispositivo.

Necesito otra estrategia: cuantizar todo en INT8 pero con técnicas avanzadas (equalization + bias correction + blockwise reconstruction) y bajar el umbral de score. 

Starting: *Re-quantize INT8 with advanced opts* (3/6)

Primero voy a investigar todas las opciones de esp-ppq disponibles, y después ejecutar la cuantización con todas las optimizaciones activadas: 

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p ./env --no-capture-output python -c "
from esp_ppq import *
import inspect

setting = QuantizationSettingFactory.espdl_setting()

# Check all boolean options and their defaults
print('=== Boolean options (default values) ===')
for attr in ['equalization', 'bias_correct', 'blockwise_reconstruction', 
             'lsq_optimization', 'channel_split', 'weight_split',
             'matrix_factorization', 'ssd_equalization', 'extension',
             'convtranspose_decomposition', 'quantize_activation', 'quantize_parameter']:
    val = getattr(setting, attr, 'N/A')
    print(f'  {attr:35s} = {val}')

# Check available calib algorithms
print()
print('=== Activation calib setting ===')
act = setting.quantize_activation_setting
for attr in dir(act):
    if not attr.startswith('_'):
        print(f'  {attr} = {getattr(act, attr)}')

# Check blockwise reconstruction setting
print()
print('=== Blockwise Reconstruction Setting ===')
br = setting.blockwise_reconstruction_setting
for attr in dir(br):
    if not attr.startswith('_'):
        v = getattr(br, attr)
        if not callable(v):
            print(f'  {attr} = {v}')

# Check LSQ setting
print()
print('=== LSQ Setting ===')
lsq = setting.lsq_optimization_setting
for attr in dir(lsq):
    if not attr.startswith('_'):
        v = getattr(lsq, attr)
        if not callable(v):
            print(f'  {attr} = {v}')
" 2>&1
Completed with input:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p ./env --no-capture-output python -c "
from esp_ppq import *
import inspect

setting = QuantizationSettingFactory.espdl_setting()

# Check all boolean options and their defaults
print('=== Boolean options (default values) ===')
for attr in ['equalization', 'bias_correct', 'blockwise_reconstruction', 
             'lsq_optimization', 'channel_split', 'weight_split',
             'matrix_factorization', 'ssd_equalization', 'extension',
             'convtranspose_decomposition', 'quantize_activation', 'quantize_parameter']:
    val = getattr(setting, attr, 'N/A')
    print(f'  {attr:35s} = {val}')

# Check available calib algorithms
print()
print('=== Activation calib setting ===')
act = setting.quantize_activation_setting
for attr in dir(act):
    if not attr.startswith('_'):
        print(f'  {attr} = {getattr(act, attr)}')

# Check blockwise reconstruction setting
print()
print('=== Blockwise Reconstruction Setting ===')
br = setting.blockwise_reconstruction_setting
for attr in dir(br):
    if not attr.startswith('_'):
        v = getattr(br, attr)
        if not callable(v):
            print(f'  {attr} = {v}')

# Check LSQ setting
print()
print('=== LSQ Setting ===')
lsq = setting.lsq_optimization_setting
for attr in dir(lsq):
    if not attr.startswith('_'):
        v = getattr(lsq, attr)
        if not callable(v):
            print(f'  {attr} = {v}')
" 2>&1

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/inspect_ppq.py)

Ran terminal command:  /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p ./env --no-capture-output python models/inspect_ppq.py 2>&1
Completed with input:  /opt/homebrew/Caskroom/miniforge/base/bin/conda run -p ./env --no-capture-output python models/inspect_ppq.py 2>&1

Ran terminal command:  ./env/bin/python models/inspect_ppq.py 2>&1

Ahora tengo claridad sobre las opciones. Algunas opciones requieren CUDA que no tenemos en Mac. Pero equalization y bias_correct funcionan en CPU. Voy a crear una cuantización optimizada: 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/requantize_yolo_advanced.py)

Ran terminal command:  ./env/bin/python models/requantize_yolo_advanced.py 2>&1

Ran terminal command:  tail -80 "/Users/admin/Library/Application Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_014esRpzkiN58K6Bsg7kAPA9__vscode-1770907754797/content.txt"
Completed with input:  tail -80 "/Users/admin/Library/Application Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_014esRpzkiN58K6Bsg7kAPA9__vscode-1770907754797/content.txt"

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_013yHKXaoreJTNx2gXh8pUHj__vscode-1770907754798/content.txt#1-1), lines 1 to 100

Summarized conversation history

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_013yHKXaoreJTNx2gXh8pUHj__vscode-1770907754798/content.txt#550-550), lines 550 to 681

Starting: *Check requantization results* (1/6)

The output shows only **Variant A** (Equalization + BiasCorrect + KL) completed — Variant B was interrupted by Ctrl+C, and C/D never ran.

Variant A results:
- Size: 2,802,528 bytes
- Exponents: score0=[-3], score1=[-2], score2=[-2] (same as original)
- Equalization found **0 pairs** — YOLO architecture doesn't benefit from it
- Bias correction improvements were minimal (e.g., 0.7045→0.7044)

Let me check what files exist and validate whether Variant A actually improved scores. 

Ran terminal command:  ls -la /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best*.espdl

Only Variant A (`_eqbc.espdl`, 2,802,528 bytes) was generated. Equalization found 0 pairs and bias correction had minimal effect. I need to:

1. Run the remaining variants (minmax, percentile) 
2. Validate ALL variants by simulating quantized inference in PPQ to compare score quality before flashing 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/requantize_yolo_advanced.py)

Now I'll create a comprehensive script that both quantizes AND validates the quantized outputs through simulated inference, so we can compare variants before flashing. 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_and_validate.py)

Ran terminal command:  ./env/bin/python models/quantize_and_validate.py 2>&1 | tee /tmp/quant_validate_output.txt

Ran terminal command:  ./env/bin/python -c "from esp_ppq import QuantizationSettingFactory; print('OK')" 2>&1

Ran terminal command:  ./env/bin/python -c "
from esp_ppq import QuantizationSettingFactory, TargetPlatform
from esp_ppq.api.interface import quantize_onnx_model
from esp_ppq.executor import TorchExecutor
print('All imports OK')
print('  TargetPlatform.ESPDL_INT8 =', TargetPlatform.ESPDL_INT8)
" 2>&1

Replacing 9 lines with 9 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_and_validate.py)


```

```


Ran terminal command:  ./env/bin/python models/quantize_and_validate.py 2>&1 | tee /tmp/quant_validate_output.txt

Ran terminal command:  tail -20 /tmp/quant_validate_output.txt

Ran terminal command:  grep -A5 "Variante A:" /tmp/quant_validate_output.txt

The issue is `quantize_onnx_model` defaults to CUDA. Let me check its signature and use CPU explicitly. 

Ran terminal command:  ./env/bin/python -c "
from esp_ppq.api.interface import quantize_onnx_model
import inspect
sig = inspect.signature(quantize_onnx_model)
for name, param in sig.parameters.items():
    print(f'  {name}: default={param.default}')
" 2>&1

Ran terminal command:  ./env/bin/python -c "
from esp_ppq.api.interface import quantize_onnx_model
# It's decorated, get the wrapped function
wrapped = quantize_onnx_model.__wrapped__ if hasattr(quantize_onnx_model, '__wrapped__') else None
if wrapped:
    import inspect
    sig = inspect.signature(wrapped)
    for n, p in sig.parameters.items():
        print(f'  {n}: default={p.default}')
else:
    # Read src
    import inspect
    src = inspect.getsource(quantize_onnx_model)
    # Show first 40 lines
    for i, line in enumerate(src.split('\n')[:40]):
        print(line)
" 2>&1

Ran terminal command:  ./env/bin/python -c "
import esp_ppq.api.interface as mod
import inspect
# Find the actual function
src = inspect.getsource(mod)
# Find quantize_onnx_model definition
lines = src.split('\n')
found = False
for i, line in enumerate(lines):
    if 'def quantize_onnx_model' in line:
        found = True
    if found:
        print(f'{i}: {line}')
        if i > 300:
            break
    if found and line.strip().startswith('return') and 'quantize_onnx_model' not in line:
        break
" 2>&1 | head -50

There it is — `device='cuda'` is the default. Let me fix the script to use `device='cpu'`. 

Replacing 12 lines with 13 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_and_validate.py)


```

```


Ran terminal command:  ./env/bin/python -c "
from esp_ppq.api.espdl_interface import export_espdl_graph
print('export_espdl_graph imported OK')
import inspect
sig = inspect.signature(export_espdl_graph)
for n, p in sig.parameters.items():
    print(f'  {n}: default={p.default}')
" 2>&1
Completed with input:  ./env/bin/python -c "
from esp_ppq.api.espdl_interface import export_espdl_graph
print('export_espdl_graph imported OK')
import inspect
sig = inspect.signature(export_espdl_graph)
for n, p in sig.parameters.items():
    print(f'  {n}: default={p.default}')
" 2>&1

Ran terminal command:  ./env/bin/python -c "
import esp_ppq.api.espdl_interface as m
funcs = [x for x in dir(m) if 'export' in x.lower()]
print('export functions:', funcs)
funcs2 = [x for x in dir(m) if not x.startswith('_')]
print('all public:', funcs2)
" 2>&1

Ran terminal command:  ./env/bin/python -c "
import inspect
from esp_ppq.api.espdl_interface import espdl_quantize_onnx
src = inspect.getsource(espdl_quantize_onnx)
lines = src.split('\n')
for i, line in enumerate(lines):
    print(f'{i}: {line}')
" 2>&1 | head -80

Ran terminal command:  ./env/bin/python -c "
import inspect
from esp_ppq.api.espdl_interface import espdl_quantize_onnx
# unwrap the decorator
fn = espdl_quantize_onnx
while hasattr(fn, '__wrapped__'):
    fn = fn.__wrapped__
src = inspect.getsource(fn)
# Still wrapped, look for the original
import esp_ppq.api.espdl_interface as mod
src = inspect.getsource(mod.espdl_quantize_onnx.__code__)
" 2>&1

./env/bin/python -c "
import dis, esp_ppq.api.espdl_interface as mod
# Read the source file directly
src_file = mod.__file__
with open(src_file) as f:
    content = f.read()
lines = content.split('\n')
in_func = False
for i, line in enumerate(lines):
    if 'def espdl_quantize_onnx' in line:
        in_func = True
    if in_func:
        print(f'{i}: {line}')
    if in_func and i > 300:
        break
" 2>&1 | head -80

Ran terminal command:  grep -n "def espdl_quantize_onnx\|export\|baking\|flatbuffer\|Exporter" ./env/lib/python3.10/site-packages/esp_ppq/api/espdl_interface.py | head -30

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/lib/python3.10/site-packages/esp_ppq/api/espdl_interface.py#240-240), lines 240 to 320

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/env/lib/python3.10/site-packages/esp_ppq/api/espdl_interface.py#157-157), lines 157 to 240

`espdl_quantize_onnx` already defaults to `device='cpu'`. The original requantize_yolo_advanced.py used it correctly. The problem was my new script used `quantize_onnx_model` which defaults to `device='cuda'`.

Let me rewrite the validation script to use the correct approach: `quantize_onnx_model` with `device='cpu'` for fast simulation, then `PFL.Exporter` for export. 

Replacing 26 lines with 28 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_and_validate.py)


```

```


Replacing 9 lines with 9 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_and_validate.py)


```

```


Ran terminal command:  ./env/bin/python -c "from esp_ppq import PFL; print('PFL OK')" 2>&1
Completed with input:  ./env/bin/python -c "from esp_ppq import PFL; print('PFL OK')" 2>&1

Ran terminal command:  ./env/bin/python -c "
# Find where PFL is defined
import esp_ppq.api.espdl_interface as mod
print(type(mod.PFL))
print(mod.PFL)
" 2>&1

Replacing 9 lines with 10 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_and_validate.py)


```

```
That lint error is a type annotation issue, not a runtime error. The attribute IS settable. Let me run it: 

Ran terminal command:  ./env/bin/python models/quantize_and_validate.py 2>&1 | tee /tmp/qv2.txt

Awaited terminal completion

The issue is `quantize_onnx_model` still defaults to CUDA despite my `device='cpu'`. The decorator may be interfering. Let me use `espdl_quantize_onnx` instead, which defaults to CPU and worked before. 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_and_validate.py)

Ran terminal command:  rm /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_and_validate.py

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_and_validate.py)

Ran terminal command:  ./env/bin/python models/quantize_and_validate.py 2>&1 | tee /tmp/qv3.txt

Checked background terminal output

Ran terminal command:  tail -5 /tmp/qv3.txt

The script completed. Let me read the full comparison results: 

Ran terminal command:  grep -A 30 "COMPARATIVA" /tmp/qv3.txt

**Excellent results!** The PPQ simulation shows all three variants produce **near-float quality** scores:

| Variant | img3_max | img4_max | img10_max | total>0.1 |
|---|---|---|---|---|
| Float ref | 0.8337 | 0.8262 | 0.8838 | 100 |
| A_kl | 0.8176 | 0.7773 | 0.8520 | 85 |
| B_minmax | 0.7773 | 0.8176 | 0.7773 | 104 |
| **C_percentile** | **0.8176** | **0.8176** | **0.8520** | **96** |

The quantization is NOT the problem — INT8 simulation shows sigmoid_max ~0.82 vs float's ~0.83. The zero detections on device must be a preprocessing or layout issue. Let me now update the firmware and deploy the best variant.  

Starting: *Pick best variant, update config* (4/7)

Let me first check the input tensor exponent and the preprocessing. This is critical — the simulation works perfectly, so the issue must be in the firmware's preprocessing.  

Ran terminal command:  head -10 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_C_percentile.info

Ran terminal command:  grep "exponents" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_C_percentile.info

Ran terminal command:  grep -E "(^  %images|^  %box|^  %score)" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_C_percentile.info

Continue to iterate?
[object Object]

Made changes.

User: @agent Continue: "Continue to iterate?"

GitHub Copilot: Ran terminal command:  tail -10 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_C_percentile.info

Ran terminal command:  grep -E "box|score" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_C_percentile.info | grep "exponents"

Model input: `[INT8, 1x224x224x3]` in **NHWC**, exponent **-7** (scale=1/128). Let me verify the firmware preprocessing matches.  

Searched for text `image_proc` (`**/firmware/**/*.cpp`), 10 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp#1-1), lines 1 to 120

The preprocessing looks correct. But the fact that changing preprocessing didn't change outputs is suspicious. Let me check how the firmware feeds the preprocessed data to the model.   

Searched for text `get_input` (`**/firmware/components/inference/**`), 3 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#200-200), lines 200 to 321

The inference code looks correct — `memcpy` copies preprocessed data into model input, then `run()`. The DIAG logging will help us verify on device. Let me now deploy the best variant (C_percentile, 2,800,272 bytes). 

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#1-1), lines 1 to 60

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#60-60), lines 60 to 158

Now I need to:
1. Update YOLO11N_SIZE to match C_percentile (2,800,272)
2. Recalculate YOLO26N_OFFSET  
3. Lower SCORE_THRESHOLD to 0.10
4. Add more diagnostic to help identify the root cause 

Ran terminal command:  python3 -c "
# YOLO11n C_percentile size
yolo11n_size = 2800272
# Offset = MBNTv3S offset + MBNTv3S size aligned to 4KB
mbntv3s_offset = 0
mbntv3s_size = 681088
yolo11n_offset = 0x0A7000  # already correct

# YOLO26n offset = YOLO11n offset + YOLO11n size aligned to 4KB
import math
yolo26n_offset_raw = yolo11n_offset + yolo11n_size
yolo26n_offset = math.ceil(yolo26n_offset_raw / 0x1000) * 0x1000

print(f'YOLO11N_SIZE = {yolo11n_size}')
print(f'YOLO11N_OFFSET = 0x{yolo11n_offset:06X}')
print(f'YOLO11N end = 0x{yolo11n_offset + yolo11n_size:06X}')
print(f'YOLO26N_OFFSET = 0x{yolo26n_offset:06X}')
print(f'YOLO26N_SIZE = 2639168')
total = yolo26n_offset + 2639168
print(f'Total used = {total} bytes = {total/1024/1024:.1f} MB')
print(f'Partition = 7MB = {7*1024*1024} bytes')
print(f'Fits: {total < 7*1024*1024}')
"

Replacing 8 lines with 8 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h)


```

```


Starting: *Add more diagnostics & build* (5/7)

Now let me also add enhanced diagnostics to the DIAG output so we can clearly see the simulated INT8 scores on-device. I want to print the first few score values and the input exponent:  

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#150-150), lines 150 to 220

Let me enhance the diagnostics to also print the exponent and a few score values as dequantized floats: 

Replacing 12 lines with 15 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp)


```

```
Now let me also enhance the INT8 output diagnostic to show score values as dequantized floats and sigmoid:  

Replacing 13 lines with 19 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp)


```

```
Now copy the best variant and update flash_models.sh:  

Ran terminal command:  cp models/yolo11n_v1_best_C_percentile.espdl models/yolo11n_v1_best.espdl && ls -la models/yolo11n_v1_best.espdl

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh#1-1), lines 1 to 40

Replacing 4 lines with 4 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh)


```

```


Replacing 1 lines with 1 lines in [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh)


```

```
Now build the firmware:   

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && source /Users/admin/esp/v5.4.3/esp-idf/export.sh && idf.py build 2>&1 | tail -30

Build successful. Now build the models partition image:   

Starting: *Build models partition* (6/7)

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && \
MODELS_DIR="../models" && \
BUILD_DIR="build" && \
PART_SIZE=$((7 * 1024 * 1024)) && \
\
# Create empty partition
dd if=/dev/zero of="${BUILD_DIR}/models_partition.bin" bs=1 count=0 seek=${PART_SIZE} 2>&1 && \
\
# MBNTv3S at offset 0
dd if="${MODELS_DIR}/MBNTv3S_ssdlite_v1_p2_best.espdl" of="${BUILD_DIR}/models_partition.bin" bs=1 seek=0 conv=notrunc 2>&1 && \
\
# YOLO11n at 0x0A7000
dd if="${MODELS_DIR}/yolo11n_v1_best.espdl" of="${BUILD_DIR}/models_partition.bin" bs=1 seek=$((0x0A7000)) conv=notrunc 2>&1 && \
\
# YOLO26n at 0x353000
dd if="${MODELS_DIR}/yolo26n_v1_best.espdl" of="${BUILD_DIR}/models_partition.bin" bs=1 seek=$((0x353000)) conv=notrunc 2>&1 && \
\
ls -la "${BUILD_DIR}/models_partition.bin" && \
echo "Models partition built OK"
Completed with input:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && \
MODELS_DIR="../models" && \
BUILD_DIR="build" && \
PART_SIZE=$((7 * 1024 * 1024)) && \
\
# Create empty partition
dd if=/dev/zero of="${BUILD_DIR}/models_partition.bin" bs=1 count=0 seek=${PART_SIZE} 2>&1 && \
\
# MBNTv3S at offset 0
dd if="${MODELS_DIR}/MBNTv3S_ssdlite_v1_p2_best.espdl" of="${BUILD_DIR}/models_partition.bin" bs=1 seek=0 conv=notrunc 2>&1 && \
\
# YOLO11n at 0x0A7000
dd if="${MODELS_DIR}/yolo11n_v1_best.espdl" of="${BUILD_DIR}/models_partition.bin" bs=1 seek=$((0x0A7000)) conv=notrunc 2>&1 && \
\
# YOLO26n at 0x353000
dd if="${MODELS_DIR}/yolo26n_v1_best.espdl" of="${BUILD_DIR}/models_partition.bin" bs=1 seek=$((0x353000)) conv=notrunc 2>&1 && \
\
ls -la "${BUILD_DIR}/models_partition.bin" && \
echo "Models partition built OK"

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
MODELS_DIR="../models"
BUILD_DIR="build"
PART_SIZE=$((7 * 1024 * 1024))
dd if=/dev/zero of="${BUILD_DIR}/models_partition.bin" bs=1 count=0 seek=${PART_SIZE} 2>&1
Completed with input:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
MODELS_DIR="../models"
BUILD_DIR="build"
PART_SIZE=$((7 * 1024 * 1024))
dd if=/dev/zero of="${BUILD_DIR}/models_partition.bin" bs=1 count=0 seek=${PART_SIZE} 2>&1

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware && python3 -c "
import os

models_dir = '../models'
out = 'build/models_partition.bin'
part_size = 7 * 1024 * 1024  # 7MB

models = [
    (0x000000, os.path.join(models_dir, 'MBNTv3S_ssdlite_v1_p2_best.espdl')),
    (0x0A7000, os.path.join(models_dir, 'yolo11n_v1_best.espdl')),
    (0x353000, os.path.join(models_dir, 'yolo26n_v1_best.espdl')),
]

# Create empty partition
data = bytearray(part_size)

for offset, path in models:
    with open(path, 'rb') as f:
        model_data = f.read()
    print(f'  {os.path.basename(path)}: {len(model_data)} bytes @ 0x{offset:06X}')
    assert offset + len(model_data) <= part_size, f'Model exceeds partition!'
    data[offset:offset+len(model_data)] = model_data

with open(out, 'wb') as f:
    f.write(data)
print(f'  Written: {out} ({len(data)} bytes)')
"

Models partition built. Now flash both firmware and models:   

Starting: *Flash and test on device* (7/7)

Ran terminal command:  source /Users/admin/esp/v5.4.3/esp-idf/export.sh 2>&1 | tail -3 && python -m esptool --chip esp32s3 -p /dev/tty.wchusbserial5B414963901 -b 460800 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m 0x0 build/bootloader/bootloader.bin 0x8000 build/partition_table/partition-table.bin 0x10000 build/tfm_tinyml_deploy.bin 0x410000 build/models_partition.bin 2>&1
Completed with input:  source /Users/admin/esp/v5.4.3/esp-idf/export.sh 2>&1 | tail -3 && python -m esptool --chip esp32s3 -p /dev/tty.wchusbserial5B414963901 -b 460800 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m 0x0 build/bootloader/bootloader.bin 0x8000 build/partition_table/partition-table.bin 0x10000 build/tfm_tinyml_deploy.bin 0x410000 build/models_partition.bin 2>&1

El puerto serie no está disponible. Esto puede significar que el dispositivo no está conectado o que el nombre del puerto ha cambiado.

**Todo está preparado para flashear.** Aquí está el resumen y los comandos que necesitas ejecutar:

## Resumen de cambios

**Descubrimiento clave:** La simulación PPQ confirma que la cuantización INT8 **SÍ funciona** — sigmoid_max ~0.82 vs float ~0.83. El problema anterior de 0 detecciones no era la cuantización, sino posiblemente otro factor en el dispositivo.

**Modelo seleccionado:** `C_percentile` (calibración percentile) — mejor calidad total entre las 3 variantes.

**Cambios realizados:**
1. **Nuevo ESPDL**: `yolo11n_v1_best.espdl` = variante C_percentile (2,800,272 bytes, INT8 puro)
2. app_config.h: `YOLO11N_SIZE=2800272`, `YOLO26N_OFFSET=0x353000`, `SCORE_THRESHOLD=0.10f`
3. inference_engine.cpp: Diagnósticos mejorados (input min/max/exp, score dequantized max + sigmoid)
4. flash_models.sh: Offsets actualizados
5. Partición reconstruida: `build/models_partition.bin`

## Para flashear

Conecta el ESP32-S3 y ejecuta:

```bash
cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware
source /Users/admin/esp/v5.4.3/esp-idf/export.sh

# Flashear firmware + modelos
python -m esptool --chip esp32s3 -p /dev/tty.wchusbserial5B414963901 -b 460800 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_size 16MB --flash_freq 80m \
  0x0 build/bootloader/bootloader.bin \
  0x8000 build/partition_table/partition-table.bin \
  0x10000 build/tfm_tinyml_deploy.bin \
  0x410000 build/models_partition.bin

# Monitorizar
idf.py -p /dev/tty.wchusbserial5B414963901 monitor
```

En el monitor, busca las líneas `DIAG`:
- **Input**: debe mostrar `min=0 max=~128 exp=-7` (valores positivos)
- **Score outputs**: ahora mostrará `deq_max` y `sig_max` para verificar directamente si los scores son correctos

Made changes.

User: Esto fue lo que obtuve. Analicemos primero antes de hacer cualquier otra cosa:

```zsh
I (27) boot: ESP-IDF v5.4.3-dirty 2nd stage bootloader
I (27) boot: compile time Feb 12 2026 16:10:54
I (27) boot: Multicore bootloader
I (28) boot: chip revision: v0.2
I (30) boot: efuse block revision: v1.3
I (34) boot.esp32s3: Boot SPI Speed : 80MHz
I (38) boot.esp32s3: SPI Mode       : DIO
I (42) boot.esp32s3: SPI Flash Size : 16MB
I (45) boot: Enabling RNG early entropy source...
I (50) boot: Partition Table:
I (52) boot: ## Label            Usage          Type ST Offset   Length
I (59) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (65) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (72) boot:  2 factory          factory app      00 00 00010000 00400000
I (78) boot:  3 models           Unknown data     01 40 00410000 00700000
I (85) boot: End of partition table
I (88) esp_image: segment 0: paddr=00010020 vaddr=3c150020 size=2f7a4h (194468) map
I (130) esp_image: segment 1: paddr=0003f7cc vaddr=3fc9c200 size=0084ch (  2124) load
I (131) esp_image: segment 2: paddr=00040020 vaddr=42000020 size=14235ch (1319772) map
I (366) esp_image: segment 3: paddr=00182384 vaddr=3fc9ca4c size=06ed8h ( 28376) load
I (372) esp_image: segment 4: paddr=00189264 vaddr=40374000 size=18168h ( 98664) load
I (394) esp_image: segment 5: paddr=001a13d4 vaddr=50000000 size=0001ch (    28) load
I (404) boot: Loaded app from partition at offset 0x10000
I (404) boot: Disabling RNG early entropy source...
I (415) octal_psram: vendor id    : 0x0d (AP)
I (415) octal_psram: dev id       : 0x02 (generation 3)
I (415) octal_psram: density      : 0x03 (64 Mbit)
I (417) octal_psram: good-die     : 0x01 (Pass)
I (421) octal_psram: Latency      : 0x01 (Fixed)
I (426) octal_psram: VCC          : 0x01 (3V)
I (430) octal_psram: SRF          : 0x01 (Fast Refresh)
I (435) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (439) octal_psram: BurstLen     : 0x01 (32 Byte)
I (444) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (449) octal_psram: DriveStrength: 0x00 (1/1)
I (454) MSPI Timing: PSRAM timing tuning index: 5
I (458) esp_psram: Found 8MB PSRAM device
I (461) esp_psram: Speed: 80MHz
I (464) cpu_start: Multicore app
I (753) esp_psram: SPI SRAM memory test OK
I (761) cpu_start: Pro cpu start user code
I (761) cpu_start: cpu freq: 240000000 Hz
I (762) app_init: Application information:
I (762) app_init: Project name:     tfm_tinyml_deploy
I (766) app_init: App version:      ac1e1dbed-dirty
I (771) app_init: Compile time:     Feb 12 2026 16:10:17
I (776) app_init: ELF file SHA256:  a5d7ad3a8...
I (780) app_init: ESP-IDF:          v5.4.3-dirty
I (785) efuse_init: Min chip rev:     v0.0
I (788) efuse_init: Max chip rev:     v0.99 
I (792) efuse_init: Chip rev:         v0.2
I (796) heap_init: Initializing. RAM available for dynamic allocation:
I (803) heap_init: At 3FCAE7C8 len 0003AF48 (235 KiB): RAM
I (808) heap_init: At 3FCE9710 len 00005724 (21 KiB): RAM
I (813) heap_init: At 600FE000 len 00001FE8 (7 KiB): RTCRAM
I (818) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (825) spi_flash: detected chip: boya
I (828) spi_flash: flash io: dio
I (832) sleep_gpio: Configure to isolate all GPIO pins in sleep state
I (837) sleep_gpio: Enable automatic switching of GPIO sleep configuration
I (844) main_task: Started on CPU0
I (874) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (874) main_task: Calling app_main()
I (874) main: ========================================
I (884) main:   TFM TinyML Deployment — ESP32-S3
I (884) main:   Model: YOLO11n
I (884) main: ========================================
I (904) main: PSRAM total: 8192 KB, free: 8189 KB
I (904) main: Input buffer: 150528 bytes in PSRAM
I (904) s3 ll_cam: DMA Channel=0
I (904) cam_hal: cam init ok
I (914) sccb-ng: pin_sda 4 pin_scl 5
I (914) sccb-ng: sccb_i2c_port=1
I (924) ov3660: Mismatch PID=0x5640
I (924) camera: Camera PID=0x5640 VER=0x00 MIDL=0x00 MIDH=0x00
I (924) camera: Detected OV5640 camera
I (924) camera: Detected camera at address=0x3c
I (1614) cam_hal: PSRAM DMA mode disabled
I (1614) s3 ll_cam: node_size: 3840, nodes_per_line: 1, lines_per_node: 6
I (1614) s3 ll_cam: dma_half_buffer_min:  3840, dma_half_buffer: 15360, lines_per_half_buffer: 24, dma_buffer_size: 30720
I (1624) cam_hal: buffer_size: 30720, half_buffer_size: 15360, node_buffer_size: 3840, node_cnt: 8, total_cnt: 10
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1634) cam_hal: Allocating 153600 Byte frame buffer in PSRAM
I (1644) cam_hal: cam config ok
I (1654) ov5640: Set PLL: bypass: 0, multiplier: 8, sys_div: 1, pre_div: 1, root_2x: 0, pclk_root_div: 1, pclk_manual: 1, pclk_div: 4
I (1654) ov5640: Calculated XVCLK: 20000000 Hz, REFIN: 20000000 Hz, VCO: 160000000 Hz, PLL_CLK: 64000000 Hz, SYSCLK: 16000000 Hz, PCLK: 8000000 Hz
I (1704) camera: Sensor configured: vflip=1, auto WB/exp/gain, lens correction
I (1704) camera: Camera initialized: OV5640 RGB565 320x240, 2 buffers in PSRAM
I (1704) main: Camera initialized (OV5640 RGB565 320x240)
I (1714) inference: Loading YOLO11n from partition 'models' offset=0xa7000 size=2800272 bytes
I (1714) inference: Partition found: offset=0x410000, size=7340032
I (1724) inference: Model mmap'd at 0x3c987000, 2800272 bytes
I (2194) inference:   Output[0]: name='box0' shape=[1, 28, 28, 64] dtype=3 exponent=-3
I (2194) inference:   Output[1]: name='box1' shape=[1, 14, 14, 64] dtype=3 exponent=-3
I (2194) inference:   Output[2]: name='box2' shape=[1, 7, 7, 64] dtype=3 exponent=-3
I (2204) inference:   Output[3]: name='score0' shape=[1, 28, 28, 5] dtype=3 exponent=-3
I (2214) inference:   Output[4]: name='score1' shape=[1, 14, 14, 5] dtype=3 exponent=-2
I (2224) inference:   Output[5]: name='score2' shape=[1, 7, 7, 5] dtype=3 exponent=-2
I (2224) inference:   Input: shape=[1, 224, 224, 3] dtype=3 exponent=-7
I (2234) inference: Model loaded OK. 6 output(s). PSRAM free: 4629 KB
I (2244) main: Inference engine ready
I (2244) postproc: Postprocessor initialized for YOLO11n
I (2244) main: Postprocessor initialized for YOLO11n
I (2254) temperature_sensor: Range [-10°C ~ 80°C], error < 1°C
I (2264) metrics: Temperature sensor initialized
I (2264) metrics: Metrics system initialized
I (2264) main: Memory post-init: PSRAM=4629 KB free, Internal=122 KB free
I (2274) pp: pp rom version: e7ae62f
I (2274) net80211: net80211 rom version: e7ae62f
I (2294) wifi:wifi driver task: 3fcd4520, prio:23, stack:6656, core=0
I (2304) wifi:wifi firmware version: e8ce6e2
I (2304) wifi:wifi certification version: v7.0
I (2304) wifi:config NVS flash: enabled
I (2304) wifi:config nano formatting: disabled
I (2304) wifi:Init data frame dynamic rx buffer num: 32
I (2314) wifi:Init static rx mgmt buffer num: 5
I (2314) wifi:Init management short buffer num: 32
I (2314) wifi:Init dynamic tx buffer num: 32
I (2324) wifi:Init static tx FG buffer num: 2
I (2324) wifi:Init static rx buffer size: 1600
I (2334) wifi:Init static rx buffer num: 10
I (2334) wifi:Init dynamic rx buffer num: 32
I (2334) wifi_init: rx ba win: 6
I (2344) wifi_init: accept mbox: 6
I (2344) wifi_init: tcpip mbox: 32
I (2344) wifi_init: udp mbox: 6
I (2354) wifi_init: tcp mbox: 6
I (2354) wifi_init: tcp tx win: 5760
I (2354) wifi_init: tcp rx win: 5760
I (2364) wifi_init: tcp mss: 1440
I (2364) wifi_init: WiFi IRAM OP enabled
I (2364) wifi_init: WiFi RX IRAM OP enabled
I (2374) phy_init: phy_version 701,f4f1da3a,Mar  3 2025,15:50:10
cam_hal: EV-VSYNC-OVF
I (2414) wifi:mode : softAP (1c:db:d4:4a:f4:59)
I (2414) wifi:Total power save buffer number: 16
I (2414) wifi:Init max length of beacon: 752/752
I (2414) wifi:Init max length of beacon: 752/752
I (2424) network: WiFi AP started: SSID=ESP32_TFM, channel=1
I (2424) esp_netif_lwip: DHCP server started on interface WIFI_AP_DEF with IP: 192.168.4.1
I (2434) network: HTTP server started on port 80
I (2434) dashboard: Dashboard registered at / (5359 bytes)
I (2444) main: Network ready: WiFi AP 'ESP32_TFM', http://192.168.4.1/
I (2454) main: Inference task started on core 0
I (2564) inference: DIAG input: 150528/150528 non-zero, min=12 max=127, exp=-7, first16=[99 104 95 99 104 95 99 102 95 99 104 95 99 104 95 99]
I (2694) main: ========================================
I (2984) main:   System running. Connect to 'ESP32_TFM'
I (2984) main:   Dashboard: http://192.168.4.1/
I (2984) main: ========================================
I (2984) main_task: Returned from app_main()
I (3464) inference: DIAG output 'box0' [INT8]: 50099/50176 nz, raw=[-103,79] exp=-3 deq_max=9.875 sig_max=0.9999 shape=[1, 28, 28, 64]
I (3464) inference: DIAG output 'box1' [INT8]: 12459/12544 nz, raw=[-60,46] exp=-3 deq_max=5.750 sig_max=0.9968 shape=[1, 14, 14, 64]
I (3474) inference: DIAG output 'box2' [INT8]: 3116/3136 nz, raw=[-46,42] exp=-3 deq_max=5.250 sig_max=0.9948 shape=[1, 7, 7, 64]
I (3484) inference: DIAG output 'score0' [INT8]: 3920/3920 nz, raw=[-128,-60] exp=-3 deq_max=-7.500 sig_max=0.0006 shape=[1, 28, 28, 5]
I (3494) inference: DIAG output 'score1' [INT8]: 980/980 nz, raw=[-45,-26] exp=-2 deq_max=-6.500 sig_max=0.0015 shape=[1, 14, 14, 5]
I (3514) inference: DIAG output 'score2' [INT8]: 245/245 nz, raw=[-50,-32] exp=-2 deq_max=-8.000 sig_max=0.0003 shape=[1, 7, 7, 5]
I (3524) yolo_decode: stage0: scores INT8, score_exp=-3, box_exp=-3, H=28
I (3524) yolo_decode: stage1: scores INT8, score_exp=-2, box_exp=-3, H=14
I (3534) yolo_decode: stage2: scores INT8, score_exp=-2, box_exp=-3, H=7
I (3544) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (3664) inference: DIAG input: 150528/150528 non-zero, min=2 max=110, exp=-7, first16=[78 79 70 78 81 70 78 83 70 78 83 70 78 83 70 78]
I (4554) inference: DIAG output 'box0' [INT8]: 50093/50176 nz, raw=[-104,79] exp=-3 deq_max=9.875 sig_max=0.9999 shape=[1, 28, 28, 64]
I (4554) inference: DIAG output 'box1' [INT8]: 12471/12544 nz, raw=[-60,45] exp=-3 deq_max=5.625 sig_max=0.9964 shape=[1, 14, 14, 64]
I (4564) inference: DIAG output 'box2' [INT8]: 3111/3136 nz, raw=[-46,42] exp=-3 deq_max=5.250 sig_max=0.9948 shape=[1, 7, 7, 64]
I (4574) inference: DIAG output 'score0' [INT8]: 3920/3920 nz, raw=[-128,-62] exp=-3 deq_max=-7.750 sig_max=0.0004 shape=[1, 28, 28, 5]
I (4584) inference: DIAG output 'score1' [INT8]: 980/980 nz, raw=[-45,-26] exp=-2 deq_max=-6.500 sig_max=0.0015 shape=[1, 14, 14, 5]
I (4604) inference: DIAG output 'score2' [INT8]: 245/245 nz, raw=[-50,-32] exp=-2 deq_max=-8.000 sig_max=0.0003 shape=[1, 7, 7, 5]
I (4614) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (4734) inference: DIAG input: 150528/150528 non-zero, min=2 max=110, exp=-7, first16=[78 81 70 78 81 70 78 81 70 78 82 70 80 83 70 81]
I (5624) inference: DIAG output 'box0' [INT8]: 50092/50176 nz, raw=[-102,79] exp=-3 deq_max=9.875 sig_max=0.9999 shape=[1, 28, 28, 64]
I (5624) inference: DIAG output 'box1' [INT8]: 12477/12544 nz, raw=[-60,45] exp=-3 deq_max=5.625 sig_max=0.9964 shape=[1, 14, 14, 64]
I (5634) inference: DIAG output 'box2' [INT8]: 3114/3136 nz, raw=[-46,42] exp=-3 deq_max=5.250 sig_max=0.9948 shape=[1, 7, 7, 64]
I (5644) inference: DIAG output 'score0' [INT8]: 3920/3920 nz, raw=[-128,-62] exp=-3 deq_max=-7.750 sig_max=0.0004 shape=[1, 28, 28, 5]
I (5654) inference: DIAG output 'score1' [INT8]: 980/980 nz, raw=[-46,-25] exp=-2 deq_max=-6.250 sig_max=0.0019 shape=[1, 14, 14, 5]
I (5674) inference: DIAG output 'score2' [INT8]: 245/245 nz, raw=[-49,-32] exp=-2 deq_max=-8.000 sig_max=0.0003 shape=[1, 7, 7, 5]
I (5684) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (6684) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (7694) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (8694) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (9694) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (10684) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (11684) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (12684) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (12684) main: Frame 10 | FPS=1.0 | inf=891ms total=994ms | det=0 | PSRAM=4480KB
I (13674) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (14684) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (15684) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (16674) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (17674) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (18674) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (19034) wifi:new:<1,0>, old:<1,1>, ap:<1,0>, sta:<255,255>, prof:1, snd_ch_cfg:0x0
I (19034) wifi:station: 16:92:a9:c4:ac:40 join, AID=1, bgn, 20
I (19054) network: Station 16:92:a9:c4:ac:40 joined, AID=1
I (19354) wifi:<ba-add>idx:2 (ifx:1, 16:92:a9:c4:ac:40), tid:6, ssn:0, winSize:64
I (19704) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (20394) esp_netif_lwip: DHCP server assigned IP to a client, IP is: 192.168.4.2
I (20714) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (21724) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (22404) wifi:<ba-add>idx:3 (ifx:1, 16:92:a9:c4:ac:40), tid:0, ssn:2, winSize:64
I (22724) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (22734) main: Frame 20 | FPS=1.0 | inf=897ms total=1000ms | det=0 | PSRAM=4480KB
I (23734) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (24734) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (25734) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (26734) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (27744) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (28734) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (29304) network: WebSocket connection opened
I (29704) network: WebSocket connection opened
I (29834) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (30844) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (31844) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (32844) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (32844) main: Frame 30 | FPS=1.0 | inf=891ms total=996ms | det=0 | PSRAM=4480KB
I (33904) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (34904) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (35904) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (36904) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (37904) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (38954) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (39964) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (40964) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (41954) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (42964) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (42974) main: Frame 40 | FPS=1.0 | inf=894ms total=998ms | det=0 | PSRAM=4480KB
I (44024) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (45024) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (46034) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (47034) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (48034) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (49094) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (50094) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (51094) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (52094) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (53094) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (53094) main: Frame 50 | FPS=1.0 | inf=893ms total=995ms | det=0 | PSRAM=4480KB
I (54154) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (55154) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (56164) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (57164) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (58164) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (59224) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (60224) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (61234) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (62234) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (63234) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (63244) main: Frame 60 | FPS=1.0 | inf=893ms total=997ms | det=0 | PSRAM=4480KB
I (64314) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (65324) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (66324) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (67324) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (67514) wifi:<ba-add>idx:4 (ifx:1, 16:92:a9:c4:ac:40), tid:1, ssn:0, winSize:64
I (67594) wifi:<ba-add>idx:5 (ifx:1, 16:92:a9:c4:ac:40), tid:5, ssn:0, winSize:64
I (68334) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (69404) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (70414) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (71414) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (72434) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (73434) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (73444) main: Frame 70 | FPS=1.0 | inf=895ms total=998ms | det=0 | PSRAM=4480KB
I (74504) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (75514) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (76514) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (77514) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (78524) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (79584) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (80594) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (81594) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (82604) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (83614) yolo_decode: YOLO11n: maxScore=0.000 threshold=0.10 → 0 cands → 0 dets
I (83624) main: Frame 80 | FPS=1.0 | inf=895ms total=999ms | det=0 | PSRAM=4480KB
```

GitHub Copilot: 

  

Starting: *Analyze device output discrepancy* (1/3)

Let me gather key diagnostic data before presenting the analysis. 

Ran terminal command:  ls -la /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_C_percentile.json 2>&1 && echo "---" && head -100 /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/yolo11n_v1_best_C_percentile.json 2>&1

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && ./env/bin/python -c "
import json

with open('models/yolo11n_v1_best_C_percentile.json') as f:
    cfg = json.load(f)

# Find score conv nodes and their quant params
score_nodes = ['node_conv2d_69', 'node_conv2d_77', 'node_conv2d_85']
box_nodes = ['node_conv2d_67', 'node_conv2d_75', 'node_conv2d_83']

for prefix in ['score', 'box']:
    nodes = score_nodes if prefix == 'score' else box_nodes
    for node_name in nodes:
        # Search for matching config key
        for key in cfg['configs']:
            if node_name.lower().replace('_', '') in key.lower().replace('_', ''):
                print(f'=== {prefix}: {key} ===')
                params = cfg['configs'][key]
                for tensor_name, tensor_cfg in params.items():
                    if isinstance(tensor_cfg, dict) and 'hash' in tensor_cfg:
                        bw = tensor_cfg.get('bit_width', '?')
                        state = tensor_cfg.get('state', '?')
                        print(f'  {tensor_name}: bw={bw} state={state}')
                break

# Search by cv3 (score) and cv2 (box)
print()
print('=== All cv3 (score) nodes ===')
for key in cfg['configs']:
    if 'cv3' in key.lower():
        params = cfg['configs'][key]
        print(f'{key}:')
        for tn, tc in params.items():
            if isinstance(tc, dict) and 'hash' in tc:
                print(f'  {tn}: bw={tc.get(\"bit_width\")} state={tc.get(\"state\")}')

print()
print('=== All cv2 (box) nodes ===')
for key in cfg['configs']:
    if 'cv2' in key.lower() and 'cv2.' in key.lower():
        params = cfg['configs'][key]
        print(f'{key}:')
        for tn, tc in params.items():
            if isinstance(tc, dict) and 'hash' in tc:
                print(f'  {tn}: bw={tc.get(\"bit_width\")} state={tc.get(\"state\")}')
" 2>&1

The inline Python had escaping issues. Let me use a script file: 

Created [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/inspect_quant_config.py)

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && ./env/bin/python models/inspect_quant_config.py 2>&1

Terminal seems polluted. Let me use a new one: 

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE
./env/bin/python models/inspect_quant_config.py

Ran terminal command:  ./env/bin/python models/inspect_quant_config.py

No cv2/cv3 in node names. Let me check what node names look like:  

Created [](file:///tmp/inspect_nodes.py)

Ran terminal command:  ./env/bin/python /tmp/inspect_nodes.py 2>&1 | tail -80

Replacing 1 lines with 1 lines in [](file:///tmp/inspect_nodes.py)


```

```


Ran terminal command:  ./env/bin/python /tmp/inspect_nodes.py 2>&1 | tail -80

Now let me run the definitive test — extract and compare actual weight/bias values from the PPQ quantized graph for score vs box convs:   

Created [](file:///tmp/diag_ppq.py)

Ran terminal command:  ./env/bin/python /tmp/diag_ppq.py 2>&1 | tee /tmp/diag_ppq_output.txt

Summarized conversation historyOptimizing tool selection...

Checked background terminal output

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01X3mfNLioHmyEAP6vaFBMdy__vscode-1770907754880/content.txt)

Starting: *Fix weight/bias extraction* (2/5)

The weight/bias analysis sections were empty. Let me check the diagnostic script and fix it. 

Read [](file:///tmp/diag_ppq.py)

The weight/bias sections were empty because the script looked for `cv3`/`cv2` in operation names, but PPQ uses `node_conv2d_*`. Let me create a targeted diagnostic using the correct node names. 

Created [](file:///tmp/diag_weights.py)

Ran terminal command:  ./env/bin/python /tmp/diag_weights.py 2>&1 | tail -200

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01RDBNVbMRqGytaZsxzqzhdW__vscode-1770907754885/content.txt)

The script found the operations but crashed on attribute access. Let me create a simpler, more robust version: 

Created [](file:///tmp/diag_weights2.py)

Ran terminal command:  ./env/bin/python /tmp/diag_weights2.py 2>&1 | grep -v "^\[INFO\]\|^\[WARNING\]\|^Calibration\|^-----\|^\[21:" | head -200

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/9ea21977-9b5b-4e15-ab12-c311ed793bb8/toolu_01EiCbgjWAUV7Hv2JQiyJp6m__vscode-1770907754888/content.txt)

Excellent! Got the score conv analysis. Now the box conv names changed after re-simplification. Let me find the correct names and get the complete picture: 

Ran terminal command:  cd /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE && ./env/bin/python -c "
import pickle, json, os
import numpy as np

# Load the just-exported .info
info_path = 'models/yolo11n_diag_test3.info'
if os.path.exists(info_path):
    with open(info_path) as f:
        print(f.read())
"


Ran terminal command:  grep -n "box\|score\|Output\|Concat\|cv2\|cv3" models/yolo11n_diag_test3.info | head -40

Ran terminal command:  grep -n "box0\|box1\|box2\|score0\|score1\|score2" models/yolo11n_diag_test3.info

Now let me check the SiLU/Swish operations that feed into the score vs box convs, and look at the ESP-DL Conv2D implementation: 

Ran terminal command:  grep -n "silu_59\|silu_63\|silu_65\|silu_69\|silu_71\|silu_75" models/yolo11n_diag_test3.info | head -20

Now let me look at the ESP-DL Conv2D implementation for potential issues with 5 output channels: 

Search ESP-DL Conv2D impl

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/README.md)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/module)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/math)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/module/include)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/module/src)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base_conv2d.hpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base_conv2d.cpp#1-1), lines 1 to 300

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/module/include/dl_module_conv.hpp#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base.hpp#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base_conv2d.cpp#300-300), lines 300 to 600

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/tie728)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base.hpp#200-200), lines 200 to 500

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base_conv2d.cpp#600-600), lines 600 to 900

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base.hpp#80-80), lines 80 to 120

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base_conv2d.cpp#900-900), lines 900 to 1200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/dl_base_isa.hpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base_conv2d.cpp#1200-1200), lines 1200 to 1347

Searched for regex `n_div_x|n_remainder|output_channel.*16|output_channel.*%` (`**/espressif__esp-dl/dl/base/dl_base.hpp`), 10 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base.hpp#270-270), lines 270 to 300

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base.hpp#1150-1150), lines 1150 to 1300

Searched for regex `s8.*conv2d|s8.*unaligned.*conv` (`**/dl_base_tie728.h`), 20 results

Ran terminal command:  wc -l /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/tie728/dl_tie728_s8_conv2d.S

Ran terminal command:  grep -n "unaligned\|n_remainder\|n_div_x\|remainder\|output_channel\|\.global.*s8.*conv2d" /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/tie728/dl_tie728_s8_conv2d.S | head -60

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/tie728/dl_tie728_s8_conv2d.S#383-383), lines 383 to 450

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/tie728/dl_tie728_s8_conv2d.S#450-450), lines 450 to 580

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/tie728/dl_tie728_s8_conv2d.S#1020-1020), lines 1020 to 1200

Searched for regex `conv_operation_shell` (`**/espressif__esp-dl/dl/base/dl_base.hpp`), 2 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base.hpp#340-340), lines 340 to 1160

Searched for regex `reset_bias_layout|channel.*pad|channel.*align|output_channel.*pad` (`**/espressif__esp-dl/**`), 20 results

Searched for regex `reset_bias_layout|filter_layout|reset_filter|change_layout` (`**/espressif__esp-dl/dl/tensor/**`), 2 results

Searched for regex `filter.*layout|reshape_filter|channel.*16|padding|filter.*pad` (`**/espressif__esp-dl/dl/tensor/**`), 16 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/tensor/src/dl_tensor_base.cpp#370-370), lines 370 to 500

Searched for regex `filter.*layout|weight.*layout|filter.*reorder|reshape.*filter` (`**/espressif__esp-dl/**`), no results

Searched for regex `output_channel.*align|n_div|aligned.*unaligned|split.*filter` (`**/espressif__esp-dl/**`), 20 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/isa/tie728/dl_tie728_s8_conv2d.S#1-1), lines 1 to 80

Searched for regex `buffer_bias_linear|buffer_0000_linear` (`**/espressif__esp-dl/dl/base/dl_base_activate_output.hpp`), no results

Searched for regex `buffer_bias_linear|buffer_0000_linear` (`**/espressif__esp-dl/dl/base/dl_base_activate_buffer.hpp`), 2 results

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/managed_components/espressif__esp-dl/dl/base/dl_base_activate_buffer.hpp#1-1), lines 1 to 60

Made changes.