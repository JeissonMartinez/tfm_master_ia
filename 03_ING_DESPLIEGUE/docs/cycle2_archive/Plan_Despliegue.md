User: Vas a actuar como un experto en Visión Artificial con Deep Learning aplicado a sistemas embebidos bajo el paradigma de TinyML, en específico en microcontroladores ESP32-S3 WROOM N16R8 con el uso de la cámara OV5640. 

Vas a organizar, detallar y planificar el despliegue e implementación de los 2 mejores modelos seleccionados en la sección 4 del notebook: `02_ING_MODELOS/GoogleCloudAI/notebooks/Informe_Ciclo1_Train.ipynb` .

Para hacerlo, debemos tener en cuenta:
- La sección 5 del mismo notebook, donde se especifican los requerimientos de software del firmware ESP32-S3.
- La sección 6 del mismo notebook, donde se especifician los riesgos identificados y mitigaciones necesarias.
- La sección 7.2 del mismo notebook, donde se detalla el Roadmap de implementación del firmware.

Todo lo que vamos a desarrollar, se hará en la carpeta `03_ING_DESPLIEGUE` la cual ya tenemos una planificación de la estructura de archivos en la sección 5.2 del mismo notebook.

Además vamos a usar C++ como lenguaje base.

Adicionalmente, considerar la necesidad de implementar un notebook auxiliar (03_ING_DESPLIEGUE/Conversion_ModelosTFLite.ipynb) donde se realice de forma previa la conversión de los mejores modelos seleccionados al formato requerido para la implementación en la ESP32-S3. Yo previamente había hecho pruebas transformando al formato .espdl y me había funcionado bien. Tal vez se pueda tener ese formato de exportación como alternativa al .h planificado (si es que es compatible).

Por último, explorar la alternativa de que el dashboard HTML planificado para visualizar las métricas y el streaming, se codifique de forma local y no en la memoria de la ESP32, para no recargarla más (evalúa si es buena idea o no).





GitHub Copilot: 

 I'll start by gathering context from the notebook and the existing deployment structure. Let me research thoroughly.     

Retrieving Notebook summary.

Retrieving Notebook summary.

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/main/03_ING_DESPLIEGUE.c)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/CMakeLists.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/Configuracion_ESP32-S3.md#1-1), lines 1 to 100

Research notebook sections 4-7

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/notebooks/Informe_Ciclo1_Train.ipynb#516-516), lines 516 to 568

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/notebooks/Informe_Ciclo1_Train.ipynb#571-571), lines 571 to 666

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/notebooks/Informe_Ciclo1_Train.ipynb#669-669), lines 669 to 760

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/notebooks/Informe_Ciclo1_Train.ipynb#763-763), lines 763 to 866

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/notebooks/Informe_Ciclo1_Train.ipynb#869-869), lines 869 to 1001

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/notebooks/Informe_Ciclo1_Train.ipynb#1004-1004), lines 1004 to 1099

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/notebooks/Informe_Ciclo1_Train.ipynb#1102-1102), lines 1102 to 1244

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/e6aea651da8b505186c9a3cfa45e41c8/GitHub.copilot-chat/chat-session-resources/0cd68bb7-00f5-4068-905c-f30f19903743/toolu_vrtx_01DBiQC2HYq1Yy15HLj8Cjqi__vscode-1770829238864/content.txt#1-1), lines 1 to 300

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/e6aea651da8b505186c9a3cfa45e41c8/GitHub.copilot-chat/chat-session-resources/0cd68bb7-00f5-4068-905c-f30f19903743/toolu_vrtx_01DBiQC2HYq1Yy15HLj8Cjqi__vscode-1770829238864/content.txt#300-300), lines 300 to 671

Now let me gather additional context about the existing deployment structure and the `.espdl` format the user mentioned.    

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/Configuracion_ESP32-S3.md#100-100), lines 100 to 178

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/main)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/main/CMakeLists.txt)

Research ESPDL format ESP32



Fetched 3 resources



Fetched 3 resources



Fetched 3 resources



Fetched 2 resources



Fetched https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-reference/protocols/esp_http_server.html

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/e6aea651da8b505186c9a3cfa45e41c8/GitHub.copilot-chat/chat-session-resources/0cd68bb7-00f5-4068-905c-f30f19903743/toolu_vrtx_01LQNFEZHYmxPocQZouwS4ZX__vscode-1770829238878/content.txt)

Now let me check for existing .tflite model files and the Google Cloud AI outputs: 

Find tflite model files

Searched for files matching `**/*.tflite`, 20 matches

Searched for files matching `**/*.espdl`, no matches

Searched for files matching `**/*.onnx`, 7 matches

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/outputs)

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/scripts)

Searched for regex `convert|export|tflite|onnx|espdl|conversion` (`**/02_ING_MODELOS/**/*.py`), 20 results

Searched for files matching `**/*.tflite`, 53 matches

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/outputs/MBNTv2_ssdlite_v1)

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/outputs/MBNTv2_ssdlite_v2)

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/outputs/MBNTv3S_ssdlite_v1)

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/outputs/MBNTv3S_ssdlite_v2)

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/outputs/MBNTv3S_ssdlite_v3)

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/outputs/yolo11n_v1)

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/outputs/yolo26n_v1)

Searched for regex `convert|export|tflite|onnx|espdl|conversion` (`**/02_ING_MODELOS/**/*.py`), 50 results

Searched for files matching `**/02_ING_MODELOS/**/export*.py`, 3 matches

Searched for files matching `**/02_ING_MODELOS/**/convert*.py`, no matches

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/Local_2/scripts)

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/src_colab)

Read [](file:///Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/GoogleCloudAI/trainer)

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/e6aea651da8b505186c9a3cfa45e41c8/GitHub.copilot-chat/chat-session-resources/0cd68bb7-00f5-4068-905c-f30f19903743/toolu_vrtx_012ku4niG9s2QSqP4KfJ1FV2__vscode-1770829238885/content.txt)

I've gathered comprehensive context from the notebook, existing codebase, and research on ESP-DL vs TFLite Micro. Before drafting the plan, I need to clarify a few points: 

Asked 4 questions (2 modelos, Runtime, Dashboard, Placa)

Perfecto. Con todas las decisiones claras, voy a elaborar el plan detallado. 

Asked 2 questions (ESP-IDF, ONNX MBNTv2)

Perfecto — ESP-IDF v5.4.3 es totalmente compatible con ambos frameworks. Tengo todo lo necesario para el plan. Aquí va:

---

## Plan: Despliegue TinyML en ESP32-S3 — Firmware Dual-Path con 3 Modelos

### TL;DR

Implementar el firmware completo para la Freenove ESP32-S3 CAM Board (WROOM N16R8 + OV5640) que permita cargar **uno de los 3 modelos seleccionados** (MBNTv2_ssdlite_v1, YOLO11n_v1, YOLO26n_v1) y ejecutar inferencia de detección de objetos en tiempo real. El sistema soportará **dos runtimes de inferencia** — ESP-DL (.espdl) y TFLite Micro (.tflite) — permitiendo comparar rendimiento real en hardware. Se incluye un notebook de conversión previo, dashboard HTML embebido mínimo con WebSocket, y métricas de performance en tiempo real. Todo el código será **C++** (GNU++23) sobre ESP-IDF v5.4.3.

La implementación se estructura en un **notebook de conversión** (pre-flash) + **4 fases de firmware** alineadas con el roadmap del notebook, priorizando tener inferencia funcional antes de completar la comunicación.

---

### Parte A: Notebook de Conversión de Modelos

**Archivo:** Conversion_ModelosTFLite.ipynb (actualmente vacío)

**Propósito:** Preparar los 3 modelos en ambos formatos de despliegue antes de flashear el firmware.

**Secciones del notebook:**

1. **Sección 1 — Setup y verificación de modelos fuente**
   - Verificar existencia de los 3 `.tflite` INT8 en `GoogleCloudAI/outputs/`:
     - `MBNTv2_ssdlite_v1/tflite/MBNTv2_ssdlite_v1_int8.tflite` (1.20 MB)
     - `yolo11n_v1/tflite/best_int8.tflite` (2.68 MB)
     - `yolo26n_v1/tflite/best_int8.tflite` (2.55 MB)
   - Verificar existencia de `.onnx` para YOLO:
     - `yolo11n_v1/train/weights/best.onnx`
     - `yolo26n_v1/train/weights/best.onnx`
   - Imprimir tamaños, shapes de entrada/salida, operadores usados (con `tflite` Python API)

2. **Sección 2 — Path TFLite Micro: conversión .tflite → C header**
   - Convertir cada `.tflite` a array C con `xxd -i` o equivalente Python (`open(rb)` → formatear bytes)
   - Generar 3 headers:
     - `mobilenetv2_ssdlite_v1_int8.h`
     - `yolo11n_v1_int8.h`
     - `yolo26n_v1_int8.h`
   - Cada header contiene `const unsigned char model_data[]` y `const unsigned int model_data_len`
   - Copiar automáticamente a `03_ING_DESPLIEGUE/main/models/tflite/`

3. **Sección 3 — Path ESP-DL: conversión ONNX → ESPDL**
   - **MBNTv2_ssdlite_v1**: Exportar `.keras` → ONNX vía `tf2onnx`, luego ONNX → `.espdl` vía `esp-ppq`
   - **YOLO11n_v1 y YOLO26n_v1**: ONNX → `.espdl` vía `esp-ppq` directamente (los `.onnx` ya existen)
   - Configurar cuantización INT8 simétrica (POWER_OF_TWO) con dataset representativo (usar imágenes del dataset de test)
   - Explorar cuantización mixta 8+16 bits para capas sensibles (especialmente en MBNTv2)
   - Generar 3 archivos `.espdl` + sus `.info` para inspección
   - Copiar a `03_ING_DESPLIEGUE/main/models/espdl/`

4. **Sección 4 — Validación cruzada**
   - Comparar salidas de inferencia: Python TFLite vs Python ESP-DL (si `esp-ppq` lo permite) con imágenes de calibración
   - Verificar que los shapes de salida coinciden con lo esperado del notebook (sección 5.5)
   - Generar tabla resumen de tamaños, shapes, operadores por formato

5. **Sección 5 — Informe resumen**
   - Tabla comparativa de ambos formatos para cada modelo
   - Documentar cualquier operador no soportado o limitación encontrada

---

### Parte B: Estructura del Proyecto Firmware

Evolución de la estructura planificada en sección 5.2 del notebook, adaptada al dual-path y a C++:

```
03_ING_DESPLIEGUE/
├── CMakeLists.txt                         # Proyecto ESP-IDF raíz (ya existe, modificar)
├── partitions.csv                         # Nuevo: esquema custom (NVS 24KB, PHY 4KB, Factory 15MB)
├── sdkconfig.defaults                     # Nuevo: PSRAM octal 80MHz, flash 16MB, C++23, WiFi
│
├── main/
│   ├── CMakeLists.txt                     # Modificar: registrar .cpp, dependencias, modelos
│   ├── idf_component.yml                  # Nuevo: esp32-camera, esp-tflite-micro, esp-dl
│   │
│   ├── main.cpp                           # Nuevo: app_main(), init HW, selección modelo, loop
│   ├── app_config.h                       # Nuevo: constantes globales, pines, tipos enumerados
│   │
│   ├── camera.h / camera.cpp             # Nuevo: init OV5640, captura RGB565, FB management
│   ├── image_proc.h / image_proc.cpp     # Nuevo: crop 224×224, RGB565→INT8
│   │
│   ├── inference_engine.h                 # Nuevo: interfaz abstracta para dual-path
│   ├── tflite_engine.h / tflite_engine.cpp  # Nuevo: runtime TFLite Micro
│   ├── espdl_engine.h / espdl_engine.cpp    # Nuevo: runtime ESP-DL
│   │
│   ├── postprocess.h / postprocess.cpp   # Nuevo: decode/NMS por tipo (MBNTv2, YOLO11, YOLO26)
│   ├── metrics.h / metrics.cpp           # Nuevo: FPS, latencia, heap, EMA, temp
│   ├── wifi_manager.h / wifi_manager.cpp # Nuevo: WiFi AP mode
│   ├── web_server.h / web_server.cpp     # Nuevo: HTTP + WebSocket + dashboard
│   ├── dashboard.h                        # Nuevo: HTML dashboard como string C (gzipped)
│   │
│   └── models/
│       ├── tflite/                        # Headers C generados por xxd -i
│       │   ├── mobilenetv2_ssdlite_v1_int8.h
│       │   ├── yolo11n_v1_int8.h
│       │   └── yolo26n_v1_int8.h
│       └── espdl/                         # Archivos .espdl (vinculados como binarios)
│           ├── mobilenetv2_ssdlite_v1.espdl
│           ├── yolo11n_v1.espdl
│           └── yolo26n_v1.espdl
│
├── components/
│   ├── esp32-camera/                      # Git submodule
│   ├── esp-tflite-micro/                  # Componente TFLite Micro (vía component registry)
│   └── esp-dl/                            # Componente ESP-DL v3.x (vía component registry)
│
├── frontend/
│   └── dashboard.html                     # HTML fuente del dashboard (para editar/mantener)
│
├── Conversion_ModelosTFLite.ipynb          # Notebook de conversión (ya existe, vacío)
├── Configuracion_ESP32-S3.md              # Documentación HW (ya existe)
│
└── docs/
    └── README.md                          # Instrucciones de build, flash y uso
```

**Cambio clave vs plan original:** Se introduce una **capa de abstracción `inference_engine.h`** que define una interfaz común para ambos runtimes (TFLite y ESP-DL), permitiendo cambiar de runtime sin modificar el loop principal. Los modelos `.espdl` se montan como binarios embebidos en flash (vía `EMBED_FILES` en CMake) en lugar de arrays C, ya que ESP-DL los carga directamente.

---

### Parte C: Implementación del Firmware — 4 Fases

#### Fase 1 — Infraestructura base (Prioridad: Alta)

1. **Configurar proyecto ESP-IDF** en CMakeLists.txt:
   - Actualizar `cmake_minimum_required`, `project()` 
   - Configurar C++23 (`-std=gnu++2b`)

2. **Crear** `partitions.csv`:
   - NVS: 24 KB, PHY: 4 KB, Factory: ~15 MB (para modelos + firmware)

3. **Crear** `sdkconfig.defaults`:
   - PSRAM: Octal 80 MHz, `heap_caps_malloc`
   - Flash: 16 MB, DIO
   - CPU: 240 MHz dual-core
   - WiFi IRAM opt habilitado
   - LWIP max sockets: 10
   - HTTPD WebSocket support habilitado

4. **Crear** `main/idf_component.yml`:
   - Declarar dependencias: `espressif/esp32-camera`, `espressif/esp-tflite-micro`, `espressif/esp-dl`

5. **Crear** `main/app_config.h`:
   - Todas las constantes (pines OV5640, resoluciones, clases, thresholds)
   - Enums: `model_type_t`, `engine_type_t` (TFLITE, ESPDL)
   - Structs: `detection_t`, `inference_metrics_t`, `model_config_t`

6. **Implementar** `camera.h / camera.cpp`:
   - `camera_init()`: configurar pines Freenove, RGB565, 320×240, double buffer en PSRAM
   - `camera_capture()` / `camera_release()`: adquirir y liberar frame buffer
   - Ajustes de sensor post-init (vflip, auto WB/exposure/gain) según Configuracion_ESP32-S3.md

7. **Implementar** `image_proc.h / image_proc.cpp`:
   - `preprocess_frame()`: crop central (offset_x=48), RGB565 → RGB888 → INT8 `[-128, 127]`
   - Buffer de salida en PSRAM (224×224×3 = ~150 KB)

8. **Verificar PSRAM**: test de inicialización, tamaño disponible, velocidad de acceso
9. **Eliminar** 03_ING_DESPLIEGUE.c (archivo placeholder, reemplazado por `main.cpp`)

#### Fase 2 — Motores de Inferencia Dual-Path (Prioridad: Alta)

10. **Crear** `inference_engine.h` — interfaz abstracta:
    - `init(model_config_t*)` → cargar modelo y alojar arena/buffers
    - `invoke(int8_t* input)` → ejecutar inferencia
    - `get_output()` → puntero a tensor(es) de salida
    - `get_arena_used()` → bytes usados
    - `deinit()` → liberar recursos

11. **Implementar** `tflite_engine.cpp`:
    - Usar `MicroMutableOpResolver` con solo los operadores necesarios de los 3 modelos (Conv2D, DepthwiseConv2D, Add, Reshape, Quantize, Dequantize, Concatenation, Logistic, etc.)
    - Arena en PSRAM vía `heap_caps_malloc(arena_size, MALLOC_CAP_SPIRAM)`
    - Manejar los 3 formatos de salida (3 tensores MBNTv2, `[1,9,1029]` YOLO11, `[1,300,6]` YOLO26)

12. **Implementar** `espdl_engine.cpp`:
    - Cargar modelo `.espdl` desde flash (embebido con `EMBED_FILES`)
    - Configurar `dl::Model` con planificador de memoria automático
    - Aprovecha scheduling dual-core automático y operadores SIMD
    - Mismo interfaz que TFLite engine

13. **Implementar** `postprocess.h / postprocess.cpp`:
    - **`postprocess_mobilenet()`**: Decode 1470 anchors + softmax + NMS O(N²)
    - **`postprocess_yolo11()`**: Transpose `[9,1029]`, filtrado por score, NMS completo. Optimizaciones: filtro agresivo (threshold ≥ 0.4), limitar a top-100 candidatos, NMS con early exit (mitiga riesgo R4)
    - **`postprocess_yolo26()`**: Filtrado lineal O(N) de 300 detecciones (NMS integrado, **no necesita NMS**)
    - NMS compartido: implementación simple con IoU threshold configurable

14. **Validar con imagen estática**: Cargar imagen de test hardcodeada, verificar que cada modelo produce detecciones coherentes con ambos runtimes

#### Fase 3 — Métricas y Comunicación (Prioridad: Media)

15. **Implementar** `metrics.h / metrics.cpp`:
    - Timers con `esp_timer_get_time()` para preprocess, inference, postprocess
    - Heap stats: `heap_caps_get_free_size(MALLOC_CAP_INTERNAL)`, `MALLOC_CAP_SPIRAM`
    - Arena usado: `interpreter->arena_used_bytes()` (TFLite) o equivalente ESP-DL
    - Temperatura: `temperature_sensor_get_celsius()`
    - EMA con ventana 30 frames (alpha ≈ 0.065)
    - Struct `inference_metrics_t` poblada cada frame

16. **Implementar** `wifi_manager.h / wifi_manager.cpp`:
    - Modo AP: SSID `ESP32_TFM`, contraseña configurable
    - DHCP server integrado, IP fija `192.168.4.1`
    - Event handlers para conexión/desconexión de clientes

17. **Implementar** `web_server.h / web_server.cpp`:
    - `GET /` → servir dashboard HTML (gzipped desde `dashboard.h`, header `Content-Encoding: gzip`)
    - `WS /ws` → WebSocket bidireccional:
      - ESP32 → Cliente: JSON con detecciones + métricas cada frame
      - Cliente → ESP32: comandos (cambiar modelo, cambiar runtime, ajustar thresholds)
    - Serialización JSON con `cJSON` (built-in ESP-IDF), buffer preasignado
    - Límite: 3 conexiones WebSocket simultáneas

18. **Crear** `frontend/dashboard.html` → `main/dashboard.h`:
    - HTML/CSS/JS mínimo (~5-10 KB pre-gzip, ~2-3 KB gzipped)
    - Panel de métricas en tiempo real (FPS, latencia desglosada, memoria)
    - Lista de detecciones activas (clase, confianza, bbox)
    - Selector de modelo (dropdown: MBNTv2, YOLO11n, YOLO26n)
    - Selector de runtime (TFLite / ESP-DL)
    - Indicadores de estado (conexión WS, modelo cargado, temperatura)
    - Script de build que convierte el HTML a C header gzipped

19. **Implementar** `main.cpp`:
    - `app_main()`: secuencia de inicialización según pseudocódigo sección 5.4 del notebook
    - Task de inferencia en Core 0 (prioridad alta)
    - WiFi/HTTP en Core 1 (prioridad media)
    - Metrics logger en Core 1 (prioridad baja)
    - Selección de modelo/runtime configurable vía NVS o WebSocket

#### Fase 4 — Benchmarking Comparativo (Prioridad: Alta)

20. **Ejecutar benchmark**: 1000 frames por modelo × 2 runtimes = 6 configuraciones
    - Métricas a recopilar: FPS, latencia (pre/inf/post), memoria (SRAM/PSRAM), estabilidad (sin OOM), temperatura
    - Exportar métricas vía WebSocket → capturar en PC con script Python

21. **Criterios de aceptación** (del notebook sección 6.2):
    - Inferencia end-to-end < 2000 ms (ideal < 1000 ms)
    - FPS > 0.5 (ideal > 1.0)
    - PSRAM libre post-init > 500 KB (ideal > 1 MB)
    - Estabilidad: 1000 frames sin crash (ideal 10000)
    - WebSocket funcional con datos JSON legibles

22. **Generar informe comparativo**: TFLite Micro vs ESP-DL en hardware real, por modelo

---

### Parte D: Mitigación de Riesgos (de sección 6.2 del notebook)

| Riesgo | Fase | Mitigación implementada |
|---|---|---|
| **R1**: PSRAM insuficiente para YOLO | Fase 2 | Single buffer de cámara como fallback, `mmap` de flash para modelo, ajustar arena empíricamente con `arena_used_bytes()`, ESP-DL tiene planificador de memoria estático que optimiza automáticamente |
| **R2**: Operadores TFLite no soportados | Fase 2 | `MicroMutableOpResolver` con ops uno a uno, verificación previa en notebook de conversión (Sección 4), ESP-DL tiene 60+ ops ONNX-aligned como alternativa |
| **R3**: WiFi interfiere con inferencia | Fase 3 | Inferencia en Core 0, WiFi en Core 1, `ESP_WIFI_IRAM_OPT` habilitado |
| **R4**: NMS lento en YOLO11n | Fase 2 | Filtro agresivo (score ≥ 0.4), top-100 candidatos, NMS con early exit |
| **R5**: Desalineación preprocesamiento | Fase 2 | Validación cruzada Python vs C con imágenes de calibración (paso 14) |
| **R6**: Sobrecalentamiento | Fase 3 | Monitoreo térmico en `metrics.cpp`, delay configurable entre frames |

---

### Verificación

- **Notebook de conversión**: Ejecutar todas las celdas, verificar que se generan los 3 `.h` (TFLite path) y los 3 `.espdl` (ESP-DL path) sin errores
- **Compilación**: `idf.py build` exitoso con ambos componentes (esp-tflite-micro + esp-dl)
- **Flash + boot**: `idf.py flash monitor` — verificar init de PSRAM, cámara y WiFi en logs
- **Inferencia**: Verificar detecciones con imagen estática (comparar con salidas Python del notebook)
- **Benchmark**: 1000 frames sin OOM, métricas dentro de umbrales de aceptación
- **Dashboard**: Conectar desde navegador a `192.168.4.1`, verificar métricas en tiempo real y cambio de modelo/runtime

### Decisiones

- **3 modelos, 1 cargado a la vez**: Se compilan los 3 en flash pero solo se instancia un modelo + runtime en cada sesión (usuario selecciona vía WebSocket o NVS)
- **Dual runtime ESP-DL + TFLite Micro**: Ambos como componentes ESP-IDF, interfaz abstracta `inference_engine.h` para intercambiabilidad, permite benchmark comparativo real
- **Dashboard embebido en ESP32**: HTML mínimo gzipped (~2-3 KB), sistema standalone sin dependencias externas
- **C++ (GNU++23)**: Alineado con tu configuración probada y permite abstracciones de interfaz limpias
- **Modelos `.espdl` como binary embed**: Montados con `EMBED_FILES` en CMake (zero-copy load), los `.tflite` como arrays C con `xxd -i`

---

### Orden de implementación sugerido

El orden **no es estrictamente secuencial** — Fase 1 y la Parte A (notebook) pueden ejecutarse en paralelo. La Fase 2 depende de tener los modelos convertidos (Parte A). Las Fases 3 y 4 son secuenciales respecto a la 2.

```
Semana 1:  Parte A (notebook conversión) + Fase 1 (infra base)
Semana 2:  Fase 2 (motores de inferencia + postproceso)
Semana 3:  Fase 3 (métricas + WiFi + dashboard)
Semana 4:  Fase 4 (benchmark + informe)
```

¿Quieres ajustar algo del plan antes de proceder a la implementación?