# TFM TinyML Detector — ESP32-S3 Firmware

Firmware de detección de objetos en tiempo real para ESP32-S3, desplegando 3 modelos TinyML entrenados sobre un dataset personalizado de 5 clases (**dog, door, obstacle, person, stair**).

## Hardware

| Componente | Especificación |
|---|---|
| **Placa** | Freenove ESP32-S3 CAM Board |
| **SoC** | ESP32-S3 WROOM N16R8 |
| **Flash** | 16 MB (DIO) |
| **PSRAM** | 8 MB Octal (80 MHz) |
| **CPU** | Dual-core Xtensa LX7 @ 240 MHz |
| **Cámara** | OV5640 (QVGA 320×240 RGB565) |

## Modelos

| Modelo | Tipo | Input dtype | Tamaño TFLite | Output |
|---|---|---|---|---|
| MBNTv2_ssdlite_v1 | SSD anchor-based | INT8 | 1.20 MB | 3 tensores: obj(1470,1) + cls(1470,5) + box(1470,4) |
| YOLO11n_v1 | YOLO11 nano | float32 | 2.68 MB | 1 tensor: (9, 1029) transposed |
| YOLO26n_v1 | YOLO26 nano e2e | float32 | 2.55 MB | 1 tensor: (300, 6) NMS-free |

## Arquitectura del Firmware

```
app_main()
  ├── metrics_init()
  ├── camera_init()         → OV5640 QVGA RGB565
  ├── image_proc_init()     → buffers PSRAM (INT8 + float)
  ├── postprocess_init()
  ├── engine->init()        → TFLite Micro o ESP-DL
  ├── wifi_init_ap()        → AP "ESP32_TFM" / "tfm2026esp"
  ├── webserver_start()     → HTTP :80 + WebSocket /ws
  └── inference_task (Core 0)
        capture → preprocess → invoke → postprocess → metrics → WS broadcast
```

### Pipeline por frame

1. **Captura** — `camera_capture()` → frame RGB565 320×240
2. **Preproceso** — Crop central 224×224 (offset 48,8) → RGB888 → normalización
   - INT8: `pixel - 128` → [-128, 127]
   - Float: `pixel / 255.0` → [0.0, 1.0]
3. **Inferencia** — `engine->invoke()` (TFLite Micro o ESP-DL)
4. **Postproceso** — Decode + NMS (específico por modelo)
5. **Métricas** — Tiempos (EMA α=0.065), heap, temperatura
6. **Broadcast** — JSON via WebSocket a dashboard

### Dual Runtime

- **TFLite Micro** (`tflite_engine.cpp`) — 26 operadores registrados, arena en PSRAM
- **ESP-DL** (`espdl_engine.cpp`) — Condicional `#if HAS_ESP_DL`, requiere modelos `.espdl`

## Estructura de archivos

```
03_ING_DESPLIEGUE/
├── CMakeLists.txt              # Proyecto ESP-IDF raíz
├── partitions.csv              # NVS + Factory ~15MB
├── sdkconfig.defaults          # PSRAM, Flash, WiFi, C++23
├── Conversion_ModelosTFLite.ipynb  # Notebook de conversión
├── convert_onnx_to_espdl.py    # Script standalone ONNX→ESPDL
├── main/
│   ├── CMakeLists.txt          # Sources, EMBED_FILES, C++23 flags
│   ├── idf_component.yml       # Dependencies: camera, tflite, esp-dl
│   ├── app_config.h            # Pines, resoluciones, tipos, clases
│   ├── camera.h/cpp            # OV5640 init/capture/release
│   ├── image_proc.h/cpp        # RGB565→RGB888, crop, normalización
│   ├── inference_engine.h      # Interfaz abstracta
│   ├── tflite_engine.h/cpp     # TFLite Micro implementation
│   ├── espdl_engine.h/cpp      # ESP-DL implementation (skeleton)
│   ├── postprocess.h/cpp       # MBNTv2 decode+NMS, YOLO11 transpose+NMS, YOLO26 filter
│   ├── metrics.h/cpp           # Timers, heap stats, EMA, temperatura
│   ├── wifi_manager.h/cpp      # WiFi AP mode
│   ├── web_server.h/cpp        # HTTP GET + WebSocket broadcast
│   ├── dashboard.h             # Gzip'd HTML blob (auto-generated)
│   ├── main.cpp                # app_main + inference task
│   ├── frontend/
│   │   └── dashboard.html      # Dashboard source (dark mode, WS)
│   └── models/
│       └── tflite/
│           ├── mobilenetv2_ssdlite_v1_int8.h   # 1.20 MB
│           ├── yolo11n_v1_int8.h                # 2.68 MB
│           └── yolo26n_v1_int8.h                # 2.55 MB
```

## Compilación

```bash
# Requisitos: ESP-IDF v5.4.3 instalado
cd 03_ING_DESPLIEGUE/

# Seleccionar target
idf.py set-target esp32s3

# Configurar (opcional — sdkconfig.defaults aplica automáticamente)
idf.py menuconfig

# Compilar
idf.py build

# Flash + monitor
idf.py -p /dev/ttyUSB0 flash monitor
```

### Selección de modelo (compile-time)

Editar en `main.cpp`:
```cpp
#define ACTIVE_MODEL_TYPE   ModelType::YOLO11N      // o MOBILENET_SSD, YOLO26N
#define ACTIVE_ENGINE_TYPE  EngineType::TFLITE_MICRO // o ESP_DL
```

## Dashboard

Acceder via WiFi:
1. Conectar a **ESP32_TFM** (password: `tfm2026esp`)
2. Abrir `http://192.168.4.1/`

El dashboard muestra en tiempo real:
- FPS y latencia (preproceso / inferencia / postproceso)
- Uso de memoria (heap interno, PSRAM, arena TFLite)
- Temperatura del SoC
- Lista de detecciones con clase y confianza

## Notas

- Solo 1 modelo cargado a la vez (limitación de PSRAM)
- Los modelos YOLO requieren input float32; MBNTv2 usa INT8
- YOLO26n tiene NMS integrado (end-to-end) — no requiere NMS on-device
- Los archivos `.espdl` deben generarse con `convert_onnx_to_espdl.py` en un entorno con PyTorch
- Arena TFLite se aloja en PSRAM con fallback progresivo (3× → 2.25× → 1.69× tamaño modelo)
