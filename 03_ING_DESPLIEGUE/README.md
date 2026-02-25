# TFM TinyML Detector — Ingeniería de Despliegue

> **Trabajo Fin de Máster** — Detección de objetos en tiempo real sobre ESP32-S3 para asistencia a personas con movilidad reducida (LCMR).

[![ESP-IDF](https://img.shields.io/badge/ESP--IDF-v5.4.3-blue?logo=espressif)](https://docs.espressif.com/projects/esp-idf/)
[![Target](https://img.shields.io/badge/MCU-ESP32--S3-orange?logo=espressif)](https://www.espressif.com/en/products/socs/esp32-s3)
[![License](https://img.shields.io/badge/license-Academic-lightgrey)]()
[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus)](https://en.cppreference.com/w/cpp/23)

---

## Tabla de Contenidos

1. [Descripción General](#1-descripción-general)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Hardware Requerido](#3-hardware-requerido)
4. [Modelos Desplegados](#4-modelos-desplegados)
5. [Estructura del Proyecto](#5-estructura-del-proyecto)
6. [Pipeline de Despliegue](#6-pipeline-de-despliegue)
7. [Configuración del Entorno](#7-configuración-del-entorno)
8. [Compilación y Flasheo](#8-compilación-y-flasheo)
9. [Dashboard Web y Protocolo WebSocket](#9-dashboard-web-y-protocolo-websocket)
10. [Ciclos de Despliegue Iterativo](#10-ciclos-de-despliegue-iterativo)
11. [Resultados On-Device](#11-resultados-on-device)
12. [Tabla de Particiones Flash](#12-tabla-de-particiones-flash)
13. [Scripts de Utilidad](#13-scripts-de-utilidad)
14. [Lecciones Aprendidas](#14-lecciones-aprendidas)
15. [Referencias](#15-referencias)

---

## 1. Descripción General

Este módulo (`03_ING_DESPLIEGUE`) implementa el **despliegue de modelos de detección de objetos TinyML** en un microcontrolador ESP32-S3, como parte de un sistema de asistencia visual para personas con movilidad reducida. El firmware captura imágenes de una cámara OV5640, ejecuta inferencia INT8 a bordo y transmite resultados en tiempo real a un dashboard web mediante WebSocket y streaming MJPEG.

**5 clases detectadas:** `dog` · `door` · `obstacle` · `person` · `stair`

### Características principales

| Característica | Detalle |
|---|---|
| **Dual-engine runtime** | Soporte simultáneo para TFLite Micro y ESP-DL v3.x |
| **Hot-swap de modelos** | Cambio de modelo en caliente desde la interfaz web, con rollback automático |
| **Dashboard embebido** | HTML/CSS/JS gzip'd servido desde flash, con MJPEG stream y bounding boxes en canvas |
| **Modos de inferencia** | Continuo (cada frame) y bajo demanda (trigger manual) |
| **Métricas en tiempo real** | Latencia por fase (pre/inf/post), FPS, temperatura CPU, uso de memoria |
| **Umbrales dinámicos** | Ajuste de confianza e IoU desde sliders en el dashboard |
| **Arquitectura dual-core** | Core 0 → inferencia, Core 1 → WiFi + HTTP/WS |

---

## 2. Arquitectura del Sistema

```
┌─────────────────────────── ESP32-S3 ───────────────────────────┐
│                                                                 │
│  Core 0 (Inference Task)              Core 1 (WiFi + HTTP/WS)  │
│  ┌──────────────────────┐             ┌──────────────────────┐  │
│  │  Camera Capture      │             │  WiFi STA Manager    │  │
│  │  (OV5640 RGB565)     │             │  HTTP Server :80     │  │
│  │         ↓            │             │  ├─ GET /  (dashboard)│  │
│  │  Center Crop         │             │  ├─ WS  /ws (metrics)│  │
│  │  320×240 → 224×224   │             │  └─ GET /stream      │  │
│  │         ↓            │             │      (MJPEG :81)     │  │
│  │  Preprocessing       │             └──────────┬───────────┘  │
│  │  RGB565→INT8 norm    │                        │              │
│  │         ↓            │                        │              │
│  │  ┌──────────────┐   │    WebSocket JSON       │              │
│  │  │ ESP-DL Engine │◄──┼────────────────────────┘              │
│  │  │ or            │   │    (model switch,                     │
│  │  │ TFLite Engine │   │     threshold, mode)                  │
│  │  └──────┬───────┘   │                                       │
│  │         ↓            │                                       │
│  │  Postprocessing      │    stream_buf                         │
│  │  (FCOS/DFL/NMS)      ├──────────────►  MJPEG Stream         │
│  │         ↓            │                                       │
│  │  Metrics EMA         ├──────────────►  WS Broadcast          │
│  └──────────────────────┘                 (JSON metrics+dets)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                          ▲
                    ┌─────┴─────┐
                    │  Browser  │
                    │ Dashboard │
                    └───────────┘
```

### Flujo de datos por frame

```
OV5640 → framebuffer (PSRAM, RGB565 320×240)
       → crop center 224×224
       → JPEG encode (stream)
       → preprocess INT8 (TFLite: pixel-128 | ESPDL: round(pixel/255×128))
       → engine.invoke() → output tensors
       → postprocess (decode + sigmoid + NMS)
       → metrics update (EMA α=0.065)
       → WS broadcast JSON + stream_buf publish JPEG
```

---

## 3. Hardware Requerido

| Componente | Especificación |
|---|---|
| **Placa de desarrollo** | Freenove ESP32-S3 WROOM CAM Board |
| **SoC** | ESP32-S3 WROOM-1 N16R8 (dual-core Xtensa LX7 @ 240 MHz) |
| **Flash** | 16 MB (Quad SPI, modo DIO) |
| **PSRAM** | 8 MB (Octal SPI @ 80 MHz) |
| **Cámara** | OV5640 (5 MP), modo QVGA 320×240 RGB565, double buffering |
| **Conectividad** | WiFi 802.11 b/g/n (modo Station) |

### Asignación de pines de cámara (OV5640)

| Señal | GPIO | Señal | GPIO |
|---|---|---|---|
| XCLK | 15 | VSYNC | 6 |
| SDA (SIOD) | 4 | HREF | 7 |
| SCL (SIOC) | 5 | PCLK | 13 |
| D7–D0 | 16, 17, 18, 12, 10, 8, 9, 11 | PWDN | -1 (N/C) |

---

## 4. Modelos Desplegados

### Producción (Ciclo 3)

| Modelo | Runtime | Parámetros | Tamaño | Entrada | Salidas | Latencia |
|---|---|---|---|---|---|---|
| **ESPDet Pico T4** | ESP-DL v3.x | 361 K | 545 KB | INT8 [1,224,224,3] exp=-7 | 6 tensores (score0-2, box0-2) FCOS 3-scale | ~405 ms |
| **YOLO26n T3 ESP** | ESP-DL v3.x | 2.5 M | 2.57 MB | INT8 [1,224,224,3] exp=-7 | 6 tensores (score0-2, box0-2) direct dist | ~2885 ms |

### Legacy (Ciclos 1–2, TFLite Micro)

| Modelo | Runtime | Formato | Notas |
|---|---|---|---|
| MobileNetV2 SSD-Lite | TFLite Micro | INT8 `.tflite` embebido | Primer modelo con detecciones exitosas (Ciclo 1) |
| YOLO11n v1 | TFLite Micro | Full INT8 `.tflite` embebido | 0 detecciones on-device por discrepancia PPQ vs runtime |
| YOLO26n v1 | TFLite Micro | Full INT8 `.tflite` embebido | End-to-end (NMS integrado, [1,300,6]) |

### Arquitectura de post-procesamiento por modelo

```
ESPDet Pico (FCOS)              YOLO26n T2 ESP (DFL)           YOLO26n T3 ESP (Direct)
score[s]: [1,H,W,5] INT8       score[s]: [1,H,W,5] INT8       score[s]: [1,H,W,5] INT8
box[s]:   [1,H,W,4] INT8       box[s]:   [1,H,W,64] INT8      box[s]:   [1,H,W,4] INT8
     ↓                              ↓                               ↓
dequant(2^exp)                  dequant(2^exp)                  dequant(2^exp)
     ↓                              ↓                               ↓
sigmoid(score)                  sigmoid(score)                  sigmoid(score)
     ↓                              ↓                               ↓
direct l,t,r,b → bbox          4×16 bins softmax → dist        direct l,t,r,b → bbox
     ↓                              ↓                               ↓
per-class NMS                   per-class NMS                   per-class NMS
```

Escalas: `s ∈ {0,1,2}` con strides `{8,16,32}` y grids `{28×28, 14×14, 7×7}` sobre input 224×224.

---

## 5. Estructura del Proyecto

```
03_ING_DESPLIEGUE/
├── CMakeLists.txt              # Top-level ESP-IDF build (proyecto activo)
├── sdkconfig.defaults          # Configuración base ESP-IDF (Ciclo 3)
├── partitions.csv              # Tabla de particiones flash (Ciclo 3)
├── requirements.txt            # Dependencias Python (esp-ppq, torch, tf, onnx...)
│
├── main/                       # ── Componente principal del firmware ──
│   ├── app_config.h            #   Configuración global (pines, resoluciones, umbrales)
│   ├── main.cpp                #   Entry point: orchestrator (init → inference loop)
│   ├── camera.cpp/.h           #   Driver OV5640 (RGB565, QVGA, double buffer)
│   ├── image_proc.cpp/.h       #   Preprocesamiento: RGB565→INT8 (TFLite/ESPDL)
│   ├── inference_engine.h      #   Interfaz abstracta (virtual) para motores
│   ├── tflite_engine.cpp/.h    #   Implementación TFLite Micro (26 ops registrados)
│   ├── espdl_engine.cpp/.h     #   Implementación ESP-DL v3.x (mmap desde partición)
│   ├── postprocess.cpp/.h      #   Decodificación: MBNTv2/YOLO11/YOLO26/FCOS/DFL
│   ├── metrics.cpp/.h          #   Métricas EMA + sensor de temperatura
│   ├── wifi_manager.cpp/.h     #   WiFi STA (conexión a red doméstica)
│   ├── web_server.cpp/.h       #   HTTP + WebSocket + MJPEG en puertos 80/81
│   ├── stream_buf.cpp/.h       #   Buffer thread-safe JPEG (producer-consumer)
│   ├── dashboard.h             #   HTML gzip'd auto-generado
│   ├── CMakeLists.txt          #   Registro de componente (C++23, REQUIRES)
│   ├── idf_component.yml       #   Dependencias: esp32-camera, esp-tflite-micro, esp-dl
│   ├── frontend/
│   │   └── dashboard.html      #   Dashboard interactivo (MJPEG + canvas bbox + WS)
│   └── models/tflite/          #   Modelos TFLite embebidos como arrays C (legacy)
│
├── models/                     # ── Modelos entrenados ──
│   ├── espdl/                  #   Modelos ESPDL producción (.espdl + .info + .json)
│   ├── cycle1_archive/         #   Variantes experimentales Ciclo 1
│   └── cycle2_archive/         #   Modelos Ciclo 2 + scripts de pipeline
│
├── scripts/                    # ── Scripts de soporte ──
│   ├── convert_onnx_to_espdl.py    # Conversión ONNX→ESPDL via esp-ppq
│   ├── flash_models.sh             # Flasheo independiente de modelos
│   ├── gen_dashboard_header.py     # HTML→gzip→C header
│   ├── validate_partitions.py      # Validación de tabla de particiones
│   └── ...                         # inspect_tflite, list_ops, etc.
│
├── deployment_cycles/          # ── Documentación por ciclo iterativo ──
│   ├── README_cycle1.md        #   Ciclo 1: Despliegue base (3 modelos TFLite/ESPDL)
│   ├── README_cycle2.md        #   Ciclo 2: Diagnóstico cuantización YOLO
│   └── README_cycle3.md        #   Ciclo 3: Integración final ESPDL
│
├── docs/                       # ── Instructivos técnicos ──
│   ├── Configuracion_ESP32-S3.md
│   ├── Instructivo_Despliegue_ESPDL.md
│   └── Instructivo_Despliegue_YOLO26_T3.md
│
├── firmware/                   # ── Firmware legacy (Ciclo 1, archivado) ──
│
├── outputs/                    # ── Capturas de inferencia on-device ──
│   ├── Inference_Embebed_1.txt #   Frame JPEG capturado desde el dashboard
│   ├── Inference_Embebed_2.txt
│   └── Inference_Embebed_3.txt
│
└── components/                 # ── Componentes ESP-IDF locales ──
    └── espressif__esp-tflite-micro/
```

---

## 6. Pipeline de Despliegue

### 6.1 Flujo completo: Entrenamiento → On-device

```
Entrenamiento (PyTorch/Keras)
     │
     ▼
Exportación ONNX (opset 13)
     │
     ├── onnxsim (simplificación de grafo)
     │
     ▼
Cuantización INT8 (esp-ppq)
     │
     ├── Calibración: 64 imágenes del dataset (NHWC para ESPDL)
     ├── Esquema: Power-of-2 (exponent-based, no zero-point)
     │
     ▼
.espdl (FlatBuffers, modelo cuantizado)
     │
     ├── Validación offline: eval_quantized.py (mAP50 gate: Δ < 15%)
     │
     ▼
Flash a partición dedicada
     │  esptool.py write_flash 0xA10000 espdet_pico_t4.espdl
     │  esptool.py write_flash 0xB10000 yolo26n_t3_esp.espdl
     │
     ▼
ESP-DL runtime: dl::Model(partition_label, IN_FLASH_PARTITION)
     │
     ├── mmap automático desde flash
     ├── param_copy=true → copia a PSRAM para rendimiento
     ├── MEMORY_MANAGER_GREEDY
     │
     ▼
Inferencia on-device (INT8, dual-core scheduling)
```

### 6.2 Normalización de entrada

La normalización correcta fue una de las **lecciones críticas** del proyecto:

| Runtime | Fórmula | Rango | Parámetro |
|---|---|---|---|
| **TFLite Micro** | `int8 = pixel_uint8 - 128` | [-128, 127] | zero_point=128 |
| **ESP-DL** | `int8 = round(pixel/255 × 128)` | [0, 127] | exponent=-7 |

> ⚠️ **Mezclar estas normalizaciones** fue la causa raíz de 0 detecciones en los Ciclos 1–2 para modelos ESPDL.

### 6.3 Dequantización de salidas (ESP-DL)

```
float_value = int8_value × 2^exponent
```

Cada tensor de salida tiene su propio exponent (no uniforme):

| Tensor | Ejemplo exponent |
|---|---|
| `score0` | -3 |
| `score1` | -2 |
| `box0` | -3 |
| `box2` | -4 |

---

## 7. Configuración del Entorno

### Requisitos previos

- **ESP-IDF** v5.3.0+ (recomendado v5.4.3)
- **Python** 3.10+ con entorno virtual
- **esptool.py** (incluido en ESP-IDF)

### Instalación

```bash
# 1. Clonar ESP-IDF
git clone -b v5.4.3 --recursive https://github.com/espressif/esp-idf.git ~/esp/v5.4.3/esp-idf
cd ~/esp/v5.4.3/esp-idf && ./install.sh esp32s3

# 2. Activar entorno
source ~/esp/v5.4.3/esp-idf/export.sh

# 3. Instalar dependencias Python (para scripts de conversión)
pip install -r requirements.txt

# 4. Instalar componentes ESP-IDF gestionados
cd /ruta/a/03_ING_DESPLIEGUE
idf.py reconfigure   # descarga esp32-camera, esp-tflite-micro, esp-dl
```

### Configuración sdkconfig destacada

```ini
# CPU
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y

# Flash 16 MB
CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y
CONFIG_ESPTOOLPY_FLASHMODE_DIO=y

# PSRAM Octal 8 MB @ 80 MHz
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_SPEED_80M=y

# C++ moderno
CONFIG_COMPILER_CXX_EXCEPTIONS=y
CONFIG_COMPILER_CXX_RTTI=y

# WebSocket
CONFIG_HTTPD_WS_SUPPORT=y

# Task WDT generoso (inferencia ~3s para YOLO26)
CONFIG_ESP_TASK_WDT_TIMEOUT_S=30
```

---

## 8. Compilación y Flasheo

### 8.1 Compilar firmware

```bash
source ~/esp/v5.4.3/esp-idf/export.sh
cd /ruta/a/03_ING_DESPLIEGUE

idf.py set-target esp32s3
idf.py build
```

### 8.2 Flashear firmware

```bash
idf.py -p /dev/tty.usbmodem* flash monitor
```

### 8.3 Flashear modelos (independiente del firmware)

Los modelos ESPDL se almacenan en particiones flash dedicadas y pueden actualizarse **sin recompilar** el firmware:

```bash
# Flashear ambos modelos
./scripts/flash_models.sh

# O manualmente:
esptool.py --port /dev/tty.usbmodem* write_flash 0xA10000 models/espdl/espdet_pico_t4.espdl
esptool.py --port /dev/tty.usbmodem* write_flash 0xB10000 models/espdl/yolo26n_t3_esp.espdl
```

### 8.4 Regenerar dashboard embebido

Tras modificar `main/frontend/dashboard.html`:

```bash
python scripts/gen_dashboard_header.py
idf.py build
```

---

## 9. Dashboard Web y Protocolo WebSocket

Una vez iniciado el firmware, el dashboard está disponible en la IP asignada por el router (visible en los logs del monitor serial).

### Endpoints HTTP

| Ruta | Puerto | Descripción |
|---|---|---|
| `GET /` | 80 | Dashboard HTML (gzip, ~5.4 KB) |
| `WS /ws` | 80 | WebSocket bidireccional (métricas + comandos) |
| `GET /stream` | 81 | MJPEG live stream (224×224 crop) |

### Protocolo WebSocket

#### Mensajes servidor → cliente (JSON)

```json
{
  "type": "metrics",
  "frame_id": 1234,
  "model": "ESPDet_Pico_T4",
  "model_idx": 0,
  "fps": 1.8,
  "total_ms": 550.2,
  "preprocess_ms": 12.3,
  "inference_ms": 405.6,
  "postprocess_ms": 2.1,
  "ema_fps": 1.7,
  "heap_kb": 120,
  "psram_kb": 3200,
  "arena_kb": 1850,
  "temp_c": 48.5,
  "detections": [
    {"class": "person", "conf": 0.87, "x1": 0.12, "y1": 0.15, "x2": 0.65, "y2": 0.95}
  ]
}
```

#### Comandos cliente → servidor (JSON)

| Comando | Ejemplo | Descripción |
|---|---|---|
| Cambio de modo | `{"cmd":"mode","value":"continuous"}` | `continuous` o `ondemand` |
| Trigger manual | `{"cmd":"infer"}` | Ejecutar una inferencia (modo on-demand) |
| Cambio de modelo | `{"cmd":"model","index":1,"req_id":42}` | Hot-swap con correlación |
| Umbrales | `{"cmd":"threshold","conf":0.5,"iou":0.4}` | Ajuste dinámico [0.05, 0.95] |

### Funcionalidades del dashboard

- **Video MJPEG** en tiempo real con overlay de bounding boxes en canvas
- **Selector de modelo** con indicador de estado (switching/ok/error)
- **Toggle continuo/on-demand** con botón de captura
- **Panel de frame capturado** (calidad JPEG alta, 80%)
- **Métricas en vivo**: FPS, latencias por fase, temperatura CPU, uso de heap/PSRAM
- **Sliders de confianza e IoU** con feedback inmediato
- **Detections tags** coloreados por clase (dog=violeta, door=azul, obstacle=ámbar, person=verde, stair=rosa)

---

## 10. Ciclos de Despliegue Iterativo

El desarrollo se articuló en **3 ciclos iterativos** documentados en `deployment_cycles/`:

### Ciclo 1 — Despliegue Base (ene–feb 2026)

**Objetivo:** Desplegar 3 modelos de detección en ESP32-S3.

| Hito | Resultado |
|---|---|
| Firmware modular: 8 componentes C++ | ✅ Completado |
| Pipeline ONNX→ESPDL con esp-ppq | ✅ Funcional |
| MobileNetV2 SSD-Lite on-device | ✅ **4 dets/frame, 846 ms, 1.1 FPS** |
| YOLO11n on-device | ❌ 0 detecciones (scores negativos) |
| YOLO26n on-device | ⏸️ No probado por prioridad YOLO11n |

**Lección clave:** Optimización de latencia de 3,514 ms → 896 ms (−74%) mediante PSRAM, `param_copy`, cache tuning.

### Ciclo 2 — Diagnóstico de Cuantización (feb 2026)

**Objetivo:** Investigar causa raíz de 0 detecciones YOLO11n.

| Investigación | Hallazgo |
|---|---|
| Equalization en PPQ | 0 pares elegibles (expected para YOLO) |
| `model->minimize()` | Sin cambio en detecciones |
| `eval_quantized.py` (mAP) | Float=0.7925, INT8=0.7418 → **Δ=6.4% → PASS** |

**Conclusión:** "La cuantización INT8 NO es la causa. El problema está en el runtime/firmware (normalización o output parsing)."

### Ciclo 3 — Integración Final ESPDL (feb 2026)

**Objetivo:** Corregir causas raíz e integrar modelos optimizados.

| Corrección | Detalle |
|---|---|
| Normalización ESPDL | `pixel-128` → `round(pixel/255×128)` (exponent=-7) |
| `espdl_engine.cpp` | Reescritura completa con API esp-dl v3.x |
| Output access | Por índice → **por nombre** (`get_output("score0")`) |
| MJPEG streaming | stream_buf producer-consumer + puerto 81 separado |
| WiFi AP → STA | Acceso desde red doméstica |
| Hot-swap modelos | vía IPC inter-core con rollback automático |

**Resultado final:**

| Modelo | Inferencia | FPS | Detecciones | Viabilidad |
|---|---|---|---|---|
| ESPDet Pico T4 | ~405 ms | 1.5–1.8 | 1–3/frame | ✅ **Viable para LCMR** |
| YOLO26n T2 ESP | ~2,885 ms | 0.3 | 0–1/frame | ❌ Demasiado lento |

---

## 11. Resultados On-Device

### ESPDet Pico T4 (modelo principal)

```
[#1234] ESPDet_Pico_T4 | 550.2 ms (pre:12.3 inf:405.6 post:2.1) | 1.8 FPS | 3 dets | heap:120K psram:3200K
```

| Métrica | Valor |
|---|---|
| Tiempo total | ~550 ms/frame |
| Preprocesamiento | ~12 ms |
| Inferencia ESP-DL | ~405 ms |
| Post-procesamiento | ~2 ms |
| FPS efectivos | 1.5–1.8 |
| RAM del modelo (PSRAM) | ~1.8 MB |
| Temperatura CPU | ~48 °C |

### Distribución de memoria (ESP32-S3 N16R8)

| Recurso | Total | Usado | Libre |
|---|---|---|---|
| Heap interno (SRAM) | ~512 KB | ~392 KB | ~120 KB |
| PSRAM | 8 MB | ~4.8 MB | ~3.2 MB |
| Flash | 16 MB | ~14 MB | ~2 MB |

---

## 12. Tabla de Particiones Flash

```
# Name            Type    SubType   Offset      Size       Descripción
nvs               data    nvs       0x9000      24 KB      Non-Volatile Storage
phy_init          data    phy       0xF000       4 KB      WiFi PHY calibration
factory           app     factory   0x10000     10 MB      Firmware ESP-IDF
model_espdet      data    0x40      0xA10000     1 MB      ESPDet Pico T4 (545 KB)
model_yolo26      data    0x40      0xB10000     3 MB      YOLO26n T3 ESP (2.57 MB)
─────────────────────────────────────────────────────────────────────
                                    TOTAL       ~14 MB     Margen libre: ~2 MB
```

Los modelos se cargan mediante **mmap** desde sus particiones dedicadas, sin consumir heap adicional para almacenamiento (solo para parámetros si `param_copy=true`).

---

## 13. Scripts de Utilidad

| Script | Descripción | Uso |
|---|---|---|
| `scripts/convert_onnx_to_espdl.py` | Conversión ONNX → ESPDL con esp-ppq (calibración INT8) | `python scripts/convert_onnx_to_espdl.py --model yolo26n` |
| `scripts/flash_models.sh` | Flasheo de modelos `.espdl` a particiones dedicadas | `./scripts/flash_models.sh` |
| `scripts/gen_dashboard_header.py` | Genera `dashboard.h` (HTML → gzip → C array) | `python scripts/gen_dashboard_header.py` |
| `scripts/validate_partitions.py` | Valida coherencia de `partitions.csv` | `python scripts/validate_partitions.py` |
| `scripts/inspect_tflite.py` | Inspección de modelos TFLite (ops, tensores, shapes) | `python scripts/inspect_tflite.py model.tflite` |
| `scripts/list_ops.py` | Lista operadores TFLite requeridos | `python scripts/list_ops.py model.tflite` |

---

## 14. Lecciones Aprendidas

### Errores críticos resueltos

1. **Normalización divergente TFLite vs ESP-DL**
   - TFLite usa `zero_point=128` → `int8 = pixel - 128`
   - ESP-DL usa `exponent=-7` → `int8 = round(pixel / 255 × 128)`
   - Mezclarlas produce scores completamente inválidos.

2. **Orden de tensores de salida no determinista**
   - `get_output(index)` retorna en orden de inserción del map interno, que puede variar entre compilaciones.
   - **Solución:** acceso por nombre (`get_output_by_name("score0")`).

3. **Exponents no uniformes entre tensores**
   - Cada tensor (`score0`, `box0`, etc.) puede tener un exponent diferente.
   - Ignorar esto produce coordenadas o scores incorrectos.

4. **Stack overflow en tarea de inferencia**
   - ESP-DL + WiFi fragmentan el heap interno.
   - **Solución:** `xTaskCreatePinnedToCoreWithCaps(..., MALLOC_CAP_SPIRAM)` para stack en PSRAM.

5. **Mismatch de tamaño en flash multi-modelo**
   - Dos modelos con offsets fijos dentro de una sola partición causaron solapamiento de 768 bytes.
   - **Solución:** una partición dedicada por modelo en Ciclo 3.

### Principios de diseño

- **Abstracción de runtime:** La interfaz `InferenceEngine` permite intercambiar TFLite Micro ↔ ESP-DL sin modificar el main loop.
- **No-regression:** Cada ciclo añade funcionalidad sin modificar código existente validado.
- **Desacoplamiento firmware/modelos:** Los modelos se flashean de forma independiente, permitiendo iteración rápida sin recompilar.
- **Observabilidad:** Métricas EMA + logs estructurados + dashboard en vivo facilitan el diagnóstico on-device.

---

## 15. Referencias

| Recurso | URL / Referencia |
|---|---|
| ESP-IDF Programming Guide | https://docs.espressif.com/projects/esp-idf/en/v5.4.3/ |
| ESP-DL v3.x Documentation | https://github.com/espressif/esp-dl |
| esp-ppq (Quantization Toolkit) | https://github.com/espressif/esp-ppq |
| TFLite Micro for ESP | https://github.com/espressif/esp-tflite-micro |
| ESP32-Camera Driver | https://github.com/espressif/esp32-camera |
| Freenove ESP32-S3 CAM | https://github.com/Freenove/Freenove_ESP32_S3_WROOM_Board |
| YOLO26n (Ultralytics) | Custom architecture, see `02_ING_MODELO` |
| ESPDet (Espressif) | Part of esp-dl model zoo |

---

<p align="center">
  <em>Desarrollado como parte del Trabajo Fin de Máster — UNIR, 2026</em>
</p>
