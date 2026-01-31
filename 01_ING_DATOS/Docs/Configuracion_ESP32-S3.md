# Guía de configuración completa que funcionó en este proyecto

## 🔧 Configuraciones ESP-IDF (menuconfig)

### PSRAM (CRÍTICO)
```
Component config → ESP PSRAM
├─ Support for external PSRAM: [✓] Enabled
├─ PSRAM Speed: Octal 80MHz
├─ PSRAM Mode: Octal
├─ SPI RAM access method: [✓] Make RAM allocatable using heap_caps_malloc
└─ PSRAM clock source: Main XTAL
└─ CPU frequency: 240 MHz
```

### Flash
```
Serial flasher config
├─ Flash size: 16 MB
└─ Flash SPI mode: DIO

Partition Table
└─ Partition Table: Custom partition table CSV
   (Ver partitions.csv: NVS 24KB, PHY 4KB, Factory 15MB)
```

### Compilador C++
```
Compiler options
└─ C++ language standard: GNU++23 (-std=gnu++2b)
```

### WiFi/Networking
```
Component config → Wi-Fi
├─ WiFi IRAM speed optimization: [✓] Enabled
└─ WiFi RX IRAM speed optimization: [✓] Enabled

Component config → LWIP
└─ Max number of open sockets: 10
```

---

## 📦 Componentes/Librerías

### Archivo main/CMakeLists.txt
```cmake
idf_component_register(SRCS "main.cpp"
                    INCLUDE_DIRS ".")

```

### Archivo CMakeLists.txt
```cmake
idf_component_register(
    SRCS "02_ResizeImage.cpp"
    INCLUDE_DIRS "."
    REQUIRES 
        nvs_flash
        esp_wifi
        esp_event
        esp_netif
        esp_http_server
        esp_timer
        esp32-camera      # Componente clonado de GitHub
)

set_source_files_properties(02_ResizeImage.cpp PROPERTIES COMPILE_FLAGS -std=gnu++2b)
```

### Componente esp32-camera
```bash
# En components/
git clone https://github.com/espressif/esp32-camera.git
```

---

## 🎥 Configuración Cámara OV5640

### Pines Hardware (Freenove ESP32-S3 WROOM-1)
```cpp
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4   // I2C SDA
#define SIOC_GPIO_NUM     5   // I2C SCL
#define XCLK_FREQ_HZ      20000000

#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11

#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13
```

### Configuración Cámara
```cpp
camera_config_t config;
config.pixel_format = PIXFORMAT_RGB565;  // ⚠️ NO usar JPEG (problemas con OV5640)
config.frame_size = FRAMESIZE_QVGA;      // 320x240
config.jpeg_quality = 10;                // No usado en RGB565
config.fb_count = 2;                     // Double buffering
config.fb_location = CAMERA_FB_IN_PSRAM; // ⚠️ Usar PSRAM
config.grab_mode = CAMERA_GRAB_LATEST;
config.sccb_i2c_port = 1;
config.xclk_freq_hz = 20000000;
config.ledc_timer = LEDC_TIMER_0;
config.ledc_channel = LEDC_CHANNEL_0;
```

### Ajustes Sensor (Post-init)
```cpp
sensor_t *s = esp_camera_sensor_get();
s->set_vflip(s, 1);          // ⚠️ Flip vertical (imagen al revés sin esto)
s->set_brightness(s, 0);
s->set_contrast(s, 0);
s->set_saturation(s, 0);
s->set_whitebal(s, 1);       // Auto white balance
s->set_awb_gain(s, 1);
s->set_exposure_ctrl(s, 1);  // Auto exposure
s->set_gain_ctrl(s, 1);      // Auto gain
s->set_lenc(s, 1);           // Lens correction
```

---

## 🖼️ Procesamiento de Imágenes Esperado

### Formatos Usados
- **Entrada cámara**: RGB565 (2 bytes/pixel) - 320x240 = 153,600 bytes
- **Resize**: RGB565 - 224x224 = 100,352 bytes (center crop)

### Funciones Clave
```cpp
// Resize: Center crop 320x240 → 224x224
offset_x = (320 - 224) / 2 = 48 pixels

```

### Allocación de Memoria
```cpp
// SIEMPRE usar PSRAM para imágenes grandes
heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
```



### Modelos Recomendados por Prioridad

**Para ESP32-S3 (con PSRAM):**

1. **FOMO (Edge Impulse)** ⭐ TOP CHOICE
   - 224x224 grayscale
   - ~40-80KB model
   - ~200-400ms inferencia (umbral máximo objetivo, si se puede menor, mejor)
   - 6-8 clases perfecto
   
2. **MobileNetV2 (0.35 alpha, quantized)**
   - 224x224 grayscale o RGB
   - ~150-250KB model  
   - ~300-600ms inferencia (umbral máximo objetivo, si se puede menor, mejor)
   - Transfer learning fácil

3. **YOLOv5 Nano (TinyML)**
   - 224x224 RGB
   - ~400KB model
   - ~800ms-1.5s inferencia (umbral máximo objetivo, si se puede menor, mejor)
   - Mejor detección multi-objeto
