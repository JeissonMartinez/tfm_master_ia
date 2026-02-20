// =============================================================================
// app_config.h — Configuración central del firmware de captura de imágenes
//
// IMPORTANTE: Editar WIFI_STA_SSID y WIFI_STA_PASSWORD antes de compilar
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>

// =============================================================================
// WIFI — Modo Station (se conecta a tu red doméstica)
// =============================================================================
#define WIFI_STA_SSID           "FLIA Martínez Capacho"       // ← EDITAR: nombre de tu WiFi
#define WIFI_STA_PASSWORD       "123@sjmc"    // ← EDITAR: contraseña de tu WiFi
#define WIFI_STA_MAX_RETRY      10                    // Reintentos de conexión
#define WIFI_STA_RETRY_DELAY_MS 2000                  // Delay entre reintentos

// mDNS hostname: acceder como http://esp32-capture.local/
#define MDNS_HOSTNAME           "esp32-capture"

// =============================================================================
// CAMERA — Pin mapping Freenove ESP32-S3 CAM Board (WROOM N16R8 + OV5640)
// Fuente: 03_ING_DESPLIEGUE/firmware/components/camera_handler/camera_handler.cpp
// =============================================================================
#define CAM_PIN_PWDN    (-1)    // Not used
#define CAM_PIN_RESET   (-1)    // Not used
#define CAM_PIN_XCLK    15      // Clock output, 20 MHz
#define CAM_PIN_SIOD    4       // I2C SDA
#define CAM_PIN_SIOC    5       // I2C SCL
#define CAM_PIN_D7      16      // Y9
#define CAM_PIN_D6      17      // Y8
#define CAM_PIN_D5      18      // Y7
#define CAM_PIN_D4      12      // Y6
#define CAM_PIN_D3      10      // Y5
#define CAM_PIN_D2      8       // Y4
#define CAM_PIN_D1      9       // Y3
#define CAM_PIN_D0      11      // Y2
#define CAM_PIN_VSYNC   6       // Vertical sync
#define CAM_PIN_HREF    7       // Horizontal ref
#define CAM_PIN_PCLK    13      // Pixel clock

// Camera settings
#define CAM_XCLK_FREQ_HZ   20000000    // 20 MHz
#define CAM_FRAME_SIZE      FRAMESIZE_SVGA  // 800×600
#define CAM_JPEG_QUALITY    12          // 1-63, lower = higher quality
#define CAM_FB_COUNT        2           // Double buffer in PSRAM (SVGA frames are larger)

// =============================================================================
// SD CARD — Freenove ESP32-S3 WROOM microSD slot (SD_MMC 1-bit mode)
//
// NOTA: Estos pines son los estándar de la placa Freenove ESP32-S3 WROOM.
// Si la SD no monta, verificar contra el esquemático de tu placa específica.
// =============================================================================
#define SD_MMC_CLK_PIN  39      // SD Clock
#define SD_MMC_CMD_PIN  38      // SD Command
#define SD_MMC_D0_PIN   40      // SD Data 0

// Punto de montaje FAT
#define SD_MOUNT_POINT  "/sdcard"
#define SD_CAPTURE_DIR  "/sdcard/captures"

// =============================================================================
// HTTP SERVER
// =============================================================================
#define HTTP_PORT               80
#define HTTP_SERVER_STACK_SIZE  8192
#define HTTP_MAX_URI_HANDLERS   12

// =============================================================================
// MJPEG STREAM
// =============================================================================
#define MJPEG_BOUNDARY      "frame_boundary"
#define MJPEG_FRAME_DELAY_MS  100     // ~10 FPS max en preview

// =============================================================================
// CAPTURE SETTINGS
// =============================================================================
#define CAPTURE_MAX_BURST           50      // Máximo fotos por ráfaga
#define CAPTURE_BURST_DELAY_MS      200     // Delay entre fotos en ráfaga
#define CAPTURE_FILE_PREFIX         "IMG"   // Prefijo: IMG_000001.jpg
#define CAPTURE_NVS_NAMESPACE       "capture"
#define CAPTURE_NVS_COUNTER_KEY     "img_counter"
