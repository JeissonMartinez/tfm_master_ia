// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Configuración global del firmware
// ESP32-S3 WROOM N16R8 + OV5640 (Freenove ESP32-S3 CAM Board)
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstdint>
#include <cstring>

// ─── Pines de cámara OV5640 (Freenove ESP32-S3 WROOM-1) ─────────────────
#define CAM_PIN_PWDN    (-1)
#define CAM_PIN_RESET   (-1)
#define CAM_PIN_XCLK     15
#define CAM_PIN_SIOD      4   // I2C SDA
#define CAM_PIN_SIOC      5   // I2C SCL
#define CAM_PIN_D7       16   // Y9
#define CAM_PIN_D6       17   // Y8
#define CAM_PIN_D5       18   // Y7
#define CAM_PIN_D4       12   // Y6
#define CAM_PIN_D3       10   // Y5
#define CAM_PIN_D2        8   // Y4
#define CAM_PIN_D1        9   // Y3
#define CAM_PIN_D0       11   // Y2
#define CAM_PIN_VSYNC     6
#define CAM_PIN_HREF      7
#define CAM_PIN_PCLK     13
#define CAM_XCLK_FREQ_HZ 20000000

// ─── Resoluciones ────────────────────────────────────────────────────────
#define CAMERA_WIDTH     320
#define CAMERA_HEIGHT    240
#define CAMERA_FB_COUNT    2   // Double buffering
#define INPUT_WIDTH      224
#define INPUT_HEIGHT     224
#define INPUT_CHANNELS     3
#define INPUT_SIZE       (INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS)

// Offset para crop central: (320 - 224) / 2 = 48
#define CROP_OFFSET_X     48
#define CROP_OFFSET_Y      8   // (240 - 224) / 2 = 8

// ─── Detección ───────────────────────────────────────────────────────────
#define MAX_DETECTIONS    20
#define NUM_CLASSES        5
#define DEFAULT_CONF_THRESHOLD  0.25f
#define DEFAULT_IOU_THRESHOLD   0.45f

// Umbrales por modelo ESPDL (ajustados del entrenamiento)
#define ESPDET_CONF_THRESHOLD   0.35f
#define ESPDET_IOU_THRESHOLD    0.40f
#define YOLO26ESP_CONF_THRESHOLD 0.25f
#define YOLO26ESP_IOU_THRESHOLD  0.45f

// FCOS & DFL grid strides (224×224 → 28, 14, 7)
#define NUM_SCALES          3
static constexpr int GRID_STRIDES[NUM_SCALES]   = {8, 16, 32};
static constexpr int GRID_SIZES[NUM_SCALES]     = {28, 14, 7};
#define DFL_REG_MAX        16   // YOLO26 DFL bins

// Nombres de clases (orden: 0-4)
static const char* const CLASS_NAMES[NUM_CLASSES] = {
    "dog", "door", "obstacle", "person", "stair"
};

// ─── WiFi STA (red doméstica) ────────────────────────────────────────────
#define WIFI_STA_SSID     "JM"
#define WIFI_STA_PASS     "Meca1020@"
#define WIFI_MAX_RETRY    10
#define WEB_SERVER_PORT   80
#define STREAM_SERVER_PORT 81   // MJPEG en puerto separado (no bloquea WS)
#define WS_MAX_CLIENTS    3

// ─── MJPEG Stream ────────────────────────────────────────────────────────
#define STREAM_JPEG_QUALITY   12   // Calidad baja → rápido (~5-10 KB/frame)
#define CAPTURE_JPEG_QUALITY  80   // Calidad alta para frame capturado
#define STREAM_BUF_MAX     (60 * 1024)  // Max JPEG buffer size (PSRAM)
#define STREAM_MAX_CLIENTS    2    // Conexiones MJPEG simultáneas

// ─── Modo de inferencia ──────────────────────────────────────────────────
enum class InferMode : uint8_t {
    CONTINUOUS,    // Inferencia en cada frame (comportamiento original)
    ON_DEMAND,     // Inferencia solo al presionar "Capturar"
};

// ─── Métricas ────────────────────────────────────────────────────────────
#define EMA_ALPHA         0.065f   // alpha = 2/(N+1), N=30
#define METRICS_REPORT_INTERVAL  1 // Cada N frames

// ─── Task priorities & cores ─────────────────────────────────────────────
#define INFERENCE_TASK_CORE      0
#define INFERENCE_TASK_PRIORITY  (configMAX_PRIORITIES - 1)
#define INFERENCE_TASK_STACK     (32 * 1024)

#define WIFI_TASK_CORE           1
// WiFi/HTTP usan prioridades por defecto de ESP-IDF

// ═══════════════════════════════════════════════════════════════════════════
// Tipos y estructuras
// ═══════════════════════════════════════════════════════════════════════════

// ─── Tipo de modelo ──────────────────────────────────────────────────────
enum class ModelType : uint8_t {
    MOBILENET_SSD,   // MBNTv2_ssdlite_v1: 3 tensores, decode anchors + NMS
    YOLO11N,         // yolo11n_v1: [1,9,1029], NMS completo en device
    YOLO26N,         // yolo26n_v1: [1,300,6], NMS integrado (TFLite)
    ESPDET_PICO,     // ESPDet Pico T4: FCOS 3-scale, 6 outputs (ESP-DL)
    YOLO26N_ESP,     // YOLO26n T2 ESP: DFL 3-scale, 6 outputs (ESP-DL)
};

// ─── Runtime de inferencia ───────────────────────────────────────────────
enum class EngineType : uint8_t {
    TFLITE_MICRO,    // esp-tflite-micro (.tflite embebido como array C)
    ESP_DL,          // esp-dl (.espdl embebido como binario)
};

// ─── Detección individual ────────────────────────────────────────────────
struct Detection {
    float x1, y1, x2, y2;      // Bounding box normalizado [0,1]
    float confidence;            // Score de confianza
    int   class_id;              // Índice de clase (0-4)

    const char* class_name() const {
        return (class_id >= 0 && class_id < NUM_CLASSES)
            ? CLASS_NAMES[class_id] : "unknown";
    }
};

// ─── Resultado de detección por frame ────────────────────────────────────
struct DetectionResult {
    Detection detections[MAX_DETECTIONS];
    int       count = 0;

    void add(const Detection& d) {
        if (count < MAX_DETECTIONS) {
            detections[count++] = d;
        }
    }

    void clear() { count = 0; }
};

// ─── Métricas de inferencia ──────────────────────────────────────────────
struct InferenceMetrics {
    float    preprocess_ms  = 0;    // Captura + crop + normalización
    float    inference_ms   = 0;    // TFLite/ESP-DL invoke()
    float    postprocess_ms = 0;    // Decode + NMS
    float    total_ms       = 0;    // Suma end-to-end
    float    fps            = 0;    // 1000.0 / total_ms
    uint32_t heap_internal_free = 0;
    uint32_t heap_internal_used = 0;
    uint32_t psram_free     = 0;
    uint32_t psram_used     = 0;
    uint32_t arena_used     = 0;    // Bytes usados del arena TFLite
    int      n_detections   = 0;
    uint32_t frame_id       = 0;
    float    cpu_temp_c     = 0;

    // Promedios móviles exponenciales
    float    ema_preprocess_ms  = 0;
    float    ema_inference_ms   = 0;
    float    ema_postprocess_ms = 0;
    float    ema_total_ms       = 0;
    float    ema_fps            = 0;
};

// ─── Configuración de modelo ─────────────────────────────────────────────
struct ModelConfig {
    const char*          name;            // Nombre para logs y WebSocket
    ModelType            type;            // Tipo de modelo
    EngineType           engine;          // Runtime activo

    // TFLite path
    const unsigned char* tflite_data;     // Puntero a array C del modelo
    size_t               tflite_size;     // Tamaño en bytes

    // ESP-DL path (binarios embebidos)
    const char*          espdl_partition; // Nombre del binario embebido

    // Parámetros
    int   input_w         = INPUT_WIDTH;
    int   input_h         = INPUT_HEIGHT;
    int   input_channels  = INPUT_CHANNELS;
    int   arena_size      = 0;            // Se determina empíricamente
    float conf_threshold  = DEFAULT_CONF_THRESHOLD;
    float iou_threshold   = DEFAULT_IOU_THRESHOLD;
};

// ─── Utilidad: EMA update ────────────────────────────────────────────────
inline float ema_update(float prev, float value) {
    return EMA_ALPHA * value + (1.0f - EMA_ALPHA) * prev;
}
