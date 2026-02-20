// =============================================================================
// app_config.h — Configuración central del firmware TFM TinyML
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>

// =============================================================================
// MODEL SELECTION (cambiar este #define para benchmarking)
// =============================================================================
// Opciones: MODEL_MBNTV3S, MODEL_YOLO11N, MODEL_YOLO26N
#define ACTIVE_MODEL  MODEL_YOLO11N

// Enum de modelos
enum ModelType {
    MODEL_MBNTV3S = 0,   // MobileNetV3-Small SSD-Lite (3 salidas)
    MODEL_YOLO11N = 1,   // YOLOv11 Nano (1 salida, NMS on-device)
    MODEL_YOLO26N = 2,   // YOLOv26 Nano (1 salida, NMS on-device)
    MODEL_COUNT   = 3
};

// Nombres legibles
static const char *MODEL_NAMES[MODEL_COUNT] = {
    "MBNTv3S_SSDLite",
    "YOLO11n",
    "YOLO26n"
};

inline const char *get_model_name(ModelType m) {
    return (m < MODEL_COUNT) ? MODEL_NAMES[m] : "Unknown";
}

// =============================================================================
// PARTICIÓN DE MODELOS — offsets dentro de la partición "models"
// Alineados a 4 KB (0x1000). Calculados desde tamaños reales de los .espdl
// =============================================================================
#define MODELS_PARTITION_LABEL  "models"

// MBNTv3S: 681,088 bytes → siguiente frontera 4KB = 0xA7000
#define MODEL_MBNTV3S_OFFSET    0x000000
#define MODEL_MBNTV3S_SIZE      681088

// YOLO11n (6 salidas, INT8 percentile calib): 2,800,272 bytes
#define MODEL_YOLO11N_OFFSET    0x0A7000
#define MODEL_YOLO11N_SIZE      2800272

// YOLO26n (6 salidas, sin detect head): 2,639,168 bytes
#define MODEL_YOLO26N_OFFSET    0x353000
#define MODEL_YOLO26N_SIZE      2639168

// Helper: obtener offset/size del modelo activo
inline void get_model_partition_info(ModelType m, size_t &offset, size_t &size) {
    switch (m) {
        case MODEL_MBNTV3S: offset = MODEL_MBNTV3S_OFFSET; size = MODEL_MBNTV3S_SIZE; break;
        case MODEL_YOLO11N: offset = MODEL_YOLO11N_OFFSET; size = MODEL_YOLO11N_SIZE; break;
        case MODEL_YOLO26N: offset = MODEL_YOLO26N_OFFSET; size = MODEL_YOLO26N_SIZE; break;
        default:            offset = 0; size = 0; break;
    }
}

// =============================================================================
// TENSORES DE ENTRADA/SALIDA (extraídos de los .info de cada modelo)
// Todos los modelos: input INT8 NHWC 1×224×224×3, exponent = -7
// =============================================================================

// --- Imagen ---
#define CAMERA_WIDTH        320
#define CAMERA_HEIGHT       240
#define MODEL_INPUT_W       224
#define MODEL_INPUT_H       224
#define MODEL_INPUT_C       3
#define MODEL_INPUT_SIZE    (MODEL_INPUT_W * MODEL_INPUT_H * MODEL_INPUT_C)  // 150,528 bytes

// Crop: 320→240 (40px cada lado), luego 240→224 resize
#define CROP_OFFSET_X       40
#define CROP_SIZE           240   // 320 - 2*40

// --- Salidas MBNTv3S (3 tensores, todos exp=-7 → scale=1/128) ---
// Output 0: class_out  [1, 1470, 5]  — class scores post-Sigmoid
// Output 1: bbox_out   [1, 1470, 4]  — bbox deltas (cx, cy, w, h)
// Output 2: objectness [1, 1470, 1]  — objectness post-Sigmoid
#define SSD_NUM_ANCHORS     1470
#define SSD_NUM_CLASSES     5
#define SSD_OUTPUT_EXP      (-7)
#define SSD_DEQUANT_SCALE   0.0078125f  // 2^(-7) = 1/128

// --- Salidas YOLO11n (6 tensores: box0/score0, box1/score1, box2/score2) ---
// Sin detection head (DFL+sigmoid+concat se hacen on-device en float32)
// box{i}:   [1, H, W, 64]  raw DFL logits (reg_max=16)
// score{i}: [1, H, W, 5]   raw class logits (pre-sigmoid)
// P3: 28×28 (stride 8), P4: 14×14 (stride 16), P5: 7×7 (stride 32)
#define YOLO11N_REG_MAX         16
#define YOLO11N_NUM_CANDIDATES  1029   // 28*28 + 14*14 + 7*7

// --- Salidas YOLO26n (6 tensores: box0/score0, box1/score1, box2/score2) ---
// box{i}:   [1, H, W, 4]   direct bbox offsets (reg_max=1, no DFL)
// score{i}: [1, H, W, 5]   raw class logits (pre-sigmoid)
#define YOLO26N_REG_MAX         1
#define YOLO26N_NUM_CANDIDATES  1029

// =============================================================================
// DETECCIÓN — Parámetros de postprocesamiento
// =============================================================================
#define NUM_CLASSES         5
#define SCORE_THRESHOLD     0.10f   // Umbral bajo para capturar detecciones INT8
#define NMS_IOU_THRESHOLD   0.45f   // IoU para NMS
#define MAX_DETECTIONS      100     // Máx detecciones finales
#define MAX_CANDIDATES      200     // Máx candidatos pre-NMS

[[maybe_unused]] static const char *CLASS_NAMES[NUM_CLASSES] = {
    "dog", "door", "obstacle", "person", "stair"
};

// =============================================================================
// SSD ANCHORS — Feature maps de MBNTv3S-SSDLite para input 224×224
// 14×14 (stride 16) × 6 anchors = 1176
//  7×7  (stride 32) × 6 anchors = 294
// Total: 1470
// =============================================================================
#define SSD_NUM_FEATURE_MAPS    2
// NOTA: min_sizes/max_sizes/aspect_ratios deben verificarse vs config de entrenamiento
// Valores estándar para SSD-Lite con MobileNetV3-Small
static const int    SSD_FM_SIZES[SSD_NUM_FEATURE_MAPS]        = {14, 7};
static const int    SSD_FM_STRIDES[SSD_NUM_FEATURE_MAPS]      = {16, 32};
static const int    SSD_ANCHORS_PER_CELL                       = 6;
static const float  SSD_MIN_SIZES[SSD_NUM_FEATURE_MAPS]       = {0.2f, 0.35f};
static const float  SSD_MAX_SIZES[SSD_NUM_FEATURE_MAPS]       = {0.35f, 0.5f};
// Aspect ratios: [1, 2, 0.5] + extra 1:1 (min-max) + [3, 1/3] = 6 total
static const float  SSD_ASPECT_RATIOS[]                        = {1.0f, 2.0f, 0.5f, 3.0f, 1.0f/3.0f};
static const int    SSD_NUM_ASPECT_RATIOS                      = 5;
// Variances para decodificación de bbox (estándar SSD)
static const float  SSD_VARIANCE_XY                            = 0.1f;
static const float  SSD_VARIANCE_WH                            = 0.2f;

// =============================================================================
// WIFI ACCESS POINT
// =============================================================================
#define WIFI_SSID           "ESP32_TFM"
#define WIFI_PASSWORD       "tfm2026esp"
#define WIFI_CHANNEL        1
#define MAX_STA_CONN        4

// =============================================================================
// RED / SERVIDOR
// =============================================================================
#define HTTP_PORT           80
#define WS_MAX_CLIENTS      4
#define WS_BROADCAST_INTERVAL_MS  100   // Mínimo entre broadcasts

// =============================================================================
// INFERENCIA
// =============================================================================
#define INFERENCE_TASK_STACK     (8 * 1024)
#define INFERENCE_TASK_PRIORITY  5
#define INFERENCE_TASK_CORE      0
#define INFERENCE_YIELD_MS       10     // Yield entre frames
