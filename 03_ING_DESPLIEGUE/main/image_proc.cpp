// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Image preprocessing implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "image_proc.h"
#include "app_config.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

static const char* TAG = "image_proc";

// Buffers en PSRAM
static int8_t* s_int8_buffer  = nullptr;
static float*  s_float_buffer = nullptr;

esp_err_t image_proc_init() {
    // Buffer INT8: 224 × 224 × 3 = 150,528 bytes
    s_int8_buffer = static_cast<int8_t*>(
        heap_caps_malloc(INPUT_SIZE * sizeof(int8_t), MALLOC_CAP_SPIRAM)
    );
    if (!s_int8_buffer) {
        ESP_LOGE(TAG, "No se pudo alojar buffer INT8 en PSRAM (%d bytes)", INPUT_SIZE);
        return ESP_ERR_NO_MEM;
    }

    // Buffer float32: 224 × 224 × 3 × 4 = 602,112 bytes
    s_float_buffer = static_cast<float*>(
        heap_caps_malloc(INPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM)
    );
    if (!s_float_buffer) {
        ESP_LOGE(TAG, "No se pudo alojar buffer float32 en PSRAM (%zu bytes)",
                 INPUT_SIZE * sizeof(float));
        // No es fatal — solo el path float no estará disponible
        ESP_LOGW(TAG, "Preprocesamiento float32 no disponible");
    }

    ESP_LOGI(TAG, "✅ Buffers de preprocesamiento alojados en PSRAM");
    ESP_LOGI(TAG, "   INT8:    %d bytes @ %p", INPUT_SIZE, s_int8_buffer);
    if (s_float_buffer) {
        ESP_LOGI(TAG, "   Float32: %zu bytes @ %p", INPUT_SIZE * sizeof(float), s_float_buffer);
    }

    return ESP_OK;
}

/// Convierte un pixel RGB565 a componentes RGB888.
static inline void rgb565_to_rgb888(uint16_t pixel, uint8_t& r, uint8_t& g, uint8_t& b) {
    // RGB565: RRRRR GGGGGG BBBBB (big-endian en el frame buffer)
    // El ESP32 camera devuelve RGB565 en byte order: [byte0=GGGBBBBB, byte1=RRRRRGGG]
    r = (pixel >> 8) & 0xF8;  // 5 bits → 8 bits
    g = (pixel >> 3) & 0xFC;  // 6 bits → 8 bits
    b = (pixel << 3) & 0xF8;  // 5 bits → 8 bits
}

int8_t* image_preprocess(const camera_fb_t* fb, int8_t* output) {
    if (!fb || !fb->buf) {
        ESP_LOGE(TAG, "Frame buffer nulo");
        return nullptr;
    }

    int8_t* out = output ? output : s_int8_buffer;
    if (!out) {
        ESP_LOGE(TAG, "Buffer de salida no disponible");
        return nullptr;
    }

    const uint16_t* src = reinterpret_cast<const uint16_t*>(fb->buf);
    int idx = 0;

    // Crop central 224×224 desde 320×240
    for (int y = CROP_OFFSET_Y; y < CROP_OFFSET_Y + INPUT_HEIGHT; y++) {
        for (int x = CROP_OFFSET_X; x < CROP_OFFSET_X + INPUT_WIDTH; x++) {
            uint16_t pixel = src[y * CAMERA_WIDTH + x];

            uint8_t r, g, b;
            rgb565_to_rgb888(pixel, r, g, b);

            // Normalizar a INT8 [-128, 127]: valor_uint8 - 128
            out[idx++] = static_cast<int8_t>(r - 128);
            out[idx++] = static_cast<int8_t>(g - 128);
            out[idx++] = static_cast<int8_t>(b - 128);
        }
    }

    return out;
}

float* image_preprocess_float(const camera_fb_t* fb, float* output) {
    if (!fb || !fb->buf) {
        ESP_LOGE(TAG, "Frame buffer nulo");
        return nullptr;
    }

    float* out = output ? output : s_float_buffer;
    if (!out) {
        ESP_LOGE(TAG, "Buffer float32 no disponible");
        return nullptr;
    }

    const uint16_t* src = reinterpret_cast<const uint16_t*>(fb->buf);
    int idx = 0;

    for (int y = CROP_OFFSET_Y; y < CROP_OFFSET_Y + INPUT_HEIGHT; y++) {
        for (int x = CROP_OFFSET_X; x < CROP_OFFSET_X + INPUT_WIDTH; x++) {
            uint16_t pixel = src[y * CAMERA_WIDTH + x];

            uint8_t r, g, b;
            rgb565_to_rgb888(pixel, r, g, b);

            // Normalizar a [0.0, 1.0]
            out[idx++] = r / 255.0f;
            out[idx++] = g / 255.0f;
            out[idx++] = b / 255.0f;
        }
    }

    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  ESP-DL INT8 preprocessing (power-of-2 quantization, exponent=-7)
//
//  The ESP-DL quantization scheme uses: float_val = int8_val * 2^(-7)
//  So to encode a pixel [0,255] into INT8:
//    int8_val = round(pixel / 255.0 * 128.0)   → range [0, 128] clamped to [0,127]
//
//  This is different from TFLite's zero-point=128 scheme (pixel - 128).
// ═══════════════════════════════════════════════════════════════════════════
int8_t* image_preprocess_espdl(const camera_fb_t* fb, int8_t* output) {
    if (!fb || !fb->buf) {
        ESP_LOGE(TAG, "Frame buffer nulo");
        return nullptr;
    }

    int8_t* out = output ? output : s_int8_buffer;
    if (!out) {
        ESP_LOGE(TAG, "Buffer de salida no disponible");
        return nullptr;
    }

    const uint16_t* src = reinterpret_cast<const uint16_t*>(fb->buf);
    int idx = 0;

    // Crop central 224×224 desde 320×240
    for (int y = CROP_OFFSET_Y; y < CROP_OFFSET_Y + INPUT_HEIGHT; y++) {
        for (int x = CROP_OFFSET_X; x < CROP_OFFSET_X + INPUT_WIDTH; x++) {
            uint16_t pixel = src[y * CAMERA_WIDTH + x];

            uint8_t r, g, b;
            rgb565_to_rgb888(pixel, r, g, b);

            // Normalizar a INT8 [0, 127] para ESP-DL (exponent=-7)
            // Formula: int8_val = (pixel * 128 + 127) / 255
            //   → equivale a round(pixel / 255.0 * 128.0) con aritmética entera
            //   → clamped a máximo 127 (sólo pixel=255 daría 128→clamp)
            auto to_espdl = [](uint8_t p) -> int8_t {
                int val = (static_cast<int>(p) * 128 + 127) / 255;
                return static_cast<int8_t>(val > 127 ? 127 : val);
            };

            out[idx++] = to_espdl(r);
            out[idx++] = to_espdl(g);
            out[idx++] = to_espdl(b);
        }
    }

    return out;
}

void image_proc_deinit() {
    if (s_int8_buffer) {
        heap_caps_free(s_int8_buffer);
        s_int8_buffer = nullptr;
    }
    if (s_float_buffer) {
        heap_caps_free(s_float_buffer);
        s_float_buffer = nullptr;
    }
    ESP_LOGI(TAG, "Buffers de preprocesamiento liberados");
}
