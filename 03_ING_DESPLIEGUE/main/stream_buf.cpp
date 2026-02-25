// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Shared MJPEG stream buffer implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "stream_buf.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "freertos/semphr.h"

#include <cstring>

static const char* TAG = "stream_buf";

// ─── Inference mode globals ──────────────────────────────────────────────
std::atomic<InferMode> g_infer_mode{InferMode::CONTINUOUS};
std::atomic<bool>      g_infer_trigger{false};

// ─── Shared buffer ───────────────────────────────────────────────────────
static uint8_t*            s_jpg_buf  = nullptr;   // PSRAM buffer
static size_t              s_jpg_len  = 0;
static SemaphoreHandle_t   s_mutex    = nullptr;
static EventGroupHandle_t  s_event    = nullptr;

esp_err_t stream_buf_init() {
    s_jpg_buf = static_cast<uint8_t*>(
        heap_caps_malloc(STREAM_BUF_MAX, MALLOC_CAP_SPIRAM));
    if (!s_jpg_buf) {
        ESP_LOGE(TAG, "No se pudo asignar buffer JPEG (%d bytes en PSRAM)",
                 STREAM_BUF_MAX);
        return ESP_ERR_NO_MEM;
    }

    s_mutex = xSemaphoreCreateMutex();
    s_event = xEventGroupCreate();

    if (!s_mutex || !s_event) {
        ESP_LOGE(TAG, "Error creando primitivas FreeRTOS");
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "✅ Stream buffer inicializado (%d KB en PSRAM)",
             STREAM_BUF_MAX / 1024);
    return ESP_OK;
}

void stream_buf_publish(const uint8_t* jpg, size_t len) {
    if (!s_jpg_buf || !jpg || len == 0) return;
    if (len > STREAM_BUF_MAX) {
        ESP_LOGW(TAG, "Frame JPEG demasiado grande: %zu > %d", len, STREAM_BUF_MAX);
        return;
    }

    if (xSemaphoreTake(s_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        std::memcpy(s_jpg_buf, jpg, len);
        s_jpg_len = len;
        xSemaphoreGive(s_mutex);

        // Signal all waiting stream handlers
        xEventGroupSetBits(s_event, STREAM_NEW_FRAME_BIT);
    }
}

bool stream_buf_read(uint8_t* out, size_t* out_len) {
    if (!s_jpg_buf || !out || !out_len) return false;

    bool ok = false;
    if (xSemaphoreTake(s_mutex, pdMS_TO_TICKS(20)) == pdTRUE) {
        if (s_jpg_len > 0) {
            std::memcpy(out, s_jpg_buf, s_jpg_len);
            *out_len = s_jpg_len;
            ok = true;
        }
        xSemaphoreGive(s_mutex);
    }
    return ok;
}

EventGroupHandle_t stream_buf_event_group() {
    return s_event;
}

void stream_buf_deinit() {
    if (s_jpg_buf) {
        heap_caps_free(s_jpg_buf);
        s_jpg_buf = nullptr;
    }
    if (s_mutex) {
        vSemaphoreDelete(s_mutex);
        s_mutex = nullptr;
    }
    if (s_event) {
        vEventGroupDelete(s_event);
        s_event = nullptr;
    }
    s_jpg_len = 0;
}
