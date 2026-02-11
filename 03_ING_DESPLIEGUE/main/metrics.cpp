// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Runtime metrics implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "metrics.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#if SOC_TEMP_SENSOR_SUPPORTED
#include "driver/temperature_sensor.h"
#endif

#include <cstring>

static const char* TAG = "metrics";

// ─── State ───────────────────────────────────────────────────────────────
static InferenceMetrics s_current{};
static SemaphoreHandle_t s_mutex = nullptr;

static int64_t s_frame_start_us  = 0;
static int64_t s_preproc_end_us  = 0;
static int64_t s_infer_end_us    = 0;

static uint32_t s_initial_internal = 0;
static uint32_t s_initial_psram    = 0;

#if SOC_TEMP_SENSOR_SUPPORTED
static temperature_sensor_handle_t s_temp_handle = nullptr;
#endif

// ─── Helpers ─────────────────────────────────────────────────────────────
static inline int64_t now_us() { return esp_timer_get_time(); }

static void update_heap(InferenceMetrics& m) {
    m.heap_internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m.heap_internal_used = s_initial_internal - m.heap_internal_free;
    m.psram_free         = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    m.psram_used         = s_initial_psram - m.psram_free;
}

static float read_temperature() {
#if SOC_TEMP_SENSOR_SUPPORTED
    if (s_temp_handle) {
        float t = 0;
        if (temperature_sensor_get_celsius(s_temp_handle, &t) == ESP_OK) {
            return t;
        }
    }
#endif
    return 0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════
//  API
// ═══════════════════════════════════════════════════════════════════════════

esp_err_t metrics_init() {
    s_mutex = xSemaphoreCreateMutex();
    if (!s_mutex) return ESP_ERR_NO_MEM;

    s_initial_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    s_initial_psram    = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    std::memset(&s_current, 0, sizeof(s_current));

#if SOC_TEMP_SENSOR_SUPPORTED
    temperature_sensor_config_t temp_cfg = TEMPERATURE_SENSOR_CONFIG_DEFAULT(-10, 80);
    if (temperature_sensor_install(&temp_cfg, &s_temp_handle) == ESP_OK) {
        temperature_sensor_enable(s_temp_handle);
        ESP_LOGI(TAG, "Sensor de temperatura habilitado");
    }
#endif

    ESP_LOGI(TAG, "✅ Métricas inicializadas");
    ESP_LOGI(TAG, "   Heap interno libre: %lu KB", s_initial_internal / 1024);
    ESP_LOGI(TAG, "   PSRAM libre:        %lu KB", s_initial_psram / 1024);
    return ESP_OK;
}

void metrics_frame_begin() {
    s_frame_start_us = now_us();
}

void metrics_preprocess_end() {
    s_preproc_end_us = now_us();
}

void metrics_inference_end() {
    s_infer_end_us = now_us();
}

void metrics_postprocess_end(int n_detections) {
    int64_t end_us = now_us();

    float preproc_ms  = (s_preproc_end_us - s_frame_start_us) / 1000.0f;
    float infer_ms    = (s_infer_end_us - s_preproc_end_us) / 1000.0f;
    float postproc_ms = (end_us - s_infer_end_us) / 1000.0f;
    float total_ms    = (end_us - s_frame_start_us) / 1000.0f;
    float fps         = (total_ms > 0) ? (1000.0f / total_ms) : 0.0f;

    if (xSemaphoreTake(s_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        s_current.preprocess_ms  = preproc_ms;
        s_current.inference_ms   = infer_ms;
        s_current.postprocess_ms = postproc_ms;
        s_current.total_ms       = total_ms;
        s_current.fps            = fps;
        s_current.n_detections   = n_detections;
        s_current.frame_id++;
        s_current.cpu_temp_c     = read_temperature();

        update_heap(s_current);

        // EMA updates
        s_current.ema_preprocess_ms  = ema_update(s_current.ema_preprocess_ms,  preproc_ms);
        s_current.ema_inference_ms   = ema_update(s_current.ema_inference_ms,   infer_ms);
        s_current.ema_postprocess_ms = ema_update(s_current.ema_postprocess_ms, postproc_ms);
        s_current.ema_total_ms       = ema_update(s_current.ema_total_ms,       total_ms);
        s_current.ema_fps            = ema_update(s_current.ema_fps,            fps);

        xSemaphoreGive(s_mutex);
    }
}

InferenceMetrics metrics_get() {
    InferenceMetrics copy{};
    if (xSemaphoreTake(s_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        copy = s_current;
        xSemaphoreGive(s_mutex);
    }
    return copy;
}

uint32_t metrics_frame_id() {
    return s_current.frame_id;
}

void metrics_deinit() {
#if SOC_TEMP_SENSOR_SUPPORTED
    if (s_temp_handle) {
        temperature_sensor_disable(s_temp_handle);
        temperature_sensor_uninstall(s_temp_handle);
        s_temp_handle = nullptr;
    }
#endif
    if (s_mutex) {
        vSemaphoreDelete(s_mutex);
        s_mutex = nullptr;
    }
    ESP_LOGI(TAG, "Métricas liberadas");
}
