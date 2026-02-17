// =============================================================================
// metrics.cpp — Implementación de métricas de rendimiento
// Usa esp_timer para timing de alta resolución (microsegundos)
// =============================================================================
#include "metrics.h"
#include "postprocess.h"   // for DetectionResult
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

// Temperature sensor (ESP32-S3 tiene sensor integrado)
#include "driver/temperature_sensor.h"

static const char *TAG = "metrics";

// Ventana para cálculo de FPS (1 segundo)
static constexpr int64_t FPS_WINDOW_US = 1000000;  // 1 seg

// Puntero a métricas globales (para metrics_get_fps)
static GlobalMetrics *s_global = nullptr;

// Handle del sensor de temperatura
static temperature_sensor_handle_t s_temp_sensor = nullptr;

// =============================================================================
// metrics_init
// =============================================================================
void metrics_init(GlobalMetrics *gm)
{
    if (!gm) return;

    *gm = {};
    gm->window_start_us = esp_timer_get_time();
    gm->min_free_psram_kb = 0xFFFFFFFF;
    s_global = gm;

    // Inicializar sensor de temperatura
    temperature_sensor_config_t temp_config = TEMPERATURE_SENSOR_CONFIG_DEFAULT(-10, 80);
    esp_err_t err = temperature_sensor_install(&temp_config, &s_temp_sensor);
    if (err == ESP_OK) {
        temperature_sensor_enable(s_temp_sensor);
        ESP_LOGI(TAG, "Temperature sensor initialized");
    } else {
        ESP_LOGW(TAG, "Temperature sensor init failed: %s (non-critical)", esp_err_to_name(err));
        s_temp_sensor = nullptr;
    }

    ESP_LOGI(TAG, "Metrics system initialized");
}

// =============================================================================
// Frame timing
// =============================================================================
void metrics_start_frame(FrameMetrics *fm)
{
    if (!fm) return;
    fm->frame_start_us = esp_timer_get_time();
    for (int i = 0; i < PHASE_COUNT; i++) {
        fm->phase_ms[i] = 0;
        fm->phase_start_us[i] = 0;
    }
    fm->total_ms = 0;
}

void metrics_start_phase(FrameMetrics *fm, MetricsPhase phase)
{
    if (!fm || phase >= PHASE_COUNT) return;
    fm->phase_start_us[phase] = esp_timer_get_time();
}

void metrics_end_phase(FrameMetrics *fm, MetricsPhase phase)
{
    if (!fm || phase >= PHASE_COUNT) return;
    int64_t elapsed = esp_timer_get_time() - fm->phase_start_us[phase];
    fm->phase_ms[phase] = (float)elapsed / 1000.0f;
}

void metrics_end_frame(FrameMetrics *fm)
{
    if (!fm) return;
    int64_t elapsed = esp_timer_get_time() - fm->frame_start_us;
    fm->total_ms = (float)elapsed / 1000.0f;
}

// =============================================================================
// metrics_update_global
// =============================================================================
void metrics_update_global(GlobalMetrics *gm, const FrameMetrics *fm,
                           const DetectionResult *result)
{
    if (!gm || !fm) return;

    gm->total_frames++;
    gm->window_frames++;

    // FPS: ventana deslizante de 1 segundo
    int64_t now = esp_timer_get_time();
    int64_t window_elapsed = now - gm->window_start_us;
    if (window_elapsed >= FPS_WINDOW_US) {
        gm->avg_fps = (float)gm->window_frames * 1e6f / (float)window_elapsed;
        gm->window_start_us = now;
        gm->window_frames = 0;
    }

    // Promedios exponenciales (alpha=0.1 para suavizado)
    const float alpha = 0.1f;
    if (gm->total_frames == 1) {
        gm->avg_inference_ms = fm->phase_ms[PHASE_INFERENCE];
        gm->avg_total_ms = fm->total_ms;
    } else {
        gm->avg_inference_ms = alpha * fm->phase_ms[PHASE_INFERENCE] +
                               (1.0f - alpha) * gm->avg_inference_ms;
        gm->avg_total_ms = alpha * fm->total_ms +
                           (1.0f - alpha) * gm->avg_total_ms;
    }

    // Máximos
    if (fm->phase_ms[PHASE_INFERENCE] > gm->max_inference_ms) {
        gm->max_inference_ms = fm->phase_ms[PHASE_INFERENCE];
    }
    if (fm->total_ms > gm->max_total_ms) {
        gm->max_total_ms = fm->total_ms;
    }

    // Memoria
    gm->free_psram_kb = (uint32_t)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
    gm->free_internal_kb = (uint32_t)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024);
    if (gm->free_psram_kb < gm->min_free_psram_kb) {
        gm->min_free_psram_kb = gm->free_psram_kb;
    }

    // Temperatura (lectura cada 10 frames para no saturar el bus)
    if (s_temp_sensor && (gm->total_frames % 10 == 0)) {
        float temp = 0;
        if (temperature_sensor_get_celsius(s_temp_sensor, &temp) == ESP_OK) {
            gm->temperature_c = temp;
        }
    }

    // Detecciones
    if (result) {
        gm->last_num_detections = result->num_detections;
    }
}

// =============================================================================
// metrics_get_fps
// =============================================================================
float metrics_get_fps(void)
{
    return s_global ? s_global->avg_fps : 0.0f;
}
