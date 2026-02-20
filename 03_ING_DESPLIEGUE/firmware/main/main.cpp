// =============================================================================
// main.cpp — Entry point del firmware TFM TinyML
// ESP32-S3 + OV5640 + ESP-DL → Detección de objetos para movilidad
// =============================================================================
#include <cstdio>
#include <cstring>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"

#include "app_config.h"
#include "camera_handler.h"
#include "image_proc.h"
#include "inference_engine.h"
#include "postprocess.h"
#include "metrics.h"
#include "network.h"
#include "dashboard.h"

static const char *TAG = "main";

// Buffer de imagen preprocesada en PSRAM (INT8 224×224×3)
static int8_t *s_input_buffer = nullptr;

// Resultado de detección del último frame
static DetectionResult s_result = {};

// Métricas globales compartidas con el módulo de red
static GlobalMetrics s_global_metrics = {};

// ---- Tarea de inferencia (Core 0) ----
static void inference_task(void *arg)
{
    ESP_LOGI(TAG, "Inference task started on core %d", xPortGetCoreID());

    FrameMetrics frame = {};
    uint32_t frame_count = 0;

    while (true) {
        metrics_start_frame(&frame);

        // 1. Captura
        metrics_start_phase(&frame, PHASE_CAPTURE);
        camera_fb_t *fb = camera_capture();
        if (!fb) {
            ESP_LOGW(TAG, "Frame capture failed, retrying...");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        metrics_end_phase(&frame, PHASE_CAPTURE);

        // 2. Preprocesamiento: RGB565 320×240 → INT8 224×224×3
        metrics_start_phase(&frame, PHASE_PREPROCESS);
        preprocess_image((const uint16_t *)fb->buf, fb->width, fb->height,
                         s_input_buffer);
        camera_release(fb);
        metrics_end_phase(&frame, PHASE_PREPROCESS);

        // 3. Inferencia ESP-DL
        metrics_start_phase(&frame, PHASE_INFERENCE);
        esp_err_t ret = inference_run(s_input_buffer);
        metrics_end_phase(&frame, PHASE_INFERENCE);

        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "Inference failed: %s", esp_err_to_name(ret));
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // 4. Postprocesamiento (decode + NMS)
        metrics_start_phase(&frame, PHASE_POSTPROCESS);
        postprocess_decode(&s_result);
        metrics_end_phase(&frame, PHASE_POSTPROCESS);

        // 5. Finalizar métricas del frame
        metrics_end_frame(&frame);
        frame_count++;

        // 6. Actualizar métricas globales
        metrics_update_global(&s_global_metrics, &frame, &s_result);

        // 7. Broadcast vía WebSocket (si hay clientes conectados)
        network_broadcast(&s_global_metrics, &frame, &s_result);

        // Log cada 10 frames
        if (frame_count % 10 == 0) {
            ESP_LOGI(TAG,
                "Frame %lu | FPS=%.1f | inf=%dms total=%dms | det=%d | PSRAM=%dKB",
                (unsigned long)frame_count,
                s_global_metrics.avg_fps,
                (int)frame.phase_ms[PHASE_INFERENCE],
                (int)frame.total_ms,
                s_result.num_detections,
                (int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
        }

        vTaskDelay(pdMS_TO_TICKS(INFERENCE_YIELD_MS));
    }
}

// ---- Entry point ----
extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  TFM TinyML Deployment — ESP32-S3");
    ESP_LOGI(TAG, "  Model: %s", get_model_name((ModelType)ACTIVE_MODEL));
    ESP_LOGI(TAG, "========================================");

    // ---- NVS ----
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "PSRAM total: %d KB, free: %d KB",
        (int)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024),
        (int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    // ---- Allocate preprocessed image buffer in PSRAM ----
    s_input_buffer = (int8_t *)heap_caps_malloc(MODEL_INPUT_SIZE, MALLOC_CAP_SPIRAM);
    if (!s_input_buffer) {
        ESP_LOGE(TAG, "FATAL: Cannot allocate %d bytes in PSRAM for input buffer",
                 MODEL_INPUT_SIZE);
        return;
    }
    ESP_LOGI(TAG, "Input buffer: %d bytes in PSRAM", MODEL_INPUT_SIZE);

    // ---- Init cámara OV5640 ----
    ESP_ERROR_CHECK(camera_init());
    ESP_LOGI(TAG, "Camera initialized (OV5640 RGB565 %dx%d)", CAMERA_WIDTH, CAMERA_HEIGHT);

    // ---- Init motor de inferencia (carga modelo desde flash) ----
    ESP_ERROR_CHECK(inference_init((ModelType)ACTIVE_MODEL));
    ESP_LOGI(TAG, "Inference engine ready");

    // ---- Init postprocesador ----
    postprocess_init((ModelType)ACTIVE_MODEL);
    ESP_LOGI(TAG, "Postprocessor initialized for %s", get_model_name((ModelType)ACTIVE_MODEL));

    // ---- Init métricas ----
    metrics_init(&s_global_metrics);

    // ---- Reporte de memoria post-init ----
    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    ESP_LOGI(TAG, "Memory post-init: PSRAM=%d KB free, Internal=%d KB free",
        (int)(psram_free / 1024), (int)(internal_free / 1024));
    if (psram_free < 512 * 1024) {
        ESP_LOGW(TAG, "WARNING: PSRAM libre < 512 KB. Riesgo de inestabilidad.");
    }

    // ---- Init red (WiFi AP + HTTP + WebSocket) — usa Core 1 ----
    ESP_ERROR_CHECK(network_init());
    network_set_debug_image_source(s_input_buffer);  // Para endpoint /debug/image
    dashboard_register_handlers();
    ESP_LOGI(TAG, "Network ready: WiFi AP '%s', http://192.168.4.1/", WIFI_SSID);

    // ---- Lanzar tarea de inferencia en Core 0 ----
    BaseType_t task_ret = xTaskCreatePinnedToCore(
        inference_task,
        "inference",
        INFERENCE_TASK_STACK,
        NULL,
        INFERENCE_TASK_PRIORITY,
        NULL,
        INFERENCE_TASK_CORE
    );
    if (task_ret != pdPASS) {
        ESP_LOGE(TAG, "FATAL: Cannot create inference task");
        return;
    }

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  System running. Connect to '%s'", WIFI_SSID);
    ESP_LOGI(TAG, "  Dashboard: http://192.168.4.1/");
    ESP_LOGI(TAG, "========================================");
}
