// =============================================================================
// main.cpp — Entry point del firmware de captura de imágenes
// ESP32-S3 + OV5640 + SD Card → Captura de dataset para detección de objetos
// =============================================================================
#include <cstdio>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"

#include "app_config.h"
#include "camera_handler.h"
#include "sd_storage.h"
#include "capture_server.h"

static const char *TAG = "main";

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  TFM — Image Capture for Dataset");
    ESP_LOGI(TAG, "  ESP32-S3 + OV5640 + SD Card");
    ESP_LOGI(TAG, "========================================");

    // ---- NVS (necesario para WiFi + contador de imágenes) ----
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    ESP_LOGI(TAG, "NVS initialized");

    // ---- Memoria disponible ----
    ESP_LOGI(TAG, "PSRAM total: %d KB, free: %d KB",
        (int)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024),
        (int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    ESP_LOGI(TAG, "Internal RAM free: %d KB",
        (int)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));

    // ---- Init cámara OV5640 ----
    ret = camera_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FATAL: Camera init failed (0x%x). Check hardware connections.", ret);
        ESP_LOGE(TAG, "Pins: XCLK=%d, SDA=%d, SCL=%d, VSYNC=%d, HREF=%d, PCLK=%d",
                 CAM_PIN_XCLK, CAM_PIN_SIOD, CAM_PIN_SIOC,
                 CAM_PIN_VSYNC, CAM_PIN_HREF, CAM_PIN_PCLK);
        return;
    }
    ESP_LOGI(TAG, "Camera ready: OV5640 %s",
             camera_is_jpeg_native() ? "JPEG native" : "RGB565→JPEG fallback");

    // ---- Init tarjeta SD ----
    ret = sd_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FATAL: SD card init failed (0x%x).", ret);
        ESP_LOGE(TAG, "Check: SD card inserted? Pins CLK=%d CMD=%d D0=%d correct?",
                 SD_MMC_CLK_PIN, SD_MMC_CMD_PIN, SD_MMC_D0_PIN);
        ESP_LOGE(TAG, "If new card, format as FAT32 first.");
        return;
    }

    sd_stats_t stats = {};
    sd_get_stats(&stats);
    ESP_LOGI(TAG, "SD card ready: %lu MB total, %lu MB free, %lu existing photos",
             (unsigned long)stats.total_mb,
             (unsigned long)stats.free_mb,
             (unsigned long)stats.photo_count);

    // ---- Init servidor web (WiFi STA + HTTP + MJPEG) ----
    ret = capture_server_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FATAL: Server init failed (0x%x).", ret);
        ESP_LOGE(TAG, "Check WiFi credentials: SSID='%s'", WIFI_STA_SSID);
        return;
    }

    // ---- Memoria post-init ----
    ESP_LOGI(TAG, "Memory post-init: PSRAM=%d KB free, Internal=%d KB free",
        (int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
        (int)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));

    // ---- Instrucciones de uso ----
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  System ready!");
    ESP_LOGI(TAG, "  WiFi: Connected to '%s'", WIFI_STA_SSID);
    ESP_LOGI(TAG, "  Web:  http://%s/", capture_server_get_ip());
    ESP_LOGI(TAG, "  mDNS: http://%s.local/", MDNS_HOSTNAME);
    ESP_LOGI(TAG, "  SD:   %lu MB free (%lu photos)",
             (unsigned long)stats.free_mb, (unsigned long)stats.photo_count);
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Open the URL in your browser to start capturing images.");
    ESP_LOGI(TAG, "Use the Python script to batch-download photos:");
    ESP_LOGI(TAG, "  python capture_download.py --ip %s download", capture_server_get_ip());
    ESP_LOGI(TAG, "========================================");

    // El servidor HTTP maneja todo vía callbacks.
    // app_main puede terminar — FreeRTOS mantiene las tareas vivas.
}
