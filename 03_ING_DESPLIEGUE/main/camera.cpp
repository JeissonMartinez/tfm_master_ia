// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Camera driver (OV5640) implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "camera.h"
#include "app_config.h"
#include "esp_log.h"

static const char* TAG = "camera";

esp_err_t camera_init() {
    camera_config_t config = {};
    config.pin_pwdn    = CAM_PIN_PWDN;
    config.pin_reset   = CAM_PIN_RESET;
    config.pin_xclk    = CAM_PIN_XCLK;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_d7      = CAM_PIN_D7;
    config.pin_d6      = CAM_PIN_D6;
    config.pin_d5      = CAM_PIN_D5;
    config.pin_d4      = CAM_PIN_D4;
    config.pin_d3      = CAM_PIN_D3;
    config.pin_d2      = CAM_PIN_D2;
    config.pin_d1      = CAM_PIN_D1;
    config.pin_d0      = CAM_PIN_D0;
    config.pin_vsync   = CAM_PIN_VSYNC;
    config.pin_href    = CAM_PIN_HREF;
    config.pin_pclk    = CAM_PIN_PCLK;

    config.xclk_freq_hz = CAM_XCLK_FREQ_HZ;
    config.ledc_timer   = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;

    // RGB565 sin compresión JPEG (OV5640 tiene problemas con JPEG)
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size   = FRAMESIZE_QVGA;      // 320×240
    config.jpeg_quality = 10;                   // No usado en RGB565
    config.fb_count     = CAMERA_FB_COUNT;      // Double buffering
    config.fb_location  = CAMERA_FB_IN_PSRAM;   // Frame buffers en PSRAM
    config.grab_mode    = CAMERA_GRAB_LATEST;   // Siempre el frame más reciente
    config.sccb_i2c_port = 1;

    ESP_LOGI(TAG, "Inicializando cámara OV5640...");
    ESP_LOGI(TAG, "  Pixel format: RGB565");
    ESP_LOGI(TAG, "  Frame size:   %dx%d (QVGA)", CAMERA_WIDTH, CAMERA_HEIGHT);
    ESP_LOGI(TAG, "  FB count:     %d (en PSRAM)", CAMERA_FB_COUNT);
    ESP_LOGI(TAG, "  XCLK freq:    %d Hz", CAM_XCLK_FREQ_HZ);

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error inicializando cámara: %s", esp_err_to_name(err));
        return err;
    }

    // ─── Ajustes de sensor post-init ─────────────────────────────────────
    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        s->set_vflip(s, 1);           // Flip vertical (imagen al revés sin esto)
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
        s->set_whitebal(s, 1);        // Auto white balance
        s->set_awb_gain(s, 1);
        s->set_exposure_ctrl(s, 1);   // Auto exposure
        s->set_gain_ctrl(s, 1);       // Auto gain
        s->set_lenc(s, 1);            // Lens correction
        ESP_LOGI(TAG, "Sensor configurado: vflip=1, auto WB/exposure/gain");
    }

    ESP_LOGI(TAG, "✅ Cámara OV5640 inicializada correctamente");
    return ESP_OK;
}

camera_fb_t* camera_capture() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Error capturando frame");
        return nullptr;
    }

    if (fb->width != CAMERA_WIDTH || fb->height != CAMERA_HEIGHT) {
        ESP_LOGW(TAG, "Frame inesperado: %dx%d (esperado %dx%d)",
                 fb->width, fb->height, CAMERA_WIDTH, CAMERA_HEIGHT);
    }

    return fb;
}

void camera_release_fb(camera_fb_t* fb) {
    if (fb) {
        esp_camera_fb_return(fb);
    }
}
