// =============================================================================
// camera_handler.cpp — OV5640 driver para Freenove ESP32-S3 CAM Board
// Toda la configuración proviene de docs/Configuracion_ESP32-S3.md (validada)
// =============================================================================
#include "camera_handler.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "camera";

// Pin mapping Freenove ESP32-S3 CAM Board (WROOM N16R8 + OV5640)
// Fuente: docs/Configuracion_ESP32-S3.md
#define CAM_PIN_PWDN    (-1)
#define CAM_PIN_RESET   (-1)
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4    // I2C SDA
#define CAM_PIN_SIOC    5    // I2C SCL
#define CAM_PIN_D7      16   // Y9
#define CAM_PIN_D6      17   // Y8
#define CAM_PIN_D5      18   // Y7
#define CAM_PIN_D4      12   // Y6
#define CAM_PIN_D3      10   // Y5
#define CAM_PIN_D2      8    // Y4
#define CAM_PIN_D1      9    // Y3
#define CAM_PIN_D0      11   // Y2
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

esp_err_t camera_init(void)
{
    camera_config_t config = {};

    // Pines
    config.pin_pwdn     = CAM_PIN_PWDN;
    config.pin_reset    = CAM_PIN_RESET;
    config.pin_xclk     = CAM_PIN_XCLK;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_d7       = CAM_PIN_D7;
    config.pin_d6       = CAM_PIN_D6;
    config.pin_d5       = CAM_PIN_D5;
    config.pin_d4       = CAM_PIN_D4;
    config.pin_d3       = CAM_PIN_D3;
    config.pin_d2       = CAM_PIN_D2;
    config.pin_d1       = CAM_PIN_D1;
    config.pin_d0       = CAM_PIN_D0;
    config.pin_vsync    = CAM_PIN_VSYNC;
    config.pin_href     = CAM_PIN_HREF;
    config.pin_pclk     = CAM_PIN_PCLK;

    // Configuración validada
    config.xclk_freq_hz = 20000000;          // 20 MHz XCLK
    config.ledc_timer   = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;

    // PIXFORMAT_RGB565 — NO JPEG (problemas conocidos con OV5640)
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size   = FRAMESIZE_QVGA;    // 320×240

    // Doble buffer en PSRAM (GRAB_LATEST para siempre obtener frame más reciente)
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_LATEST;

    // JPEG quality (no aplica para RGB565, pero requerido por el struct)
    config.jpeg_quality = 12;

    // Inicializar driver
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x (%s)", err, esp_err_to_name(err));
        return err;
    }

    // Ajustes post-init del sensor (CRÍTICO: vflip necesario en esta placa)
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_vflip(s, 1);            // Imagen invertida sin esto
        s->set_hmirror(s, 0);
        s->set_whitebal(s, 1);         // Auto white balance
        s->set_exposure_ctrl(s, 1);    // Auto exposición
        s->set_gain_ctrl(s, 1);        // Auto ganancia
        s->set_awb_gain(s, 1);
        s->set_lenc(s, 1);             // Corrección de lente
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
        ESP_LOGI(TAG, "Sensor configured: vflip=1, auto WB/exp/gain, lens correction");
    } else {
        ESP_LOGW(TAG, "Could not get sensor handle for post-init config");
    }

    ESP_LOGI(TAG, "Camera initialized: OV5640 RGB565 320x240, 2 buffers in PSRAM");
    return ESP_OK;
}

camera_fb_t *camera_capture(void)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGW(TAG, "Frame buffer acquisition failed");
        return nullptr;
    }

    // Verificar formato esperado
    if (fb->format != PIXFORMAT_RGB565) {
        ESP_LOGW(TAG, "Unexpected pixel format: %d (expected RGB565)", fb->format);
    }

    return fb;
}

void camera_release(camera_fb_t *fb)
{
    if (fb) {
        esp_camera_fb_return(fb);
    }
}

const uint8_t *camera_get_last_frame(size_t *out_len)
{
    // Para debug visual: captura rápida sin retener
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        if (out_len) *out_len = 0;
        return nullptr;
    }
    // NOTA: esto retorna el puntero del buffer compartido.
    // El caller debe usar los datos antes de que el driver los sobreescriba.
    if (out_len) *out_len = fb->len;
    const uint8_t *ptr = fb->buf;
    esp_camera_fb_return(fb);
    return ptr;
}
