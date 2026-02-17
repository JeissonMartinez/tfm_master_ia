// =============================================================================
// camera_handler.cpp — OV5640 driver para captura de imágenes en JPEG
//
// Pin mapping idéntico al firmware de despliegue:
//   03_ING_DESPLIEGUE/firmware/components/camera_handler/camera_handler.cpp
//
// Diferencia clave: usa PIXFORMAT_JPEG para almacenamiento eficiente en SD.
// Si JPEG nativo falla con OV5640 (problema conocido), usa RGB565 + frame2jpg.
// =============================================================================
#include "camera_handler.h"
#include "app_config.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "camera";
static bool s_jpeg_native = false;

// =============================================================================
// Intentar inicializar en modo JPEG nativo
// =============================================================================
static esp_err_t try_init_jpeg(void)
{
    camera_config_t config = {};

    // Pines — idénticos al firmware de despliegue
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

    config.xclk_freq_hz = CAM_XCLK_FREQ_HZ;
    config.ledc_timer   = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;

    // JPEG nativo — más eficiente para guardado en SD
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size   = CAM_FRAME_SIZE;
    config.jpeg_quality = CAM_JPEG_QUALITY;

    config.fb_count     = CAM_FB_COUNT;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_LATEST;

    return esp_camera_init(&config);
}

// =============================================================================
// Fallback: inicializar en RGB565 (si JPEG falla con OV5640)
// =============================================================================
static esp_err_t try_init_rgb565(void)
{
    camera_config_t config = {};

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

    config.xclk_freq_hz = CAM_XCLK_FREQ_HZ;
    config.ledc_timer   = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;

    // Fallback: RGB565 (igual que firmware de despliegue)
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size   = CAM_FRAME_SIZE;
    config.jpeg_quality = CAM_JPEG_QUALITY;

    config.fb_count     = CAM_FB_COUNT;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_LATEST;

    return esp_camera_init(&config);
}

// =============================================================================
// Ajustes post-init del sensor (CRÍTICO: vflip necesario en esta placa)
// Idénticos al firmware de despliegue
// =============================================================================
static void configure_sensor(void)
{
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
}

// =============================================================================
// camera_init — Intenta JPEG, fallback a RGB565
// =============================================================================
esp_err_t camera_init(void)
{
    ESP_LOGI(TAG, "Initializing camera (attempting JPEG native)...");

    esp_err_t err = try_init_jpeg();
    if (err == ESP_OK) {
        s_jpeg_native = true;
        ESP_LOGI(TAG, "Camera initialized in JPEG native mode");
    } else {
        ESP_LOGW(TAG, "JPEG init failed (0x%x), trying RGB565 fallback...", err);

        // Deinit antes de reintentar
        esp_camera_deinit();

        err = try_init_rgb565();
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Camera init failed in both modes: 0x%x (%s)",
                     err, esp_err_to_name(err));
            return err;
        }
        s_jpeg_native = false;
        ESP_LOGW(TAG, "Camera initialized in RGB565 mode (JPEG via frame2jpg)");
    }

    configure_sensor();

    ESP_LOGI(TAG, "Camera ready: OV5640 %s 320x240, %d buffers in PSRAM",
             s_jpeg_native ? "JPEG" : "RGB565→JPEG", CAM_FB_COUNT);
    return ESP_OK;
}

// =============================================================================
// camera_capture_jpeg — Captura frame como JPEG
// =============================================================================
camera_fb_t *camera_capture_jpeg(void)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGW(TAG, "Frame buffer acquisition failed");
        return nullptr;
    }

    // Si estamos en modo JPEG nativo, el frame ya es JPEG
    if (s_jpeg_native) {
        if (fb->format != PIXFORMAT_JPEG) {
            ESP_LOGW(TAG, "Expected JPEG but got format %d", fb->format);
        }
        return fb;
    }

    // Fallback: convertir RGB565 a JPEG usando frame2jpg del driver
    if (fb->format == PIXFORMAT_RGB565) {
        uint8_t *jpg_buf = nullptr;
        size_t jpg_len = 0;

        bool ok = frame2jpg(fb, CAM_JPEG_QUALITY, &jpg_buf, &jpg_len);
        esp_camera_fb_return(fb);

        if (!ok || !jpg_buf) {
            ESP_LOGE(TAG, "frame2jpg conversion failed");
            return nullptr;
        }

        // Crear un fb "virtual" con los datos JPEG
        // NOTA: Usamos un fb estático para evitar malloc/free en cada captura.
        // El caller DEBE llamar camera_release() que liberará jpg_buf.
        static camera_fb_t jpeg_fb;
        jpeg_fb.buf = jpg_buf;
        jpeg_fb.len = jpg_len;
        jpeg_fb.width = 320;
        jpeg_fb.height = 240;
        jpeg_fb.format = PIXFORMAT_JPEG;
        jpeg_fb.timestamp.tv_sec = 0;
        jpeg_fb.timestamp.tv_usec = 0;

        return &jpeg_fb;
    }

    ESP_LOGW(TAG, "Unexpected pixel format: %d", fb->format);
    esp_camera_fb_return(fb);
    return nullptr;
}

// =============================================================================
// camera_release — Liberar frame buffer
// =============================================================================
void camera_release(camera_fb_t *fb)
{
    if (!fb) return;

    if (!s_jpeg_native && fb->format == PIXFORMAT_JPEG) {
        // Frame convertido: liberar buffer JPEG allocado por frame2jpg
        free(fb->buf);
        fb->buf = nullptr;
        fb->len = 0;
    } else {
        // Frame nativo: devolver al driver
        esp_camera_fb_return(fb);
    }
}

bool camera_is_jpeg_native(void)
{
    return s_jpeg_native;
}
