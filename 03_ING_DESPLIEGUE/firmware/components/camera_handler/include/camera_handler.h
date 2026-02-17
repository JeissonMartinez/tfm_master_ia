// =============================================================================
// camera_handler.h — Driver de cámara OV5640 para Freenove ESP32-S3 CAM
// Configuración validada en piloto (docs/Configuracion_ESP32-S3.md)
// =============================================================================
#pragma once

#include "esp_err.h"
#include "esp_camera.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Inicializar cámara OV5640 con configuración probada.
 *        RGB565, QVGA (320×240), doble buffer en PSRAM.
 * @return ESP_OK o código de error
 */
esp_err_t camera_init(void);

/**
 * @brief Capturar un frame (grab latest).
 * @return Puntero al frame buffer, o NULL si falla.
 *         DEBE llamarse camera_release() después de usar.
 */
camera_fb_t *camera_capture(void);

/**
 * @brief Liberar frame buffer de vuelta al driver.
 */
void camera_release(camera_fb_t *fb);

/**
 * @brief Obtener puntero al último frame capturado (para debug visual).
 *        No tomar ownership — solo lectura.
 */
const uint8_t *camera_get_last_frame(size_t *out_len);

#ifdef __cplusplus
}
#endif
