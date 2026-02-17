// =============================================================================
// camera_handler.h — Driver de cámara OV5640 para captura de imágenes
// Pin mapping idéntico al firmware de despliegue
// =============================================================================
#pragma once

#include "esp_err.h"
#include "esp_camera.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Inicializar cámara OV5640 en modo JPEG para captura de dataset.
 *        Intenta JPEG nativo primero; fallback a RGB565 si falla.
 * @return ESP_OK o código de error
 */
esp_err_t camera_init(void);

/**
 * @brief Capturar un frame JPEG.
 * @return Puntero al frame buffer (JPEG), o NULL si falla.
 *         DEBE llamarse camera_release() después de usar.
 */
camera_fb_t *camera_capture_jpeg(void);

/**
 * @brief Liberar frame buffer de vuelta al driver.
 */
void camera_release(camera_fb_t *fb);

/**
 * @brief Obtener si la cámara está usando JPEG nativo o fallback RGB565.
 * @return true si JPEG nativo, false si fallback
 */
bool camera_is_jpeg_native(void);

#ifdef __cplusplus
}
#endif
