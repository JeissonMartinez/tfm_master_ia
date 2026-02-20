// =============================================================================
// capture_server.h — WiFi STA + HTTP Server + MJPEG Stream + REST API
// =============================================================================
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Inicializar WiFi en modo STA, mDNS, y servidor HTTP.
 *        Registra todos los endpoints (MJPEG stream, captura, galería, status).
 * @return ESP_OK o código de error
 */
esp_err_t capture_server_init(void);

/**
 * @brief Obtener IP asignada como string.
 *        Solo válido después de capture_server_init() exitoso.
 */
const char *capture_server_get_ip(void);

#ifdef __cplusplus
}
#endif
