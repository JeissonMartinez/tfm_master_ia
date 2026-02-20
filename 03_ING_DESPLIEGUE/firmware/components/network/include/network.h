// =============================================================================
// network.h — WiFi AP + HTTP Server + WebSocket broadcast
// =============================================================================
#pragma once

#include "esp_err.h"
#include "metrics.h"
#include "postprocess.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Inicializar WiFi en modo AP y servidor HTTP con WebSocket.
 *        SSID/password desde app_config.h.
 * @return ESP_OK o código de error
 */
esp_err_t network_init(void);

/**
 * @brief Broadcast de métricas y detecciones a todos los clientes WebSocket.
 *        Construye JSON y envía frame WS.
 *        Safe to call from any task — usa mutex interno.
 */
void network_broadcast(const GlobalMetrics *gm, const FrameMetrics *fm,
                       const DetectionResult *result);

/**
 * @brief Obtener handle del servidor HTTP (para registrar handlers adicionales).
 */
void *network_get_server_handle(void);

/**
 * @brief Obtener puntero al buffer de imagen preprocesado para debug.
 *        Se setea desde main antes de iniciar inferencia.
 */
void network_set_debug_image_source(const int8_t *input_buffer);

#ifdef __cplusplus
}
#endif
