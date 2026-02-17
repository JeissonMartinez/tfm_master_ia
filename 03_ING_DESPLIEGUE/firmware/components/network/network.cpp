// =============================================================================
// network.cpp — WiFi AP + HTTP Server + WebSocket + Debug Image endpoint
//
// Endpoints:
//   GET /ws           — WebSocket (broadcast JSON de métricas + detecciones)
//   GET /debug/image  — BMP 224×224 del buffer preprocesado (diagnóstico)
//   GET /api/status   — JSON status (modelo activo, memoria, uptime)
//   GET /             — Dashboard HTML (servido por componente dashboard)
//
// WiFi: AP mode, SSID y password de app_config.h
// Mitigación R3: WiFi/HTTP en Core 1, inferencia en Core 0
// =============================================================================
#include "network.h"
#include "app_config.h"
#include "image_proc.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <cstdio>
#include <cstring>

static const char *TAG = "network";

// ---- Estado global ----
static httpd_handle_t s_server = nullptr;
static const int8_t *s_debug_image = nullptr;
static SemaphoreHandle_t s_ws_mutex = nullptr;

// Buffer BMP para debug image (en PSRAM)
static uint8_t *s_bmp_buffer = nullptr;

// =============================================================================
// WiFi Event Handler
// =============================================================================
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t *event = (wifi_event_ap_staconnected_t *)event_data;
        ESP_LOGI(TAG, "Station " MACSTR " joined, AID=%d",
                 MAC2STR(event->mac), event->aid);
    } else if (event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t *event = (wifi_event_ap_stadisconnected_t *)event_data;
        ESP_LOGI(TAG, "Station " MACSTR " left, AID=%d",
                 MAC2STR(event->mac), event->aid);
    }
}

// =============================================================================
// WiFi AP init
// =============================================================================
static esp_err_t wifi_init_ap(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {};
    strlcpy((char *)wifi_config.ap.ssid, WIFI_SSID, sizeof(wifi_config.ap.ssid));
    strlcpy((char *)wifi_config.ap.password, WIFI_PASSWORD, sizeof(wifi_config.ap.password));
    wifi_config.ap.ssid_len = strlen(WIFI_SSID);
    wifi_config.ap.channel = WIFI_CHANNEL;
    wifi_config.ap.max_connection = MAX_STA_CONN;
    wifi_config.ap.authmode = WIFI_AUTH_WPA2_PSK;
    wifi_config.ap.pmf_cfg.required = false;

    if (strlen(WIFI_PASSWORD) < 8) {
        wifi_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "WiFi AP started: SSID=%s, channel=%d", WIFI_SSID, WIFI_CHANNEL);
    return ESP_OK;
}

// =============================================================================
// WebSocket handler
// =============================================================================
static esp_err_t ws_handler(httpd_req_t *req)
{
    if (req->method == HTTP_GET) {
        ESP_LOGI(TAG, "WebSocket connection opened");
        return ESP_OK;
    }

    // Recibir frame (para ping/pong o mensajes del dashboard)
    httpd_ws_frame_t ws_frame = {};
    ws_frame.type = HTTPD_WS_TYPE_TEXT;

    esp_err_t ret = httpd_ws_recv_frame(req, &ws_frame, 0);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "WS recv failed: %s", esp_err_to_name(ret));
        return ret;
    }

    // Podríamos procesar mensajes del dashboard aquí (e.g., cambio de umbral)
    return ESP_OK;
}

// =============================================================================
// Debug image endpoint: GET /debug/image
// Sirve BMP 224×224 del buffer preprocesado — indispensable para diagnóstico
// (Lección clave del piloto: detectar domain shift y errores de preprocessing)
// =============================================================================
static esp_err_t debug_image_handler(httpd_req_t *req)
{
    if (!s_debug_image || !s_bmp_buffer) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "No image available");
        return ESP_FAIL;
    }

    size_t bmp_size = 0;
    preprocess_generate_debug_bmp(s_debug_image, s_bmp_buffer, &bmp_size);

    httpd_resp_set_type(req, "image/bmp");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_send(req, (const char *)s_bmp_buffer, bmp_size);

    return ESP_OK;
}

// =============================================================================
// Status endpoint: GET /api/status
// =============================================================================
static esp_err_t status_handler(httpd_req_t *req)
{
    char buf[256];
    int len = snprintf(buf, sizeof(buf),
        "{\"model\":\"%s\","
        "\"uptime_s\":%lld,"
        "\"free_psram_kb\":%lu,"
        "\"free_internal_kb\":%lu,"
        "\"temperature_c\":0}",
        get_model_name((ModelType)ACTIVE_MODEL),
        (long long)(esp_timer_get_time() / 1000000),
        (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
        (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, buf, len);
    return ESP_OK;
}

// =============================================================================
// HTTP Server init
// =============================================================================
static esp_err_t http_server_init(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = HTTP_PORT;
    config.max_uri_handlers = 8;
    config.stack_size = 8192;
    config.core_id = 1;   // HTTP server en Core 1 (inferencia en Core 0)

    esp_err_t ret = httpd_start(&s_server, &config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "HTTP server start failed: %s", esp_err_to_name(ret));
        return ret;
    }

    // WebSocket endpoint
    httpd_uri_t ws_uri = {
        .uri = "/ws",
        .method = HTTP_GET,
        .handler = ws_handler,
        .user_ctx = nullptr,
        .is_websocket = true,
        .handle_ws_control_frames = false,
        .supported_subprotocol = nullptr,
    };
    httpd_register_uri_handler(s_server, &ws_uri);

    // Debug image endpoint
    httpd_uri_t debug_uri = {
        .uri = "/debug/image",
        .method = HTTP_GET,
        .handler = debug_image_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &debug_uri);

    // Status endpoint
    httpd_uri_t status_uri = {
        .uri = "/api/status",
        .method = HTTP_GET,
        .handler = status_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &status_uri);

    ESP_LOGI(TAG, "HTTP server started on port %d", HTTP_PORT);
    return ESP_OK;
}

// =============================================================================
// network_init
// =============================================================================
esp_err_t network_init(void)
{
    s_ws_mutex = xSemaphoreCreateMutex();
    if (!s_ws_mutex) {
        ESP_LOGE(TAG, "Failed to create WS mutex");
        return ESP_ERR_NO_MEM;
    }

    // Allocar buffer BMP en PSRAM
    s_bmp_buffer = (uint8_t *)heap_caps_malloc(preprocess_bmp_size(), MALLOC_CAP_SPIRAM);
    if (!s_bmp_buffer) {
        ESP_LOGW(TAG, "Could not allocate BMP buffer in PSRAM");
    }

    // WiFi AP
    esp_err_t ret = wifi_init_ap();
    if (ret != ESP_OK) return ret;

    // HTTP Server
    ret = http_server_init();
    if (ret != ESP_OK) return ret;

    return ESP_OK;
}

// =============================================================================
// network_broadcast — Enviar JSON a todos los clientes WebSocket
// =============================================================================
void network_broadcast(const GlobalMetrics *gm, const FrameMetrics *fm,
                       const DetectionResult *result)
{
    if (!s_server || !gm || !fm || !result) return;

    if (xSemaphoreTake(s_ws_mutex, pdMS_TO_TICKS(10)) != pdTRUE) return;

    // Construir JSON
    // Buffer en stack (suficiente para ~10 detecciones)
    char json[1024];
    int pos = 0;

    pos += snprintf(json + pos, sizeof(json) - pos,
        "{\"model\":\"%s\","
        "\"frame\":%lu,"
        "\"fps\":%.1f,"
        "\"capture_ms\":%.0f,"
        "\"preprocess_ms\":%.0f,"
        "\"inference_ms\":%.0f,"
        "\"postprocess_ms\":%.0f,"
        "\"total_ms\":%.0f,"
        "\"free_psram_kb\":%lu,"
        "\"free_internal_kb\":%lu,"
        "\"temperature_c\":%.1f,"
        "\"detections\":[",
        get_model_name(result->model),
        (unsigned long)gm->total_frames,
        gm->avg_fps,
        fm->phase_ms[PHASE_CAPTURE],
        fm->phase_ms[PHASE_PREPROCESS],
        fm->phase_ms[PHASE_INFERENCE],
        fm->phase_ms[PHASE_POSTPROCESS],
        fm->total_ms,
        (unsigned long)gm->free_psram_kb,
        (unsigned long)gm->free_internal_kb,
        gm->temperature_c
    );

    // Detecciones (limitar a las primeras 10 para no desbordar el buffer)
    int max_det = (result->num_detections < 10) ? result->num_detections : 10;
    for (int i = 0; i < max_det && pos < (int)sizeof(json) - 100; i++) {
        const Detection &d = result->detections[i];
        if (i > 0) pos += snprintf(json + pos, sizeof(json) - pos, ",");
        pos += snprintf(json + pos, sizeof(json) - pos,
            "{\"class\":\"%s\",\"score\":%.2f,\"bbox\":[%.3f,%.3f,%.3f,%.3f]}",
            (d.class_id >= 0 && d.class_id < NUM_CLASSES) ? CLASS_NAMES[d.class_id] : "?",
            d.score,
            d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2
        );
    }

    pos += snprintf(json + pos, sizeof(json) - pos, "]}");

    // Enviar a todos los clientes WS conectados
    httpd_ws_frame_t ws_frame = {};
    ws_frame.final = true;
    ws_frame.fragmented = false;
    ws_frame.type = HTTPD_WS_TYPE_TEXT;
    ws_frame.payload = (uint8_t *)json;
    ws_frame.len = pos;

    // Enumerar sockets conectados
    size_t max_fds = WS_MAX_CLIENTS;
    int fds[WS_MAX_CLIENTS];
    if (httpd_get_client_list(s_server, &max_fds, fds) == ESP_OK) {
        for (size_t i = 0; i < max_fds; i++) {
            if (httpd_ws_get_fd_info(s_server, fds[i]) == HTTPD_WS_CLIENT_WEBSOCKET) {
                httpd_ws_send_frame_async(s_server, fds[i], &ws_frame);
            }
        }
    }

    xSemaphoreGive(s_ws_mutex);
}

// =============================================================================
// network_get_server_handle / network_set_debug_image_source
// =============================================================================
void *network_get_server_handle(void)
{
    return (void *)s_server;
}

void network_set_debug_image_source(const int8_t *input_buffer)
{
    s_debug_image = input_buffer;
}
