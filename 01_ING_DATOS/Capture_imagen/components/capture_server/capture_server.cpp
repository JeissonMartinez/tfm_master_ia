// =============================================================================
// capture_server.cpp — WiFi STA + HTTP + REST API para captura de imágenes
//
// Endpoints:
//   GET /             — Dashboard HTML embebido
//   GET /api/frame    — Captura un frame JPEG (para preview polling)
//   POST /api/capture         — Capturar 1 foto → SD
//   POST /api/capture/burst   — Capturar N fotos → SD (?count=10)
//   GET /api/photos           — Listar fotos (?page=1&limit=20)
//   GET /api/photos/*         — Descargar foto individual
//   DELETE /api/photos/*      — Eliminar foto individual
//   DELETE /api/photos        — Eliminar todas las fotos
//   GET /api/status           — Estado del sistema
//
// NOTA: Se usa un endpoint de frame único (/api/frame) en lugar de MJPEG
// stream para que el servidor HTTP pueda atender múltiples requests
// (captura, status, etc.) sin bloquearse.
//
// WiFi: Station mode, se conecta al WiFi doméstico.
// mDNS: http://esp32-capture.local/
// =============================================================================
#include "capture_server.h"
#include "app_config.h"
#include "camera_handler.h"
#include "sd_storage.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "mdns.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
// semphr.h removed — no longer needed (MJPEG stream mutex eliminated)
#include <cstdio>
#include <cstring>
#include <cstdlib>

static const char *TAG = "capture_srv";

// ---- WiFi state ----
static EventGroupHandle_t s_wifi_event_group = nullptr;
#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1

static char s_ip_str[20] = "0.0.0.0";
static int s_retry_count = 0;
static httpd_handle_t s_server = nullptr;

// HTML embebido
extern const uint8_t index_html_start[] asm("_binary_index_html_start");
extern const uint8_t index_html_end[]   asm("_binary_index_html_end");

// =============================================================================
// WiFi Event Handler
// =============================================================================
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_count < WIFI_STA_MAX_RETRY) {
            s_retry_count++;
            ESP_LOGW(TAG, "WiFi disconnected, retry %d/%d...",
                     s_retry_count, WIFI_STA_MAX_RETRY);
            vTaskDelay(pdMS_TO_TICKS(WIFI_STA_RETRY_DELAY_MS));
            esp_wifi_connect();
        } else {
            ESP_LOGE(TAG, "WiFi connection failed after %d retries", WIFI_STA_MAX_RETRY);
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        snprintf(s_ip_str, sizeof(s_ip_str), IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "Connected! IP: %s", s_ip_str);
        s_retry_count = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

// =============================================================================
// WiFi STA init
// =============================================================================
static esp_err_t wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t inst_any_id;
    esp_event_handler_instance_t inst_got_ip;

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &inst_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &inst_got_ip));

    wifi_config_t wifi_config = {};
    strlcpy((char *)wifi_config.sta.ssid, WIFI_STA_SSID, sizeof(wifi_config.sta.ssid));
    strlcpy((char *)wifi_config.sta.password, WIFI_STA_PASSWORD, sizeof(wifi_config.sta.password));
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    wifi_config.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to WiFi '%s'...", WIFI_STA_SSID);

    // Esperar conexión o fallo
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE, pdMS_TO_TICKS(30000));

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "WiFi connected to '%s', IP: %s", WIFI_STA_SSID, s_ip_str);
        return ESP_OK;
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGE(TAG, "Failed to connect to '%s'", WIFI_STA_SSID);
        return ESP_FAIL;
    } else {
        ESP_LOGE(TAG, "WiFi connection timeout");
        return ESP_ERR_TIMEOUT;
    }
}

// =============================================================================
// mDNS init
// =============================================================================
static void mdns_init_service(void)
{
    esp_err_t err = mdns_init();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "mDNS init failed: %s", esp_err_to_name(err));
        return;
    }
    mdns_hostname_set(MDNS_HOSTNAME);
    mdns_instance_name_set("ESP32-S3 Image Capture");
    mdns_service_add(NULL, "_http", "_tcp", HTTP_PORT, NULL, 0);
    ESP_LOGI(TAG, "mDNS: http://%s.local/", MDNS_HOSTNAME);
}

// =============================================================================
// Helper: parse query param as int
// =============================================================================
static int parse_query_int(httpd_req_t *req, const char *key, int default_val)
{
    char query[128] = {};
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
        return default_val;
    }
    char val[16] = {};
    if (httpd_query_key_value(query, key, val, sizeof(val)) != ESP_OK) {
        return default_val;
    }
    int v = atoi(val);
    return (v > 0) ? v : default_val;
}

// =============================================================================
// GET / — Dashboard HTML
// =============================================================================
static esp_err_t root_handler(httpd_req_t *req)
{
    size_t len = index_html_end - index_html_start;
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_send(req, (const char *)index_html_start, len);
    return ESP_OK;
}

// =============================================================================
// GET /api/frame — Capturar un frame JPEG (para preview por polling)
// No-bloqueante: retorna inmediatamente después de enviar el frame.
// El HTML llama este endpoint repetidamente con JavaScript.
// =============================================================================
static esp_err_t frame_handler(httpd_req_t *req)
{
    camera_fb_t *fb = camera_capture_jpeg();
    if (!fb) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Frame capture failed");
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    httpd_resp_set_hdr(req, "Pragma", "no-cache");

    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    camera_release(fb);
    return res;
}

// =============================================================================
// POST /api/capture — Capturar 1 foto
// =============================================================================
static esp_err_t capture_handler(httpd_req_t *req)
{
    camera_fb_t *fb = camera_capture_jpeg();
    if (!fb) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Capture failed");
        return ESP_FAIL;
    }

    char filename[32] = {};
    esp_err_t ret = sd_save_jpeg(fb->buf, fb->len, filename);
    size_t photo_size = fb->len;
    camera_release(fb);

    if (ret != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "SD save failed");
        return ESP_FAIL;
    }

    // Respuesta JSON
    char json[256];
    int len = snprintf(json, sizeof(json),
        "{\"ok\":true,\"filename\":\"%s\",\"size\":%d,\"total\":%lu}",
        filename, (int)photo_size, (unsigned long)sd_get_photo_count());

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, len);
    return ESP_OK;
}

// =============================================================================
// POST /api/capture/burst?count=10 — Capturar N fotos
// =============================================================================
static esp_err_t burst_handler(httpd_req_t *req)
{
    int count = parse_query_int(req, "count", 10);
    if (count < 1) count = 1;
    if (count > CAPTURE_MAX_BURST) count = CAPTURE_MAX_BURST;

    ESP_LOGI(TAG, "Burst capture: %d photos", count);

    // Buffer para JSON response
    size_t json_size = 128 + count * 64;
    char *json = (char *)malloc(json_size);
    if (!json) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "No memory");
        return ESP_FAIL;
    }

    int pos = 0;
    pos += snprintf(json + pos, json_size - pos, "{\"ok\":true,\"count\":%d,\"files\":[", count);

    int success = 0;
    for (int i = 0; i < count; i++) {
        camera_fb_t *fb = camera_capture_jpeg();
        if (!fb) {
            ESP_LOGW(TAG, "Burst: frame %d capture failed", i);
            continue;
        }

        char filename[32] = {};
        esp_err_t ret = sd_save_jpeg(fb->buf, fb->len, filename);
        camera_release(fb);

        if (ret == ESP_OK) {
            if (success > 0) {
                pos += snprintf(json + pos, json_size - pos, ",");
            }
            pos += snprintf(json + pos, json_size - pos, "\"%s\"", filename);
            success++;
        }

        // Delay entre capturas para estabilizar exposición
        if (i < count - 1) {
            vTaskDelay(pdMS_TO_TICKS(CAPTURE_BURST_DELAY_MS));
        }
    }

    pos += snprintf(json + pos, json_size - pos,
        "],\"saved\":%d,\"total\":%lu}",
        success, (unsigned long)sd_get_photo_count());

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, pos);
    free(json);

    ESP_LOGI(TAG, "Burst complete: %d/%d saved", success, count);
    return ESP_OK;
}

// =============================================================================
// GET /api/photos?page=1&limit=20 — Listar fotos
// =============================================================================
static esp_err_t list_photos_handler(httpd_req_t *req)
{
    int page = parse_query_int(req, "page", 1);
    int limit = parse_query_int(req, "limit", 20);
    if (limit > 100) limit = 100;
    int offset = (page - 1) * limit;

    // Buffer para JSON (20 archivos × ~50 chars cada uno + overhead)
    size_t json_size = 256 + limit * 64;
    char *json = (char *)malloc(json_size);
    if (!json) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "No memory");
        return ESP_FAIL;
    }

    // Wrapper JSON con metadata de paginación
    char *files_json = json + 128; // Reservar espacio para el header
    size_t files_size = json_size - 128;

    int count = sd_list_files(offset, limit, files_json, files_size);
    uint32_t total = sd_get_photo_count();

    int pos = snprintf(json, 128,
        "{\"page\":%d,\"limit\":%d,\"total\":%lu,\"files\":",
        page, limit, (unsigned long)total);

    // Mover files_json al lugar correcto (justo después del header)
    if (files_json != json + pos) {
        memmove(json + pos, files_json, strlen(files_json) + 1);
    }
    pos += strlen(json + pos);
    pos += snprintf(json + pos, json_size - pos, "}");

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, pos);
    free(json);
    return ESP_OK;
}

// =============================================================================
// GET /api/photos/* — Descargar foto individual
// =============================================================================
static esp_err_t download_photo_handler(httpd_req_t *req)
{
    // Extraer filename del URI: /api/photos/IMG_000001.jpg
    const char *uri = req->uri;
    const char *raw = uri + strlen("/api/photos/");

    // Copiar filename truncando en '?' (query params) o '\0'
    char filename[64];
    size_t i = 0;
    for (; raw[i] && raw[i] != '?' && i < sizeof(filename) - 1; i++) {
        filename[i] = raw[i];
    }
    filename[i] = '\0';

    // Filename vacío? (p.ej. /api/photos/ sin nada)
    if (filename[0] == '\0') {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing filename");
        return ESP_FAIL;
    }

    // Sanitizar: solo letras, números, guión bajo, punto, guión
    for (const char *p = filename; *p; p++) {
        if (!(*p >= 'a' && *p <= 'z') && !(*p >= 'A' && *p <= 'Z') &&
            !(*p >= '0' && *p <= '9') && *p != '_' && *p != '.' && *p != '-') {
            httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid filename");
            return ESP_FAIL;
        }
    }

    uint8_t *data = nullptr;
    size_t len = 0;
    esp_err_t ret = sd_read_file(filename, &data, &len);
    if (ret != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "File not found");
        return ESP_FAIL;
    }

    // Headers para descarga/preview — si viene ?dl=1 forzar descarga
    char query[32] = {};
    bool force_download = false;
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char val[4] = {};
        if (httpd_query_key_value(query, "dl", val, sizeof(val)) == ESP_OK && val[0] == '1') {
            force_download = true;
        }
    }

    char disposition[128];
    if (force_download) {
        snprintf(disposition, sizeof(disposition), "attachment; filename=\"%s\"", filename);
    } else {
        snprintf(disposition, sizeof(disposition), "inline; filename=\"%s\"", filename);
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", disposition);
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");

    // Enviar en chunks para no agotar stack
    size_t sent = 0;
    const size_t chunk_size = 4096;
    while (sent < len) {
        size_t to_send = (len - sent > chunk_size) ? chunk_size : (len - sent);
        if (httpd_resp_send_chunk(req, (const char *)(data + sent), to_send) != ESP_OK) {
            free(data);
            return ESP_FAIL;
        }
        sent += to_send;
    }
    httpd_resp_send_chunk(req, NULL, 0); // Finalizar chunked response

    free(data);
    return ESP_OK;
}

// =============================================================================
// DELETE /api/photos/* — Eliminar foto individual  (via POST con ?action=delete)
// DELETE /api/photos   — Eliminar todas           (via POST con ?action=delete_all)
// Nota: Usamos POST para compatibilidad con browsers (DELETE no siempre soportado)
// =============================================================================
static esp_err_t delete_photo_handler(httpd_req_t *req)
{
    const char *uri = req->uri;

    // DELETE /api/photos (eliminar todas)
    if (strlen(uri) <= strlen("/api/photos/") ||
        strcmp(uri, "/api/photos") == 0 ||
        strcmp(uri, "/api/photos/") == 0) {
        esp_err_t ret = sd_delete_all();
        char json[64];
        int len = snprintf(json, sizeof(json), "{\"ok\":%s}", ret == ESP_OK ? "true" : "false");
        httpd_resp_set_type(req, "application/json");
        httpd_resp_send(req, json, len);
        return ESP_OK;
    }

    // DELETE /api/photos/IMG_000001.jpg
    const char *filename = uri + strlen("/api/photos/");
    esp_err_t ret = sd_delete_file(filename);

    char json[128];
    int len = snprintf(json, sizeof(json),
        "{\"ok\":%s,\"filename\":\"%s\",\"total\":%lu}",
        ret == ESP_OK ? "true" : "false",
        filename,
        (unsigned long)sd_get_photo_count());

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, len);
    return ESP_OK;
}

// =============================================================================
// GET /api/status — Estado del sistema
// =============================================================================
static esp_err_t status_handler(httpd_req_t *req)
{
    sd_stats_t stats = {};
    sd_get_stats(&stats);

    // WiFi RSSI
    wifi_ap_record_t ap_info;
    int rssi = 0;
    if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
        rssi = ap_info.rssi;
    }

    char json[512];
    int len = snprintf(json, sizeof(json),
        "{\"ip\":\"%s\","
        "\"hostname\":\"%s.local\","
        "\"wifi_rssi\":%d,"
        "\"uptime_s\":%lld,"
        "\"sd_total_mb\":%lu,"
        "\"sd_free_mb\":%lu,"
        "\"photo_count\":%lu,"
        "\"next_counter\":%lu,"
        "\"jpeg_native\":%s,"
        "\"free_psram_kb\":%lu,"
        "\"free_internal_kb\":%lu}",
        s_ip_str,
        MDNS_HOSTNAME,
        rssi,
        (long long)(esp_timer_get_time() / 1000000),
        (unsigned long)stats.total_mb,
        (unsigned long)stats.free_mb,
        (unsigned long)stats.photo_count,
        (unsigned long)stats.next_counter,
        camera_is_jpeg_native() ? "true" : "false",
        (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
        (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_send(req, json, len);
    return ESP_OK;
}

// =============================================================================
// POST /api/delete_all — Endpoint simplificado para eliminar todo (desde HTML)
// =============================================================================
static esp_err_t delete_all_handler(httpd_req_t *req)
{
    esp_err_t ret = sd_delete_all();
    char json[64];
    int len = snprintf(json, sizeof(json),
        "{\"ok\":%s}", ret == ESP_OK ? "true" : "false");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, len);
    return ESP_OK;
}

// =============================================================================
// HTTP Server init — Registrar todos los endpoints
// =============================================================================
static esp_err_t http_server_init(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = HTTP_PORT;
    config.max_uri_handlers = HTTP_MAX_URI_HANDLERS;
    config.stack_size = HTTP_SERVER_STACK_SIZE;
    config.core_id = 1;
    config.uri_match_fn = httpd_uri_match_wildcard;  // Para /api/photos/*
    config.lru_purge_enable = true;   // Permite reutilizar sockets cuando se agotan
    config.max_open_sockets = 5;      // Suficiente para preview + API concurrentes

    esp_err_t ret = httpd_start(&s_server, &config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "HTTP server start failed: %s", esp_err_to_name(ret));
        return ret;
    }

    // Dashboard
    httpd_uri_t root_uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = root_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &root_uri);

    // Favicon — responder 204 para evitar que matchee el wildcard /api/photos/*
    httpd_uri_t favicon_uri = {
        .uri = "/favicon.ico",
        .method = HTTP_GET,
        .handler = +[](httpd_req_t *req) -> esp_err_t {
            httpd_resp_set_status(req, "204 No Content");
            httpd_resp_send(req, NULL, 0);
            return ESP_OK;
        },
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &favicon_uri);

    // Frame endpoint (single JPEG, non-blocking)
    httpd_uri_t frame_uri = {
        .uri = "/api/frame",
        .method = HTTP_GET,
        .handler = frame_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &frame_uri);

    // Captura — 1 foto
    httpd_uri_t capture_uri = {
        .uri = "/api/capture",
        .method = HTTP_POST,
        .handler = capture_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &capture_uri);

    // Captura — burst
    httpd_uri_t burst_uri = {
        .uri = "/api/capture/burst",
        .method = HTTP_POST,
        .handler = burst_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &burst_uri);

    // Listar fotos
    httpd_uri_t list_uri = {
        .uri = "/api/photos",
        .method = HTTP_GET,
        .handler = list_photos_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &list_uri);

    // Descargar foto individual (wildcard)
    httpd_uri_t download_uri = {
        .uri = "/api/photos/*",
        .method = HTTP_GET,
        .handler = download_photo_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &download_uri);

    // Eliminar foto individual (wildcard) — usando POST para compatibilidad
    httpd_uri_t delete_uri = {
        .uri = "/api/photos/*",
        .method = HTTP_DELETE,
        .handler = delete_photo_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &delete_uri);

    // Eliminar todas las fotos
    httpd_uri_t delete_all_uri = {
        .uri = "/api/delete_all",
        .method = HTTP_POST,
        .handler = delete_all_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &delete_all_uri);

    // Status
    httpd_uri_t status_uri = {
        .uri = "/api/status",
        .method = HTTP_GET,
        .handler = status_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &status_uri);

    ESP_LOGI(TAG, "HTTP server started on port %d with %d endpoints",
             HTTP_PORT, 10);
    return ESP_OK;
}

// =============================================================================
// capture_server_init
// =============================================================================
esp_err_t capture_server_init(void)
{
    // WiFi STA
    esp_err_t ret = wifi_init_sta();
    if (ret != ESP_OK) return ret;

    // mDNS
    mdns_init_service();

    // HTTP Server
    ret = http_server_init();
    if (ret != ESP_OK) return ret;

    return ESP_OK;
}

const char *capture_server_get_ip(void)
{
    return s_ip_str;
}
