// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — HTTP + WebSocket server implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "web_server.h"
#include "dashboard.h"         // gzip'd HTML blob

#include "esp_log.h"
#include "esp_http_server.h"
#include "cJSON.h"

#include <cstring>
#include <algorithm>
#include <unistd.h>

static const char* TAG = "webserv";

static httpd_handle_t s_server = nullptr;

// ─── WebSocket file descriptors ──────────────────────────────────────────
static int  s_ws_fds[WS_MAX_CLIENTS] = {};
static int  s_ws_count = 0;

static void ws_add(int fd) {
    if (s_ws_count < WS_MAX_CLIENTS) {
        s_ws_fds[s_ws_count++] = fd;
        ESP_LOGI(TAG, "WS cliente conectado fd=%d (total=%d)", fd, s_ws_count);
    }
}

static void ws_remove(int fd) {
    for (int i = 0; i < s_ws_count; ++i) {
        if (s_ws_fds[i] == fd) {
            s_ws_fds[i] = s_ws_fds[--s_ws_count];
            ESP_LOGI(TAG, "WS cliente desconectado fd=%d (total=%d)", fd, s_ws_count);
            return;
        }
    }
}

// ─── GET / — serve dashboard ─────────────────────────────────────────────
static esp_err_t root_get_handler(httpd_req_t* req) {
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
    httpd_resp_send(req, reinterpret_cast<const char*>(dashboard_html_gz),
                    dashboard_html_gz_len);
    return ESP_OK;
}

// ─── WebSocket /ws ───────────────────────────────────────────────────────
static esp_err_t ws_handler(httpd_req_t* req) {
    if (req->method == HTTP_GET) {
        // Handshake — register fd
        ws_add(httpd_req_to_sockfd(req));
        return ESP_OK;
    }

    // Receive frame (ping/pong or text — we don't expect incoming data)
    httpd_ws_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.type = HTTPD_WS_TYPE_TEXT;

    esp_err_t ret = httpd_ws_recv_frame(req, &frame, 0);
    if (ret != ESP_OK) return ret;

    if (frame.type == HTTPD_WS_TYPE_CLOSE) {
        ws_remove(httpd_req_to_sockfd(req));
    }

    return ESP_OK;
}

// ─── Close callback (detect disconnect) ──────────────────────────────────
static void ws_close_cb(httpd_handle_t hd, int sockfd) {
    ws_remove(sockfd);
    close(sockfd);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════════

esp_err_t webserver_start() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = WEB_SERVER_PORT;
    config.max_open_sockets = WS_MAX_CLIENTS + 2;  // WS + HTTP
    config.close_fn = ws_close_cb;
    config.lru_purge_enable = true;

    esp_err_t ret = httpd_start(&s_server, &config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Error al iniciar HTTP server: %s", esp_err_to_name(ret));
        return ret;
    }

    // Register GET /
    static const httpd_uri_t root_uri = {
        .uri      = "/",
        .method   = HTTP_GET,
        .handler  = root_get_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_server, &root_uri);

    // Register WS /ws
    static const httpd_uri_t ws_uri = {
        .uri          = "/ws",
        .method       = HTTP_GET,
        .handler      = ws_handler,
        .user_ctx     = nullptr,
        .is_websocket = true,
    };
    httpd_register_uri_handler(s_server, &ws_uri);

    ESP_LOGI(TAG, "✅ HTTP server en puerto %d", WEB_SERVER_PORT);
    return ESP_OK;
}

void webserver_broadcast(const InferenceMetrics& m, const DetectionResult& dets) {
    if (s_ws_count == 0) return;

    cJSON* root = cJSON_CreateObject();
    if (!root) return;

    // ─── Metrics ─────────────────────────────────────────────────────
    cJSON_AddNumberToObject(root, "frame",       m.frame_id);
    cJSON_AddNumberToObject(root, "pre_ms",      m.preprocess_ms);
    cJSON_AddNumberToObject(root, "inf_ms",      m.inference_ms);
    cJSON_AddNumberToObject(root, "post_ms",     m.postprocess_ms);
    cJSON_AddNumberToObject(root, "total_ms",    m.total_ms);
    cJSON_AddNumberToObject(root, "fps",         m.fps);
    cJSON_AddNumberToObject(root, "ema_fps",     m.ema_fps);
    cJSON_AddNumberToObject(root, "ema_inf_ms",  m.ema_inference_ms);
    cJSON_AddNumberToObject(root, "heap_int_kb", m.heap_internal_free / 1024);
    cJSON_AddNumberToObject(root, "psram_kb",    m.psram_free / 1024);
    cJSON_AddNumberToObject(root, "arena_kb",    m.arena_used / 1024);
    cJSON_AddNumberToObject(root, "temp_c",      m.cpu_temp_c);

    // ─── Detections ──────────────────────────────────────────────────
    cJSON* arr = cJSON_AddArrayToObject(root, "dets");
    for (int i = 0; i < dets.count; ++i) {
        const Detection& d = dets.detections[i];
        cJSON* obj = cJSON_CreateObject();
        cJSON_AddStringToObject(obj, "cls", d.class_name());
        cJSON_AddNumberToObject(obj, "cf",  d.confidence);
        cJSON_AddNumberToObject(obj, "x1",  d.x1);
        cJSON_AddNumberToObject(obj, "y1",  d.y1);
        cJSON_AddNumberToObject(obj, "x2",  d.x2);
        cJSON_AddNumberToObject(obj, "y2",  d.y2);
        cJSON_AddItemToArray(arr, obj);
    }

    char* json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json_str) return;

    size_t len = std::strlen(json_str);

    httpd_ws_frame_t frame = {};
    frame.type    = HTTPD_WS_TYPE_TEXT;
    frame.payload = reinterpret_cast<uint8_t*>(json_str);
    frame.len     = len;

    // Send to all connected WS clients
    for (int i = 0; i < s_ws_count; /* no increment */) {
        esp_err_t ret = httpd_ws_send_frame_async(s_server, s_ws_fds[i], &frame);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "Error enviando a fd=%d, eliminando", s_ws_fds[i]);
            ws_remove(s_ws_fds[i]);
            // Don't increment — ws_remove shifts elements
        } else {
            ++i;
        }
    }

    free(json_str);
}

void webserver_stop() {
    if (s_server) {
        httpd_stop(s_server);
        s_server = nullptr;
    }
    s_ws_count = 0;
    ESP_LOGI(TAG, "HTTP server detenido");
}
