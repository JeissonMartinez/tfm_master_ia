// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — HTTP + WebSocket server implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "web_server.h"
#include "dashboard.h"         // gzip'd HTML blob
#include "stream_buf.h"        // shared JPEG buffer + infer mode globals

#include "esp_log.h"
#include "esp_http_server.h"
#include "esp_heap_caps.h"
#include "cJSON.h"

#include <cstring>
#include <algorithm>
#include <unistd.h>

static const char* TAG = "webserv";

static httpd_handle_t s_server = nullptr;        // port 80 — root + WS
static httpd_handle_t s_stream_server = nullptr;  // port 81 — MJPEG stream

// ─── WebSocket file descriptors ──────────────────────────────────────────
static int  s_ws_fds[WS_MAX_CLIENTS] = {};
static int  s_ws_count = 0;

static void ws_remove(int fd);

static void ws_broadcast_text(const char* text, size_t len) {
    if (!s_server || s_ws_count == 0 || !text || len == 0) return;

    httpd_ws_frame_t frame = {};
    frame.type    = HTTPD_WS_TYPE_TEXT;
    frame.payload = reinterpret_cast<uint8_t*>(const_cast<char*>(text));
    frame.len     = len;

    for (int i = 0; i < s_ws_count; /* no increment */) {
        esp_err_t ret = httpd_ws_send_frame_async(s_server, s_ws_fds[i], &frame);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "WS event error fd=%d: %s, eliminando",
                     s_ws_fds[i], esp_err_to_name(ret));
            ws_remove(s_ws_fds[i]);
        } else {
            ++i;
        }
    }
}

static void ws_add(int fd) {
    if (s_ws_count < WS_MAX_CLIENTS) {
        s_ws_fds[s_ws_count++] = fd;
        ESP_LOGI(TAG, "WS cliente conectado fd=%d (total=%d)", fd, s_ws_count);
    } else {
        ESP_LOGW(TAG, "WS máximo de clientes alcanzado (%d), rechazando fd=%d",
                 WS_MAX_CLIENTS, fd);
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

// ─── GET /stream — MJPEG live stream ─────────────────────────────────────
static const char* MJPEG_BOUNDARY = "frameboundary";
static const char* MJPEG_CT = "multipart/x-mixed-replace;boundary=frameboundary";

static esp_err_t stream_handler(httpd_req_t* req) {
    ESP_LOGI(TAG, "Stream MJPEG cliente conectado");

    httpd_resp_set_type(req, MJPEG_CT);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");

    // Allocate a local read buffer from PSRAM to avoid holding the shared
    // mutex during the (slow) HTTP send.
    uint8_t* local_buf = static_cast<uint8_t*>(
        heap_caps_malloc(STREAM_BUF_MAX, MALLOC_CAP_SPIRAM));
    if (!local_buf) {
        ESP_LOGE(TAG, "No se pudo asignar buffer local para stream");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    EventGroupHandle_t evt = stream_buf_event_group();
    char part_header[128];

    while (true) {
        // Wait for a new frame signal (timeout 1 s — keeps connection alive)
        EventBits_t bits = xEventGroupWaitBits(
            evt, STREAM_NEW_FRAME_BIT, pdTRUE, pdFALSE, pdMS_TO_TICKS(1000));

        if (!(bits & STREAM_NEW_FRAME_BIT)) {
            // Timeout — send empty boundary to keep connection alive
            continue;
        }

        // Copy latest frame under short mutex
        size_t jpg_len = 0;
        if (!stream_buf_read(local_buf, &jpg_len) || jpg_len == 0) continue;

        // Build multipart header
        int hdr_len = snprintf(part_header, sizeof(part_header),
            "\r\n--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
            MJPEG_BOUNDARY, jpg_len);

        // Send header + JPEG data
        esp_err_t ret = httpd_resp_send_chunk(req, part_header, hdr_len);
        if (ret != ESP_OK) break;

        ret = httpd_resp_send_chunk(req, reinterpret_cast<const char*>(local_buf),
                                    jpg_len);
        if (ret != ESP_OK) break;
    }

    heap_caps_free(local_buf);
    ESP_LOGI(TAG, "Stream MJPEG cliente desconectado");
    return ESP_OK;
}

// ─── WebSocket /ws ───────────────────────────────────────────────────────
static void handle_ws_command(const char* payload, size_t len) {
    ESP_LOGI(TAG, "WS cmd recibido: %.*s", (int)len, payload);

    cJSON* root = cJSON_ParseWithLength(payload, len);
    if (!root) {
        ESP_LOGW(TAG, "WS cmd JSON parse failed");
        return;
    }

    const cJSON* cmd = cJSON_GetObjectItem(root, "cmd");
    if (!cJSON_IsString(cmd)) {
        ESP_LOGW(TAG, "WS cmd sin campo 'cmd'");
        cJSON_Delete(root);
        return;
    }

    if (std::strcmp(cmd->valuestring, "mode") == 0) {
        const cJSON* val = cJSON_GetObjectItem(root, "value");
        if (cJSON_IsString(val)) {
            if (std::strcmp(val->valuestring, "continuous") == 0) {
                g_infer_mode.store(InferMode::CONTINUOUS, std::memory_order_relaxed);
                ESP_LOGI(TAG, "Modo → CONTINUOUS");
            } else if (std::strcmp(val->valuestring, "ondemand") == 0) {
                g_infer_mode.store(InferMode::ON_DEMAND, std::memory_order_relaxed);
                ESP_LOGI(TAG, "Modo → ON_DEMAND");
            } else {
                ESP_LOGW(TAG, "Modo desconocido: %s", val->valuestring);
            }
        }
    } else if (std::strcmp(cmd->valuestring, "infer") == 0) {
        g_infer_trigger.store(true, std::memory_order_relaxed);
        ESP_LOGI(TAG, "Trigger de inferencia recibido");
#if DYNAMIC_THRESHOLDS
    } else if (std::strcmp(cmd->valuestring, "threshold") == 0) {
        const cJSON* conf_val = cJSON_GetObjectItem(root, "conf");
        const cJSON* iou_val  = cJSON_GetObjectItem(root, "iou");
        if (cJSON_IsNumber(conf_val)) {
            float c = static_cast<float>(conf_val->valuedouble);
            if (c >= 0.05f && c <= 0.95f) {
                g_conf_threshold.store(c, std::memory_order_relaxed);
                ESP_LOGI(TAG, "conf_threshold → %.2f", c);
            }
        }
        if (cJSON_IsNumber(iou_val)) {
            float u = static_cast<float>(iou_val->valuedouble);
            if (u >= 0.05f && u <= 0.95f) {
                g_iou_threshold.store(u, std::memory_order_relaxed);
                ESP_LOGI(TAG, "iou_threshold → %.2f", u);
            }
        }
#endif
    } else if (std::strcmp(cmd->valuestring, "model") == 0) {
        const cJSON* idx = cJSON_GetObjectItem(root, "index");
        uint32_t req_id = 0;
        const cJSON* req = cJSON_GetObjectItem(root, "req_id");
        if (cJSON_IsNumber(req) && req->valuedouble >= 0) {
            req_id = static_cast<uint32_t>(req->valuedouble);
        }
        if (cJSON_IsNumber(idx)) {
            int i = idx->valueint;
            if (i >= 0 && i < NUM_AVAILABLE_MODELS) {
                g_next_model.store(static_cast<uint8_t>(i), std::memory_order_relaxed);
                g_model_req_id.store(req_id, std::memory_order_relaxed);
                g_model_switch.store(true, std::memory_order_release);
                ESP_LOGI(TAG, "Model switch solicitado → modelo %d", i);
                webserver_notify_model_switch("started", false, i, -1, nullptr, req_id, nullptr);
            } else {
                ESP_LOGW(TAG, "Model index fuera de rango: %d", i);
                webserver_notify_model_switch("done", false, i, -1, nullptr,
                                              req_id, "index_out_of_range");
            }
        }
    } else {
        ESP_LOGW(TAG, "WS cmd desconocido: %s", cmd->valuestring);
    }

    cJSON_Delete(root);
}

void webserver_notify_model_switch(const char* phase,
                                   bool ok,
                                   int target_idx,
                                   int active_idx,
                                   const char* model_name,
                                   uint32_t req_id,
                                   const char* error)
{
    cJSON* root = cJSON_CreateObject();
    if (!root) return;

    cJSON_AddStringToObject(root, "evt", "model_switch");
    cJSON_AddStringToObject(root, "phase", phase ? phase : "done");
    cJSON_AddBoolToObject(root, "ok", ok);
    cJSON_AddNumberToObject(root, "target_idx", target_idx);
    cJSON_AddNumberToObject(root, "active_idx", active_idx);
    if (model_name) cJSON_AddStringToObject(root, "model", model_name);
    if (req_id > 0) cJSON_AddNumberToObject(root, "req_id", req_id);
    if (error) cJSON_AddStringToObject(root, "error", error);

    char* json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!json_str) return;

    ws_broadcast_text(json_str, std::strlen(json_str));
    free(json_str);
}

static esp_err_t ws_handler(httpd_req_t* req) {
    if (req->method == HTTP_GET) {
        // Handshake — register fd
        ws_add(httpd_req_to_sockfd(req));
        return ESP_OK;
    }

    // Receive frame (ping/pong, text commands, or close)
    httpd_ws_frame_t frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.type = HTTPD_WS_TYPE_TEXT;

    // First call with 0 length to get the frame info
    esp_err_t ret = httpd_ws_recv_frame(req, &frame, 0);
    if (ret != ESP_OK) return ret;

    if (frame.type == HTTPD_WS_TYPE_CLOSE) {
        ws_remove(httpd_req_to_sockfd(req));
        return ESP_OK;
    }

    // Read the payload if present
    if (frame.len > 0 && frame.len < 256) {
        uint8_t buf[256] = {};
        frame.payload = buf;
        ret = httpd_ws_recv_frame(req, &frame, frame.len);
        if (ret == ESP_OK && frame.type == HTTPD_WS_TYPE_TEXT) {
            handle_ws_command(reinterpret_cast<const char*>(buf), frame.len);
        }
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
    // ─── Server 1: Port 80 — Dashboard + WebSocket ───────────────────
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = WEB_SERVER_PORT;
    config.ctrl_port   = 32768;  // control port for server 1
    config.max_open_sockets = WS_MAX_CLIENTS + 2;
    config.close_fn = ws_close_cb;
    config.lru_purge_enable = true;
    config.stack_size = 6144;

    esp_err_t ret = httpd_start(&s_server, &config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Error al iniciar HTTP server p%d: %s",
                 WEB_SERVER_PORT, esp_err_to_name(ret));
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

    ESP_LOGI(TAG, "✅ HTTP server en puerto %d (dashboard + WS)", WEB_SERVER_PORT);

    // ─── Server 2: Port 81 — MJPEG stream (blocking, separate task) ──
    httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
    stream_config.server_port = STREAM_SERVER_PORT;
    stream_config.ctrl_port   = 32769;  // control port for server 2 (must differ)
    stream_config.max_open_sockets = STREAM_MAX_CLIENTS + 1;
    stream_config.lru_purge_enable = true;
    stream_config.stack_size = 8192;  // Stream handler needs more stack

    ret = httpd_start(&s_stream_server, &stream_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Error al iniciar stream server p%d: %s",
                 STREAM_SERVER_PORT, esp_err_to_name(ret));
        return ret;
    }

    // Register GET /stream (MJPEG) on the stream server
    static const httpd_uri_t stream_uri = {
        .uri      = "/stream",
        .method   = HTTP_GET,
        .handler  = stream_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(s_stream_server, &stream_uri);

    ESP_LOGI(TAG, "✅ MJPEG stream en puerto %d", STREAM_SERVER_PORT);
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

    // ─── Inference mode ──────────────────────────────────────────────
    cJSON_AddStringToObject(root, "mode",
        g_infer_mode.load(std::memory_order_relaxed) == InferMode::CONTINUOUS
        ? "continuous" : "ondemand");

    // ─── Active model ────────────────────────────────────────────────
    if (m.model_name) {
        cJSON_AddStringToObject(root, "model", m.model_name);
    }
    cJSON_AddNumberToObject(root, "model_idx", m.model_idx);

#if DYNAMIC_THRESHOLDS
    // ─── Current thresholds (for slider sync) ────────────────────────
    cJSON_AddNumberToObject(root, "conf_thr",
        g_conf_threshold.load(std::memory_order_relaxed));
    cJSON_AddNumberToObject(root, "iou_thr",
        g_iou_threshold.load(std::memory_order_relaxed));
#endif

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
    int sent = 0;
    for (int i = 0; i < s_ws_count; /* no increment */) {
        esp_err_t ret = httpd_ws_send_frame_async(s_server, s_ws_fds[i], &frame);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "WS broadcast error fd=%d: %s, eliminando",
                     s_ws_fds[i], esp_err_to_name(ret));
            ws_remove(s_ws_fds[i]);
            // Don't increment — ws_remove shifts elements
        } else {
            ++sent;
            ++i;
        }
    }

    // Log broadcast status periodically (every 10th frame)
    if (m.frame_id % 10 == 1) {
        ESP_LOGI(TAG, "WS broadcast #%lu → %d/%d clientes (%zu bytes)",
                 m.frame_id, sent, s_ws_count, len);
    }

    free(json_str);
}

void webserver_send_capture(const uint8_t* jpg, size_t len) {
    if (s_ws_count == 0 || !jpg || len == 0) return;

    httpd_ws_frame_t frame = {};
    frame.type    = HTTPD_WS_TYPE_BINARY;
    frame.payload = const_cast<uint8_t*>(jpg);
    frame.len     = len;

    for (int i = 0; i < s_ws_count; /* no increment */) {
        esp_err_t ret = httpd_ws_send_frame_async(s_server, s_ws_fds[i], &frame);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "Error enviando captura a fd=%d, eliminando", s_ws_fds[i]);
            ws_remove(s_ws_fds[i]);
        } else {
            ++i;
        }
    }
}

void webserver_stop() {
    if (s_stream_server) {
        httpd_stop(s_stream_server);
        s_stream_server = nullptr;
    }
    if (s_server) {
        httpd_stop(s_server);
        s_server = nullptr;
    }
    s_ws_count = 0;
    ESP_LOGI(TAG, "HTTP servers detenidos");
}
