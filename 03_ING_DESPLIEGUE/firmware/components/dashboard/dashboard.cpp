// =============================================================================
// dashboard.cpp — Servir dashboard HTML embebido en flash
// El HTML se incluye vía EMBED_FILES en CMakeLists.txt
// =============================================================================
#include "dashboard.h"
#include "network.h"
#include "esp_log.h"
#include "esp_http_server.h"

static const char *TAG = "dashboard";

// Símbolos generados por EMBED_FILES (linker)
// El nombre deriva del path: "web/index.html" → "_binary_index_html_start/end"
extern const uint8_t index_html_start[] asm("_binary_index_html_start");
extern const uint8_t index_html_end[]   asm("_binary_index_html_end");

static esp_err_t root_handler(httpd_req_t *req)
{
    size_t len = index_html_end - index_html_start;
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_send(req, (const char *)index_html_start, len);
    return ESP_OK;
}

void dashboard_register_handlers(void)
{
    httpd_handle_t server = (httpd_handle_t)network_get_server_handle();
    if (!server) {
        ESP_LOGE(TAG, "HTTP server not available");
        return;
    }

    httpd_uri_t root_uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = root_handler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(server, &root_uri);

    size_t html_size = index_html_end - index_html_start;
    ESP_LOGI(TAG, "Dashboard registered at / (%d bytes)", (int)html_size);
}
