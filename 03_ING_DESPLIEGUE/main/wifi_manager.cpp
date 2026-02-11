// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — WiFi AP implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "wifi_manager.h"
#include "app_config.h"

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"

#include <cstring>

static const char* TAG = "wifi_ap";

static esp_netif_t* s_netif = nullptr;

// ─── Event handler ───────────────────────────────────────────────────────
static void wifi_event_handler(void* arg, esp_event_base_t base,
                               int32_t id, void* data)
{
    if (base == WIFI_EVENT) {
        switch (id) {
        case WIFI_EVENT_AP_STACONNECTED: {
            auto* ev = static_cast<wifi_event_ap_staconnected_t*>(data);
            ESP_LOGI(TAG, "Estación conectada — AID=%d", ev->aid);
            break;
        }
        case WIFI_EVENT_AP_STADISCONNECTED: {
            auto* ev = static_cast<wifi_event_ap_stadisconnected_t*>(data);
            ESP_LOGI(TAG, "Estación desconectada — AID=%d", ev->aid);
            break;
        }
        default:
            break;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
esp_err_t wifi_init_ap() {
    // NVS (requerido por WiFi driver)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Netif + event loop
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_netif = esp_netif_create_default_wifi_ap();

    // WiFi init
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, nullptr, nullptr));

    // AP config
    wifi_config_t ap_config = {};
    std::strncpy(reinterpret_cast<char*>(ap_config.ap.ssid),
                 WIFI_AP_SSID, sizeof(ap_config.ap.ssid) - 1);
    std::strncpy(reinterpret_cast<char*>(ap_config.ap.password),
                 WIFI_AP_PASS, sizeof(ap_config.ap.password) - 1);
    ap_config.ap.ssid_len       = std::strlen(WIFI_AP_SSID);
    ap_config.ap.channel        = WIFI_AP_CHANNEL;
    ap_config.ap.max_connection = WIFI_AP_MAX_CONN;
    ap_config.ap.authmode       = WIFI_AUTH_WPA2_PSK;
    ap_config.ap.pmf_cfg.required = false;

    if (std::strlen(WIFI_AP_PASS) < 8) {
        ap_config.ap.authmode = WIFI_AUTH_OPEN;
        ESP_LOGW(TAG, "Contraseña < 8 chars → AP abierto");
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "═══════════════════════════════════════════════");
    ESP_LOGI(TAG, "✅ WiFi AP iniciado");
    ESP_LOGI(TAG, "   SSID:     %s", WIFI_AP_SSID);
    ESP_LOGI(TAG, "   Password: %s", WIFI_AP_PASS);
    ESP_LOGI(TAG, "   IP:       %s", wifi_get_ip());
    ESP_LOGI(TAG, "   Canal:    %d", WIFI_AP_CHANNEL);
    ESP_LOGI(TAG, "═══════════════════════════════════════════════");

    return ESP_OK;
}

void wifi_deinit() {
    esp_wifi_stop();
    esp_wifi_deinit();
    if (s_netif) {
        esp_netif_destroy_default_wifi(s_netif);
        s_netif = nullptr;
    }
    ESP_LOGI(TAG, "WiFi AP detenido");
}

const char* wifi_get_ip() {
    // Default AP IP on ESP-IDF
    return "192.168.4.1";
}
