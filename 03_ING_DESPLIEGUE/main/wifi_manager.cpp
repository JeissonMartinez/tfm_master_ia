// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — WiFi STA implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "wifi_manager.h"
#include "app_config.h"

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#include <cstring>
#include <cstdio>

static const char* TAG = "wifi_sta";

static esp_netif_t* s_netif = nullptr;
static EventGroupHandle_t s_wifi_event_group = nullptr;
static int s_retry_num = 0;
static char s_ip_str[16] = "0.0.0.0";

#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1

// ─── Event handler ───────────────────────────────────────────────────────
static void wifi_event_handler(void* arg, esp_event_base_t base,
                               int32_t id, void* data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < WIFI_MAX_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGW(TAG, "Reintentando conexión (%d/%d)...", s_retry_num, WIFI_MAX_RETRY);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
            ESP_LOGE(TAG, "Conexión fallida tras %d intentos", WIFI_MAX_RETRY);
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        auto* ev = static_cast<ip_event_got_ip_t*>(data);
        snprintf(s_ip_str, sizeof(s_ip_str), IPSTR, IP2STR(&ev->ip_info.ip));
        ESP_LOGI(TAG, "IP obtenida: %s", s_ip_str);
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
esp_err_t wifi_init_sta() {
    // NVS (requerido por WiFi driver)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    s_wifi_event_group = xEventGroupCreate();

    // Netif + event loop
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_netif = esp_netif_create_default_wifi_sta();

    // WiFi init
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, nullptr, nullptr));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, nullptr, nullptr));

    // STA config
    wifi_config_t sta_config = {};
    std::strncpy(reinterpret_cast<char*>(sta_config.sta.ssid),
                 WIFI_STA_SSID, sizeof(sta_config.sta.ssid) - 1);
    std::strncpy(reinterpret_cast<char*>(sta_config.sta.password),
                 WIFI_STA_PASS, sizeof(sta_config.sta.password) - 1);
    sta_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    sta_config.sta.pmf_cfg.capable    = true;
    sta_config.sta.pmf_cfg.required   = false;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Conectando a '%s'...", WIFI_STA_SSID);

    // Esperar conexión o fallo
    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE, portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "═══════════════════════════════════════════════");
        ESP_LOGI(TAG, "✅ WiFi STA conectado");
        ESP_LOGI(TAG, "   SSID: %s", WIFI_STA_SSID);
        ESP_LOGI(TAG, "   IP:   %s", s_ip_str);
        ESP_LOGI(TAG, "═══════════════════════════════════════════════");
        return ESP_OK;
    }

    ESP_LOGE(TAG, "No se pudo conectar a '%s'", WIFI_STA_SSID);
    return ESP_FAIL;
}

void wifi_deinit() {
    esp_wifi_stop();
    esp_wifi_deinit();
    if (s_netif) {
        esp_netif_destroy_default_wifi(s_netif);
        s_netif = nullptr;
    }
    if (s_wifi_event_group) {
        vEventGroupDelete(s_wifi_event_group);
        s_wifi_event_group = nullptr;
    }
    ESP_LOGI(TAG, "WiFi STA detenido");
}

const char* wifi_get_ip() {
    return s_ip_str;
}
