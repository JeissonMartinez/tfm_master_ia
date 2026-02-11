// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — WiFi Access-Point manager
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "esp_err.h"

/// Start WiFi in AP mode (SSID / password from app_config.h).
/// Initialises NVS, netif, event loop, and DHCP server.
esp_err_t wifi_init_ap();

/// Stop WiFi AP.
void wifi_deinit();

/// Get the AP's IP address string (e.g. "192.168.4.1").
const char* wifi_get_ip();
