// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — WiFi Station manager
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "esp_err.h"

/// Start WiFi in STA mode and connect to the configured network.
/// Initialises NVS, netif, event loop. Blocks until IP obtained or max retries.
esp_err_t wifi_init_sta();

/// Stop WiFi STA.
void wifi_deinit();

/// Get the STA IP address string (e.g. "192.168.1.55").
const char* wifi_get_ip();
