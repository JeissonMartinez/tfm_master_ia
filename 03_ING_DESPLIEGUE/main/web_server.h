// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Embedded HTTP + WebSocket server
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "app_config.h"
#include "esp_err.h"

/// Start the HTTP server on WEB_SERVER_PORT (default 80).
///   GET  /       → serves the embedded dashboard (gzip'd HTML)
///   WS   /ws     → real-time JSON metrics stream
esp_err_t webserver_start();

/// Push a metrics snapshot + detection list to all connected WS clients.
/// Serialises to JSON via cJSON.
void webserver_broadcast(const InferenceMetrics& m, const DetectionResult& dets);

/// Stop the server and close all connections.
void webserver_stop();
