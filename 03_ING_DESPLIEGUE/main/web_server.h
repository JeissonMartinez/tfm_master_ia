// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Embedded HTTP + WebSocket server
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "app_config.h"
#include "esp_err.h"

/// Start the HTTP server on WEB_SERVER_PORT (default 80).
///   GET  /        → serves the embedded dashboard (gzip'd HTML)
///   GET  /stream  → MJPEG live stream (multipart/x-mixed-replace)
///   WS   /ws      → real-time JSON metrics + bidirectional commands
esp_err_t webserver_start();

/// Push a metrics snapshot + detection list to all connected WS clients.
/// Serialises to JSON via cJSON.
void webserver_broadcast(const InferenceMetrics& m, const DetectionResult& dets);

/// Push a captured JPEG frame to all WS clients as a binary message.
/// Used in on-demand mode after inference to show the analysed frame.
void webserver_send_capture(const uint8_t* jpg, size_t len);

/// Stop the server and close all connections.
void webserver_stop();
