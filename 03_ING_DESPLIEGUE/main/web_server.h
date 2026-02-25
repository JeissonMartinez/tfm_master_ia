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

/// Broadcast explicit model-switch ACK/status events to WS clients.
/// phase: "started" or "done"
/// ok: true when final switch succeeded (used when phase="done")
/// req_id: optional request identifier echoed from client command
/// error: optional error text for failed switches
void webserver_notify_model_switch(const char* phase,
								   bool ok,
								   int target_idx,
								   int active_idx,
								   const char* model_name,
								   uint32_t req_id = 0,
								   const char* error = nullptr);

/// Stop the server and close all connections.
void webserver_stop();
