// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Shared MJPEG stream buffer
//
// Thread-safe double buffer for JPEG frames: inference_task writes,
// MJPEG handler(s) read.  Uses FreeRTOS EventGroup to signal new frames.
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "app_config.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#include <cstdint>
#include <cstddef>
#include <atomic>

// Event group bit signalling a new frame is ready
#define STREAM_NEW_FRAME_BIT  (1 << 0)

// ─── Inference mode control (shared with main.cpp) ───────────────────────
extern std::atomic<InferMode>  g_infer_mode;
extern std::atomic<bool>       g_infer_trigger;

#if DYNAMIC_THRESHOLDS
// ─── Dynamic thresholds (adjustable from web UI) ─────────────────────────
// These are NOT std::atomic<float> because FreeRTOS on Xtensa doesn't
// guarantee atomic float loads.  We use a simple volatile + relaxed reads
// which is safe for single-writer (WS task) / single-reader (inference task)
// on ESP32-S3 (both cores share coherent RAM).
extern std::atomic<float>  g_conf_threshold;
extern std::atomic<float>  g_iou_threshold;
#endif

// ─── Runtime model switching (shared with main.cpp) ──────────────────────
extern std::atomic<bool>    g_model_switch;   // flag: switch requested
extern std::atomic<uint8_t> g_next_model;     // index into AVAILABLE_MODELS
extern std::atomic<uint32_t> g_model_req_id;  // optional request correlation id

/// Initialise the stream buffer (allocates PSRAM buffer + FreeRTOS primitives).
esp_err_t stream_buf_init();

/// Publish a new JPEG frame (called from inference_task on Core 0).
/// Copies `len` bytes from `jpg` into the shared buffer under a mutex,
/// then signals all waiting stream handlers via the event group.
/// @param jpg   Pointer to JPEG data (will be copied).
/// @param len   Size in bytes.
void stream_buf_publish(const uint8_t* jpg, size_t len);

/// Read the latest JPEG frame into a caller-provided buffer.
/// Thread-safe (takes mutex).  Returns the number of bytes copied.
/// @param out       Destination buffer (must be ≥ STREAM_BUF_MAX).
/// @param out_len   On return, the JPEG size in bytes.
/// @return true if a frame was available, false if buffer was empty.
bool stream_buf_read(uint8_t* out, size_t* out_len);

/// Get the event group handle for waiting on new frames.
EventGroupHandle_t stream_buf_event_group();

/// Deinitialise.
void stream_buf_deinit();
