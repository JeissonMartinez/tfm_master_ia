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
