// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Runtime metrics collection
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "esp_err.h"
#include "app_config.h"

/// Initialise metrics (timers, baseline heap snapshot).
esp_err_t metrics_init();

/// Begin a new frame measurement cycle. Call before capture.
void metrics_frame_begin();

/// Mark the end of preprocessing.  Called after image_preprocess().
void metrics_preprocess_end();

/// Mark the end of inference.  Called after engine->invoke().
void metrics_inference_end();

/// Mark the end of postprocessing and finalise the frame.
/// Updates EMA values and heap stats.
void metrics_postprocess_end(int n_detections);

/// Get the latest completed metrics snapshot (thread-safe copy).
InferenceMetrics metrics_get();

/// Get the running frame counter.
uint32_t metrics_frame_id();

/// Deinitialise.
void metrics_deinit();
