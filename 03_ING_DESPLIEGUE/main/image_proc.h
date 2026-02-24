// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Image preprocessing
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstdint>
#include "esp_camera.h"

/// Inicializa el buffer de preprocesamiento en PSRAM.
/// @return ESP_OK si se alocó el buffer correctamente.
esp_err_t image_proc_init();

/// Preprocesa un frame de cámara para inferencia.
///
/// Pipeline:
///   1. Crop central 224×224 desde 320×240 (offset_x=48, offset_y=8)
///   2. RGB565 → RGB888
///   3. Normalización a INT8 [-128, 127]
///
/// @param fb       Frame buffer de la cámara (RGB565, 320×240)
/// @param output   Buffer de salida (224×224×3, INT8). Si nullptr, usa buffer interno.
/// @return Puntero al buffer de salida (INT8), nullptr si error.
int8_t* image_preprocess(const camera_fb_t* fb, int8_t* output = nullptr);

/// Preprocesa un frame para modelos que esperan float32.
/// Pipeline igual pero normaliza a [0.0, 1.0].
/// @return Puntero al buffer de salida (float32), nullptr si error.
float* image_preprocess_float(const camera_fb_t* fb, float* output = nullptr);

/// Preprocesa un frame para modelos ESP-DL con cuantización INT8 power-of-2.
///
/// Pipeline:
///   1. Crop central 224×224 desde 320×240
///   2. RGB565 → RGB888
///   3. Normalización a INT8 [0, 127]: val = round(pixel / 255.0 * 128)
///      compatible con exponent=-7 → float = int8 * 2^(-7) ≈ pixel / 255.0
///
/// @param fb       Frame buffer de la cámara (RGB565, 320×240)
/// @param output   Buffer de salida (224×224×3, INT8). Si nullptr, usa buffer interno.
/// @return Puntero al buffer de salida (INT8), nullptr si error.
int8_t* image_preprocess_espdl(const camera_fb_t* fb, int8_t* output = nullptr);

/// Libera los buffers de preprocesamiento.
void image_proc_deinit();
