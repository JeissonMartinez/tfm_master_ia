// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Camera driver (OV5640)
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "esp_camera.h"

/// Inicializa la cámara OV5640 con RGB565 320×240 y double buffer en PSRAM.
/// @return ESP_OK si la inicialización fue exitosa.
esp_err_t camera_init();

/// Captura un frame de la cámara.
/// @return Puntero al framebuffer (caller debe llamar camera_release_fb).
///         nullptr si la captura falló.
camera_fb_t* camera_capture();

/// Libera el framebuffer capturado.
void camera_release_fb(camera_fb_t* fb);
