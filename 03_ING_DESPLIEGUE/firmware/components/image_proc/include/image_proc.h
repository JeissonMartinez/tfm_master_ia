// =============================================================================
// image_proc.h — Preprocesamiento de imagen para modelos .espdl
// Pipeline: RGB565 320×240 → Center crop → Resize → RGB888 INT8 224×224×3
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Preprocesar frame de cámara para inferencia.
 *
 * Operaciones en una sola pasada (sin buffers intermedios):
 *   1. Center crop: 320×240 → 240×240 (40px cada lado)
 *   2. Resize bilineal: 240×240 → 224×224
 *   3. Conversión RGB565 → RGB888
 *   4. Normalización: uint8 [0,255] → int8 [-128,127] (pixel - 128)
 *
 * Los 3 modelos .espdl aceptan INT8 NHWC con exponent=-7, lo cual
 * corresponde a la normalización pixel/255.0 ≈ (pixel-128)/128.
 *
 * @param src_rgb565   Buffer RGB565 de la cámara (en PSRAM)
 * @param src_width    Ancho fuente (320)
 * @param src_height   Alto fuente (240)
 * @param dst_int8     Buffer destino INT8 [224][224][3] (en PSRAM, pre-allocado)
 */
void preprocess_image(const uint16_t *src_rgb565, int src_width, int src_height,
                      int8_t *dst_int8);

/**
 * @brief Generar imagen BMP desde el buffer preprocesado (para debug visual).
 *        Convierte INT8 [-128,127] de vuelta a uint8 [0,255] y genera BMP 224×224.
 *
 * @param input_int8   Buffer INT8 224×224×3 (el mismo que se pasa a inferencia)
 * @param bmp_buf      Buffer destino para BMP (debe tener ≥ bmp_get_size() bytes)
 * @param bmp_size     Tamaño del BMP generado (out)
 */
void preprocess_generate_debug_bmp(const int8_t *input_int8,
                                   uint8_t *bmp_buf, size_t *bmp_size);

/**
 * @brief Obtener tamaño en bytes de un BMP 224×224 RGB888.
 */
size_t preprocess_bmp_size(void);

#ifdef __cplusplus
}
#endif
