// =============================================================================
// image_proc.cpp — Preprocesamiento en una sola pasada
// RGB565 320×240 → crop+resize+convert → INT8 224×224×3 NHWC
// =============================================================================
#include "image_proc.h"
#include "app_config.h"
#include "esp_log.h"
#include <cmath>
#include <cstring>
#include <algorithm>

[[maybe_unused]] static const char *TAG = "img_proc";

// Dimensiones locales (no usar nombres que colisionen con macros de app_config.h)
static constexpr int IMG_SRC_W       = 320;
static constexpr int IMG_SRC_H       = 240;
static constexpr int IMG_DST_W       = 224;
static constexpr int IMG_DST_H       = 224;
static constexpr int IMG_CROP_X      = 40;   // (320-240)/2
static constexpr int IMG_CROP_SZ     = 240;  // Lado del cuadrado after crop

// ---- Helpers RGB565 ----

// RGB565: RRRRRGGG GGGBBBBB (big endian en ESP32 camera)
static inline void rgb565_to_rgb888(uint16_t pixel, uint8_t &r, uint8_t &g, uint8_t &b)
{
    // ESP32 camera puede entregar el byte order invertido (little endian)
    // Swap bytes si es necesario
    pixel = __builtin_bswap16(pixel);

    uint8_t r5 = (pixel >> 11) & 0x1F;
    uint8_t g6 = (pixel >> 5)  & 0x3F;
    uint8_t b5 =  pixel        & 0x1F;

    // Expansión precisa a 8 bits
    r = (r5 << 3) | (r5 >> 2);
    g = (g6 << 2) | (g6 >> 4);
    b = (b5 << 3) | (b5 >> 2);
}

// ---- Bilinear interpolation helpers ----

static inline float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

// =============================================================================
// preprocess_image — Pasada única: crop + resize bilineal + RGB565→RGB888 + INT8
// =============================================================================
void preprocess_image(const uint16_t *src_rgb565, int src_width, int src_height,
                      int8_t *dst_int8)
{
    // Factores de escala: mapear coordenada destino → coordenada fuente (dentro del crop)
    const float scale_x = (float)IMG_CROP_SZ / (float)IMG_DST_W;
    const float scale_y = (float)(src_height) / (float)IMG_DST_H;  // 240→224

    for (int dy = 0; dy < IMG_DST_H; dy++) {
        // Coordenada fuente Y (en el frame completo)
        float sy = dy * scale_y;
        int y0 = (int)sy;
        int y1 = std::min(y0 + 1, src_height - 1);
        float fy = sy - y0;

        for (int dx = 0; dx < IMG_DST_W; dx++) {
            // Coordenada fuente X (con offset de crop)
            float sx = dx * scale_x + IMG_CROP_X;
            int x0 = (int)sx;
            int x1 = std::min(x0 + 1, src_width - 1);
            float fx = sx - x0;

            // Leer 4 píxeles vecinos del frame RGB565
            uint8_t r00, g00, b00, r01, g01, b01;
            uint8_t r10, g10, b10, r11, g11, b11;

            rgb565_to_rgb888(src_rgb565[y0 * src_width + x0], r00, g00, b00);
            rgb565_to_rgb888(src_rgb565[y0 * src_width + x1], r01, g01, b01);
            rgb565_to_rgb888(src_rgb565[y1 * src_width + x0], r10, g10, b10);
            rgb565_to_rgb888(src_rgb565[y1 * src_width + x1], r11, g11, b11);

            // Interpolación bilineal por canal
            float r = lerp(lerp(r00, r01, fx), lerp(r10, r11, fx), fy);
            float g = lerp(lerp(g00, g01, fx), lerp(g10, g11, fx), fy);
            float b = lerp(lerp(b00, b01, fx), lerp(b10, b11, fx), fy);

            // Normalización a INT8 dependiente del modelo:
            //
            // YOLO (exponent=-7): calibrado con pixel/255 → [0, 1]
            //   int8 = round(pixel * 128 / 255) → rango [0, 127]
            //   model interpreta: int8 * 2^(-7) = int8/128 → [0, 0.99]
            //
            // MBNTv3S (exponent=0, dtype float internamente):
            //   int8 = pixel - 128 → rango [-128, 127]
            //   Funciona porque el modelo TF absorbe el offset
            int dst_idx = (dy * IMG_DST_W + dx) * 3;
#if (ACTIVE_MODEL == MODEL_YOLO11N || ACTIVE_MODEL == MODEL_YOLO26N)
            // YOLO: pixel [0,255] → float [0,1] representado como INT8 [0,127]
            dst_int8[dst_idx + 0] = (int8_t)std::min(127, (int)std::round(r * 128.0f / 255.0f));
            dst_int8[dst_idx + 1] = (int8_t)std::min(127, (int)std::round(g * 128.0f / 255.0f));
            dst_int8[dst_idx + 2] = (int8_t)std::min(127, (int)std::round(b * 128.0f / 255.0f));
#else
            // MBNTv3S: pixel - 128 centering
            dst_int8[dst_idx + 0] = (int8_t)((int)std::round(r) - 128);
            dst_int8[dst_idx + 1] = (int8_t)((int)std::round(g) - 128);
            dst_int8[dst_idx + 2] = (int8_t)((int)std::round(b) - 128);
#endif
        }
    }
}

// =============================================================================
// BMP generation for debug visual (endpoint /debug/image)
// =============================================================================

// BMP header sizes
static constexpr int BMP_HEADER_SIZE = 54;   // 14 (file) + 40 (info)
static constexpr int BMP_ROW_BYTES   = IMG_DST_W * 3;
// BMP rows must be 4-byte aligned: 224*3=672, 672%4=0 → no padding needed
static constexpr int BMP_DATA_SIZE   = IMG_DST_H * BMP_ROW_BYTES;
static constexpr int BMP_FILE_SIZE   = BMP_HEADER_SIZE + BMP_DATA_SIZE;

size_t preprocess_bmp_size(void)
{
    return BMP_FILE_SIZE;
}

void preprocess_generate_debug_bmp(const int8_t *input_int8,
                                   uint8_t *bmp_buf, size_t *bmp_size)
{
    if (!input_int8 || !bmp_buf || !bmp_size) return;

    memset(bmp_buf, 0, BMP_HEADER_SIZE);

    // ---- BMP File Header (14 bytes) ----
    bmp_buf[0] = 'B'; bmp_buf[1] = 'M';
    uint32_t fsize = BMP_FILE_SIZE;
    memcpy(&bmp_buf[2], &fsize, 4);
    uint32_t offset = BMP_HEADER_SIZE;
    memcpy(&bmp_buf[10], &offset, 4);

    // ---- BMP Info Header (40 bytes) ----
    uint32_t hdr_size = 40;
    memcpy(&bmp_buf[14], &hdr_size, 4);
    int32_t w = IMG_DST_W;
    int32_t h = IMG_DST_H;   // Positivo = bottom-up
    memcpy(&bmp_buf[18], &w, 4);
    memcpy(&bmp_buf[22], &h, 4);
    uint16_t planes = 1;
    memcpy(&bmp_buf[26], &planes, 2);
    uint16_t bpp = 24;
    memcpy(&bmp_buf[28], &bpp, 2);
    uint32_t img_size = BMP_DATA_SIZE;
    memcpy(&bmp_buf[34], &img_size, 4);

    // ---- Pixel data (bottom-up, BGR order for BMP) ----
    for (int row = 0; row < IMG_DST_H; row++) {
        // BMP es bottom-up: fila 0 del BMP = última fila de la imagen
        int src_row = IMG_DST_H - 1 - row;
        uint8_t *dst_row = &bmp_buf[BMP_HEADER_SIZE + row * BMP_ROW_BYTES];

        for (int col = 0; col < IMG_DST_W; col++) {
            int src_idx = (src_row * IMG_DST_W + col) * 3;
            // INT8→uint8: inversa de la normalización
#if (ACTIVE_MODEL == MODEL_YOLO11N || ACTIVE_MODEL == MODEL_YOLO26N)
            // YOLO: int8 [0,127] → pixel = int8 * 255 / 128
            uint8_t r = (uint8_t)std::min(255, std::max(0, (int)(input_int8[src_idx + 0] * 255.0f / 128.0f)));
            uint8_t g = (uint8_t)std::min(255, std::max(0, (int)(input_int8[src_idx + 1] * 255.0f / 128.0f)));
            uint8_t b = (uint8_t)std::min(255, std::max(0, (int)(input_int8[src_idx + 2] * 255.0f / 128.0f)));
#else
            // MBNTv3S: int8 + 128 → pixel
            uint8_t r = (uint8_t)((int)input_int8[src_idx + 0] + 128);
            uint8_t g = (uint8_t)((int)input_int8[src_idx + 1] + 128);
            uint8_t b = (uint8_t)((int)input_int8[src_idx + 2] + 128);
#endif

            // BMP usa orden BGR
            dst_row[col * 3 + 0] = b;
            dst_row[col * 3 + 1] = g;
            dst_row[col * 3 + 2] = r;
        }
    }

    *bmp_size = BMP_FILE_SIZE;
}
