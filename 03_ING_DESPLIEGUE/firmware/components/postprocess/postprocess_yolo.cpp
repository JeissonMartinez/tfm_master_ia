// =============================================================================
// postprocess_yolo.cpp — Decoder para YOLO11n y YOLO26n (formato ESP-DL)
//
// Los modelos se exportan SIN detection head (sin DFL/sigmoid/concat),
// produciendo 6 salidas independientes con exponents individuales:
//
//   YOLO11n (reg_max=16):
//     box0 [1,28,28,64] score0 [1,28,28,5]  (P3, stride=8)
//     box1 [1,14,14,64] score1 [1,14,14,5]  (P4, stride=16)
//     box2 [1,7,7,64]   score2 [1,7,7,5]    (P5, stride=32)
//
//   YOLO26n (reg_max=1):
//     box0 [1,28,28,4]  score0 [1,28,28,5]  (P3, stride=8)
//     box1 [1,14,14,4]  score1 [1,14,14,5]  (P4, stride=16)
//     box2 [1,7,7,4]    score2 [1,7,7,5]    (P5, stride=32)
//
// Post-procesamiento on-device (en float32):
//   1. Para cada celda: dequantizar score → sigmoid → filtrar
//   2. Dequantizar box → DFL softmax + integral → dist2bbox
//   3. NMS
//
// Layout en memoria: NHWC (ESP-DL convierte internamente)
//   box:   data[(y * W + x) * C_box + c]
//   score: data[(y * W + x) * C_cls + c]
// =============================================================================
#include "postprocess.h"
#include "inference_engine.h"
#include "app_config.h"
#include "esp_log.h"
#include <cmath>
#include <algorithm>
#include <cstring>

static const char *TAG = "yolo_decode";

// =============================================================================
// Constantes
// =============================================================================

// Feature map sizes para input 224×224
static const int FM_SIZES[3]  = {28, 14, 7};   // H=W para P3, P4, P5
static const int STRIDES[3]   = {8, 16, 32};    // stride por nivel

// Nombres de tensores de salida
static const char *BOX_NAMES[3]   = {"box0", "box1", "box2"};
static const char *SCORE_NAMES[3] = {"score0", "score1", "score2"};

// =============================================================================
// Funciones matemáticas on-device
// =============================================================================

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float inverse_sigmoid(float y)
{
    // logit(y) = ln(y / (1-y))
    y = fmaxf(y, 1e-6f);
    y = fminf(y, 1.0f - 1e-6f);
    return logf(y / (1.0f - y));
}

/**
 * DFL (Distribution Focal Loss) integral:
 * Aplica softmax sobre `reg_max` bins y calcula la media ponderada.
 * Esto convierte la distribución de probabilidad en un offset de distancia.
 */
static float dfl_integral(const float *data, int reg_max)
{
    // Softmax
    float max_val = data[0];
    for (int i = 1; i < reg_max; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    float sum_exp = 0.0f;
    float weighted_sum = 0.0f;
    for (int i = 0; i < reg_max; i++) {
        float e = expf(data[i] - max_val);
        sum_exp += e;
        weighted_sum += (float)i * e;
    }

    return weighted_sum / sum_exp;
}

// =============================================================================
// parse_stage — Procesar un nivel de detección (P3/P4/P5)
//
// Template para ambos modelos (YOLO11n con DFL, YOLO26n sin DFL).
// =============================================================================

static void parse_stage(
    int stage_idx,           // 0=P3, 1=P4, 2=P5
    int reg_max,             // 16 para YOLO11n, 1 para YOLO26n
    Detection *candidates,
    int *p_num_cands,
    int max_cands,
    float *p_max_score_debug
)
{
    int H = FM_SIZES[stage_idx];
    int W = FM_SIZES[stage_idx];
    int stride = STRIDES[stage_idx];

    // Obtener tensores de salida por nombre
    const void *score_data_raw = nullptr;
    const void *box_data_raw = nullptr;
    int score_size = 0, box_size = 0;
    int score_exp = 0, box_exp = 0;
    int score_dtype = -1;

    esp_err_t err;
    err = inference_get_output_by_name(SCORE_NAMES[stage_idx], &score_data_raw, &score_size, &score_exp, &score_dtype);
    if (err != ESP_OK || !score_data_raw) {
        ESP_LOGE(TAG, "Failed to get %s", SCORE_NAMES[stage_idx]);
        return;
    }
    err = inference_get_output_by_name(BOX_NAMES[stage_idx], &box_data_raw, &box_size, &box_exp, nullptr);
    if (err != ESP_OK || !box_data_raw) {
        ESP_LOGE(TAG, "Failed to get %s", BOX_NAMES[stage_idx]);
        return;
    }

    // Determinar si scores son float (dtype==0 en ESP-DL DATA_TYPE_FLOAT)
    // También verificar exponent: float outputs tienen exponent=0
    bool score_is_float = (score_dtype == 0);  // DATA_TYPE_FLOAT = 0

    const int8_t *score_data_i8 = score_is_float ? nullptr : (const int8_t *)score_data_raw;
    const float  *score_data_f  = score_is_float ? (const float *)score_data_raw : nullptr;
    const int8_t *box_data = (const int8_t *)box_data_raw;

    float score_scale = powf(2.0f, score_exp);     // solo para INT8
    float box_scale = powf(2.0f, box_exp);

    // Umbral pre-sigmoid en dominio cuantizado (solo para path INT8)
    float score_thr_logit = inverse_sigmoid(SCORE_THRESHOLD);
    int8_t score_thr_quantized = (int8_t)fmaxf(fminf(roundf(score_thr_logit / score_scale), 127.0f), -128.0f);

    int box_channels = reg_max * 4;   // 64 para YOLO11n, 4 para YOLO26n
    int num_cands = *p_num_cands;

    // Log info para primera invocación
    static int s_stage_log = 0;
    if (s_stage_log < 3) {
        ESP_LOGI(TAG, "stage%d: scores %s, score_exp=%d, box_exp=%d, H=%d",
                 stage_idx, score_is_float ? "FLOAT" : "INT8",
                 score_exp, box_exp, H);
        s_stage_log++;
    }

    // Iterar sobre cada celda del feature map
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (num_cands >= max_cands) goto done;

            float score;
            int best_cls;

            if (score_is_float) {
                // === PATH FLOAT: scores ya son logits float ===
                const float *cell_score = &score_data_f[(y * W + x) * NUM_CLASSES];
                float max_logit = cell_score[0];
                best_cls = 0;
                for (int c = 1; c < NUM_CLASSES; c++) {
                    if (cell_score[c] > max_logit) {
                        max_logit = cell_score[c];
                        best_cls = c;
                    }
                }

                // Filtro rápido en dominio float (pre-sigmoid)
                if (max_logit < score_thr_logit) {
                    // Track max para debug incluso si no pasa el umbral
                    float s = sigmoid(max_logit);
                    if (s > *p_max_score_debug) *p_max_score_debug = s;
                    continue;
                }

                score = sigmoid(max_logit);
            } else {
                // === PATH INT8: dequantizar + sigmoid ===
                const int8_t *cell_score = &score_data_i8[(y * W + x) * NUM_CLASSES];
                int8_t max_raw = cell_score[0];
                best_cls = 0;
                for (int c = 1; c < NUM_CLASSES; c++) {
                    if (cell_score[c] > max_raw) {
                        max_raw = cell_score[c];
                        best_cls = c;
                    }
                }

                // Filtro rápido cuantizado
                if (max_raw < score_thr_quantized) continue;

                score = sigmoid(max_raw * score_scale);
            }

            // Debug: rastrear máximo
            if (score > *p_max_score_debug) {
                *p_max_score_debug = score;
            }

            if (score < SCORE_THRESHOLD) continue;

            // --- Decodificar bbox ---
            // Puntero a box data: layout NHWC → offset = (y*W + x)*box_channels
            const int8_t *cell_box = &box_data[(y * W + x) * box_channels];

            float cx, cy, w_half, h_half;

            if (reg_max > 1) {
                // YOLO11n: DFL decode (4 distribuciones de reg_max bins)
                // box_channels = 64 = 4 * 16
                // Layout: [lt0..lt15, rt0..rt15, tb0..tb15, rb0..rb15]
                float dfl_vals[64];  // max reg_max*4 = 64
                for (int i = 0; i < box_channels; i++) {
                    dfl_vals[i] = cell_box[i] * box_scale;
                }

                // dist2bbox: distancias (left, top, right, bottom) → (x1,y1,x2,y2)
                float dist_l = dfl_integral(&dfl_vals[0 * reg_max], reg_max);
                float dist_t = dfl_integral(&dfl_vals[1 * reg_max], reg_max);
                float dist_r = dfl_integral(&dfl_vals[2 * reg_max], reg_max);
                float dist_b = dfl_integral(&dfl_vals[3 * reg_max], reg_max);

                // Centro de la celda en píxeles
                float cell_cx = ((float)x + 0.5f) * stride;
                float cell_cy = ((float)y + 0.5f) * stride;

                // bbox en píxeles (x1, y1, x2, y2)
                float x1 = (cell_cx - dist_l * stride);
                float y1 = (cell_cy - dist_t * stride);
                float x2 = (cell_cx + dist_r * stride);
                float y2 = (cell_cy + dist_b * stride);

                // Normalizar a [0, 1]
                Detection &det = candidates[num_cands];
                det.bbox.x1 = x1 / (float)MODEL_INPUT_W;
                det.bbox.y1 = y1 / (float)MODEL_INPUT_H;
                det.bbox.x2 = x2 / (float)MODEL_INPUT_W;
                det.bbox.y2 = y2 / (float)MODEL_INPUT_H;
                bbox_clamp(det.bbox);

                if (det.bbox.x2 <= det.bbox.x1 || det.bbox.y2 <= det.bbox.y1) continue;

                det.class_id = best_cls;
                det.score = score;
                num_cands++;

            } else {
                // YOLO26n (reg_max=1): predicción directa de (x1,y1,x2,y2)
                // box_channels = 4
                // El modelo predice offsets relativos al centro de la celda:
                //   box[0..3] = (lt, tb, rt, rb) distancias normalizadas
                float d0 = cell_box[0] * box_scale;
                float d1 = cell_box[1] * box_scale;
                float d2 = cell_box[2] * box_scale;
                float d3 = cell_box[3] * box_scale;

                float cell_cx = ((float)x + 0.5f) * stride;
                float cell_cy = ((float)y + 0.5f) * stride;

                float x1 = cell_cx - d0 * stride;
                float y1 = cell_cy - d1 * stride;
                float x2 = cell_cx + d2 * stride;
                float y2 = cell_cy + d3 * stride;

                Detection &det = candidates[num_cands];
                det.bbox.x1 = x1 / (float)MODEL_INPUT_W;
                det.bbox.y1 = y1 / (float)MODEL_INPUT_H;
                det.bbox.x2 = x2 / (float)MODEL_INPUT_W;
                det.bbox.y2 = y2 / (float)MODEL_INPUT_H;
                bbox_clamp(det.bbox);

                if (det.bbox.x2 <= det.bbox.x1 || det.bbox.y2 <= det.bbox.y1) continue;

                det.class_id = best_cls;
                det.score = score;
                num_cands++;
            }
        }
    }

done:
    *p_num_cands = num_cands;
}

// =============================================================================
// decode_yolo11n — 6 salidas, reg_max=16, DFL+sigmoid on-device
// =============================================================================
void decode_yolo11n(DetectionResult *result)
{
    Detection candidates[MAX_CANDIDATES];
    int num_cands = 0;
    float max_score = 0.0f;

    // Procesar los 3 niveles de detección
    for (int stage = 0; stage < 3; stage++) {
        parse_stage(stage, 16, candidates, &num_cands, MAX_CANDIDATES, &max_score);
    }

    // NMS
    nms_process(candidates, num_cands, NMS_IOU_THRESHOLD,
                result->detections, &result->num_detections, MAX_DETECTIONS);

    ESP_LOGI(TAG, "YOLO11n: maxScore=%.3f threshold=%.2f → %d cands → %d dets",
             max_score, SCORE_THRESHOLD, num_cands, result->num_detections);
}

// =============================================================================
// decode_yolo26n — 6 salidas, reg_max=1, sigmoid on-device (no DFL)
// =============================================================================
void decode_yolo26n(DetectionResult *result)
{
    Detection candidates[MAX_CANDIDATES];
    int num_cands = 0;
    float max_score = 0.0f;

    for (int stage = 0; stage < 3; stage++) {
        parse_stage(stage, 1, candidates, &num_cands, MAX_CANDIDATES, &max_score);
    }

    // NMS
    nms_process(candidates, num_cands, NMS_IOU_THRESHOLD,
                result->detections, &result->num_detections, MAX_DETECTIONS);

    ESP_LOGI(TAG, "YOLO26n: maxScore=%.3f threshold=%.2f → %d cands → %d dets",
             max_score, SCORE_THRESHOLD, num_cands, result->num_detections);
}

