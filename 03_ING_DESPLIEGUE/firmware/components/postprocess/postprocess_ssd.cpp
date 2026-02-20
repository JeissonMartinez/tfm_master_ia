// =============================================================================
// postprocess_ssd.cpp — Decoder para MBNTv3S_SSDLite
//
// Salidas del modelo (.info):
//   Output 0: class_out  [1, 1470, 5]  INT8, exp=-7 (post-Sigmoid)
//   Output 1: bbox_out   [1, 1470, 4]  INT8, exp=-7
//   Output 2: objectness [1, 1470, 1]  INT8, exp=-7 (post-Sigmoid)
//
// Pipeline:
//   1. Dequantizar INT8 → float (scale = 2^(-7) = 1/128)
//   2. Filtrar por objectness > threshold
//   3. Decodificar bbox con anchors SSD: decoded = anchor + delta * variance
//   4. Clasificar: argmax de class_out
//   5. NMS
//
// NOTA: Los anchors deben verificarse contra la configuración de entrenamiento.
//       Los valores por defecto aquí son estándar para SSD-Lite + MobileNetV3.
// =============================================================================
#include "postprocess.h"
#include "inference_engine.h"
#include "app_config.h"
#include "esp_log.h"
#include <cmath>
#include <algorithm>

static const char *TAG = "ssd_decode";

// ---- Anchors pre-computados ----
// Formato: [num_anchors][4] = {cx, cy, w, h} normalizados a [0,1]
static float s_anchors[SSD_NUM_ANCHORS][4] = {};
static bool  s_anchors_ready = false;

// =============================================================================
// ssd_generate_anchors — Pre-computar prior boxes
// =============================================================================
void ssd_generate_anchors(void)
{
    int idx = 0;

    for (int l = 0; l < SSD_NUM_FEATURE_MAPS; l++) {
        int fm = SSD_FM_SIZES[l];
        float min_s = SSD_MIN_SIZES[l];
        float max_s = SSD_MAX_SIZES[l];

        for (int y = 0; y < fm; y++) {
            for (int x = 0; x < fm; x++) {
                // Centro del anchor (normalizado)
                float cx = (x + 0.5f) / (float)fm;
                float cy = (y + 0.5f) / (float)fm;

                // Anchor 1: min_size × min_size (aspect ratio 1:1)
                if (idx < SSD_NUM_ANCHORS) {
                    s_anchors[idx][0] = cx;
                    s_anchors[idx][1] = cy;
                    s_anchors[idx][2] = min_s;
                    s_anchors[idx][3] = min_s;
                    idx++;
                }

                // Anchor 2: sqrt(min_size * max_size) (aspect ratio 1:1, tamaño intermedio)
                if (idx < SSD_NUM_ANCHORS) {
                    float s_mid = sqrtf(min_s * max_s);
                    s_anchors[idx][0] = cx;
                    s_anchors[idx][1] = cy;
                    s_anchors[idx][2] = s_mid;
                    s_anchors[idx][3] = s_mid;
                    idx++;
                }

                // Anchors 3-6: diferentes aspect ratios
                for (int a = 0; a < SSD_NUM_ASPECT_RATIOS; a++) {
                    float ar = SSD_ASPECT_RATIOS[a];
                    if (fabsf(ar - 1.0f) < 1e-6f) continue;  // ya cubierto arriba

                    if (idx < SSD_NUM_ANCHORS) {
                        s_anchors[idx][0] = cx;
                        s_anchors[idx][1] = cy;
                        s_anchors[idx][2] = min_s * sqrtf(ar);
                        s_anchors[idx][3] = min_s / sqrtf(ar);
                        idx++;
                    }
                }
            }
        }
    }

    s_anchors_ready = true;
    ESP_LOGI(TAG, "SSD anchors generated: %d (expected %d)", idx, SSD_NUM_ANCHORS);

    if (idx != SSD_NUM_ANCHORS) {
        ESP_LOGW(TAG, "Anchor count mismatch! Check SSD_ASPECT_RATIOS config");
    }
}

// =============================================================================
// decode_ssd — Decodificar salidas MBNTv3S
// =============================================================================
void decode_ssd(DetectionResult *result)
{
    if (!s_anchors_ready) {
        ESP_LOGE(TAG, "Anchors not generated — call postprocess_init first");
        return;
    }

    // Obtener tensores de salida POR NOMBRE (no por índice, que depende del
    // orden de std::map y sería: bbox_out=0, class_out=1, objectness=2)
    const void *class_data_raw = nullptr;
    const void *bbox_data_raw  = nullptr;
    const void *obj_data_raw   = nullptr;
    int cls_size = 0, bbox_size = 0, obj_size = 0;
    int cls_exp = 0, bbox_exp = 0, obj_exp = 0;

    inference_get_output_by_name("class_out",  &class_data_raw, &cls_size,  &cls_exp,  nullptr);
    inference_get_output_by_name("bbox_out",   &bbox_data_raw,  &bbox_size, &bbox_exp, nullptr);
    inference_get_output_by_name("objectness", &obj_data_raw,   &obj_size,  &obj_exp,  nullptr);

    const int8_t *class_data = (const int8_t *)class_data_raw;
    const int8_t *bbox_data  = (const int8_t *)bbox_data_raw;
    const int8_t *obj_data   = (const int8_t *)obj_data_raw;

    if (!class_data || !bbox_data || !obj_data) {
        ESP_LOGE(TAG, "Failed to get SSD output tensors");
        return;
    }

    // Escalas de dequantización
    float cls_scale  = powf(2.0f, cls_exp);   // 2^(-7) = 1/128
    float bbox_scale = powf(2.0f, bbox_exp);
    float obj_scale  = powf(2.0f, obj_exp);

    // --- Diagnóstico para los primeros frames ---
    static int ssd_diag_count = 0;
    if (ssd_diag_count < 5) {
        // Scan objectness
        int obj_nonzero = 0;
        int8_t obj_max_raw = -128;
        for (int i = 0; i < SSD_NUM_ANCHORS; i++) {
            if (obj_data[i] != 0) obj_nonzero++;
            if (obj_data[i] > obj_max_raw) obj_max_raw = obj_data[i];
        }
        // Scan class scores
        int cls_nonzero = 0;
        int8_t cls_max_raw = -128;
        for (int i = 0; i < SSD_NUM_ANCHORS * SSD_NUM_CLASSES; i++) {
            if (class_data[i] != 0) cls_nonzero++;
            if (class_data[i] > cls_max_raw) cls_max_raw = class_data[i];
        }
        ESP_LOGW(TAG, "DIAG[%d] obj: %d/%d nonzero, max_raw=%d (%.4f) | "
                 "cls: %d/%d nonzero, max_raw=%d (%.4f) | "
                 "exp: cls=%d bbox=%d obj=%d",
                 ssd_diag_count,
                 obj_nonzero, SSD_NUM_ANCHORS, (int)obj_max_raw, obj_max_raw * obj_scale,
                 cls_nonzero, SSD_NUM_ANCHORS * SSD_NUM_CLASSES, (int)cls_max_raw, cls_max_raw * cls_scale,
                 cls_exp, bbox_exp, obj_exp);
        ssd_diag_count++;
    }

    // ---- Fase 1: Filtrar por objectness y construir candidatos ----
    Detection candidates[MAX_CANDIDATES];
    int num_candidates = 0;

    for (int i = 0; i < SSD_NUM_ANCHORS && num_candidates < MAX_CANDIDATES; i++) {
        // Dequantizar objectness (post-Sigmoid en el modelo)
        float objectness = obj_data[i] * obj_scale;

        if (objectness < SCORE_THRESHOLD) continue;

        // Encontrar mejor clase
        int best_cls = 0;
        float best_score = -1e9f;
        for (int c = 0; c < SSD_NUM_CLASSES; c++) {
            float score = class_data[i * SSD_NUM_CLASSES + c] * cls_scale;
            if (score > best_score) {
                best_score = score;
                best_cls = c;
            }
        }

        // Score combinado = objectness * class_score
        float final_score = objectness * best_score;
        if (final_score < SCORE_THRESHOLD) continue;

        // Decodificar bbox con anchors
        // delta format: (dx, dy, dw, dh) relativo al anchor
        float dx = bbox_data[i * 4 + 0] * bbox_scale;
        float dy = bbox_data[i * 4 + 1] * bbox_scale;
        float dw = bbox_data[i * 4 + 2] * bbox_scale;
        float dh = bbox_data[i * 4 + 3] * bbox_scale;

        // Decodificación SSD estándar con varianzas
        float cx = s_anchors[i][0] + dx * SSD_VARIANCE_XY * s_anchors[i][2];
        float cy = s_anchors[i][1] + dy * SSD_VARIANCE_XY * s_anchors[i][3];
        float w  = s_anchors[i][2] * expf(dw * SSD_VARIANCE_WH);
        float h  = s_anchors[i][3] * expf(dh * SSD_VARIANCE_WH);

        // Convertir (cx, cy, w, h) → (x1, y1, x2, y2) normalizado
        Detection &det = candidates[num_candidates];
        det.bbox.x1 = cx - w / 2.0f;
        det.bbox.y1 = cy - h / 2.0f;
        det.bbox.x2 = cx + w / 2.0f;
        det.bbox.y2 = cy + h / 2.0f;
        bbox_clamp(det.bbox);

        det.class_id = best_cls;
        det.score = final_score;
        num_candidates++;
    }

    // ---- Fase 2: NMS ----
    nms_process(candidates, num_candidates, NMS_IOU_THRESHOLD,
                result->detections, &result->num_detections, MAX_DETECTIONS);

    ESP_LOGD(TAG, "SSD: %d candidates → %d detections after NMS",
             num_candidates, result->num_detections);
}
