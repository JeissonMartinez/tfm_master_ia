// =============================================================================
// postprocess_common.cpp — NMS y utilidades comunes de postprocesamiento
// =============================================================================
#include "postprocess.h"
#include "inference_engine.h"
#include "esp_log.h"
#include <cmath>
#include <algorithm>

static const char *TAG = "postproc";

// Modelo activo (set en postprocess_init)
static ModelType s_model = MODEL_COUNT;

// ---- Forward declarations de los decoders específicos ----
extern void decode_ssd(DetectionResult *result);
extern void decode_yolo11n(DetectionResult *result);
extern void decode_yolo26n(DetectionResult *result);

// =============================================================================
// postprocess_init
// =============================================================================
void postprocess_init(ModelType model)
{
    s_model = model;
    ESP_LOGI(TAG, "Postprocessor initialized for %s", get_model_name(model));

    if (model == MODEL_MBNTV3S) {
        // Pre-computar anchors SSD (ver postprocess_ssd.cpp)
        extern void ssd_generate_anchors(void);
        ssd_generate_anchors();
    }
}

// =============================================================================
// postprocess_decode — Dispatcher al decoder correcto
// =============================================================================
void postprocess_decode(DetectionResult *result)
{
    if (!result) return;

    result->num_detections = 0;
    result->model = s_model;

    switch (s_model) {
        case MODEL_MBNTV3S:
            decode_ssd(result);
            break;
        case MODEL_YOLO11N:
            decode_yolo11n(result);
            break;
        case MODEL_YOLO26N:
            decode_yolo26n(result);
            break;
        default:
            ESP_LOGE(TAG, "Unknown model type: %d", (int)s_model);
            break;
    }
}

// =============================================================================
// bbox_iou — Intersection over Union
// =============================================================================
float bbox_iou(const BBox &a, const BBox &b)
{
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    return (union_area > 1e-6f) ? (inter_area / union_area) : 0.0f;
}

// =============================================================================
// bbox_clamp — Clamp a [0, 1]
// =============================================================================
void bbox_clamp(BBox &box)
{
    box.x1 = std::max(0.0f, std::min(1.0f, box.x1));
    box.y1 = std::max(0.0f, std::min(1.0f, box.y1));
    box.x2 = std::max(0.0f, std::min(1.0f, box.x2));
    box.y2 = std::max(0.0f, std::min(1.0f, box.y2));
}

// =============================================================================
// nms_process — Greedy NMS con sort previo
//
// Optimizaciones MCU:
//   - Insertion sort (eficiente para arrays pequeños, sin overhead malloc)
//   - Early exit cuando output lleno
//   - Candidatos ya filtrados por score threshold antes de llamar
// =============================================================================
void nms_process(Detection *candidates, int num_candidates,
                 float iou_threshold,
                 Detection *output, int *out_count, int max_output)
{
    *out_count = 0;
    if (num_candidates == 0) return;

    // Limitar candidatos
    if (num_candidates > MAX_CANDIDATES) {
        num_candidates = MAX_CANDIDATES;
    }

    // Insertion sort descendente por score (eficiente para N < 200)
    for (int i = 1; i < num_candidates; i++) {
        Detection key = candidates[i];
        int j = i - 1;
        while (j >= 0 && candidates[j].score < key.score) {
            candidates[j + 1] = candidates[j];
            j--;
        }
        candidates[j + 1] = key;
    }

    // Array de flags "suprimido"
    // Usamos un array local (MAX_CANDIDATES es 200, cabe en stack)
    bool suppressed[MAX_CANDIDATES] = {};

    for (int i = 0; i < num_candidates && *out_count < max_output; i++) {
        if (suppressed[i]) continue;

        // Aceptar esta detección
        output[*out_count] = candidates[i];
        (*out_count)++;

        // Suprimir detecciones solapadas de la misma clase
        for (int j = i + 1; j < num_candidates; j++) {
            if (suppressed[j]) continue;
            if (candidates[j].class_id != candidates[i].class_id) continue;

            if (bbox_iou(candidates[i].bbox, candidates[j].bbox) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
}
