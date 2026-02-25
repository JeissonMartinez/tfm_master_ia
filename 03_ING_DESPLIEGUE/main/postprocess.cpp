// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Post-processing implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "postprocess.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

static const char* TAG = "postproc";

// ─── Constantes de salida por modelo ─────────────────────────────────────
static constexpr int MBNT_NUM_ANCHORS    = 1470;
static constexpr int MBNT_NUM_CLASSES    = NUM_CLASSES;   // 5
static constexpr int MBNT_BOX_DIM        = 4;

static constexpr int Y11_ROWS            = 9;             // 4 box + 5 cls
static constexpr int Y11_COLS            = 1029;          // proposals

static constexpr int Y26_MAX_DETS        = 300;
static constexpr int Y26_DET_DIM         = 6;             // x1,y1,x2,y2,conf,cls

// ─── NMS auxiliar ────────────────────────────────────────────────────────

/// Compute IoU between two boxes in [x1,y1,x2,y2] format.
static float iou_xyxy(const Detection& a, const Detection& b) {
    float xi1 = std::max(a.x1, b.x1);
    float yi1 = std::max(a.y1, b.y1);
    float xi2 = std::min(a.x2, b.x2);
    float yi2 = std::min(a.y2, b.y2);

    float inter_w = std::max(0.0f, xi2 - xi1);
    float inter_h = std::max(0.0f, yi2 - yi1);
    float inter   = inter_w * inter_h;

    float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float uni    = area_a + area_b - inter;

    return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

/// Greedy per-class NMS.  Modifies `result` in-place.
static void nms_per_class(DetectionResult& result, float iou_thr) {
    if (result.count <= 1) return;

    // Sort descending by confidence
    std::sort(result.detections, result.detections + result.count,
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    Detection kept[MAX_DETECTIONS];
    int kept_count = 0;

    bool suppressed[MAX_DETECTIONS] = {};

    for (int i = 0; i < result.count && kept_count < MAX_DETECTIONS; ++i) {
        if (suppressed[i]) continue;
        kept[kept_count++] = result.detections[i];

        for (int j = i + 1; j < result.count; ++j) {
            if (suppressed[j]) continue;
            if (result.detections[i].class_id != result.detections[j].class_id) continue;
            if (iou_xyxy(result.detections[i], result.detections[j]) >= iou_thr) {
                suppressed[j] = true;
            }
        }
    }

    std::memcpy(result.detections, kept, kept_count * sizeof(Detection));
    result.count = kept_count;
}

// ─── Clamp helper ────────────────────────────────────────────────────────
static inline float clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }

// ═══════════════════════════════════════════════════════════════════════════
//  MBNTv2 SSD-Lite  (3 output tensors)
//
//  objectness  : (1470, 1)  — sigmoid score
//  class_probs : (1470, 5)  — sigmoid per-class score
//  bbox_preds  : (1470, 4)  — normalised [xc, yc, w, h]
//
//  Pipeline:
//    1. Filter by objectness > threshold
//    2. combined_conf = objectness * max(class_probs)
//    3. Decode [xc,yc,w,h] → [x1,y1,x2,y2] normalised
//    4. Per-class NMS
// ═══════════════════════════════════════════════════════════════════════════
DetectionResult postprocess_mobilenet(
    const float* objectness,
    const float* class_probs,
    const float* bbox_preds,
    float conf_thr,
    float iou_thr)
{
    DetectionResult result;
    result.clear();

    for (int i = 0; i < MBNT_NUM_ANCHORS; ++i) {
        float obj = objectness[i];               // (N,1) layout → stride 1
        if (obj < conf_thr) continue;

        // Find best class
        const float* cls = &class_probs[i * MBNT_NUM_CLASSES];
        int best_cls = 0;
        float best_score = cls[0];
        for (int c = 1; c < MBNT_NUM_CLASSES; ++c) {
            if (cls[c] > best_score) {
                best_score = cls[c];
                best_cls = c;
            }
        }

        float combined = obj * best_score;
        if (combined < conf_thr) continue;

        // Decode box: [xc, yc, w, h] → [x1, y1, x2, y2] normalised
        const float* box = &bbox_preds[i * MBNT_BOX_DIM];
        float xc = clamp01(box[0]);
        float yc = clamp01(box[1]);
        float w  = clamp01(box[2]);
        float h  = clamp01(box[3]);

        Detection det;
        det.x1 = clamp01(xc - w * 0.5f);
        det.y1 = clamp01(yc - h * 0.5f);
        det.x2 = clamp01(xc + w * 0.5f);
        det.y2 = clamp01(yc + h * 0.5f);
        det.confidence = combined;
        det.class_id   = best_cls;

        result.add(det);
    }

    if (result.count > 1) {
        nms_per_class(result, iou_thr);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
//  YOLO11n  (1 tensor, transposed layout)
//
//  Raw output: (1, 9, 1029)  row-major
//    row 0 = xc   (normalised)
//    row 1 = yc
//    row 2 = w
//    row 3 = h
//    row 4..8 = class scores (5 classes)
//
//  Pipeline:
//    1. Transpose effectively — access (row, col) = raw[row * 1029 + col]
//    2. For each of the 1029 proposals: max(class_score) > threshold?
//    3. Decode [xc,yc,w,h] → [x1,y1,x2,y2] normalised
//    4. Per-class NMS
// ═══════════════════════════════════════════════════════════════════════════
DetectionResult postprocess_yolo11(
    const float* raw_output,
    float conf_thr,
    float iou_thr)
{
    DetectionResult result;
    result.clear();

    // raw_output layout: [row][col]  = raw_output[row * Y11_COLS + col]
    for (int p = 0; p < Y11_COLS; ++p) {
        // Find best class score
        int best_cls = 0;
        float best_score = raw_output[4 * Y11_COLS + p];   // class 0
        for (int c = 1; c < NUM_CLASSES; ++c) {
            float s = raw_output[(4 + c) * Y11_COLS + p];
            if (s > best_score) {
                best_score = s;
                best_cls = c;
            }
        }

        if (best_score < conf_thr) continue;

        // Decode box (normalised [0,1])
        float xc = raw_output[0 * Y11_COLS + p];
        float yc = raw_output[1 * Y11_COLS + p];
        float w  = raw_output[2 * Y11_COLS + p];
        float h  = raw_output[3 * Y11_COLS + p];

        // YOLO outputs are in pixel coords (0–224), normalise
        xc /= static_cast<float>(INPUT_WIDTH);
        yc /= static_cast<float>(INPUT_HEIGHT);
        w  /= static_cast<float>(INPUT_WIDTH);
        h  /= static_cast<float>(INPUT_HEIGHT);

        Detection det;
        det.x1 = clamp01(xc - w * 0.5f);
        det.y1 = clamp01(yc - h * 0.5f);
        det.x2 = clamp01(xc + w * 0.5f);
        det.y2 = clamp01(yc + h * 0.5f);
        det.confidence = best_score;
        det.class_id   = best_cls;

        result.add(det);
    }

    if (result.count > 1) {
        nms_per_class(result, iou_thr);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
//  YOLO26n end-to-end  (1 tensor, NMS-free)
//
//  Raw output: (1, 300, 6)  row-major
//    col 0 = x1  (absolute pixel coords 0–224)
//    col 1 = y1
//    col 2 = x2
//    col 3 = y2
//    col 4 = confidence
//    col 5 = class_id  (float, cast to int)
//
//  Pipeline:
//    1. Filter by confidence > threshold
//    2. Normalise pixel coords to [0,1]
//    3. No NMS needed (end-to-end model)
// ═══════════════════════════════════════════════════════════════════════════
DetectionResult postprocess_yolo26(
    const float* raw_output,
    float conf_thr,
    bool coords_normalized)
{
    DetectionResult result;
    result.clear();

    const float inv_w = coords_normalized ? 1.0f : (1.0f / static_cast<float>(INPUT_WIDTH));
    const float inv_h = coords_normalized ? 1.0f : (1.0f / static_cast<float>(INPUT_HEIGHT));

    for (int d = 0; d < Y26_MAX_DETS; ++d) {
        const float* row = &raw_output[d * Y26_DET_DIM];
        float conf = row[4];

        if (conf < conf_thr) continue;

        // class_id is stored as float, round to int
        int cls_id = static_cast<int>(std::round(row[5]));
        if (cls_id < 0 || cls_id >= NUM_CLASSES) continue;

        Detection det;
        det.x1 = clamp01(row[0] * inv_w);
        det.y1 = clamp01(row[1] * inv_h);
        det.x2 = clamp01(row[2] * inv_w);
        det.y2 = clamp01(row[3] * inv_h);
        det.confidence = conf;
        det.class_id   = cls_id;

        result.add(det);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Lifecycle
// ═══════════════════════════════════════════════════════════════════════════
esp_err_t postprocess_init() {
    ESP_LOGI(TAG, "✅ Postprocessing inicializado");
    ESP_LOGI(TAG, "   MBNTv2: %d anchors × (%d cls + 4 box + 1 obj)",
             MBNT_NUM_ANCHORS, MBNT_NUM_CLASSES);
    ESP_LOGI(TAG, "   YOLO11: %d×%d transposed", Y11_ROWS, Y11_COLS);
    ESP_LOGI(TAG, "   YOLO26: %d×%d end2end", Y26_MAX_DETS, Y26_DET_DIM);
    ESP_LOGI(TAG, "   ESPDet FCOS: 3 scales (%d,%d,%d) strides (%d,%d,%d)",
             GRID_SIZES[0], GRID_SIZES[1], GRID_SIZES[2],
             GRID_STRIDES[0], GRID_STRIDES[1], GRID_STRIDES[2]);
    ESP_LOGI(TAG, "   YOLO26 DFL: 3 scales, reg_max=%d", DFL_REG_MAX);
    ESP_LOGI(TAG, "   YOLO26 T3: 3 scales, direct dist (no DFL, box_ch=4)");
    return ESP_OK;
}

void postprocess_deinit() {
    ESP_LOGI(TAG, "Postprocessing liberado");
}

// ═══════════════════════════════════════════════════════════════════════════
//  ESP-DL Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Sigmoid activation (inlined for hot loop)
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/// Dequantize INT8 value using power-of-2 exponent: float = int8 * 2^exp
static inline float dequant(int8_t val, int exp) {
    float scale = (exp >= 0) ? static_cast<float>(1 << exp)
                             : (1.0f / static_cast<float>(1 << (-exp)));
    return static_cast<float>(val) * scale;
}

// ═══════════════════════════════════════════════════════════════════════════
//  ESPDet Pico T4 — FCOS anchor-free (3-scale, direct distances)
//
//  Output tensors:
//    score0 [1,28,28,5] exp=-3    score1 [1,14,14,5] exp=-3    score2 [1,7,7,5] exp=-3
//    box0   [1,28,28,4] exp=-3    box1   [1,14,14,4] exp=-3    box2   [1,7,7,4] exp=-4
//
//  Box layout: [left, top, right, bottom] — ReLU clamped distances from grid center
//  Decode:
//    l,t,r,b = max(0, dequant(box)) * stride
//    x1 = (cx - l) / 224, y1 = (cy - t) / 224
//    x2 = (cx + r) / 224, y2 = (cy + b) / 224
// ═══════════════════════════════════════════════════════════════════════════
DetectionResult postprocess_espdet_espdl(
    const InferenceEngine* engine,
    float conf_thr,
    float iou_thr)
{
    DetectionResult result;
    result.clear();

    // Scale names
    static const char* score_names[NUM_SCALES] = {"score0", "score1", "score2"};
    static const char* box_names[NUM_SCALES]   = {"box0", "box1", "box2"};

    const float inv_dim = 1.0f / static_cast<float>(INPUT_WIDTH);

    for (int s = 0; s < NUM_SCALES; ++s) {
        const int8_t* score_data = static_cast<const int8_t*>(
            engine->get_output_by_name(score_names[s]));
        const int8_t* box_data = static_cast<const int8_t*>(
            engine->get_output_by_name(box_names[s]));

        if (!score_data || !box_data) {
            ESP_LOGW(TAG, "ESPDet: output '%s' o '%s' no encontrado",
                     score_names[s], box_names[s]);
            continue;
        }

        int score_exp = engine->get_output_exponent(score_names[s]);
        int box_exp   = engine->get_output_exponent(box_names[s]);
        int grid_h    = GRID_SIZES[s];
        int grid_w    = GRID_SIZES[s];
        int stride    = GRID_STRIDES[s];

        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                int offset = (gy * grid_w + gx) * NUM_CLASSES;

                // Find best class score after dequant + sigmoid
                int best_cls = 0;
                float best_score = -1e9f;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    float s_val = dequant(score_data[offset + c], score_exp);
                    if (s_val > best_score) {
                        best_score = s_val;
                        best_cls = c;
                    }
                }

                float conf = sigmoid(best_score);
                if (conf < conf_thr) continue;

                // Decode box: [l, t, r, b] from grid center
                int box_offset = (gy * grid_w + gx) * 4;
                float l = std::max(0.0f, dequant(box_data[box_offset + 0], box_exp));
                float t = std::max(0.0f, dequant(box_data[box_offset + 1], box_exp));
                float r = std::max(0.0f, dequant(box_data[box_offset + 2], box_exp));
                float b = std::max(0.0f, dequant(box_data[box_offset + 3], box_exp));

                // Grid center in pixel coords
                float cx = (static_cast<float>(gx) + 0.5f) * stride;
                float cy = (static_cast<float>(gy) + 0.5f) * stride;

                Detection det;
                det.x1 = clamp01((cx - l * stride) * inv_dim);
                det.y1 = clamp01((cy - t * stride) * inv_dim);
                det.x2 = clamp01((cx + r * stride) * inv_dim);
                det.y2 = clamp01((cy + b * stride) * inv_dim);
                det.confidence = conf;
                det.class_id   = best_cls;

                result.add(det);
                if (result.count >= MAX_DETECTIONS) break;
            }
            if (result.count >= MAX_DETECTIONS) break;
        }
        if (result.count >= MAX_DETECTIONS) break;
    }

    if (result.count > 1) {
        nms_per_class(result, iou_thr);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
//  YOLO26n T2 ESP — DFL integral (3-scale, 4×16 bins)
//
//  Output tensors:
//    score0 [1,28,28,5] exp=-3    score1 [1,14,14,5] exp=-2    score2 [1,7,7,5] exp=-3
//    box0   [1,28,28,64] exp=-3   box1   [1,14,14,64] exp=-3   box2   [1,7,7,64] exp=-3
//
//  Box layout: [l_bins(16), t_bins(16), r_bins(16), b_bins(16)]
//  DFL decode:
//    For each direction d ∈ {l,t,r,b}:
//      bins[0..15] = dequant → softmax → weighted sum = distance
//    dist2bbox same as ESPDet but with DFL-computed distances
// ═══════════════════════════════════════════════════════════════════════════
DetectionResult postprocess_yolo26_espdl(
    const InferenceEngine* engine,
    float conf_thr,
    float iou_thr)
{
    DetectionResult result;
    result.clear();

    static const char* score_names[NUM_SCALES] = {"score0", "score1", "score2"};
    static const char* box_names[NUM_SCALES]   = {"box0", "box1", "box2"};

    const float inv_dim = 1.0f / static_cast<float>(INPUT_WIDTH);

    for (int s = 0; s < NUM_SCALES; ++s) {
        const int8_t* score_data = static_cast<const int8_t*>(
            engine->get_output_by_name(score_names[s]));
        const int8_t* box_data = static_cast<const int8_t*>(
            engine->get_output_by_name(box_names[s]));

        if (!score_data || !box_data) {
            ESP_LOGW(TAG, "YOLO26ESP: output '%s' o '%s' no encontrado",
                     score_names[s], box_names[s]);
            continue;
        }

        int score_exp = engine->get_output_exponent(score_names[s]);
        int box_exp   = engine->get_output_exponent(box_names[s]);
        int grid_h    = GRID_SIZES[s];
        int grid_w    = GRID_SIZES[s];
        int stride    = GRID_STRIDES[s];

        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                int score_offset = (gy * grid_w + gx) * NUM_CLASSES;

                // Find best class
                int best_cls = 0;
                float best_score = -1e9f;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    float s_val = dequant(score_data[score_offset + c], score_exp);
                    if (s_val > best_score) {
                        best_score = s_val;
                        best_cls = c;
                    }
                }

                float conf = sigmoid(best_score);
                if (conf < conf_thr) continue;

                // DFL integral for 4 directions: l, t, r, b
                // Each direction has DFL_REG_MAX=16 bins
                int box_offset = (gy * grid_w + gx) * (4 * DFL_REG_MAX);
                float distances[4];

                for (int d = 0; d < 4; ++d) {
                    const int8_t* bins_raw = &box_data[box_offset + d * DFL_REG_MAX];

                    // Softmax over DFL_REG_MAX bins
                    float max_val = -1e9f;
                    float bins[DFL_REG_MAX];
                    for (int k = 0; k < DFL_REG_MAX; ++k) {
                        bins[k] = dequant(bins_raw[k], box_exp);
                        if (bins[k] > max_val) max_val = bins[k];
                    }

                    float sum_exp = 0.0f;
                    for (int k = 0; k < DFL_REG_MAX; ++k) {
                        bins[k] = std::exp(bins[k] - max_val);
                        sum_exp += bins[k];
                    }

                    // Weighted sum = expected distance
                    float dist = 0.0f;
                    for (int k = 0; k < DFL_REG_MAX; ++k) {
                        dist += (bins[k] / sum_exp) * static_cast<float>(k);
                    }
                    distances[d] = dist;
                }

                // Grid center in pixel coords
                float cx = (static_cast<float>(gx) + 0.5f) * stride;
                float cy = (static_cast<float>(gy) + 0.5f) * stride;

                // dist2bbox: l,t,r,b → x1,y1,x2,y2 normalized
                Detection det;
                det.x1 = clamp01((cx - distances[0] * stride) * inv_dim);
                det.y1 = clamp01((cy - distances[1] * stride) * inv_dim);
                det.x2 = clamp01((cx + distances[2] * stride) * inv_dim);
                det.y2 = clamp01((cy + distances[3] * stride) * inv_dim);
                det.confidence = conf;
                det.class_id   = best_cls;

                result.add(det);
                if (result.count >= MAX_DETECTIONS) break;
            }
            if (result.count >= MAX_DETECTIONS) break;
        }
        if (result.count >= MAX_DETECTIONS) break;
    }

    if (result.count > 1) {
        nms_per_class(result, iou_thr);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
//  YOLO26n T3 ESP — Direct distances (3-scale, 4 channels, NO DFL)
//
//  Output tensors:
//    score0 [1,28,28,5] exp=-3   score1 [1,14,14,5] exp=-2   score2 [1,7,7,5] exp=-3
//    box0   [1,28,28,4] exp=-3   box1   [1,14,14,4] exp=-3   box2   [1,7,7,4] exp=-4
//
//  Box layout: [l, t, r, b] — direct predicted distances (like ESPDet)
//  NO DFL required: reg_max=1, dfl=Identity
//
//  Decode identical to ESPDet:
//    l,t,r,b = dequant(box) (no ReLU needed — values already positive from training)
//    x1 = (cx - l * stride) / 224, etc.
// ═══════════════════════════════════════════════════════════════════════════
DetectionResult postprocess_yolo26_t3_espdl(
    const InferenceEngine* engine,
    float conf_thr,
    float iou_thr)
{
    DetectionResult result;
    result.clear();

    static const char* score_names[NUM_SCALES] = {"score0", "score1", "score2"};
    static const char* box_names[NUM_SCALES]   = {"box0", "box1", "box2"};

    const float inv_dim = 1.0f / static_cast<float>(INPUT_WIDTH);

    for (int s = 0; s < NUM_SCALES; ++s) {
        const int8_t* score_data = static_cast<const int8_t*>(
            engine->get_output_by_name(score_names[s]));
        const int8_t* box_data = static_cast<const int8_t*>(
            engine->get_output_by_name(box_names[s]));

        if (!score_data || !box_data) {
            ESP_LOGW(TAG, "YOLO26T3: output '%s' o '%s' no encontrado",
                     score_names[s], box_names[s]);
            continue;
        }

        int score_exp = engine->get_output_exponent(score_names[s]);
        int box_exp   = engine->get_output_exponent(box_names[s]);
        int grid_h    = GRID_SIZES[s];
        int grid_w    = GRID_SIZES[s];
        int stride    = GRID_STRIDES[s];

        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                int score_offset = (gy * grid_w + gx) * NUM_CLASSES;

                // Find best class (dequant + compare in logit space)
                int best_cls = 0;
                float best_score = -1e9f;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    float s_val = dequant(score_data[score_offset + c], score_exp);
                    if (s_val > best_score) {
                        best_score = s_val;
                        best_cls = c;
                    }
                }

                float conf = sigmoid(best_score);
                if (conf < conf_thr) continue;

                // Direct distance decode (NO DFL) — only 4 channels
                int box_offset = (gy * grid_w + gx) * 4;
                float l = dequant(box_data[box_offset + 0], box_exp);
                float t = dequant(box_data[box_offset + 1], box_exp);
                float r = dequant(box_data[box_offset + 2], box_exp);
                float b = dequant(box_data[box_offset + 3], box_exp);

                // Grid center
                float cx = (static_cast<float>(gx) + 0.5f) * stride;
                float cy = (static_cast<float>(gy) + 0.5f) * stride;

                // dist2bbox normalised
                Detection det;
                det.x1 = clamp01((cx - l * stride) * inv_dim);
                det.y1 = clamp01((cy - t * stride) * inv_dim);
                det.x2 = clamp01((cx + r * stride) * inv_dim);
                det.y2 = clamp01((cy + b * stride) * inv_dim);
                det.confidence = conf;
                det.class_id   = best_cls;

                result.add(det);
                if (result.count >= MAX_DETECTIONS) break;
            }
            if (result.count >= MAX_DETECTIONS) break;
        }
        if (result.count >= MAX_DETECTIONS) break;
    }

    if (result.count > 1) {
        nms_per_class(result, iou_thr);
    }

    return result;
}
