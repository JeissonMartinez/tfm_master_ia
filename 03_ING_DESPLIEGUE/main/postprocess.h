// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Post-processing for 6 model architectures
//
//  MBNTv2_ssdlite_v1 : 3 tensors  → (1,1470,1) obj + (1,1470,5) cls + (1,1470,4) box
//  YOLO11n_v1        : 1 tensor   → (1,9,1029)  transposed [box+cls, proposals]
//  YOLO26n_v1        : 1 tensor   → (1,300,6)   end2end [x1,y1,x2,y2,conf,cls_id]
//  ESPDet Pico T4    : 6 tensors  → FCOS 3-scale (score0-2, box0-2), direct dist
//  YOLO26n T2 ESP    : 6 tensors  → DFL 3-scale (score0-2, box0-2), DFL integral
//  YOLO26n T3 ESP    : 6 tensors  → Direct 3-scale (score0-2, box0-2), no DFL
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "app_config.h"
#include "inference_engine.h"
#include "esp_err.h"

/// Initialise postprocessing (may pre-allocate scratch buffers in PSRAM).
esp_err_t postprocess_init();

/// MBNTv2 SSD-Lite: 3 output tensors → DetectionResult.
/// @param objectness  (1470,1) objectness scores [0,1]
/// @param class_probs (1470,5) per-class sigmoid scores
/// @param bbox_preds  (1470,4) normalised [xc,yc,w,h]
/// @param conf_thr    combined confidence threshold (obj * cls)
/// @param iou_thr     IoU threshold for NMS
DetectionResult postprocess_mobilenet(
    const float* objectness,   // 1470 × 1
    const float* class_probs,  // 1470 × 5
    const float* bbox_preds,   // 1470 × 4
    float conf_thr = DEFAULT_CONF_THRESHOLD,
    float iou_thr  = DEFAULT_IOU_THRESHOLD);

/// YOLO11n: single output (9, 1029) → DetectionResult.
/// Layout: rows 0-3 = [xc, yc, w, h], rows 4-8 = class scores.
/// @param raw_output  pointer to 9×1029 floats (row-major)
/// @param conf_thr    class-score threshold
/// @param iou_thr     IoU threshold for NMS
DetectionResult postprocess_yolo11(
    const float* raw_output,   // 9 × 1029
    float conf_thr = DEFAULT_CONF_THRESHOLD,
    float iou_thr  = DEFAULT_IOU_THRESHOLD);

/// YOLO26n end-to-end: single output (300, 6) → DetectionResult.
/// Layout: [x1, y1, x2, y2, confidence, class_id] in absolute pixel coords.
/// NMS already applied by the model — only confidence filtering needed.
/// @param raw_output  pointer to 300×6 floats
/// @param conf_thr    confidence threshold
/// @param coords_normalized  true if coords are already in [0,1] (fullint8 models)
DetectionResult postprocess_yolo26(
    const float* raw_output,   // 300 × 6
    float conf_thr = DEFAULT_CONF_THRESHOLD,
    bool coords_normalized = false);

/// Release scratch buffers.
void postprocess_deinit();

// ═══════════════════════════════════════════════════════════════════════════
//  ESP-DL ESPDL postprocessors  (multi-output, INT8, power-of-2 quantized)
// ═══════════════════════════════════════════════════════════════════════════

/// ESPDet Pico T4: FCOS anchor-free detector — 3 scales, 6 output tensors.
/// Each scale has score[HxWx5] + box[HxWx4] (direct distances l,t,r,b).
/// Pipeline: dequant → sigmoid(scores) → filter → decode dist2bbox → NMS.
/// @param engine  Pointer to EspDlEngine (for get_output_by_name)
/// @param conf_thr  Confidence threshold after sigmoid
/// @param iou_thr   IoU threshold for NMS
DetectionResult postprocess_espdet_espdl(
    const InferenceEngine* engine,
    float conf_thr = ESPDET_CONF_THRESHOLD,
    float iou_thr  = ESPDET_IOU_THRESHOLD);

/// YOLO26n T2 ESP: DFL-based detector — 3 scales, 6 output tensors.
/// Each scale has score[HxWx5] + box[HxWx64] (4 × DFL_REG_MAX=16 bins).
/// Pipeline: dequant → sigmoid(scores) → filter → DFL integral → dist2bbox → NMS.
/// @param engine  Pointer to EspDlEngine (for get_output_by_name)
/// @param conf_thr  Confidence threshold after sigmoid
/// @param iou_thr   IoU threshold for NMS
DetectionResult postprocess_yolo26_espdl(
    const InferenceEngine* engine,
    float conf_thr = YOLO26ESP_CONF_THRESHOLD,
    float iou_thr  = YOLO26ESP_IOU_THRESHOLD);

/// YOLO26n T3 ESP: Direct distance detector — 3 scales, 6 output tensors.
/// Each scale has score[HxWx5] + box[HxWx4] (direct distances l,t,r,b).
/// Pipeline: dequant → sigmoid(scores) → filter → dist2bbox → NMS.
/// NO DFL integral required (reg_max=1, Identity).
/// @param engine  Pointer to EspDlEngine (for get_output_by_name)
/// @param conf_thr  Confidence threshold after sigmoid
/// @param iou_thr   IoU threshold for NMS
DetectionResult postprocess_yolo26_t3_espdl(
    const InferenceEngine* engine,
    float conf_thr = YOLO26T3ESP_CONF_THRESHOLD,
    float iou_thr  = YOLO26T3ESP_IOU_THRESHOLD);
