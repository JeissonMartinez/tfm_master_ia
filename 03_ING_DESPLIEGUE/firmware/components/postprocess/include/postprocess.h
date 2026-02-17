// =============================================================================
// postprocess.h — Decodificación de salidas + NMS para los 3 modelos
//
// MBNTv3S: 3 tensores (class, bbox, objectness) + anchors SSD + NMS
// YOLO11n: 1 tensor [1,9,1029] → decode + NMS
// YOLO26n: 1 tensor [1,1029,9] → decode + NMS (transposed)
// =============================================================================
#pragma once

#include <cstdint>
#include "app_config.h"
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---- Estructuras de detección ----

struct BBox {
    float x1, y1, x2, y2;   // Coordenadas normalizadas [0,1] relativas a 224×224
};

struct Detection {
    BBox  bbox;
    int   class_id;
    float score;
};

struct DetectionResult {
    Detection detections[MAX_DETECTIONS];
    int       num_detections;
    ModelType model;
};

// ---- API pública ----

/**
 * @brief Inicializar postprocesador para el modelo seleccionado.
 *        Para MBNTv3S: pre-computa anchors SSD.
 */
void postprocess_init(ModelType model);

/**
 * @brief Decodificar salidas del modelo activo en detecciones.
 *        Llama internamente a inference_get_output() para obtener tensores.
 *        Aplica dequantización, filtrado por score, decode bbox, y NMS.
 */
void postprocess_decode(DetectionResult *result);

// ---- NMS (reutilizable) ----

/**
 * @brief Non-Maximum Suppression.
 *
 * @param candidates    Array de detecciones candidatas (será modificado/reordenado)
 * @param num_candidates Número de candidatos
 * @param iou_threshold IoU threshold para supresión
 * @param output        Array destino para detecciones finales
 * @param out_count     Número de detecciones finales (out)
 * @param max_output    Capacidad del array de salida
 */
void nms_process(Detection *candidates, int num_candidates,
                 float iou_threshold,
                 Detection *output, int *out_count, int max_output);

// ---- Utilidad ----

/**
 * @brief Calcular IoU entre dos bounding boxes.
 */
float bbox_iou(const BBox &a, const BBox &b);

/**
 * @brief Clamp bbox a [0,1].
 */
void bbox_clamp(BBox &box);

#ifdef __cplusplus
}
#endif
