// =============================================================================
// metrics.h — Recolección de métricas de rendimiento por frame
// Timing, memoria, temperatura, FPS
// =============================================================================
#pragma once

#include <cstdint>

// Forward declaration
struct DetectionResult;

#ifdef __cplusplus
extern "C" {
#endif

// ---- Fases de temporización ----
enum MetricsPhase {
    PHASE_CAPTURE     = 0,
    PHASE_PREPROCESS  = 1,
    PHASE_INFERENCE   = 2,
    PHASE_POSTPROCESS = 3,
    PHASE_COUNT       = 4
};

// ---- Métricas por frame ----
struct FrameMetrics {
    int64_t  frame_start_us;                    // Timestamp inicio frame
    int64_t  phase_start_us[PHASE_COUNT];       // Timestamp inicio cada fase
    float    phase_ms[PHASE_COUNT];             // Duración de cada fase (ms)
    float    total_ms;                          // Duración total del frame (ms)
};

// ---- Métricas globales acumuladas ----
struct GlobalMetrics {
    uint32_t total_frames;         // Frames procesados desde inicio
    float    avg_fps;              // FPS media (ventana deslizante)
    float    avg_inference_ms;     // Inferencia media (ventana deslizante)
    float    avg_total_ms;         // Total medio por frame
    float    max_inference_ms;     // Peor caso inferencia
    float    max_total_ms;         // Peor caso total
    float    temperature_c;        // Temperatura SoC (°C)
    uint32_t free_psram_kb;        // PSRAM libre (KB)
    uint32_t free_internal_kb;     // RAM interna libre (KB)
    uint32_t min_free_psram_kb;    // Mínimo histórico PSRAM libre
    int      last_num_detections;  // Detecciones en último frame

    // Para cálculo de FPS (ventana deslizante)
    int64_t  window_start_us;
    uint32_t window_frames;
};

// ---- API ----

/**
 * @brief Inicializar métricas globales y sensor de temperatura.
 */
void metrics_init(GlobalMetrics *gm);

/**
 * @brief Marcar inicio de un frame.
 */
void metrics_start_frame(FrameMetrics *fm);

/**
 * @brief Marcar inicio de una fase dentro del frame.
 */
void metrics_start_phase(FrameMetrics *fm, MetricsPhase phase);

/**
 * @brief Marcar fin de una fase (calcula duración).
 */
void metrics_end_phase(FrameMetrics *fm, MetricsPhase phase);

/**
 * @brief Marcar fin del frame (calcula total_ms).
 */
void metrics_end_frame(FrameMetrics *fm);

/**
 * @brief Actualizar métricas globales con datos del frame actual.
 */
void metrics_update_global(GlobalMetrics *gm, const FrameMetrics *fm,
                           const DetectionResult *result);

/**
 * @brief Obtener FPS actual (atajo a global metrics).
 */
float metrics_get_fps(void);

#ifdef __cplusplus
}
#endif
