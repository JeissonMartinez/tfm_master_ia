// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — ESP-DL engine
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "inference_engine.h"

/// Motor de inferencia basado en ESP-DL (Espressif Deep Learning).
/// Carga modelos .espdl embebidos en flash.
/// Aprovecha scheduling dual-core y operadores SIMD nativos.
class EspDlEngine : public InferenceEngine {
public:
    EspDlEngine() = default;
    ~EspDlEngine() override;

    esp_err_t init(const ModelConfig* config) override;
    esp_err_t invoke(const int8_t* input) override;
    esp_err_t invoke_float(const float* input) override;
    const void* get_output(int index = 0) const override;
    esp_err_t get_output_shape(int index, int* dims, int* n_dims) const override;
    int get_output_count() const override;
    size_t get_arena_used() const override;
    void deinit() override;
    const char* runtime_name() const override { return "ESP-DL"; }
    EngineType engine_type() const override { return EngineType::ESP_DL; }

private:
    struct Impl;
    Impl* m_impl = nullptr;
};
