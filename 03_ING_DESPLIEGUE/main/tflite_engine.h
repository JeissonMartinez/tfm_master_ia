// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — TFLite Micro engine
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "inference_engine.h"

/// Motor de inferencia basado en TensorFlow Lite for Microcontrollers.
/// Usa MicroMutableOpResolver con los 26 operadores necesarios.
/// Arena de tensores en PSRAM.
class TFLiteEngine : public InferenceEngine {
public:
    TFLiteEngine() = default;
    ~TFLiteEngine() override;

    esp_err_t init(const ModelConfig* config) override;
    esp_err_t invoke(const int8_t* input) override;
    esp_err_t invoke_float(const float* input) override;
    const void* get_output(int index = 0) const override;
    esp_err_t get_output_shape(int index, int* dims, int* n_dims) const override;
    int get_output_count() const override;
    size_t get_arena_used() const override;
    void deinit() override;
    const char* runtime_name() const override { return "TFLite Micro"; }
    EngineType engine_type() const override { return EngineType::TFLITE_MICRO; }

private:
    struct Impl;
    Impl* m_impl = nullptr;
};
