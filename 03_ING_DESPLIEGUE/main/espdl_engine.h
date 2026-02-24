// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — ESP-DL engine
//
// Motor de inferencia basado en ESP-DL v3.x (Espressif Deep Learning).
// Carga modelos .espdl desde particiones flash independientes.
// Aprovecha scheduling dual-core y operadores SIMD nativos del LX7.
//
// API key:
//   dl::Model constructor → carga desde partición flash (mmap automático)
//   TensorBase::assign()  → alimentar input INT8
//   model->run(MULTI_CORE) → inferencia dual-core
//   model->get_output("name") → acceso a tensores por nombre
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "inference_engine.h"

/// Motor de inferencia basado en ESP-DL (Espressif Deep Learning).
/// Carga modelos .espdl desde particiones flash independientes.
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

    // ESP-DL outputs are always INT8 with power-of-2 exponent
    bool is_output_int8(int index = 0) const override { return true; }
    float get_output_scale(int index = 0) const override;
    int32_t get_output_zero_point(int index = 0) const override { return 0; }

    // ESP-DL multi-output access by name
    const void* get_output_by_name(const char* name) const override;
    int get_output_exponent(const char* name) const override;
    esp_err_t get_output_shape_by_name(const char* name,
                                        int* dims, int* n_dims) const override;

private:
    struct Impl;
    Impl* m_impl = nullptr;
};
