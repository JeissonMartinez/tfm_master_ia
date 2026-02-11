// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — Inference engine interface (abstract)
//
// Capa de abstracción para dual-path: TFLite Micro y ESP-DL.
// Permite cambiar de runtime sin modificar el loop principal.
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "app_config.h"
#include "esp_err.h"

/// Interfaz abstracta para motores de inferencia.
/// Implementada por TFLiteEngine y EspDlEngine.
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    /// Inicializa el motor con la configuración del modelo.
    /// Aloja arena/buffers en PSRAM, carga el modelo, etc.
    /// @return ESP_OK si la inicialización fue exitosa.
    virtual esp_err_t init(const ModelConfig* config) = 0;

    /// Ejecuta inferencia sobre el input proporcionado.
    /// @param input Puntero al tensor de entrada (INT8, 224×224×3).
    /// @return ESP_OK si la inferencia fue exitosa.
    virtual esp_err_t invoke(const int8_t* input) = 0;

    /// Ejecuta inferencia con input float32.
    /// @param input Puntero al tensor de entrada (float32, 224×224×3).
    /// @return ESP_OK si la inferencia fue exitosa.
    virtual esp_err_t invoke_float(const float* input) = 0;

    /// Obtiene el puntero al tensor de salida por índice.
    /// @param index Índice del tensor de salida (0-based).
    /// @return Puntero al tensor, nullptr si índice inválido.
    virtual const void* get_output(int index = 0) const = 0;

    /// Obtiene las dimensiones del tensor de salida.
    /// @param index Índice del tensor de salida.
    /// @param dims  Array donde se escriben las dimensiones.
    /// @param n_dims Número de dimensiones escritas (out).
    /// @return ESP_OK si éxito.
    virtual esp_err_t get_output_shape(int index, int* dims, int* n_dims) const = 0;

    /// Número de tensores de salida.
    virtual int get_output_count() const = 0;

    /// Bytes de arena/memoria usados por el motor.
    virtual size_t get_arena_used() const = 0;

    /// Libera todos los recursos del motor.
    virtual void deinit() = 0;

    /// Nombre del runtime ("TFLite Micro" o "ESP-DL").
    virtual const char* runtime_name() const = 0;

    /// Tipo de motor.
    virtual EngineType engine_type() const = 0;
};
