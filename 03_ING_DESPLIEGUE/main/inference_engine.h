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

    /// Returns true if the output tensor at `index` is quantized INT8.
    virtual bool is_output_int8(int index = 0) const = 0;

    /// Quantization scale of the output tensor (valid only for INT8 outputs).
    virtual float get_output_scale(int index = 0) const = 0;

    /// Quantization zero-point of the output tensor (valid only for INT8 outputs).
    virtual int32_t get_output_zero_point(int index = 0) const = 0;

    /// Obtiene un tensor de salida por nombre (ESP-DL multi-output).
    /// Por defecto retorna nullptr (sólo EspDlEngine lo implementa).
    virtual const void* get_output_by_name(const char* /*name*/) const { return nullptr; }

    /// Exponent de cuantización power-of-2 para un tensor de salida por nombre.
    /// Fórmula: float_value = int8_value * 2^exponent.
    virtual int get_output_exponent(const char* /*name*/) const { return 0; }

    /// Obtiene las dimensiones del tensor de salida por nombre.
    virtual esp_err_t get_output_shape_by_name(const char* /*name*/,
                                                int* /*dims*/, int* /*n_dims*/) const {
        return ESP_ERR_NOT_SUPPORTED;
    }
};
