// =============================================================================
// inference_engine.h — Wrapper del motor ESP-DL para modelos .espdl
// Carga modelo desde partición flash vía mmap (zero-copy)
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include "esp_err.h"
#include "app_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Inicializar motor de inferencia.
 *   1. Localiza la partición "models"
 *   2. Memory-map del modelo seleccionado (zero-copy desde flash)
 *   3. Crea instancia dl::Model
 *
 * @param model  Modelo a cargar (MODEL_MBNTV3S, MODEL_YOLO11N, MODEL_YOLO26N)
 * @return ESP_OK o código de error
 */
esp_err_t inference_init(ModelType model);

/**
 * @brief Ejecutar inferencia.
 *   1. Copia input_data al tensor de entrada del modelo
 *   2. Ejecuta forward pass
 *   3. Las salidas quedan accesibles vía inference_get_output()
 *
 * @param input_data  Buffer INT8 [224][224][3] NHWC (en PSRAM)
 * @return ESP_OK o código de error
 */
esp_err_t inference_run(const int8_t *input_data);

/**
 * @brief Obtener tensor de salida por índice.
 *
 * @param index       Índice del tensor (0-based)
 * @param out_data    Puntero a los datos INT8 del tensor (out)
 * @param out_size    Número total de elementos (out)
 * @param out_exponent Exponente para dequantización: float = int8 * 2^exp (out)
 * @return ESP_OK o ESP_ERR_INVALID_ARG
 */
esp_err_t inference_get_output(int index, const int8_t **out_data,
                               int *out_size, int *out_exponent);

/**
 * @brief Obtener shape del tensor de salida.
 *
 * @param index    Índice del tensor
 * @param dims     Array de dimensiones (out, mínimo 4 elementos)
 * @param num_dims Número de dimensiones (out)
 * @return ESP_OK o ESP_ERR_INVALID_ARG
 */
esp_err_t inference_get_output_shape(int index, int *dims, int *num_dims);

/**
 * @brief Número de tensores de salida del modelo activo.
 */
int inference_get_num_outputs(void);

/**
 * @brief Obtener tensor de salida por nombre (evita problemas de ordenación).
 *
 * @param name        Nombre del tensor (e.g., "class_out", "bbox_out")
 * @param out_data    Puntero a los datos del tensor (void*, cast según dtype) (out)
 * @param out_size    Número total de elementos (out)
 * @param out_exponent Exponente para dequantización INT8: float = int8 * 2^exp (out)
 * @param out_dtype   Tipo de dato: 0=FLOAT, otras=INT8,INT16,etc (out, puede ser NULL)
 * @return ESP_OK o ESP_ERR_NOT_FOUND
 */
esp_err_t inference_get_output_by_name(const char *name, const void **out_data,
                                       int *out_size, int *out_exponent,
                                       int *out_dtype);

/**
 * @brief Liberar recursos del motor de inferencia.
 */
void inference_deinit(void);

#ifdef __cplusplus
}
#endif
