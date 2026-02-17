// =============================================================================
// sd_storage.h — Almacenamiento de imágenes en tarjeta SD vía SD_MMC
// =============================================================================
#pragma once

#include "esp_err.h"
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/// Información de estado de la SD
typedef struct {
    uint32_t total_mb;          // Capacidad total en MB
    uint32_t free_mb;           // Espacio libre en MB
    uint32_t photo_count;       // Número de fotos en el directorio
    uint32_t next_counter;      // Siguiente número de archivo
} sd_stats_t;

/**
 * @brief Inicializar tarjeta SD en modo SD_MMC 1-bit.
 *        Monta FAT filesystem y crea directorio de capturas.
 *        Recupera el contador de imágenes desde NVS.
 * @return ESP_OK o código de error
 */
esp_err_t sd_init(void);

/**
 * @brief Guardar buffer JPEG como archivo en la SD.
 * @param data      Puntero a datos JPEG
 * @param len       Tamaño en bytes
 * @param out_name  Buffer para recibir el nombre del archivo (mín 32 chars)
 * @return ESP_OK o código de error
 */
esp_err_t sd_save_jpeg(const uint8_t *data, size_t len, char *out_name);

/**
 * @brief Listar archivos en formato JSON.
 * @param offset    Número de archivos a saltar (paginación)
 * @param limit     Máximo de archivos a retornar
 * @param out_json  Buffer de salida para JSON
 * @param json_size Tamaño del buffer
 * @return Número de archivos escritos al JSON, -1 si error
 */
int sd_list_files(int offset, int limit, char *out_json, size_t json_size);

/**
 * @brief Leer archivo desde la SD.
 * @param filename  Nombre del archivo (solo nombre, no path completo)
 * @param out_data  Puntero que recibirá el buffer allocado (caller debe free())
 * @param out_len   Tamaño del archivo
 * @return ESP_OK o código de error
 */
esp_err_t sd_read_file(const char *filename, uint8_t **out_data, size_t *out_len);

/**
 * @brief Eliminar un archivo.
 * @param filename  Nombre del archivo (solo nombre)
 * @return ESP_OK o código de error
 */
esp_err_t sd_delete_file(const char *filename);

/**
 * @brief Eliminar todas las capturas.
 * @return ESP_OK o código de error
 */
esp_err_t sd_delete_all(void);

/**
 * @brief Obtener estadísticas de la SD.
 */
esp_err_t sd_get_stats(sd_stats_t *stats);

/**
 * @brief Obtener el número total de fotos.
 */
uint32_t sd_get_photo_count(void);

#ifdef __cplusplus
}
#endif
