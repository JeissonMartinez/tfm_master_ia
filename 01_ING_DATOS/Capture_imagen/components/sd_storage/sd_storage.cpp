// =============================================================================
// sd_storage.cpp — Almacenamiento de imágenes JPEG en tarjeta SD
//
// Usa SD_MMC en modo 1-bit para la Freenove ESP32-S3 WROOM.
// Archivos nombrados secuencialmente: IMG_000001.jpg, IMG_000002.jpg, ...
// Contador persistido en NVS para sobrevivir reboots.
// =============================================================================
#include "sd_storage.h"
#include "app_config.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "ff.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/unistd.h>

static const char *TAG = "sd_storage";

static sdmmc_card_t *s_card = nullptr;
static uint32_t s_img_counter = 0;
static SemaphoreHandle_t s_sd_mutex = nullptr;

// =============================================================================
// Persistir / recuperar contador de NVS
// =============================================================================
static void load_counter(void)
{
    nvs_handle_t h;
    if (nvs_open(CAPTURE_NVS_NAMESPACE, NVS_READONLY, &h) == ESP_OK) {
        nvs_get_u32(h, CAPTURE_NVS_COUNTER_KEY, &s_img_counter);
        nvs_close(h);
        ESP_LOGI(TAG, "Restored image counter: %lu", (unsigned long)s_img_counter);
    } else {
        s_img_counter = 0;
        ESP_LOGI(TAG, "No saved counter, starting at 0");
    }
}

static void save_counter(void)
{
    nvs_handle_t h;
    if (nvs_open(CAPTURE_NVS_NAMESPACE, NVS_READWRITE, &h) == ESP_OK) {
        nvs_set_u32(h, CAPTURE_NVS_COUNTER_KEY, s_img_counter);
        nvs_commit(h);
        nvs_close(h);
    }
}

// =============================================================================
// sd_init
// =============================================================================
esp_err_t sd_init(void)
{
    ESP_LOGI(TAG, "Initializing SD card (SD_MMC 1-bit mode)...");

    s_sd_mutex = xSemaphoreCreateMutex();
    if (!s_sd_mutex) {
        ESP_LOGE(TAG, "Failed to create SD mutex");
        return ESP_ERR_NO_MEM;
    }

    // Configurar host SD_MMC
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.flags = SDMMC_HOST_FLAG_1BIT;   // 1-bit mode
    host.max_freq_khz = SDMMC_FREQ_DEFAULT;

    // Configurar slot con pines GPIO
    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.clk = (gpio_num_t)SD_MMC_CLK_PIN;
    slot_config.cmd = (gpio_num_t)SD_MMC_CMD_PIN;
    slot_config.d0  = (gpio_num_t)SD_MMC_D0_PIN;
    slot_config.d1  = GPIO_NUM_NC;
    slot_config.d2  = GPIO_NUM_NC;
    slot_config.d3  = GPIO_NUM_NC;
    slot_config.width = 1;   // 1-bit bus
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    // Montar FAT filesystem
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024
    };

    esp_err_t ret = esp_vfs_fat_sdmmc_mount(
        SD_MOUNT_POINT, &host, &slot_config, &mount_config, &s_card);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount FAT filesystem on SD card.");
            ESP_LOGE(TAG, "If the card is new, format it as FAT32 first.");
        } else {
            ESP_LOGE(TAG, "SD card mount failed: %s (0x%x)", esp_err_to_name(ret), ret);
            ESP_LOGE(TAG, "Verify SD card is inserted and pins CLK=%d CMD=%d D0=%d are correct.",
                     SD_MMC_CLK_PIN, SD_MMC_CMD_PIN, SD_MMC_D0_PIN);
        }
        return ret;
    }

    // Info de la tarjeta
    sdmmc_card_print_info(stdout, s_card);

    // Crear directorio de capturas si no existe
    struct stat st;
    if (stat(SD_CAPTURE_DIR, &st) != 0) {
        ESP_LOGI(TAG, "Creating capture directory: %s", SD_CAPTURE_DIR);
        if (mkdir(SD_CAPTURE_DIR, 0775) != 0) {
            ESP_LOGE(TAG, "Failed to create directory %s", SD_CAPTURE_DIR);
            return ESP_FAIL;
        }
    }

    // Recuperar contador de NVS
    load_counter();

    ESP_LOGI(TAG, "SD card ready. Mount: %s, Captures: %s", SD_MOUNT_POINT, SD_CAPTURE_DIR);
    return ESP_OK;
}

// =============================================================================
// sd_save_jpeg
// =============================================================================
esp_err_t sd_save_jpeg(const uint8_t *data, size_t len, char *out_name)
{
    if (!data || len == 0) return ESP_ERR_INVALID_ARG;
    if (xSemaphoreTake(s_sd_mutex, pdMS_TO_TICKS(5000)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    s_img_counter++;

    // Generar nombre de archivo
    char filename[64];
    snprintf(filename, sizeof(filename), "%s_%06lu.jpg",
             CAPTURE_FILE_PREFIX, (unsigned long)s_img_counter);

    char filepath[128];
    snprintf(filepath, sizeof(filepath), "%s/%s", SD_CAPTURE_DIR, filename);

    // Escribir archivo
    FILE *f = fopen(filepath, "wb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to open file: %s", filepath);
        xSemaphoreGive(s_sd_mutex);
        return ESP_FAIL;
    }

    size_t written = fwrite(data, 1, len, f);
    fclose(f);

    if (written != len) {
        ESP_LOGE(TAG, "Write error: wrote %d of %d bytes to %s",
                 (int)written, (int)len, filepath);
        xSemaphoreGive(s_sd_mutex);
        return ESP_FAIL;
    }

    // Persistir contador cada 10 fotos (balance entre wear y fiabilidad)
    if (s_img_counter % 10 == 0) {
        save_counter();
    }

    if (out_name) {
        strlcpy(out_name, filename, 32);
    }

    ESP_LOGI(TAG, "Saved: %s (%d bytes)", filename, (int)len);
    xSemaphoreGive(s_sd_mutex);
    return ESP_OK;
}

// =============================================================================
// sd_list_files — Lista archivos con paginación, retorna JSON
// =============================================================================
int sd_list_files(int offset, int limit, char *out_json, size_t json_size)
{
    if (xSemaphoreTake(s_sd_mutex, pdMS_TO_TICKS(5000)) != pdTRUE) {
        return -1;
    }

    DIR *dir = opendir(SD_CAPTURE_DIR);
    if (!dir) {
        ESP_LOGE(TAG, "Failed to open directory: %s", SD_CAPTURE_DIR);
        xSemaphoreGive(s_sd_mutex);
        return -1;
    }

    int pos = 0;
    int count = 0;
    int skipped = 0;

    pos += snprintf(out_json + pos, json_size - pos, "[");

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        // Solo archivos .jpg
        size_t name_len = strlen(entry->d_name);
        if (name_len < 5) continue;
        if (strcmp(entry->d_name + name_len - 4, ".jpg") != 0 &&
            strcmp(entry->d_name + name_len - 4, ".JPG") != 0) continue;

        if (skipped < offset) {
            skipped++;
            continue;
        }
        if (count >= limit) break;

        // Obtener tamaño del archivo
        char filepath[128];
        snprintf(filepath, sizeof(filepath), "%s/%s", SD_CAPTURE_DIR, entry->d_name);
        struct stat st;
        size_t fsize = 0;
        if (stat(filepath, &st) == 0) {
            fsize = st.st_size;
        }

        if (count > 0) {
            pos += snprintf(out_json + pos, json_size - pos, ",");
        }
        pos += snprintf(out_json + pos, json_size - pos,
            "{\"name\":\"%s\",\"size\":%d}",
            entry->d_name, (int)fsize);

        if ((size_t)pos >= json_size - 100) break; // Safety margin
        count++;
    }

    pos += snprintf(out_json + pos, json_size - pos, "]");
    closedir(dir);
    xSemaphoreGive(s_sd_mutex);
    return count;
}

// =============================================================================
// sd_read_file — Leer archivo completo (caller debe free() el buffer)
// =============================================================================
esp_err_t sd_read_file(const char *filename, uint8_t **out_data, size_t *out_len)
{
    if (!filename || !out_data || !out_len) return ESP_ERR_INVALID_ARG;
    if (xSemaphoreTake(s_sd_mutex, pdMS_TO_TICKS(5000)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    char filepath[128];
    snprintf(filepath, sizeof(filepath), "%s/%s", SD_CAPTURE_DIR, filename);

    struct stat st;
    if (stat(filepath, &st) != 0) {
        ESP_LOGW(TAG, "File not found: %s", filepath);
        xSemaphoreGive(s_sd_mutex);
        return ESP_ERR_NOT_FOUND;
    }

    size_t fsize = st.st_size;
    uint8_t *buf = (uint8_t *)malloc(fsize);
    if (!buf) {
        ESP_LOGE(TAG, "Cannot allocate %d bytes for file read", (int)fsize);
        xSemaphoreGive(s_sd_mutex);
        return ESP_ERR_NO_MEM;
    }

    FILE *f = fopen(filepath, "rb");
    if (!f) {
        free(buf);
        xSemaphoreGive(s_sd_mutex);
        return ESP_FAIL;
    }

    size_t read = fread(buf, 1, fsize, f);
    fclose(f);

    if (read != fsize) {
        ESP_LOGE(TAG, "Read error: got %d of %d bytes", (int)read, (int)fsize);
        free(buf);
        xSemaphoreGive(s_sd_mutex);
        return ESP_FAIL;
    }

    *out_data = buf;
    *out_len = fsize;
    xSemaphoreGive(s_sd_mutex);
    return ESP_OK;
}

// =============================================================================
// sd_delete_file
// =============================================================================
esp_err_t sd_delete_file(const char *filename)
{
    if (!filename) return ESP_ERR_INVALID_ARG;
    if (xSemaphoreTake(s_sd_mutex, pdMS_TO_TICKS(5000)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    char filepath[128];
    snprintf(filepath, sizeof(filepath), "%s/%s", SD_CAPTURE_DIR, filename);

    int ret = unlink(filepath);
    xSemaphoreGive(s_sd_mutex);

    if (ret != 0) {
        ESP_LOGW(TAG, "Failed to delete: %s", filepath);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Deleted: %s", filename);
    return ESP_OK;
}

// =============================================================================
// sd_delete_all
// =============================================================================
esp_err_t sd_delete_all(void)
{
    if (xSemaphoreTake(s_sd_mutex, pdMS_TO_TICKS(10000)) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    DIR *dir = opendir(SD_CAPTURE_DIR);
    if (!dir) {
        xSemaphoreGive(s_sd_mutex);
        return ESP_FAIL;
    }

    int deleted = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        size_t name_len = strlen(entry->d_name);
        if (name_len < 5) continue;
        if (strcmp(entry->d_name + name_len - 4, ".jpg") != 0 &&
            strcmp(entry->d_name + name_len - 4, ".JPG") != 0) continue;

        char filepath[128];
        snprintf(filepath, sizeof(filepath), "%s/%s", SD_CAPTURE_DIR, entry->d_name);
        if (unlink(filepath) == 0) deleted++;
    }
    closedir(dir);

    // Reset counter
    s_img_counter = 0;
    save_counter();

    ESP_LOGI(TAG, "Deleted %d files, counter reset to 0", deleted);
    xSemaphoreGive(s_sd_mutex);
    return ESP_OK;
}

// =============================================================================
// sd_get_stats
// =============================================================================
esp_err_t sd_get_stats(sd_stats_t *stats)
{
    if (!stats) return ESP_ERR_INVALID_ARG;
    if (!s_card) return ESP_ERR_INVALID_STATE;

    // Espacio en la SD (desde CSD del card descriptor)
    uint64_t total_bytes = (uint64_t)s_card->csd.capacity * s_card->csd.sector_size;
    stats->total_mb = (uint32_t)(total_bytes / (1024 * 1024));

    // Espacio libre vía FATFS
    FATFS *fs;
    DWORD free_clust;
    if (f_getfree("0:", &free_clust, &fs) == FR_OK) {
        uint64_t free_bytes = (uint64_t)free_clust * fs->csize *
                              s_card->csd.sector_size;
        stats->free_mb = (uint32_t)(free_bytes / (1024 * 1024));
    } else {
        // Fallback: usar info del card descriptor
        stats->free_mb = stats->total_mb;
    }

    // Contar archivos
    stats->photo_count = sd_get_photo_count();
    stats->next_counter = s_img_counter + 1;

    return ESP_OK;
}

// =============================================================================
// sd_get_photo_count
// =============================================================================
uint32_t sd_get_photo_count(void)
{
    DIR *dir = opendir(SD_CAPTURE_DIR);
    if (!dir) return 0;

    uint32_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        size_t name_len = strlen(entry->d_name);
        if (name_len < 5) continue;
        if (strcmp(entry->d_name + name_len - 4, ".jpg") == 0 ||
            strcmp(entry->d_name + name_len - 4, ".JPG") == 0) {
            count++;
        }
    }
    closedir(dir);
    return count;
}
