// =============================================================================
// inference_engine.cpp — Motor de inferencia ESP-DL 3.2.4
// Carga modelos .espdl desde partición flash vía esp_partition_mmap
//
// ESP-DL API (3.2.4):
//   - dl::Model(const char* addr, fbs::MODEL_LOCATION_IN_FLASH_RODATA, ...)
//   - model->get_input()   → TensorBase* (single input)
//   - model->get_outputs() → map<string, TensorBase*>& (all outputs)
//   - model->run(dl::RUNTIME_MODE_SINGLE_CORE)
//   - TensorBase: ->data (void*), ->shape (vector<int>), ->exponent (int), ->size (int)
// =============================================================================
#include "inference_engine.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include <cstring>
#include <vector>
#include <string>

#include "dl_model_base.hpp"

static const char *TAG = "inference";

// Estado del motor
static dl::Model *s_model = nullptr;
static esp_partition_mmap_handle_t s_mmap_handle = 0;
static ModelType s_active_model = MODEL_COUNT;

// Cache de nombres de salida para acceso por índice
static std::vector<std::string> s_output_names;
static int s_num_outputs = 0;

esp_err_t inference_init(ModelType model)
{
    if (s_model) {
        ESP_LOGW(TAG, "Engine already initialized, deinit first");
        inference_deinit();
    }

    // 1. Obtener offset y tamaño del modelo
    size_t model_offset = 0, model_size = 0;
    get_model_partition_info(model, model_offset, model_size);
    if (model_size == 0) {
        ESP_LOGE(TAG, "Invalid model type: %d", (int)model);
        return ESP_ERR_INVALID_ARG;
    }

    ESP_LOGI(TAG, "Loading %s from partition '%s' offset=0x%x size=%d bytes",
             get_model_name(model), MODELS_PARTITION_LABEL,
             (unsigned)model_offset, (int)model_size);

    // 2. Encontrar partición
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        (esp_partition_subtype_t)0x40,
        MODELS_PARTITION_LABEL
    );
    if (!part) {
        ESP_LOGE(TAG, "Partition '%s' not found", MODELS_PARTITION_LABEL);
        return ESP_ERR_NOT_FOUND;
    }

    ESP_LOGI(TAG, "Partition found: offset=0x%lx, size=%ld",
             (unsigned long)part->address, (long)part->size);

    // Verificar que el modelo cabe
    if (model_offset + model_size > part->size) {
        ESP_LOGE(TAG, "Model exceeds partition: offset(%d)+size(%d) > partition(%ld)",
                 (int)model_offset, (int)model_size, (long)part->size);
        return ESP_ERR_INVALID_SIZE;
    }

    // 3. Memory-map: zero-copy desde flash
    const void *mapped_data = nullptr;
    esp_err_t err = esp_partition_mmap(
        part, model_offset, model_size,
        ESP_PARTITION_MMAP_DATA,
        &mapped_data, &s_mmap_handle
    );
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mmap failed: %s", esp_err_to_name(err));
        return err;
    }

    ESP_LOGI(TAG, "Model mmap'd at %p, %d bytes", mapped_data, (int)model_size);

    // 4. Crear modelo ESP-DL
    //    Pasamos el puntero mmap'd como si fuera RODATA (ambos son lecturas
    //    directas desde flash). El constructor interno crea FbsLoader que
    //    parsea el flatbuffer y construye el grafo de ejecución.
    //
    //    NOTA RENDIMIENTO:
    //    - max_internal_size > 0: asigna buffers intermedios en SRAM interna
    //      (4× más rápida que PSRAM). Con cache 64KB el heap interno libre es
    //      ~31KB contiguos, así que usamos 0 (PSRAM only) para evitar crash.
    //      El mayor beneficio viene de dual-core + -O2 + cache 64KB.
    //    - param_copy=true: copia pesos de flash a PSRAM (más rápido que flash).
    s_model = new (std::nothrow) dl::Model(
        reinterpret_cast<const char *>(mapped_data),
        fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        0,                          // max_internal_size: 0 = PSRAM only (safe)
        dl::MEMORY_MANAGER_GREEDY,
        nullptr,                    // key (sin encriptar)
        true                        // param_copy: copiar pesos a PSRAM
    );
    if (!s_model) {
        ESP_LOGE(TAG, "Failed to create dl::Model (OOM?)");
        esp_partition_munmap(s_mmap_handle);
        s_mmap_handle = 0;
        return ESP_FAIL;
    }

    // Ciclo 2: liberar buffers intermedios no necesarios post-construcción.
    // Referencia: espressif/esp-detection deploy pipeline.
    s_model->minimize();

    // 5. Cache de info de salidas para acceso por índice
    s_output_names.clear();
    auto &outputs = s_model->get_outputs();
    for (auto &kv : outputs) {
        s_output_names.push_back(kv.first);
        ESP_LOGI(TAG, "  Output[%d]: name='%s' shape=%s dtype=%d exponent=%d",
                 (int)s_output_names.size() - 1,
                 kv.first.c_str(),
                 dl::vector_to_string(kv.second->shape).c_str(),
                 (int)kv.second->dtype,
                 kv.second->exponent);
    }
    s_num_outputs = (int)s_output_names.size();

    // Log input tensor info
    dl::TensorBase *input = s_model->get_input();
    if (input) {
        ESP_LOGI(TAG, "  Input: shape=%s dtype=%d exponent=%d",
                 dl::vector_to_string(input->shape).c_str(),
                 (int)input->dtype, input->exponent);
    }

    s_active_model = model;

    ESP_LOGI(TAG, "Model loaded OK. %d output(s). PSRAM free: %d KB",
             s_num_outputs,
             (int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    // Imprimir perfil de memoria (sin profile() completo que ejecuta
    // un forward pass y tarda ~3.5s — descomentar solo para debug)
    // s_model->profile();

    return ESP_OK;
}

esp_err_t inference_run(const int8_t *input_data)
{
    if (!s_model || !input_data) {
        return ESP_ERR_INVALID_STATE;
    }

    // Obtener tensor de entrada y copiar datos
    dl::TensorBase *input = s_model->get_input();
    if (!input || !input->data) {
        ESP_LOGE(TAG, "Input tensor is null");
        return ESP_FAIL;
    }

    memcpy(input->data, input_data, MODEL_INPUT_SIZE);

    // Diagnóstico: verificar que el input no está todo a cero
    static int s_diag_count = 0;
    if (s_diag_count < 3) {
        const int8_t *in = (const int8_t *)input->data;
        int nonzero_in = 0;
        int8_t in_min = 127, in_max = -128;
        for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
            if (in[i] != 0) nonzero_in++;
            if (in[i] < in_min) in_min = in[i];
            if (in[i] > in_max) in_max = in[i];
        }
        ESP_LOGI(TAG, "DIAG input: %d/%d non-zero, min=%d max=%d, exp=%d, first16=[%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d]",
                 nonzero_in, MODEL_INPUT_SIZE, (int)in_min, (int)in_max, input->exponent,
                 in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],
                 in[8],in[9],in[10],in[11],in[12],in[13],in[14],in[15]);
    }

    // Forward pass — dual-core: ESP-DL divide Conv2D entre ambos cores
    // (~2× speedup vs single-core para capas convolucionales)
    s_model->run(dl::RUNTIME_MODE_MULTI_CORE);

    // Diagnóstico: verificar salida después del forward pass
    if (s_diag_count < 3) {
        auto &outputs = s_model->get_outputs();
        for (auto &kv : outputs) {
            dl::TensorBase *out = kv.second;
            int total = out->size;
            int total_bytes = total;
            bool is_float = (out->dtype == dl::DATA_TYPE_FLOAT);

            if (is_float) {
                const float *fdata = (const float *)out->data;
                int num_floats = total / (int)sizeof(float);
                float fmin = fdata[0], fmax = fdata[0];
                int nonzero = 0;
                for (int i = 0; i < num_floats; i++) {
                    if (fdata[i] != 0.0f) nonzero++;
                    if (fdata[i] < fmin) fmin = fdata[i];
                    if (fdata[i] > fmax) fmax = fdata[i];
                }
                ESP_LOGI(TAG, "DIAG output '%s' [FLOAT]: %d/%d non-zero, min=%.3f max=%.3f, shape=%s",
                         kv.first.c_str(), nonzero, num_floats, fmin, fmax,
                         dl::vector_to_string(out->shape).c_str());
            } else {
                const int8_t *odata = (const int8_t *)out->data;
                int nonzero = 0;
                int8_t minv = 127, maxv = -128;
                for (int i = 0; i < total; i++) {
                    if (odata[i] != 0) nonzero++;
                    if (odata[i] < minv) minv = odata[i];
                    if (odata[i] > maxv) maxv = odata[i];
                }
                float scale = 1.0f;
                for (int e = 0; e < -out->exponent; e++) scale *= 0.5f;
                for (int e = 0; e < out->exponent; e++) scale *= 2.0f;
                float deq_max = maxv * scale;
                float sig_max = 1.0f / (1.0f + expf(-deq_max));
                ESP_LOGI(TAG, "DIAG output '%s' [INT8]: %d/%d nz, raw=[%d,%d] exp=%d deq_max=%.3f sig_max=%.4f shape=%s",
                         kv.first.c_str(), nonzero, total, (int)minv, (int)maxv,
                         out->exponent, deq_max, sig_max,
                         dl::vector_to_string(out->shape).c_str());
            }
        }
        s_diag_count++;
    }

    return ESP_OK;
}

esp_err_t inference_get_output(int index, const int8_t **out_data,
                               int *out_size, int *out_exponent)
{
    if (!s_model || index < 0 || index >= s_num_outputs) {
        return ESP_ERR_INVALID_ARG;
    }

    dl::TensorBase *output = s_model->get_output(s_output_names[index]);
    if (!output) {
        ESP_LOGE(TAG, "Output tensor '%s' not found", s_output_names[index].c_str());
        return ESP_FAIL;
    }

    if (out_data) {
        *out_data = static_cast<const int8_t *>(output->data);
    }
    if (out_size) {
        *out_size = output->size;   // total elements including padding
    }
    if (out_exponent) {
        *out_exponent = output->exponent;
    }

    return ESP_OK;
}

esp_err_t inference_get_output_shape(int index, int *dims, int *num_dims)
{
    if (!s_model || index < 0 || index >= s_num_outputs) {
        return ESP_ERR_INVALID_ARG;
    }

    dl::TensorBase *output = s_model->get_output(s_output_names[index]);
    if (!output) return ESP_FAIL;

    if (num_dims) {
        *num_dims = (int)output->shape.size();
    }
    if (dims) {
        for (int i = 0; i < (int)output->shape.size() && i < 4; i++) {
            dims[i] = output->shape[i];
        }
    }

    return ESP_OK;
}

int inference_get_num_outputs(void)
{
    return s_num_outputs;
}

esp_err_t inference_get_output_by_name(const char *name, const void **out_data,
                                       int *out_size, int *out_exponent,
                                       int *out_dtype)
{
    if (!s_model || !name) {
        return ESP_ERR_INVALID_ARG;
    }

    std::string key(name);
    dl::TensorBase *output = s_model->get_output(key);
    if (!output) {
        ESP_LOGE(TAG, "Output tensor '%s' not found", name);
        return ESP_ERR_NOT_FOUND;
    }

    if (out_data) {
        *out_data = output->data;
    }
    if (out_size) {
        *out_size = output->size;
    }
    if (out_exponent) {
        *out_exponent = output->exponent;
    }
    if (out_dtype) {
        *out_dtype = (int)output->dtype;
    }

    return ESP_OK;
}

void inference_deinit(void)
{
    if (s_model) {
        delete s_model;
        s_model = nullptr;
    }
    if (s_mmap_handle) {
        esp_partition_munmap(s_mmap_handle);
        s_mmap_handle = 0;
    }
    s_output_names.clear();
    s_num_outputs = 0;
    s_active_model = MODEL_COUNT;
    ESP_LOGI(TAG, "Engine deinitialized");
}
