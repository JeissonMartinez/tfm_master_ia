// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — ESP-DL engine implementation
//
// ESP-DL v3.x usa modelos .espdl (FlatBuffers) cargados desde particiones
// flash independientes.  El runtime automatiza scheduling dual-core y usa
// operadores SIMD optimizados para el Xtensa LX7 del ESP32-S3.
//
// Flujo:
//   1. Construir dl::Model con etiqueta de partición → mmap automático
//   2. model->get_input()→assign({1,224,224,3}, data, -7, INT8)
//   3. model->run(RUNTIME_MODE_MULTI_CORE)
//   4. model->get_output("score0")→get_element_ptr<int8_t>()
// ═══════════════════════════════════════════════════════════════════════════
#include "espdl_engine.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <new>
#include <string>
#include <map>
#include <vector>

// ESP-DL v3.x headers
#if __has_include("dl_model_base.hpp")
    #include "dl_model_base.hpp"
    #define HAS_ESP_DL 1
#else
    #define HAS_ESP_DL 0
#endif

static const char* TAG = "espdl_engine";

// ═══════════════════════════════════════════════════════════════════════════
//  Private implementation (PIMPL)
// ═══════════════════════════════════════════════════════════════════════════
struct EspDlEngine::Impl {
    const ModelConfig* config      = nullptr;
    bool               initialized = false;
    size_t             model_ram_bytes = 0;  // PSRAM consumed by model

#if HAS_ESP_DL
    dl::Model*         model       = nullptr;

    // Cacheamos nombres de output para acceso por índice
    std::vector<std::string>  output_names;

    // Helper: obtener TensorBase* por nombre (const-safe)
    dl::TensorBase* get_tensor(const char* name) const {
        if (!model || !name) return nullptr;
        // get_outputs() returns a non-const reference to the map
        // we const_cast because dl::Model API doesn't provide const access
        auto& outputs = const_cast<dl::Model*>(model)->get_outputs();
        auto it = outputs.find(std::string(name));
        if (it != outputs.end()) return it->second;
        return nullptr;
    }

    // Helper: obtener TensorBase* por índice
    dl::TensorBase* get_tensor_by_idx(int index) const {
        if (index < 0 || index >= static_cast<int>(output_names.size()))
            return nullptr;
        return get_tensor(output_names[index].c_str());
    }
#endif
};

// ═══════════════════════════════════════════════════════════════════════════
//  Lifecycle
// ═══════════════════════════════════════════════════════════════════════════

EspDlEngine::~EspDlEngine() {
    deinit();
}

esp_err_t EspDlEngine::init(const ModelConfig* config) {
    if (!config) {
        ESP_LOGE(TAG, "Configuración de modelo nula");
        return ESP_ERR_INVALID_ARG;
    }

    deinit();

    m_impl = new (std::nothrow) Impl();
    if (!m_impl) {
        ESP_LOGE(TAG, "No se pudo alojar Impl");
        return ESP_ERR_NO_MEM;
    }
    m_impl->config = config;

#if HAS_ESP_DL
    const char* partition_label = config->espdl_partition;
    if (!partition_label || partition_label[0] == '\0') {
        ESP_LOGE(TAG, "Partición ESPDL no configurada para modelo: %s", config->name);
        return ESP_ERR_INVALID_ARG;
    }

    ESP_LOGI(TAG, "Cargando modelo ESP-DL: %s (partición: %s)",
             config->name, partition_label);
    ESP_LOGI(TAG, "   Heap libre antes: interno=%u KB, PSRAM=%u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    size_t psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    // Construir modelo desde partición flash
    // max_internal_size=0 → máximo uso de PSRAM
    // param_copy=true → copia parámetros a RAM (mejor rendimiento)
    try {
        m_impl->model = new dl::Model(
            partition_label,
            fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
            0,                          // max_internal_size
            dl::MEMORY_MANAGER_GREEDY,  // Memory manager type
            nullptr,                    // encryption key
            true                        // param_copy
        );
    } catch (const std::exception& e) {
        ESP_LOGE(TAG, "Excepción al cargar modelo: %s", e.what());
        return ESP_ERR_NOT_FOUND;
    } catch (...) {
        ESP_LOGE(TAG, "Excepción desconocida al cargar modelo desde partición '%s'",
                 partition_label);
        return ESP_ERR_NOT_FOUND;
    }

    if (!m_impl->model) {
        ESP_LOGE(TAG, "dl::Model constructor returned null");
        return ESP_ERR_NOT_FOUND;
    }

    // Cachear nombres de outputs para acceso por índice
    auto& outputs_map = m_impl->model->get_outputs();
    for (auto& [name, tensor] : outputs_map) {
        m_impl->output_names.push_back(name);
        auto& shape = tensor->shape;
        ESP_LOGI(TAG, "   Output: '%s' dtype=%d exp=%d shape=[%d%s%s%s]",
                 name.c_str(), tensor->dtype, tensor->exponent,
                 shape.size() > 0 ? shape[0] : 0,
                 shape.size() > 1 ? ("," + std::to_string(shape[1])).c_str() : "",
                 shape.size() > 2 ? ("," + std::to_string(shape[2])).c_str() : "",
                 shape.size() > 3 ? ("," + std::to_string(shape[3])).c_str() : "");
    }

    // Verificar input
    dl::TensorBase* input = m_impl->model->get_input();
    if (input) {
        ESP_LOGI(TAG, "   Input:  dtype=%d exp=%d shape=[%d,%d,%d,%d]",
                 input->dtype, input->exponent,
                 input->shape.size() > 0 ? input->shape[0] : 0,
                 input->shape.size() > 1 ? input->shape[1] : 0,
                 input->shape.size() > 2 ? input->shape[2] : 0,
                 input->shape.size() > 3 ? input->shape[3] : 0);
    }

    ESP_LOGI(TAG, "   Heap libre después: interno=%u KB, PSRAM=%u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    // Capture RAM consumed by the model (PSRAM delta)
    size_t psram_after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    m_impl->model_ram_bytes = (psram_before > psram_after)
                              ? (psram_before - psram_after) : 0;
    ESP_LOGI(TAG, "   Modelo RAM: %u KB",
             (unsigned)(m_impl->model_ram_bytes / 1024));

    ESP_LOGI(TAG, "✅ ESP-DL inicializado: %s (%d outputs)",
             config->name, (int)m_impl->output_names.size());

    m_impl->initialized = true;
    return ESP_OK;

#else
    ESP_LOGW(TAG, "⚠️ ESP-DL no disponible en la compilación actual");
    ESP_LOGW(TAG, "   Asegúrese de que esp-dl está en idf_component.yml");
    m_impl->initialized = false;
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
//  Inference
// ═══════════════════════════════════════════════════════════════════════════

esp_err_t EspDlEngine::invoke(const int8_t* input) {
    if (!m_impl || !m_impl->initialized) return ESP_ERR_INVALID_STATE;

#if HAS_ESP_DL
    if (!input) return ESP_ERR_INVALID_ARG;

    // Alimentar input tensor
    // Los modelos ESPDL esperan exponent=-7: val_float = val_int8 * 2^(-7)
    dl::TensorBase* input_tensor = m_impl->model->get_input();
    if (!input_tensor) {
        ESP_LOGE(TAG, "No se pudo obtener tensor de entrada");
        return ESP_ERR_INVALID_STATE;
    }

    std::vector<int> input_shape = {1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS};
    input_tensor->assign(input_shape, static_cast<const void*>(input),
                         -7, dl::DATA_TYPE_INT8);

    // Ejecutar inferencia dual-core
    m_impl->model->run(dl::RUNTIME_MODE_MULTI_CORE);

    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

esp_err_t EspDlEngine::invoke_float(const float* input) {
    if (!m_impl || !m_impl->initialized) return ESP_ERR_INVALID_STATE;

#if HAS_ESP_DL
    // ESP-DL INT8 models don't support float input directly.
    // The caller must use image_preprocess_espdl() to get INT8 data.
    ESP_LOGE(TAG, "ESP-DL INT8 models no soportan input float32 — usar invoke(int8_t*)");
    return ESP_ERR_NOT_SUPPORTED;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
//  Output access  (by index)
// ═══════════════════════════════════════════════════════════════════════════

const void* EspDlEngine::get_output(int index) const {
    if (!m_impl || !m_impl->initialized) return nullptr;
#if HAS_ESP_DL
    auto* t = m_impl->get_tensor_by_idx(index);
    return t ? t->get_element_ptr() : nullptr;
#else
    return nullptr;
#endif
}

esp_err_t EspDlEngine::get_output_shape(int index, int* dims, int* n_dims) const {
    if (!m_impl || !m_impl->initialized) return ESP_ERR_INVALID_STATE;
#if HAS_ESP_DL
    auto* t = m_impl->get_tensor_by_idx(index);
    if (!t) return ESP_ERR_INVALID_ARG;
    auto& shape = t->shape;
    if (n_dims) *n_dims = static_cast<int>(shape.size());
    if (dims) {
        for (int i = 0; i < static_cast<int>(shape.size()); ++i)
            dims[i] = shape[i];
    }
    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

int EspDlEngine::get_output_count() const {
#if HAS_ESP_DL
    if (!m_impl || !m_impl->initialized) return 0;
    return static_cast<int>(m_impl->output_names.size());
#else
    return 0;
#endif
}

float EspDlEngine::get_output_scale(int index) const {
#if HAS_ESP_DL
    if (!m_impl || !m_impl->initialized) return 1.0f;
    auto* t = m_impl->get_tensor_by_idx(index);
    if (!t) return 1.0f;
    int exp = t->exponent;
    // Power-of-2: scale = 2^exponent
    return (exp >= 0) ? static_cast<float>(1 << exp)
                      : (1.0f / static_cast<float>(1 << (-exp)));
#else
    return 1.0f;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
//  Output access  (by name — ESP-DL multi-output)
// ═══════════════════════════════════════════════════════════════════════════

const void* EspDlEngine::get_output_by_name(const char* name) const {
    if (!m_impl || !m_impl->initialized) return nullptr;
#if HAS_ESP_DL
    auto* t = m_impl->get_tensor(name);
    return t ? t->get_element_ptr() : nullptr;
#else
    return nullptr;
#endif
}

int EspDlEngine::get_output_exponent(const char* name) const {
#if HAS_ESP_DL
    if (!m_impl || !m_impl->initialized) return 0;
    auto* t = m_impl->get_tensor(name);
    return t ? t->exponent : 0;
#else
    return 0;
#endif
}

esp_err_t EspDlEngine::get_output_shape_by_name(const char* name,
                                                  int* dims, int* n_dims) const {
    if (!m_impl || !m_impl->initialized) return ESP_ERR_INVALID_STATE;
#if HAS_ESP_DL
    auto* t = m_impl->get_tensor(name);
    if (!t) return ESP_ERR_INVALID_ARG;
    auto& shape = t->shape;
    if (n_dims) *n_dims = static_cast<int>(shape.size());
    if (dims) {
        for (int i = 0; i < static_cast<int>(shape.size()); ++i)
            dims[i] = shape[i];
    }
    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
//  Utilities
// ═══════════════════════════════════════════════════════════════════════════

size_t EspDlEngine::get_arena_used() const {
#if HAS_ESP_DL
    return m_impl ? m_impl->model_ram_bytes : 0;
#else
    return 0;
#endif
}

void EspDlEngine::deinit() {
    if (!m_impl) return;

#if HAS_ESP_DL
    if (m_impl->model) {
        delete m_impl->model;
        m_impl->model = nullptr;
    }
    m_impl->output_names.clear();
#endif

    delete m_impl;
    m_impl = nullptr;

    ESP_LOGI(TAG, "ESP-DL desinicializado");
}
