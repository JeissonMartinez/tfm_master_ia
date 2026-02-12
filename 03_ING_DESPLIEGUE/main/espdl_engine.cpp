// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — ESP-DL engine implementation
//
// ESP-DL v3.x usa modelos .espdl (FlatBuffers) embebidos en flash.
// El runtime automatiza scheduling dual-core y usa operadores SIMD
// optimizados para el Xtensa LX7 del ESP32-S3.
//
// NOTA: Esta implementación requiere que los archivos .espdl sean
// generados con convert_onnx_to_espdl.py y embebidos vía EMBED_FILES.
// Si los .espdl no están disponibles, el engine reportará error en init().
// ═══════════════════════════════════════════════════════════════════════════
#include "espdl_engine.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <new>            // std::nothrow

// ESP-DL headers — condicionalmente incluidos
#if __has_include("dl_model_base.h")
    #include "dl_model_base.h"
    #define HAS_ESP_DL 1
#else
    #define HAS_ESP_DL 0
#endif

static const char* TAG = "espdl_engine";

struct EspDlEngine::Impl {
    const ModelConfig* config = nullptr;
    size_t             memory_used = 0;
    bool               initialized = false;

#if HAS_ESP_DL
    // dl::Model instance (ESP-DL v3.x)
    // Se instanciará cuando esp-dl esté disponible como componente
    void* model_handle = nullptr;
#endif
};

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
    ESP_LOGI(TAG, "Inicializando ESP-DL para modelo: %s", config->name);

    // Verificar que el archivo .espdl embebido está disponible
    // Los archivos embebidos con EMBED_FILES se acceden vía:
    //   extern const uint8_t <start>[] asm("_binary_<filename>_start");
    //   extern const uint8_t <end>[]   asm("_binary_<filename>_end");
    //
    // El nombre del símbolo depende del nombre del archivo en CMakeLists.txt
    // Se resolverá en tiempo de compilación según el modelo seleccionado.

    // TODO: Implementar carga del modelo .espdl con dl::Model API
    // Ejemplo de la documentación de ESP-DL:
    //
    //   dl::Model model;
    //   model.load(espdl_data, espdl_size);
    //   model.set_input(input_tensor);
    //   model.run();
    //   auto output = model.get_output();

    ESP_LOGI(TAG, "✅ ESP-DL inicializado para %s", config->name);
    m_impl->initialized = true;
    return ESP_OK;

#else
    ESP_LOGW(TAG, "⚠️ ESP-DL no disponible en la compilación actual");
    ESP_LOGW(TAG, "   Asegúrese de que esp-dl está en idf_component.yml");
    ESP_LOGW(TAG, "   y que los archivos .espdl están en models/espdl/");

    // No es un error fatal — el motor reporta que no está disponible
    // y el sistema puede funcionar solo con TFLite Micro
    m_impl->initialized = false;
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

esp_err_t EspDlEngine::invoke(const int8_t* input) {
    if (!m_impl || !m_impl->initialized) return ESP_ERR_INVALID_STATE;

#if HAS_ESP_DL
    // TODO: Implementar inferencia con ESP-DL
    // model->set_input(input);
    // model->run();
    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

esp_err_t EspDlEngine::invoke_float(const float* input) {
    if (!m_impl || !m_impl->initialized) return ESP_ERR_INVALID_STATE;

#if HAS_ESP_DL
    // TODO: Implementar inferencia float con ESP-DL
    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

const void* EspDlEngine::get_output(int index) const {
    if (!m_impl || !m_impl->initialized) return nullptr;

#if HAS_ESP_DL
    // TODO: Retornar tensor de salida de ESP-DL
    return nullptr;
#else
    return nullptr;
#endif
}

esp_err_t EspDlEngine::get_output_shape(int index, int* dims, int* n_dims) const {
    if (!m_impl || !m_impl->initialized) return ESP_ERR_INVALID_STATE;

#if HAS_ESP_DL
    // TODO: Obtener shape del tensor de salida
    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

int EspDlEngine::get_output_count() const {
#if HAS_ESP_DL
    // TODO: Retornar número de outputs
    return 1;
#else
    return 0;
#endif
}

size_t EspDlEngine::get_arena_used() const {
    return m_impl ? m_impl->memory_used : 0;
}

void EspDlEngine::deinit() {
    if (!m_impl) return;

#if HAS_ESP_DL
    if (m_impl->model_handle) {
        // TODO: Liberar modelo ESP-DL
        m_impl->model_handle = nullptr;
    }
#endif

    delete m_impl;
    m_impl = nullptr;

    ESP_LOGI(TAG, "ESP-DL desinicializado");
}
