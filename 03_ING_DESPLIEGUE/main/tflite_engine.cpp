// ═══════════════════════════════════════════════════════════════════════════
// TFM TinyML Detector — TFLite Micro engine implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "tflite_engine.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "tflite_engine";

// ─── Número de operadores: determinado en Conversion_ModelosTFLite.ipynb
// +1 QUANTIZE + SplitV para modelos full_integer_quant
static constexpr int NUM_OPS = 26;

struct TFLiteEngine::Impl {
    const tflite::Model*                           model     = nullptr;
    tflite::MicroMutableOpResolver<NUM_OPS>*       resolver  = nullptr;
    tflite::MicroInterpreter*                      interpreter = nullptr;
    uint8_t*                                       arena     = nullptr;
    size_t                                         arena_size = 0;
    const ModelConfig*                             config    = nullptr;
};

TFLiteEngine::~TFLiteEngine() {
    deinit();
}

esp_err_t TFLiteEngine::init(const ModelConfig* config) {
    if (!config || !config->tflite_data || config->tflite_size == 0) {
        ESP_LOGE(TAG, "Configuración de modelo inválida");
        return ESP_ERR_INVALID_ARG;
    }

    deinit();  // Limpiar instancia previa si existe

    m_impl = new (std::nothrow) Impl();
    if (!m_impl) {
        ESP_LOGE(TAG, "No se pudo alojar Impl");
        return ESP_ERR_NO_MEM;
    }
    m_impl->config = config;

    // ─── 1. Parsear modelo desde el array C ──────────────────────────────
    m_impl->model = tflite::GetModel(config->tflite_data);
    if (!m_impl->model || m_impl->model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Modelo TFLite inválido o versión incompatible");
        deinit();
        return ESP_ERR_INVALID_STATE;
    }
    ESP_LOGI(TAG, "Modelo cargado: %s (%zu bytes)", config->name, config->tflite_size);

    // ─── 2. Registrar operadores ─────────────────────────────────────────
    m_impl->resolver = new (std::nothrow) tflite::MicroMutableOpResolver<NUM_OPS>();
    if (!m_impl->resolver) {
        ESP_LOGE(TAG, "No se pudo alojar OpResolver");
        deinit();
        return ESP_ERR_NO_MEM;
    }

    // Los 26 operadores necesarios para los 3 modelos
    m_impl->resolver->AddAdd();
    m_impl->resolver->AddBatchMatMul();
    m_impl->resolver->AddCast();
    m_impl->resolver->AddConcatenation();
    m_impl->resolver->AddConv2D();
    m_impl->resolver->AddDepthwiseConv2D();
    m_impl->resolver->AddDequantize();
    m_impl->resolver->AddFloorMod();
    m_impl->resolver->AddGather();
    m_impl->resolver->AddGatherNd();
    m_impl->resolver->AddLess();
    m_impl->resolver->AddLogistic();
    m_impl->resolver->AddMaxPool2D();
    m_impl->resolver->AddMul();
    m_impl->resolver->AddPack();
    m_impl->resolver->AddPad();
    m_impl->resolver->AddReduceMax();
    m_impl->resolver->AddReshape();
    m_impl->resolver->AddResizeNearestNeighbor();
    m_impl->resolver->AddShape();
    m_impl->resolver->AddSoftmax();
    m_impl->resolver->AddStridedSlice();
    m_impl->resolver->AddSub();
    m_impl->resolver->AddTranspose();
    m_impl->resolver->AddQuantize();   // full_integer_quant models
    m_impl->resolver->AddSplitV();     // full_integer_quant YOLO11n
    ESP_LOGI(TAG, "OpResolver configurado: %d operadores", NUM_OPS);

    // ─── 3. Alojar arena en PSRAM ────────────────────────────────────────
    // Arena size: ~3× tamaño del modelo como estimación inicial
    m_impl->arena_size = config->arena_size > 0
        ? config->arena_size
        : config->tflite_size * 3;

    // Intentar alojar — si falla, reducir tamaño progresivamente
    for (int attempt = 0; attempt < 3; attempt++) {
        m_impl->arena = static_cast<uint8_t*>(
            heap_caps_malloc(m_impl->arena_size, MALLOC_CAP_SPIRAM)
        );
        if (m_impl->arena) break;

        ESP_LOGW(TAG, "No se pudo alojar arena de %zu bytes, reduciendo...",
                 m_impl->arena_size);
        m_impl->arena_size = m_impl->arena_size * 3 / 4;  // Reducir 25%
    }

    if (!m_impl->arena) {
        ESP_LOGE(TAG, "Error fatal: no se pudo alojar arena en PSRAM");
        deinit();
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Arena alojada en PSRAM: %zu bytes @ %p",
             m_impl->arena_size, m_impl->arena);

    // ─── 4. Crear intérprete ─────────────────────────────────────────────
    m_impl->interpreter = new (std::nothrow) tflite::MicroInterpreter(
        m_impl->model,
        *m_impl->resolver,
        m_impl->arena,
        m_impl->arena_size
    );
    if (!m_impl->interpreter) {
        ESP_LOGE(TAG, "No se pudo crear MicroInterpreter");
        deinit();
        return ESP_ERR_NO_MEM;
    }

    // ─── 5. Alojar tensores ──────────────────────────────────────────────
    TfLiteStatus status = m_impl->interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() falló — arena demasiado pequeña o ops incompatibles");
        // Destructor of MicroInterpreter calls FreeSubgraphs() which may
        // crash if AllocateTensors() left internal state half-initialised.
        // Leak the interpreter intentionally to avoid Guru Meditation;
        // the arena free below reclaims most memory.
        m_impl->interpreter = nullptr;
        deinit();
        return ESP_ERR_NO_MEM;
    }

    size_t used = m_impl->interpreter->arena_used_bytes();
    ESP_LOGI(TAG, "✅ TFLite Micro inicializado");
    ESP_LOGI(TAG, "   Arena usada: %zu / %zu bytes (%.1f%%)",
             used, m_impl->arena_size, 100.0f * used / m_impl->arena_size);
    ESP_LOGI(TAG, "   Inputs:  %d", m_impl->interpreter->inputs_size());
    ESP_LOGI(TAG, "   Outputs: %d", m_impl->interpreter->outputs_size());

    // Log tensor shapes for diagnostics
    TfLiteTensor* in0 = m_impl->interpreter->input(0);
    if (in0 && in0->dims) {
        ESP_LOGI(TAG, "   Input[0]: type=%d, ndim=%d, shape=[%d,%d,%d,%d]",
                 in0->type, in0->dims->size,
                 in0->dims->size > 0 ? in0->dims->data[0] : -1,
                 in0->dims->size > 1 ? in0->dims->data[1] : -1,
                 in0->dims->size > 2 ? in0->dims->data[2] : -1,
                 in0->dims->size > 3 ? in0->dims->data[3] : -1);
    }
    TfLiteTensor* out0 = m_impl->interpreter->output(0);
    if (out0 && out0->dims) {
        ESP_LOGI(TAG, "   Output[0]: type=%d, ndim=%d, shape=[%d,%d,%d]",
                 out0->type, out0->dims->size,
                 out0->dims->size > 0 ? out0->dims->data[0] : -1,
                 out0->dims->size > 1 ? out0->dims->data[1] : -1,
                 out0->dims->size > 2 ? out0->dims->data[2] : -1);
    }

    return ESP_OK;
}

esp_err_t TFLiteEngine::invoke(const int8_t* input) {
    if (!m_impl || !m_impl->interpreter) return ESP_ERR_INVALID_STATE;

    // Copiar input al tensor de entrada
    TfLiteTensor* input_tensor = m_impl->interpreter->input(0);
    if (!input_tensor) return ESP_ERR_INVALID_STATE;

    // Diagnóstico pre-invoke
    ESP_LOGI(TAG, "Input tensor: type=%d dims=%d data=%p",
             input_tensor->type, input_tensor->dims->size, input_tensor->data.raw);
    ESP_LOGI(TAG, "Stack libre: %u words",
             (unsigned)uxTaskGetStackHighWaterMark(nullptr));

    memcpy(input_tensor->data.int8, input, INPUT_SIZE);

    // Ejecutar inferencia
    ESP_LOGI(TAG, "Invoke() starting...");
    TfLiteStatus status = m_impl->interpreter->Invoke();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() falló");
        return ESP_FAIL;
    }

    return ESP_OK;
}

esp_err_t TFLiteEngine::invoke_float(const float* input) {
    if (!m_impl || !m_impl->interpreter) return ESP_ERR_INVALID_STATE;

    TfLiteTensor* input_tensor = m_impl->interpreter->input(0);
    if (!input_tensor) return ESP_ERR_INVALID_STATE;

    // Si el tensor espera float, copiar directamente
    if (input_tensor->type == kTfLiteFloat32) {
        memcpy(input_tensor->data.f, input, INPUT_SIZE * sizeof(float));
    } else if (input_tensor->type == kTfLiteInt8) {
        // Cuantizar float → int8 usando los parámetros del tensor
        float scale = input_tensor->params.scale;
        int32_t zero_point = input_tensor->params.zero_point;
        for (int i = 0; i < INPUT_SIZE; i++) {
            int32_t q = static_cast<int32_t>(input[i] / scale + zero_point);
            q = (q < -128) ? -128 : (q > 127 ? 127 : q);
            input_tensor->data.int8[i] = static_cast<int8_t>(q);
        }
    }

    TfLiteStatus status = m_impl->interpreter->Invoke();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() falló");
        return ESP_FAIL;
    }

    return ESP_OK;
}

const void* TFLiteEngine::get_output(int index) const {
    if (!m_impl || !m_impl->interpreter) return nullptr;
    if (index < 0 || index >= m_impl->interpreter->outputs_size()) return nullptr;

    TfLiteTensor* tensor = m_impl->interpreter->output(index);
    return tensor ? tensor->data.data : nullptr;
}

esp_err_t TFLiteEngine::get_output_shape(int index, int* dims, int* n_dims) const {
    if (!m_impl || !m_impl->interpreter) return ESP_ERR_INVALID_STATE;
    if (index < 0 || index >= m_impl->interpreter->outputs_size()) return ESP_ERR_INVALID_ARG;

    TfLiteTensor* tensor = m_impl->interpreter->output(index);
    if (!tensor) return ESP_ERR_INVALID_STATE;

    *n_dims = tensor->dims->size;
    for (int i = 0; i < tensor->dims->size; i++) {
        dims[i] = tensor->dims->data[i];
    }
    return ESP_OK;
}

int TFLiteEngine::get_output_count() const {
    if (!m_impl || !m_impl->interpreter) return 0;
    return m_impl->interpreter->outputs_size();
}

size_t TFLiteEngine::get_arena_used() const {
    if (!m_impl || !m_impl->interpreter) return 0;
    return m_impl->interpreter->arena_used_bytes();
}

bool TFLiteEngine::is_output_int8(int index) const {
    if (!m_impl || !m_impl->interpreter) return false;
    if (index < 0 || index >= m_impl->interpreter->outputs_size()) return false;
    TfLiteTensor* tensor = m_impl->interpreter->output(index);
    return tensor && tensor->type == kTfLiteInt8;
}

float TFLiteEngine::get_output_scale(int index) const {
    if (!m_impl || !m_impl->interpreter) return 1.0f;
    if (index < 0 || index >= m_impl->interpreter->outputs_size()) return 1.0f;
    TfLiteTensor* tensor = m_impl->interpreter->output(index);
    return tensor ? tensor->params.scale : 1.0f;
}

int32_t TFLiteEngine::get_output_zero_point(int index) const {
    if (!m_impl || !m_impl->interpreter) return 0;
    if (index < 0 || index >= m_impl->interpreter->outputs_size()) return 0;
    TfLiteTensor* tensor = m_impl->interpreter->output(index);
    return tensor ? tensor->params.zero_point : 0;
}

void TFLiteEngine::deinit() {
    if (!m_impl) return;

    if (m_impl->interpreter) {
        delete m_impl->interpreter;
        m_impl->interpreter = nullptr;
    }
    if (m_impl->resolver) {
        delete m_impl->resolver;
        m_impl->resolver = nullptr;
    }
    if (m_impl->arena) {
        heap_caps_free(m_impl->arena);
        m_impl->arena = nullptr;
    }
    m_impl->model = nullptr;

    delete m_impl;
    m_impl = nullptr;

    ESP_LOGI(TAG, "TFLite Micro desinicializado");
}
