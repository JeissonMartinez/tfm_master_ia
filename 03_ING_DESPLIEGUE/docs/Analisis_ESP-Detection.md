User: Haz una revisión profunda de este repositorio: https://github.com/espressif/esp-detection

Quiero entender qué contiene, cómo usarlo para implementar modelos de YOLO en ESP32S3 y qué de esto puede servir para el desarrollo actual del despliegue que estamos intentando hacer en este proyecto local.

GitHub Copilot: 

  

Created 4 todos



Fetched 2 resources

Searched espressif/esp-detection for "repository structure main components YOLO ESP32 model deployment", 64 results

Now let me explore the key deployment files in more detail and also understand the local project. 

Starting: *Explore key repo components* (2/4)



Fetched 3 resources

Searched espressif/esp-detection for "espdet_pico model config yaml architecture definition backbone head", 64 results

Now let me get the config YAML and the requirements, and also look at the local project's key files: 

Starting: *Understand local project state* (3/4)

Analyze local project state



Fetched 3 resources

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_models_esp.py)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/CMakeLists.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/partitions.csv)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/Plan_Despliegue.md#1-1), lines 1 to 300

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/Configuracion_ESP32-S3.md)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/requirements.txt#1-1), lines 1 to 50

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py#200-200), lines 200 to 288

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/docs/Plan_Despliegue.md#300-300), lines 300 to 429

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/CMakeLists.txt)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/idf_component.yml)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/camera_handler)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/metrics)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/dashboard)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/requirements.txt#50-50), lines 50 to 172

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/main/main.cpp#100-100), lines 100 to 182

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_common.cpp#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/image_proc/image_proc.cpp#1-1), lines 1 to 150

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/camera_handler/camera_handler.cpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/metrics/metrics.cpp#1-1), lines 1 to 150

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/network/network.cpp#1-1), lines 1 to 200

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/dashboard/dashboard.cpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#200-200), lines 200 to 330

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp#200-200), lines 200 to 334

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp#100-100), lines 100 to 216

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/include/inference_engine.h)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/include/postprocess.h)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/sdkconfig.defaults#1-1), lines 1 to 50

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/dashboard/web)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h#100-100), lines 100 to 158

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/flash_models.sh#1-1), lines 1 to 100

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/dashboard/web/index.html#1-1), lines 1 to 100

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/5784bb0f-5f82-4993-9022-3266c8ea3d06/toolu_01QdyjYj4UrH8apSzTiZ3DYi__vscode-1770995437409/content.txt)

Starting: *Synthesize findings & recommendations* (4/4)

Completed: *Synthesize findings & recommendations* (4/4)

---

## Revisión profunda: `espressif/esp-detection` vs. tu proyecto local

### 1. Qué contiene `esp-detection`

Es un framework **end-to-end** oficial de Espressif para entrenar y desplegar modelos de detección ultra-ligeros en chips ESP. Está construido **sobre Ultralytics YOLOv11** y genera modelos `.espdl` listos para ESP-DL.

**Estructura del repositorio:**

| Directorio | Contenido |
|---|---|
| `cfg/models/` | Definición YAML de la arquitectura **ESPDet-Pico** (0.36M params, `reg_max=1`, sin DFL) |
| `cfg/datasets/` | YAMLs de datasets en formato YOLO (ej. `coco_cat.yaml`) |
| `nn/modules/` | Bloques custom optimizados para ESP: `DSConv`, `DSBottleneck`, `DSC3k2`, `ESPBlock`, `ESPBlockLite`, `ESPSerial`, `ESPSerialLite`, **`ESPDetect`** (cabeza de detección con `reg_max=1`) |
| https://github.com/espressif/esp-detection/tree/main/nn/esp_tasks.py#L0-L23 | `custom_parse_model()` — monkey-patch de `ultralytics.nn.tasks.parse_model` para registrar los bloques ESP custom |
| https://github.com/espressif/esp-detection/tree/main/data/esp_dataset.py#L0-L131 | Datasets custom: `YOLOPosNegDataset` (soporte pos+neg), `YOLOWeightedDataset` (balanceo de clases) |
| https://github.com/espressif/esp-detection/tree/main/train.py#L0-L36 | Entrenamiento: inyecta el parser, carga el YAML de ESPDet-Pico o pesos pretrained, entrena ~1200 epochs |
| https://github.com/espressif/esp-detection/tree/main/val.py#L0-L20 | Validación con mAP sobre COCO |
| https://github.com/espressif/esp-detection/tree/main/deploy/export.py#L0-L126 | **Exportación ONNX**: `ESP_Detect_Exporter` con **6 salidas raw** (`box0,score0,...box2,score2`), opset 13, simplify. Usa `ESPDetect.export_onnx_forward()` |
| https://github.com/espressif/esp-detection/tree/main/deploy/quantize.py#L0-L106 | **Cuantización**: `esp_ppq.QuantizationSettingFactory.espdl_setting()` + equalization + `espdl_quantize_onnx()` → `.espdl` |
| https://github.com/espressif/esp-detection/tree/main/deploy/eval_quantized_model.py#L0-L114 | Evaluación mAP del modelo cuantizado ejecutándolo con `TorchExecutor` de PPQ |
| `deploy/espdet_example_template/` | Template de proyecto C++ ESP-IDF listo para flash (app_main.cpp con JPEG decode + inferencia + log) |
| `deploy/espdet_model_template/` | Template de componente ESP-DL (`espdet_detect.cpp/hpp`) con pre/post-procesador: `ESPDetPostProcessor` con letterbox y strides `{8,16,32}` |
| https://github.com/espressif/esp-detection/tree/main/espdet_run.py#L0-L121 | **Script all-in-one**: Train → Export ONNX → Quantize → Clone esp-dl → Genera proyecto C++ listo para `idf.py flash monitor` |

**Modelo ESPDet-Pico (arquitectura YAML):**
- Backbone: `Conv(3→64)` → `DSConv` → `ESPBlockLite` → `DSConv` → `DSC3k2` → `SCDown` → `DSC3k2` → `SCDown` → `DSC3k2` → `SPPF` → `DSConv(7×7)`
- Head: FPN con `nn.Upsample` + `Concat` + `ESPBlock`, detección en P3/P4/P5
- Cabeza: **`ESPDetect`** con `reg_max=1` (sin DFL, regresión directa de 4 coords)
- Solo 0.36M parámetros, input 224×224
- **Rendimiento ESP32-S3: ~126ms latencia (~7.9 FPS) a 224×224**

**Dependencias clave:**
```
ultralytics>=8.3.112
torch==2.2.0
onnx==1.17.0
onnxsim==0.4.36
esp-ppq (git)
```

---

### 2. Cómo usarlo para implementar YOLO en ESP32-S3

El flujo completo con `esp-detection` sería:

```bash
# 1. Instalar
conda create -n espdet python=3.8
conda activate espdet
pip install -r requirements.txt

# 2. Ejecutar pipeline completo (Train + Export + Quantize + Generate C++ project)
python espdet_run.py \
  --class_name myobject \
  --pretrained_path None \
  --dataset "cfg/datasets/my_dataset.yaml" \
  --size 224 224 \
  --target "esp32s3" \
  --calib_data "deploy/my_calib" \
  --espdl "espdet_pico_224_224_myobject.espdl" \
  --img "test.jpg"

# 3. Flashear
cd esp-dl/examples/myobject_detect
idf.py set-target esp32s3
idf.py flash monitor
```

**Diferencias clave de su pipeline vs. uso estándar de YOLO11n:**

| Aspecto | ESPDet-Pico (esp-detection) | YOLO11n estándar |
|---|---|---|
| `reg_max` | **1** (regresión directa, sin DFL) | **16** (requiere DFL softmax+integral) |
| Params | 0.36M | 2.6M |
| Export ONNX | 6 salidas raw via `ESPDetect.export_onnx_forward()` | 6 salidas raw via monkey-patch de `Detect` |
| Post-procesador C++ | `ESPDetPostProcessor` (built-in en ESP-DL) | Custom `postprocess_yolo.cpp` con DFL |
| Cuantización | `espdl_setting()` con equalization | `espdl_setting()` con equalization |
| Latencia ESP32-S3 | ~126ms (224×224) | Significativamente mayor (~500ms+ estimado) |
| mAP (1 clase, gato) | 69.9 mAP50:95 | Similar en single-class |

---

### 3. Qué puede servir para tu proyecto local

Tu proyecto ya está en **fase avanzada** (late Phase 2) con 3 modelos (YOLO11n, YOLO26n, MBNTv3S-SSDLite), un firmware funcional con dashboard real-time, y pipeline de cuantización. Pero hay **puntos concretos** donde `esp-detection` puede aportar:

#### A) **Adoptar ESPDetPostProcessor de ESP-DL en lugar de postprocessors custom** (ALTA prioridad para YOLO26n)

Tu YOLO26n ya usa `reg_max=1` igual que ESPDet-Pico. El `ESPDetPostProcessor` incluido en ESP-DL maneja exactamente este caso con anchors `{{8,8,4,4}, {16,16,8,8}, {32,32,16,16}}` y letterbox. Podrías reemplazar tu postprocess_yolo26n.cpp custom por el built-in, reduciendo código y bugs.

#### B) **Referencia de cuantización con equalization** (ALTA prioridad)

Tu `quantize_models_esp.py` usa `espdl_setting()` sin equalization explícita. El repo `esp-detection` activa:
```python
quant_setting.equalization = True
quant_setting.equalization_setting.iterations = 4   # o 10 en eval
quant_setting.equalization_setting.value_threshold = .4  # o .3
quant_setting.equalization_setting.opt_level = 2
```
Dado que llevas **muchas iteraciones** de cuantización (variantes `_A_kl`, `_B_minmax`, `_C_percentile`, `_eqbc`, `_fp32scores`...), adoptar equalization con estos parámetros podría mejorar la calidad de los scores INT8 que parece ser tu principal dolor.

#### C) **Eval del modelo cuantizado ANTES de flashear** (MEDIA prioridad)

El script https://github.com/espressif/esp-detection/tree/main/deploy/eval_quantized_model.py#L0-L114 ejecuta el grafo cuantizado PPQ como `TorchExecutor` contra el validator de Ultralytics para obtener mAP real del modelo INT8. Esto evita el ciclo lento de cuantizar → flashear → observar → ajustar. Podrías adaptar esta clase `QuantizedModelValidator` para evaluar tus YOLO11n/YOLO26n cuantizados en PC antes de deployar.

#### D) **Considerar ESPDet-Pico como un 4to modelo** (MEDIA prioridad)

Con solo 0.36M params y ~126ms en ESP32-S3 (vs. ~500ms+ de YOLO11n con DFL), sería un candidato ideal para el benchmark de tu TFM. Entrenarías con tus 5 clases (dog, door, obstacle, person, stair) y tendrías un modelo que realmente corre a >7 FPS en el dispositivo.

#### E) **Template C++ de deployment** (BAJA prioridad, ya lo tienes)

Tu firmware ya es más avanzado que el template de `esp-detection` (tienes cámara real, WebSocket, dashboard, multi-model). Pero la estructura del template confirma que tu enfoque de `dl::Model` + `ImagePreprocessor` + postprocessor es el patrón canónico de Espressif.

#### F) **Export ONNX: confirmar patrón de 6 salidas** (Ya implementado)

Tu `export_onnx_esp.py` ya implementa el mismo patrón que https://github.com/espressif/esp-detection/tree/main/deploy/export.py#L0-L126: monkey-patch `Detect.forward` → 6 tensores raw (`box0,score0,...box2,score2`). Esto valida que tu enfoque es correcto.

---

### Resumen ejecutivo

| Aspecto | Tu proyecto | esp-detection | Acción recomendada |
|---|---|---|---|
| Pipeline train→deploy | Manual, scripts separados | All-in-one https://github.com/espressif/esp-detection/tree/main/espdet_run.py#L0-L121 | Tomar como referencia, no reemplazar |
| Cuantización | `espdl_setting()` básico | `espdl_setting()` + equalization | **Activar equalization** en tu pipeline |
| Eval cuantizado en PC | No existe | https://github.com/espressif/esp-detection/tree/main/deploy/eval_quantized_model.py#L0-L114 | **Adoptar** para evitar ciclo flash-debug |
| Post-proceso YOLO | Custom C++ (DFL para 11n, directo para 26n) | `ESPDetPostProcessor` built-in | **Considerar** para YOLO26n |
| Modelo ligero | YOLO11n (2.6M), YOLO26n, SSD | ESPDet-Pico (0.36M) | **Entrenar ESPDet-Pico** con tus 5 clases |
| Firmware | Avanzado (cámara, dashboard, WS) | Template básico (JPEG embed) | Mantener tu firmware |

User: A nivel de arquitectura, ¿qué diferencia tiene el modelo ESPDet-Pico con mis modelos YOLO11n, YOLO26n y SSD ?

GitHub Copilot: 

 Let me revisar los detalles arquitectónicos específicos de tus modelos locales para hacer una comparación precisa. 

Get model architecture details

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp#1-1), lines 1 to 300

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/app_config/include/app_config.h)

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp#300-300), lines 300 to 334

Searched for files matching `**/firmware/components/postprocess/postprocess_yolo26*`, no matches

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/5784bb0f-5f82-4993-9022-3266c8ea3d06/toolu_01WWpE8bdXdw449tNF4xYuwR__vscode-1770995437457/content.txt)

Ahora tengo todos los detalles. Aquí va la comparación arquitectónica:

---

## Comparación Arquitectónica: ESPDet-Pico vs. YOLO11n vs. YOLO26n vs. MBNTv3S-SSDLite

### 1. Visión general

| | **ESPDet-Pico** | **YOLO11n** | **YOLO26n** | **MBNTv3S-SSDLite** |
|---|---|---|---|---|
| Familia | YOLO11-custom (Espressif) | Ultralytics YOLO11 | Ultralytics YOLO26 | MobileNetV3-Small + SSD |
| Paradigma | Anchor-free, single-stage | Anchor-free, single-stage | Anchor-free, single-stage | Anchor-based, single-stage |
| Parámetros | **0.36M** | **2.6M** | ~2.3M | ~1.5M |
| Input | 224×224 | 224×224 | 224×224 | 224×224 |
| `reg_max` | **1** | **16** | **1** | N/A |
| Necesita DFL | No | **Sí** | No | No |
| FLOPs | 0.17G | ~6.5G* | ~5G* | ~0.6G |

### 2. Backbone

```
ESPDet-Pico                          YOLO11n
─────────────                        ─────────
Conv 3→16 (3×3, s2)                  Conv 3→16 (3×3, s2)
DSConv 16→32 (3×3, s2)              Conv 16→32 (3×3, s2)
ESPBlockLite 32→64                   C3k2 32→64
DSConv 64→64 (3×3, s2)              Conv 64→64 (3×3, s2)
DSC3k2 ×2 64→64                      C3k2 ×2 64→128
SCDown 64→64 (s2)                    Conv 128→128 (3×3, s2)
DSC3k2 ×2 64→128 (c3k=True)         C3k2 ×2 128→128
SCDown 128→128 (s2)                  Conv 128→256 (3×3, s2)
DSC3k2 ×2 128→256 (c3k=True)        C3k2 ×2 256→256
SPPF 256→256                         SPPF 256→256
DSConv 256→256 (7×7, p3)            C2PSA 256→256
```

**Diferencias clave del backbone:**

- **`DSConv` (Depthwise Separable Conv)** vs `Conv` estándar: ESPDet-Pico reemplaza la mayoría de convoluciones estándar con Depthwise Separable, que ejecutan $\frac{1}{C_{out}} + \frac{1}{k^2}$ de las multiplicaciones. Esto es **la razón principal** de tener 7× menos parámetros.

- **`ESPBlockLite`** vs `C3k2`: ESPBlockLite usa `DSBottleneck` (dos DSConv) en lugar del `Bottleneck` estándar (dos Conv). Internamente es una versión "serial" — cada bottleneck se alimenta del anterior secuencialmente, no en paralelo como C2f.

- **`DSC3k2`** vs `C3k2`: Mismo diseño que C3k2 (Split → N bottlenecks → Concat → Conv1×1) pero con `DSBottleneck` en lugar de `Bottleneck`.

- **`SCDown`** vs `Conv(s2)`: Ambos hacen downsampling, SCDown es un módulo de Ultralytics que usa una pointwise conv + depthwise conv separadas (algo más eficiente que Conv directa con stride).

- **Width multiplier 0.25** vs 0.25: Ambos usan escala "nano" (0.25), pero los bloques DSConv de ESPDet tienen menos operaciones por canal → mismo width, 7× menos FLOPs.

### 3. Neck/Head (Feature Pyramid)

```
ESPDet-Pico                          YOLO11n                    MBNTv3S-SSDLite
─────────────                        ─────────                  ─────────────────
Upsample 2×                          Upsample 2×                (sin FPN)
Concat(P4)                           Concat(P4)
ESPBlock ×2                          C3k2 ×2                    Feature Map 14×14 ──┐
Upsample 2×                          Upsample 2×                Feature Map  7×7 ──┤
Concat(P3)                           Concat(P3)                                     │
ESPBlock ×2 → P3/8                   C3k2 ×2 → P3/8            6 anchors/cell ─────┘
DSConv(s2)                           Conv(s2)
Concat(P4)                           Concat(P4)
ESPBlock ×2 → P4/16                  C3k2 ×2 → P4/16
DSConv(s2)                           Conv(s2)
Concat(P5)                           Concat(P5)
ESPBlock ×2 → P5/32                  C3k2 ×2 → P5/32
```

SSDLite **no tiene FPN** — toma directamente de 2 feature maps del backbone (14×14 y 7×7) y aplica SSD heads con anchors predefinidos. Esto reduce latencia pero pierde la fusión multi-escala.

### 4. Cabeza de Detección (la diferencia más crítica)

Aquí es donde las arquitecturas divergen drásticamente:

#### ESPDet-Pico (`ESPDetect`, `reg_max=1`)
```
Para cada nivel (P3, P4, P5):
  Box branch:  DSConv(ch→c2, 3×3) → DSConv(c2→c2, 3×3) → Conv1×1(c2→4)
  Score branch: [DWConv(ch→ch, 3×3) → Conv(ch→c3, 1×1)] → [DWConv(c3→c3, 3×3) → Conv(c3→c3, 1×1)] → Conv1×1(c3→nc)

  Salidas: 4 offsets directos + nc scores por celda
  Decodificación: x1 = cx - d0*stride, y1 = cy - d1*stride, x2 = cx + d2*stride, y2 = cy + d3*stride
```

#### YOLO11n (`Detect`, `reg_max=16`)
```
Para cada nivel (P3, P4, P5):
  Box branch:  Conv(ch→c2, 3×3) → Conv(c2→c2, 3×3) → Conv1×1(c2→64)
  Score branch: Conv(ch→c3, 3×3) → Conv(c3→c3, 3×3) → Conv1×1(c3→nc)

  Salidas: 64 valores DFL + nc scores por celda
  Decodificación:
    1. 64 valores → 4 grupos de 16 → Softmax(16) → Integral(Σ i·p[i]) para cada dir.
    2. x1 = cx - dfl(left)*stride, etc.
```

#### YOLO26n (`Detect`, `reg_max=1`)
```
Para cada nivel (P3, P4, P5):
  Box branch:  Conv(ch→c2, 3×3) → Conv(c2→c2, 3×3) → Conv1×1(c2→4)
  Score branch: Conv(ch→c3, 3×3) → Conv(c3→c3, 3×3) → Conv1×1(c3→nc)

  Salidas: 4 offsets directos + nc scores por celda
  Decodificación: idéntica a ESPDet-Pico
```

#### MBNTv3S-SSDLite (anchor-based, 1470 priors)
```
Para cada nivel (FM 14×14, FM 7×7):
  Class: Lightweight Conv → 5 scores por anchor
  Bbox:  Lightweight Conv → 4 deltas (dx, dy, dw, dh) por anchor
  Objectness: Lightweight Conv → 1 score por anchor

  Salidas: [1, 1470, 5] + [1, 1470, 4] + [1, 1470, 1]
  Decodificación:
    cx = anchor_cx + dx * 0.1 * anchor_w
    cy = anchor_cy + dy * 0.1 * anchor_h
    w  = anchor_w * exp(dw * 0.2)
    h  = anchor_h * exp(dh * 0.2)
```

### 5. Impacto en cuantización INT8

| Aspecto | ESPDet-Pico | YOLO11n | YOLO26n | SSDLite |
|---|---|---|---|---|
| Box channels | **4** (directo) | **64** (DFL) | **4** (directo) | **4** (delta) |
| DFL softmax en INT8? | No necesita | **Sí** — softmax sobre 16 bins cuantizados degrada precisión | No necesita | No necesita |
| Score range | Raw logits → sigmoid en device | Raw logits → sigmoid en device | Raw logits → sigmoid en device | **Post-sigmoid** ya en modelo |
| Riesgo de cuantización | Bajo (4 valores simples) | **Alto** (DFL discretiza distribución continua → error compuesto con INT8) | Bajo | Bajo (scores ya comprimidos) |
| Exponent sharing | 4 vals/celda, similar rango | 64 vals/celda, **bins DFL causan distribución muy plana** → exponent inadecuado | 4 vals/celda | Objectness separado ayuda |

**El DFL es tu problema principal con YOLO11n**: los 16 logits de cada distribución DFL tienen rangos similares (todos cerca de 0), al cuantizar a INT8 pierden resolución relativa. El softmax posterior amplifica estas diferencias mínimas, causando errores geométricos. **ESPDet-Pico y YOLO26n evitan esto completamente por diseño.**

### 6. Post-procesamiento en ESP32-S3 (costo computacional)

| Operación | ESPDet-Pico | YOLO11n | YOLO26n | SSDLite |
|---|---|---|---|---|
| Candidatos | 1029 | 1029 | 1029 | 1470 |
| DFL softmax+integral | **No** | **4×1029 = 4116** softmax(16) | **No** | **No** |
| Dequantize | 4 mult/celda | 64 mult/celda | 4 mult/celda | 10 mult/anchor |
| Score sigmoid | 1× | 1× | 1× | 0× (ya aplicado) |
| Anchor lookup | No (anchor-free) | No | No | **Sí** (1470 pre-computados, ~23KB) |
| Exp() calls en decode | 0 | 4116×16 = 65856 (DFL) | 0 | 2940 (dw/dh) |
| **Costo relativo** | **1×** | **~15×** | **1×** | **~2×** |

### 7. Resumen visual: el diagrama clave

```
               ┌───────────────────── DETECCIÓN ──────────────────────┐
               │                                                       │
               │  ┌──── Backbone ────┐   ┌── Neck(FPN) ──┐   ┌ Head ┐│
               │  │                  │   │                │   │      ││
  ESPDet-Pico  │  │ DSConv+ESPBlock  │→  │ ESPBlock       │→  │reg=1 ││  0.36M  ~126ms
               │  │  (DW-Separable)  │   │ (DW-Separable) │   │4ch   ││
               │  └──────────────────┘   └────────────────┘   └──────┘│
               │                                                       │
  YOLO11n      │  │ Conv + C3k2      │→  │ C3k2           │→  │reg=16││  2.6M   ~500ms+
               │  │  (Standard)      │   │ (Standard)     │   │64ch  ││
               │  │  + C2PSA(Attn)   │   │                │   │+DFL  ││
               │                                                       │
  YOLO26n      │  │ Conv + C3k2      │→  │ C3k2           │→  │reg=1 ││  ~2.3M  ~400ms+
               │  │  (Standard)      │   │ (Standard)     │   │4ch   ││
               │                                                       │
  SSDLite      │  │ MBNetV3-Small    │→  │ (sin FPN)      │→  │6 anc ││  ~1.5M  ~250ms?
               │  │  (DW-Separable)  │   │                │   │/cell ││
               └───────────────────────────────────────────────────────┘
```

### 8. Conclusión ejecutiva: ¿por qué ESPDet-Pico y tu YOLO26n son similares?

ESPDet-Pico y tu YOLO26n comparten el **mismo paradigma de cabeza de detección** (`reg_max=1`, 4 offsets directos, anchor-free, strides 8/16/32, misma decodificación `dist2bbox`). La diferencia es que ESPDet-Pico va más allá y **también optimiza backbone y neck** con bloques Depthwise Separable, reduciendo los parámetros de 2.3M a 0.36M (6.4×) y los FLOPs de ~5G a 0.17G (29×).

Dicho de otro modo: **YOLO26n es un ESPDet-Pico con un backbone "pesado"**. Si entrenaras ESPDet-Pico con tus 5 clases, tendrías la cabeza de detección más fácil de cuantizar (la misma que YOLO26n) pero con un backbone que realmente cabe cómodamente en ESP32-S3 y da >7 FPS reales.

User: Analicemos dos escenarios:

1) Cambiar las arquitecturas y reentrenar mis versiones de modelos actuales.
Teniendo en cuenta el diseño de la arquitectura de ESPDet-Pico , ¿qué cambios podría hacer en las arquitecturas de YOLO11n, YOLO26n y MobileNetV3-Small SSDLite, para competir con el desempeño de ESPDet-Pico? 

2) Mantener los modelos actuales (sin cuantizar), pero adoptar los puntos que me mencionaste inicialmente en el análisis de `espressif/esp-detection` vs. mi proyecto local:
- Post-proceso con `ESPDetPostProcessor` built-in
- Usar el `ESPDetect.export_onnx_forward()` 
- Referencia de cuantización con equialization 
- Eval los modelos cuantizados antes de flashear con `eval_quantized_model.py` con la adaptación de la clase `QuantizedModelValidator`
- Revisar el Template C++ de desployment de `esp-detection` para validar si hay algo que me sirva a mi firmware actual

---
Para ambos, ¿qué pasos debería seguir? ¿qué consecuencias podrían tener?

GitHub Copilot: 

 Let me revisar algunos detalles clave del proyecto antes de responder con precisión. 

Review firmware and export details

Searched for regex `.` (`**/models/export_onnx_esp.py`), 1 result

Searched for regex `.` (`**/models/quantize_models_esp.py`), 1 result

Searched for regex `.` (`**/firmware/components/postprocess/postprocess_yolo.cpp`), 1 result

Searched for regex `.` (`**/firmware/components/postprocess/postprocess_ssd.cpp`), 1 result

Searched for regex `.` (`**/firmware/components/inference/inference_engine.cpp`), 1 result

Ran terminal command:  wc -l /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_models_esp.py /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp /Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/export_onnx_esp.py#1-1), lines 1 to 287

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/quantize_models_esp.py#1-1), lines 1 to 144

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_yolo.cpp#1-1), lines 1 to 333

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/postprocess/postprocess_ssd.cpp#1-1), lines 1 to 215

Read [](file:///Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/firmware/components/inference/inference_engine.cpp#1-1), lines 1 to 329

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/5784bb0f-5f82-4993-9022-3266c8ea3d06/toolu_01AwnkZY5yJ4BZYSjBgTe5Ar__vscode-1770995437466/content.txt#1-1), lines 1 to 300

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/5784bb0f-5f82-4993-9022-3266c8ea3d06/toolu_01AwnkZY5yJ4BZYSjBgTe5Ar__vscode-1770995437466/content.txt#300-300), lines 300 to 600

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/5784bb0f-5f82-4993-9022-3266c8ea3d06/toolu_01AwnkZY5yJ4BZYSjBgTe5Ar__vscode-1770995437466/content.txt#600-600), lines 600 to 900

Read [](file:///Users/admin/Library/Application%20Support/Code/User/workspaceStorage/7432cc3d52ca1665ad83ad1d9a8410d4/GitHub.copilot-chat/chat-session-resources/5784bb0f-5f82-4993-9022-3266c8ea3d06/toolu_01AwnkZY5yJ4BZYSjBgTe5Ar__vscode-1770995437466/content.txt#900-900), lines 900 to 1249

Tengo todo el contexto necesario. Aquí va el análisis detallado de ambos escenarios:

---

## Escenario 1: Cambiar arquitecturas y reentrenar

### Qué cambiar en cada modelo para acercarse a ESPDet-Pico

#### YOLO11n — 3 cambios con impacto decreciente

**Cambio 1 (CRÍTICO): Reducir `reg_max` de 16 a 1**

Esto es lo que más impacto tiene. Tu YOLO11n exporta `box{i}` con 64 canales (16×4) que luego requieren DFL softmax+integral en device — 65856 llamadas a `expf()` en tu `postprocess_yolo.cpp`. Con `reg_max=1` (como ya sucede en YOLO26n y ESPDet-Pico):
- Las salidas de box pasan de `[1,64,H,W]` a `[1,4,H,W]` — **16× menos datos** de bbox por nivel
- Eliminas completamente `dfl_integral()` y el branch `if (reg_max > 1)` de `parse_stage()`
- La cuantización INT8 deja de pasar por softmax sobre bins discretizados — **la mayor fuente de error cuantizado desaparece**
- Consecuencia: sin reentrenar, no se puede cambiar `reg_max` — requiere modificar la YAML de arquitectura y reentrenar desde cero

**Cambio 2 (ALTO): Reemplazar Conv estándar con DSConv (Depthwise Separable)**

ESPDet-Pico usa `DSConv` en lugar de `Conv` en backbone y neck. En Ultralytics, el `Conv` estándar es una convolución regular de $C_{in} \times C_{out} \times k \times k$ parámetros. `DSConv` la divide en depthwise ($C_{in} \times k \times k$) + pointwise ($C_{in} \times C_{out} \times 1 \times 1$), dando una reducción de $\frac{1}{C_{out}} + \frac{1}{k^2}$ en FLOPs — para $k=3$, eso es ~8-9× menos.

- Consecuencia: necesitas definir un YAML custom que use `DSConv`, `DSC3k2`, `DSBottleneck` en lugar de `Conv`, `C3k2`, `Bottleneck` — exactamente lo que hace el YAML de ESPDet-Pico
- No puedes reentrenar un `.pt` existente con bloques diferentes — requiere training from scratch

**Cambio 3 (MEDIO): Reducir width multiplier**

Tu YOLO11n usa scale `n` (width=0.25) pero con bloques Conv estándar esos canales producen muchas más multiplicaciones que los DSConv de ESPDet-Pico con el mismo width. Reducir canales adicionalmente (e.g. `max_channels=256` en lugar de 1024) ayuda, pero con DSConv ya tienes el grueso del ahorro.

**Resumen YOLO11n → "YOLO11n-ESP":**

| Componente | Actual | Modificado | Efecto |
|---|---|---|---|
| `reg_max` | 16 | **1** | Elimina DFL, -16× datos bbox, cuantización robusta |
| Backbone convs | `Conv` | `DSConv` | -8× FLOPs por conv |
| Bottleneck | `Bottleneck` | `DSBottleneck` | -8× FLOPs por bloque |
| C3k2 | `C3k2` (conv regular) | `DSC3k2` | -8× FLOPs por bloque |
| Head blocks | `C3k2` | `ESPBlock` | Misma estructura, base DSConv |
| Head detect | `Detect(reg_max=16)` | `ESPDetect(reg_max=1)` | 4 canales directos |
| Parámetros estimados | ~2.6M | **~0.3-0.5M** | -5-8× |
| Latencia ESP32-S3 est. | ~500ms+ | **~120-150ms** | -3-4× |

**Pero en ese punto... ya estarías construyendo ESPDet-Pico.** La pregunta es si vale la pena hacer un "YOLO11n modificado" versus directamente usar ESPDet-Pico y configurarlo con `nc=5`.

#### YOLO26n — menos que cambiar

Tu YOLO26n **ya tiene `reg_max=1`**, así que el problema más grave (DFL) no le aplica. Lo que le falta vs. ESPDet-Pico:

| Componente | Actual YOLO26n | ESPDet-Pico | Ganancia estimada |
|---|---|---|---|
| Backbone convs | `Conv` estándar | `DSConv` | ~8× menos FLOPs por conv |
| Bottleneck | `Bottleneck` | `DSBottleneck` | ~8× menos FLOPs |
| Módulo C2f/C3k2 | `C3k2` (conv regular) | `DSC3k2` | ~8× menos FLOPs |
| Neck blocks | `C3k2` | `ESPBlock` | ~8× menos FLOPs |
| Head detect | `Detect(reg_max=1)` | `ESPDetect(reg_max=1)` | Menor: cv2 usa DSConv, cv3 usa DWConv+Conv |
| Parámetros | ~2.3M | 0.36M | -6× |
| Attn (C2PSA) | Sí (parcheado) | No | Elimina complexity |

Consecuencia: un "YOLO26n-ESP" con bloques Depthwise Separable sería esencialmente ESPDet-Pico con el backbone de YOLO26 rediseñado. Nuevamente, casi idéntico a entrenar ESPDet-Pico directamente.

#### MBNTv3S-SSDLite — ya usa Depthwise Separable

MobileNetV3-Small **ya usa convoluciones depthwise separable** — su backbone es inherentemente eficiente como el de ESPDet-Pico. Los cambios serían diferentes:

| Aspecto | Actual | Posible cambio | Consecuencia |
|---|---|---|---|
| Head | Anchor-based (1470 priors, 2 FM) | Anchor-free (1029 cells, 3 FM) | Rediseño completo de head + reentrenar |
| FPN | Sin FPN | Añadir FPN con ESPBlock | Más parámetros, pero mejor multi-escala |
| Objectness output | Tensor separado (1470×1) | Eliminarlo (como YOLO) | Score = class score directo |
| Activaciones post-modelo | Sigmoid dentro del modelo | Sigmoid en device | Mejor rango dinámico para cuantización |
| Score processing | `objectness × class_score` | `sigmoid(class_logit)` directo | Elimina multiplicación compuesta |

Pero cambiar MBNTv3S a architecture anchor-free equivale a **diseñar un modelo nuevo** — no es "modificar" SSDLite.

### Consecuencias del Escenario 1

| Pro | Contra |
|---|---|
| Latencia real <150ms en ESP32-S3 (>7 FPS) | Reentrenamiento completo (~1200 epochs × 3 modelos) |
| Cuantización robusta sin DFL | Necesitas dataset en formato YOLO (ya lo tienes) |
| Menor consumo de flash (~665KB vs 2.7MB) | Posible pérdida de mAP vs. modelos grandes |
| Eliminas `dfl_integral()` del firmware | Validas que bloques custom funcionen con esp-ppq |
| Comparación justa en el TFM (mismo paradigma) | Pierdes la comparación "modelo estándar vs. optimizado" |

**Veredicto**: Si la meta es maximizar FPS en ESP32-S3, entrenar ESPDet-Pico con tus 5 clases es la vía más directa. Modificar YOLO11n/26n "a mano" convergería al mismo diseño con más esfuerzo.

---

## Escenario 2: Mantener modelos actuales, adoptar mejoras del pipeline

### Paso A: Activar Equalization en la cuantización

**Qué cambiar en** quantize_models_esp.py:

Tu código actual (línea ~82):
```python
quant_setting = QuantizationSettingFactory.espdl_setting()
```

Lo que hace `esp-detection`:
```python
quant_setting = QuantizationSettingFactory.espdl_setting()
quant_setting.equalization = True
quant_setting.equalization_setting.iterations = 4
quant_setting.equalization_setting.value_threshold = 0.4
quant_setting.equalization_setting.opt_level = 2
quant_setting.equalization_setting.interested_layers = None
```

**Qué hace**: Equalization redistribuye los rangos de pesos entre capas adyacentes para que la cuantización INT8 sea más uniforme. Reduce el caso donde una capa tiene pesos muy grandes y la siguiente muy pequeños — el exponent de una aplasta la resolución de la otra.

**Consecuencias**:
- El proceso de cuantización será **más lento** (~2-4× por las iteraciones de equalization)
- Puede mejorar significativamente la calidad de scores INT8 — es la técnica específica que Espressif recomienda para sus modelos
- **No requiere cambiar nada en el firmware** — solo afecta al `.espdl` generado
- Riesgo: en algunos modelos puede empeorar si `value_threshold` es demasiado agresivo — empieza con los valores de Espressif y ajusta
- **Es el cambio de mayor impacto con menor esfuerzo** de todos los listados

**Pasos**:
1. Modificar `quantize_models_esp.py` añadiendo equalization
2. Re-cuantizar YOLO11n y YOLO26n
3. Comparar con las variantes anteriores (`_A_kl`, `_B_minmax`, etc.)
4. Flashear y probar — si los scores mejoran, ya tendrás tu línea base

### Paso B: Eval cuantizado antes de flashear

**Qué hacer**: Adaptar https://github.com/espressif/esp-detection/tree/main/deploy/eval_quantized_model.py#L0-L114 a tus modelos.

El script de Espressif hace esto:
1. Cuantiza el ONNX con `espdl_quantize_onnx()` → obtiene el `quant_ppq_graph`
2. Crea un `TorchExecutor(quant_ppq_graph)` — ejecuta el modelo cuantizado en CPU como si fuera ESP-DL
3. Conecta el executor a `DetectionValidator` de Ultralytics para calcular mAP real del modelo INT8
4. La función `ppq_graph_inference()` toma las 6 salidas, las concatena como `[box, score]` por nivel, pasa por `ESPDetect._inference()` para obtener predicciones completas

**Lo que necesitarías adaptar**:
- La función `ppq_graph_inference()` reconstruye la cabeza de detección configurando `ESPDetect(nc=1, ch=[32,64,128])` — tú tendrías que configurarla con `nc=5` y tus canales reales
- Necesitas un dataset de validación en formato YOLO (no solo el pickle de calibración)
- Para tu SSD, este enfoque **no aplica** directamente — el `DetectionValidator` de Ultralytics asume formato YOLO, no SSD

**Consecuencias**:
- **Enormemente valioso**: en lugar de ciclos de cuantizar → flashear → observar logs → ajustar (minutos por iteración), puedes evaluar mAP en segundos en tu Mac
- Complejidad media: requiere adaptar ~100 líneas de Python
- Solo aplica a YOLO11n y YOLO26n (no al SSD)

**Pasos**:
1. Crear `models/eval_quantized.py` basado en https://github.com/espressif/esp-detection/tree/main/deploy/eval_quantized_model.py#L0-L114
2. Configurar `ESPDetect(nc=5, ch=...)` con los canales correctos de tus modelos
3. Preparar tu dataset de validación en formato YOLO
4. Ejecutar: cuantizar → evaluar mAP INT8 → comparar con mAP FP32

### Paso C: Post-procesador `ESPDetPostProcessor` built-in

**Contexto**: ESP-DL incluye `dl::detect::ESPDetPostProcessor` como clase built-in en C++. Solo funciona con modelos tipo ESPDet (6 salidas, `reg_max=1`, strides 8/16/32).

**Para YOLO26n** (`reg_max=1`): podría funcionar **si** el formato de salida es compatible. Pero hay diferencias sutiles:

| Aspecto | `ESPDetPostProcessor` (ESP-DL) | Tu `postprocess_yolo.cpp` |
|---|---|---|
| Anchors param | `{{8,8,4,4}, {16,16,8,8}, {32,32,16,16}}` (strides + aux) | `{8, 16, 32}` strides directos |
| Score activation | Asume dentro del procesador | `sigmoid()` explícito en tu código |
| Threshold | Constructor param (0.25 default) | `SCORE_THRESHOLD = 0.10` |
| NMS IoU | Constructor param (0.7 default) | `NMS_IOU_THRESHOLD = 0.45` |
| Letterbox | `enable_letterbox({114,114,114})` | No usas letterbox (crop+resize directo) |
| Input preprocessor | `ImagePreprocessor` acoplado | Separado en `image_proc.cpp` |

**Problema clave**: `ESPDetPostProcessor` está **acoplado** a `ImagePreprocessor` de ESP-DL. Espera que el preprocesado se haga con la clase built-in, incluyendo letterbox con padding 114. Tu pipeline usa crop+resize custom sin letterbox. Adoptar el postprocessor significa también adoptar el preprocessor — lo que implica reescribir `image_proc.cpp`.

**Para YOLO11n** (`reg_max=16`): **no aplica** — el postprocessor built-in no soporta DFL.

**Para SSD**: **no aplica** — es un decoder anchor-free, no compatible con SSD.

**Consecuencias**:
- Solo viable para YOLO26n, y requiere también cambiar el preprocessor
- Reduces ~100 líneas de C++ en postprocess pero añades dependencia de ESP-DL classes internas
- Tu postprocessor actual ya funciona bien — la ganancia real es mínima
- **Mi recomendación: no vale la pena.** Tu código custom es más flexible y ya está debugeado

### Paso D: Usar `ESPDetect.export_onnx_forward()` de esp-detection

**Tu código actual** en `export_onnx_esp.py` ya implementa exactamente lo mismo:

```python
# Tu ESP_Detect_Forward.forward:
for i in range(self.nl):
    results.append(self.cv2[i](x[i]))   # box
    results.append(self.cv3[i](x[i]))   # score
return tuple(results)
```

```python
# esp-detection ESPDetect.export_onnx_forward:
box0 = self.cv2[0](x[0]); score0 = self.cv3[0](x[0])
box1 = self.cv2[1](x[1]); score1 = self.cv3[1](x[1])
box2 = self.cv2[2](x[2]); score2 = self.cv3[2](x[2])
return box0, score0, box1, score1, box2, score2
```

Son **funcionalmente idénticos**. Tu versión es más genérica (loop sobre `self.nl`), la de Espressif tiene los 3 niveles hardcoded. No hay nada que adoptar aquí — **ya lo tienes**.

La única diferencia es que Espressif también exporta con `output_names=["box0","score0","box1","score1","box2","score2"]` explícitos, cosa que tu código también hace. Validado.

### Paso E: Revisar template C++ de deployment

**Comparación de tu firmware vs. template de esp-detection**:

| Componente | Tu firmware | Template esp-detection | Veredicto |
|---|---|---|---|
| Model loading | `esp_partition_mmap()` + `dl::Model(rodata)` | Igual: `dl::Model(path, MODEL_LOCATION_IN_FLASH_RODATA)` | Equivalente |
| `param_copy` | `true` (PSRAM) | No configurado (usa default) | **Tu versión es mejor** |
| `max_internal_size` | `0` (PSRAM only) | No configurado | **Tu versión es más segura** |
| Memory manager | `MEMORY_MANAGER_GREEDY` | Default | Tu versión es explícita |
| `model->minimize()` | No usas | **Sí lo usa** | Podrías adoptarlo |
| Multi-core | `RUNTIME_MODE_MULTI_CORE` | No especificado | **Tu versión es mejor** |
| Input source | Cámara real + crop + resize | JPEG embed en flash | Tu versión es producción |
| Preprocessor | Custom `image_proc.cpp` | `ImagePreprocessor` built-in | Tu versión tiene más control |
| Postprocessor | Custom por modelo | `ESPDetPostProcessor` | Tu versión es más flexible |
| Output | WebSocket + dashboard | `ESP_LOGI` a serial | Tu versión es muy superior |

**Un punto interesante**: el template usa `m_model->minimize()` después de crear el modelo. Esta llamada libera buffers internos temporales del parsing. Podrías añadirlo a tu `inference_engine.cpp` después de la creación del modelo para liberar algo de memoria:

```cpp
s_model = new dl::Model(...);
s_model->minimize();  // ← libera buffers de parsing
```

**Consecuencia**: potencialmente libera unos KB de PSRAM. Costo: una línea de código. Riesgo: ninguno.

---

## Resumen ejecutivo: qué hacer y en qué orden

| # | Acción | Escenario | Esfuerzo | Impacto | Riesgo |
|---|---|---|---|---|---|
| 1 | **Activar equalization en cuantización** | 2 | 5 min (3 líneas) | **Alto** — puede resolver el problema de scores INT8 | Bajo |
| 2 | **Añadir `model->minimize()`** al inference engine | 2 | 1 min (1 línea) | Bajo — libera algo de PSRAM | Ninguno |
| 3 | **Crear eval_quantized.py** para mAP INT8 en PC | 2 | 2-3 horas | **Alto** — elimina ciclo flash-debug | Medio (adaptar validator) |
| 4 | **Entrenar ESPDet-Pico con 5 clases** | 1 | 1-2 días (training) | **Muy alto** — modelo real a >7 FPS | Bajo (framework probado) |
| 5 | Adoptar `ESPDetPostProcessor` para YOLO26n | 2 | 1 día | Bajo | Alto (reescribir preprocessor) |
| 6 | Modificar YOLO11n/26n con bloques DS | 1 | 3-5 días | Alto pero redundante con #4 | Alto (arquitectura nueva) |

**Mi recomendación**: Ejecuta 1 → 2 → 3 esta semana (mejora inmediata del pipeline actual), y luego decide si #4 (entrenar ESPDet-Pico como 4to modelo comparativo) vale la pena para el TFM. Salta #5 y #6.