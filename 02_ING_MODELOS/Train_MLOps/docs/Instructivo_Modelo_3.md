# Instructivo Técnico: Modelo 3 - ESPDet-Pico (Custom PyTorch Loop)

> **Estado**: ✅ Implementado en v2.6.0 (arquitectura oficial Espressif)

## 1. Integración en la Infraestructura Vertex AI
- **Dataset:** Formato YOLO unificado (`gs://...tfm-data/datasets/iodc_yolo.zip`).
- **Contenedor:** Ecosistema PyTorch (`pytorch-gpu.2-4.py310`).
- **Configuración YAML:** `configs/espdet_pico_v2.yaml` (v2 = arquitectura oficial). ✅ Creado.
- **Entry-point:** `trainer/task_espdet.py` para orquestar la descarga de pesos, instanciar el modelo y lanzar el bucle manual de entrenamiento. ✅ Actualizado v2.6.0.

## 2. Definición de la Arquitectura (PyTorch)
Ubicación: `src_colab/utils_model.py` + `src_colab/espdet_modules/`. ✅ Implementado.

1. **Importar Bloques de Espressif:** Copiados textualmente desde `esp-detection` repo: `DSConv`, `DSBottleneck`, `ESPBlockLite`, `DSC3k2`, `ESPBlock`, `ESPDetectHead`. Ubicados en `src_colab/espdet_modules/`.
2. **Construir la Clase Principal (`ESPDetPico`):**
   - Ensamblar secuencialmente el **Backbone** siguiendo el YAML oficial: `Conv(3->16)` -> `DSConv(16->32,s=2)` -> `ESPBlockLite(32->64)` -> `DSConv(64->64,s=2)` -> `DSC3k2(64->64)` -> `SCDown(64->64)` -> `DSC3k2(64->64,c3k=True)` -> `SCDown(64->128)` -> `DSC3k2(128->128,c3k=True)` -> `SPPF(128->128)` -> `DSConv(128->128,k=7)`.
   - Ensamblar el **Neck** (FPN ligero): Top-down `Upsample` -> Concat -> `ESPBlock`. Bottom-up `DSConv(s=2)` -> Concat -> `ESPBlock`.
   - **Head:** Instanciar `ESPDetectHead` con `reg_max=1`, `nc=5`, `ch=(32, 128, 128)`. Box: `DSConv→DSConv→Conv2d(4)`. Cls: `DWConv+Conv→DWConv+Conv→Conv2d(nc)`.
3. **Strides**: `[8, 16, 32]` (P3/P4/P5) — oficial.
4. **Modificación Crítica para Exportación ONNX:**
   - Método `export_onnx_forward()` en `ESPDetectHead`, activable con `model.set_export_mode(True)`.
   - Produce tupla interleaved: `(box0, score0, box1, score1, box2, score2)` para compatibilidad directa con `esp-ppq` → `.espdl`.

## 3. Lógica del Dataset y Aumentaciones
Ubicación sugerida: `src_colab/utils_data.py`.

- **Reutilización Estratégica:** Emplear exactamente la misma clase `IODCDataset(Dataset)` diseñada para FCOS y YOLO26n.
- **Aumentaciones (Albumentations):** Mantener el enfoque agresivo en variabilidad de iluminación (`RandomBrightnessContrast`, `HueSaturationValue`) y geométrica (`ShiftScaleRotate`) para compensar el sesgo de la fuente original.
- **Progressive Resizing:** Usar el método `set_image_size(new_size)` invocado desde el bucle para iterar sobre múltiplos de 32 (ej. 640, 416, 320, 224).

## 4. Flujo de Entrenamiento (Transfer Learning con `strict=False`)
Ubicación: `trainer/task_espdet.py` y `src_colab/utils_train.py`. ✅ Implementado.

### Paso 0: Inyección de Conocimiento (Transfer Learning)
- Descargar los pesos oficiales pre-entrenados en cat-detection: `espdet_pico_224_224_cat.pt` (del repo `espressif/esp-detection/examples/cat_detection/`).
- **Nota**: NO existe checkpoint COCO-80 para ESPDet-Pico. Los pesos de cat-detection (nc=1) son el mejor punto de partida disponible (~0.36M params, mAP50:95=69.9 en COCO-cat val).
- Instanciar la red: `model = ESPDetPico(nc=5).cuda()`.
- Cargar los pesos con key conversion (Ultralytics → nuestra nomenclatura):
  `model.load_state_dict(converted_state, strict=False)`
- *Resultado:* El Backbone y el Neck (layers 0-22) adquieren los pesos de cat-detection. Solo las capas finales `cv3.{0,1,2}.2` (clasificación, nc=1→5) se inicializan aleatoriamente. ~99.97% de params transferidos.

### Fase 1: Warm-Up y Freeze (Épocas 0 - 50)
- **Acción:** Congelar los pesos del Backbone (iterar sobre los parámetros y fijar `requires_grad = False`).
- **Optimizador:** AdamW con Learning Rate agresivo (`~1e-3`) exclusivo para el Neck (si es necesario) y el Head novato.
- **Resizing:** Forzar `train_loader.dataset.set_image_size(640)` en las primeras épocas, bajando a `416` a mitad de fase. Esto fuerza a la nueva cabeza de detección a mapear características ricas y de alta resolución.

### Fase 2: Fine-Tuning y Estabilización (Épocas 51 - 150)
- **Acción:** Descongelar el Backbone entero (`requires_grad = True`).
- **Optimizador:** Bajar el Learning Rate considerablemente (`~1e-4`) para preservar las valiosas extracciones de características de las capas DSConv pre-entrenadas.
- **Resizing Final:** En la época 120, establecer el dataset rígidamente en `224` (la resolución de inferencia real de la cámara OV5640 en el ESP32-S3) y mantenerlo así hasta la época 150 para asentar las estadísticas internas de normalización del modelo a los artefactos propios de esa baja resolución.