# Instructivo Técnico: Modelo 1 - MobileNetV3-Small + FCOS (Tiny)

## 1. Integración en la Infraestructura Vertex AI
- **Dataset:** Abandonar el formato TFRecord. Modificar `trainer/task_mobilenet.py` para que descargue y descomprima el dataset en formato YOLO (`gs://...tfm-data/datasets/yolo26.zip`).
- **Contenedor:** Migrar el contenedor de Vertex AI de TensorFlow (`tf-gpu.2-17.py310`) a PyTorch (`pytorch-gpu.2-4.py310`).
- **Configuración YAML:** Crear `configs/mobilenet_v3_fcos_v1.yaml`. Definir fases de entrenamiento explícitas (Freeze epochs vs. Fine-tuning epochs).

## 2. Definición de la Arquitectura (PyTorch)
Ubicación sugerida: `src_colab/utils_model.py`.

1. **Backbone:** Instanciar `torchvision.models.mobilenet_v3_small(weights='DEFAULT')`. Extraer mapas de características en `stage3` (stride 8), `stage4` (stride 16) y `stage5` (stride 32).
2. **Neck (Lightweight FPN):** - Convoluciones $1\times1$ para reducir todos los *stages* a `fpn_channels = 64`.
   - Fusión *Top-Down* con interpolación bilineal (`nn.Upsample`).
   - Bloques de suavizado usando `DSConv` (Depthwise Separable Convolution) para mantener un bajo consumo de PSRAM.
3. **Head (FCOS Anchor-free):**
   - Conectar un bloque de evaluación a cada nivel de FPN (P3, P4, P5).
   - **Clasificación:** Convolución a 5 canales (dog, door, obstacle, person, stair).
   - **Centerness:** Convolución a 1 canal.
   - **Regresión:** Convolución a 4 canales ($l, t, r, b$). Usar activación `ReLU` o `SiLU` (¡NUNCA `Exp()` para proteger la cuantización a INT8!).
4. **Exportación ONNX:** Sobrescribir el método `forward()` para devolver una tupla plana `(box0, score0, box1, score1, box2, score2)` donde `score = sigmoid(cls) * sigmoid(centerness)`.

## 3. Lógica del Dataset y Aumentaciones
Ubicación sugerida: `src_colab/utils_data.py`.

1. **Clase `IODCDataset(Dataset)`:** Carga imágenes de 640x640 y lee etiquetas YOLO.
2. **Aumentaciones Online (Albumentations):**
   - Incluir `RandomBrightnessContrast`, `HueSaturationValue` (énfasis en color por la fuente de iluminación).
   - Incluir `ShiftScaleRotate` y `HorizontalFlip` (robusteza espacial).
   - Configurar `bbox_params=A.BboxParams(format='yolo')`.
3. **Progressive Resizing Manual:**
   - Crear método `set_image_size(new_size)`.
   - Dentro de `__getitem__`, ejecutar `cv2.resize()` de la imagen aumentada usando `self.img_size` antes de convertirla a tensor normalizado [0, 1].

## 4. Flujo de Entrenamiento (Transfer Learning & Fine-Tuning)
Ubicación sugerida: `src_colab/utils_train.py`.

- **Fase 1: Warm-Up y Freeze (Épocas 0 - 50)**
  - **Pesos:** El Backbone inicia con pesos de ImageNet. Neck y Head inician aleatorios.
  - **Acción:** Congelar parámetros del Backbone (`requires_grad = False`).
  - **Optimizador:** AdamW con Learning Rate alto (`~1e-3`) para que el Head aprenda rápidamente.
  - **Resizing:** `epoch 0` -> 640x640. `epoch 30` -> 416x416.
- **Fase 2: Fine-Tuning de Red Completa (Épocas 51 - 150)**
  - **Acción:** Descongelar el Backbone (`requires_grad = True`).
  - **Optimizador:** Bajar Learning Rate (`~1e-4`) para no destruir los pesos pre-entrenados del Backbone.
  - **Resizing:** `epoch 90` -> 320x320. `epoch 120` -> 224x224 (Resolución objetivo para la ESP32-S3). Mantener fijo en 224x224 para asentar las estadísticas de Batch Normalization.