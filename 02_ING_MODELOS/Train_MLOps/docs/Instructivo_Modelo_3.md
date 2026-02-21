# Instructivo Técnico: Modelo 3 - ESPDet-Pico (Custom PyTorch Loop)

## 1. Integración en la Infraestructura Vertex AI
- **Dataset:** Formato YOLO unificado (`gs://...tfm-data/datasets/yolo26.zip`).
- **Contenedor:** Ecosistema PyTorch (`pytorch-gpu.2-4.py310`).
- **Configuración YAML:** Crear `configs/espdet_pico_v1.yaml`. Definir hiperparámetros de entrenamiento y documentar que es una arquitectura *custom* (0.36M parámetros) extraída de Espressif.
- **Entry-point:** Crear `trainer/task_espdet.py` para orquestar la descarga de pesos de COCO, instanciar el modelo y lanzar nuestro bucle manual de entrenamiento.

## 2. Definición de la Arquitectura (PyTorch)
Ubicación sugerida: `src_colab/utils_model.py`.

1. **Importar Bloques de Espressif:** Copiar textualmente las definiciones de las clases `DSConv`, `DSBottleneck`, `ESPBlockLite`, `DSC3k2`, `SCDown` y `ESPDetect` desde la carpeta `nn/modules/` del repositorio oficial `esp-detection`.
2. **Construir la Clase Principal (`ESPDetPico`):**
   - Ensamblar secuencialmente el **Backbone** siguiendo su YAML original: `Conv(3->16)` -> `DSConv` -> `ESPBlockLite` -> `DSConv` -> `DSC3k2` -> `SCDown` -> `DSC3k2` -> `SCDown` -> `DSC3k2` -> `SPPF` -> `DSConv`.
   - Ensamblar el **Neck** (FPN ligero): `Upsample` -> Concat -> `ESPBlock`.
   - **Head:** Instanciar la cabeza `ESPDetect` configurada con `reg_max=1` y `nc=5` (clases: dog, door, obstacle, person, stair).
3. **Modificación Crítica para Exportación ONNX:**
   - Crear un método o sobrescribir `forward()` para emular `ESPDetect.export_onnx_forward()`.
   - El modelo debe devolver una tupla plana con 6 tensores separados crudos: `(box0, score0, box1, score1, box2, score2)`. Esto garantiza que el compilador `esp-ppq` lo traduzca perfectamente a `.espdl` para el ESP32-S3.

## 3. Lógica del Dataset y Aumentaciones
Ubicación sugerida: `src_colab/utils_data.py`.

- **Reutilización Estratégica:** Emplear exactamente la misma clase `IODCDataset(Dataset)` diseñada para FCOS y YOLO26n.
- **Aumentaciones (Albumentations):** Mantener el enfoque agresivo en variabilidad de iluminación (`RandomBrightnessContrast`, `HueSaturationValue`) y geométrica (`ShiftScaleRotate`) para compensar el sesgo de la fuente original.
- **Progressive Resizing:** Usar el método `set_image_size(new_size)` invocado desde el bucle para iterar sobre múltiplos de 32 (ej. 640, 416, 320, 224).

## 4. Flujo de Entrenamiento (Transfer Learning con `strict=False`)
Ubicación sugerida: `trainer/task_espdet.py` y `src_colab/utils_train.py`.

### Paso 0: Inyección de Conocimiento (Transfer Learning)
- Descargar el archivo de pesos oficial pre-entrenado en COCO: `espdet_pico_coco.pt`.
- Instanciar la red: `model = ESPDetPico(num_classes=5).cuda()`.
- Cargar los pesos forzando la asimetría en el Head: 
  `model.load_state_dict(torch.load('espdet_pico_coco.pt'), strict=False)`
- *Resultado:* El Backbone y el Neck adquieren los pesos expertos de COCO. Las capas finales de la cabeza `ESPDetect` (que ahora esperan 5 clases en vez de 80) conservan sus pesos aleatorios iniciales.

### Fase 1: Warm-Up y Freeze (Épocas 0 - 50)
- **Acción:** Congelar los pesos del Backbone (iterar sobre los parámetros y fijar `requires_grad = False`).
- **Optimizador:** AdamW con Learning Rate agresivo (`~1e-3`) exclusivo para el Neck (si es necesario) y el Head novato.
- **Resizing:** Forzar `train_loader.dataset.set_image_size(640)` en las primeras épocas, bajando a `416` a mitad de fase. Esto fuerza a la nueva cabeza de detección a mapear características ricas y de alta resolución.

### Fase 2: Fine-Tuning y Estabilización (Épocas 51 - 150)
- **Acción:** Descongelar el Backbone entero (`requires_grad = True`).
- **Optimizador:** Bajar el Learning Rate considerablemente (`~1e-4`) para preservar las valiosas extracciones de características de las capas DSConv pre-entrenadas.
- **Resizing Final:** En la época 120, establecer el dataset rígidamente en `224` (la resolución de inferencia real de la cámara OV5640 en el ESP32-S3) y mantenerlo así hasta la época 150 para asentar las estadísticas internas de normalización del modelo a los artefactos propios de esa baja resolución.