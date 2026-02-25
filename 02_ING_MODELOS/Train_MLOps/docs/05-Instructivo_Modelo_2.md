# Instructivo Técnico: Modelo 2 - YOLO26n (PyTorch Custom Loop)

## 1. Integración en la Infraestructura Vertex AI
- **Dataset:** Formato YOLO (`gs://...tfm-data/datasets/yolo26.zip`).
- **Contenedor:** Mantener `pytorch-gpu.2-4.py310`. Asegurar `pip install ultralytics>=8.4` dinámicamente o en `requirements_yolo.txt`.
- **Configuración YAML:** Reutilizar o clonar `yolo26n_v1.yaml`, adaptando parámetros para progressive resizing manual.

## 2. Definición de la Arquitectura (PyTorch)
Ubicación sugerida: `src_colab/utils_model.py`.

1. **Instanciación:** Crear el modelo usando la API de Ultralytics configurado para 5 clases: `wrapper = YOLO('yolo26n.yaml').load('yolo26n.pt')`.
2. **Extracción:** Separar el núcleo de PyTorch para control manual: `model = wrapper.model.cuda()`.
3. **Confirmación Arquitectónica:** YOLO26n usa nativamente `reg_max=1` (sin DFL) y convoluciones estándar (`Conv` y `C3k2`).
4. **Exportación ONNX:** Reutilizar el script de `export_onnx_esp.py` que ya hace un monkey-patch al método `forward()` de la cabeza `Detect` para emitir los 6 tensores separados crudos.

## 3. Lógica del Dataset y Aumentaciones
Ubicación sugerida: Reutilizar la clase `IODCDataset` programada para FCOS.

- Al usar nuestro propio bucle en `src_colab/utils_train.py`, nos desvinculamos del cargador de datos interno de Ultralytics.
- **Mosaic/Mixup:** Si se desea Mosaico, implementar la lógica explícita en `IODCDataset.__getitem__`. Desactivar en la época 130 (`close_mosaic: 20`).
- **Progressive Resizing:** Aplicar el mismo controlador dinámico `set_image_size(new_size)` que en el Modelo 1.

## 4. Flujo de Entrenamiento (Transfer Learning & Fine-Tuning)
Ubicación sugerida: Adaptar `trainer/task_yolo.py` y `src_colab/utils_train.py`.

- **Fase 1: Warm-Up de Cabeza (Épocas 0 - 50)**
  - **Pesos:** Todos inicializados desde `yolo26n.pt` (entrenado en COCO), excepto la última convolución de clase que se reinicializó aleatoriamente para las 5 clases nuevas.
  - **Acción:** Congelar el Backbone (usualmente las primeras ~10 capas correspondientes al extractor CSP).
  - **Optimizador:** AdamW con Learning Rate alto (`~1e-3`).
  - **Loss Function:** Utilizar `compute_loss(outputs, labels)` usando la librería base de Ultralytics (v8DetectionLoss o similar), ya que calcular el *TaskAlignedAssigner* para las cajas es complejo de escribir desde cero.
  - **Resizing:** `epoch 0` -> 640x640. `epoch 30` -> 416x416.
- **Fase 2: Fine-Tuning Completo (Épocas 51 - 150)**
  - **Acción:** Descongelar el Backbone de YOLO26n.
  - **Optimizador:** Reducir Learning Rate (`~1e-4`).
  - **Resizing:** Bajar escalonadamente hasta que la época 120 fuerce el Dataset a 224x224 y se mantenga así para especializar el modelo a la resolución del dispositivo.