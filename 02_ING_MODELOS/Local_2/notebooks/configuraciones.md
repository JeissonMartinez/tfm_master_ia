# Especificaciones de Modelos Seleccionados

Este documento detalla las configuraciones finales de los modelos **SSD V6** y **YOLO V1** seleccionados tras el proceso de experimentación en `03_TrainModels.ipynb`.

## 1. Modelo Principal: YOLO V1 (YOLOv11 Nano)

Este modelo fue seleccionado como la opción primaria debido a su superioridad en precisión (mAP@50 > 0.50) y consistencia en la detección.

### 1.1 Arquitectura
- **Modelo Base:** `yolo11n.pt` (YOLOv11 Nano pre-entrenado)
- **Framework:** Ultralytics YOLO
- **Input Size:** 224x224 píxeles
- **Formato de Salida:** `[x_center, y_center, width, height]` (normalizado) + `class_id` + `confidence`

### 1.2 Hiperparámetros de Entrenamiento
| Parámetro | Valor | Descripción |
|:---|:---|:---|
| **Épocas** | 100 | Entrenamiento completo con early stopping (patience=15) |
| **Batch Size** | 32 | |
| **Optimizador** | AdamW | `momentum=0.937` |
| **Learning Rate** | `lr0=0.001` | Factor final `lrf=0.01` (Cos_lr=True) |
| **Weight Decay** | 0.0005 | Regularización L2 |
| **Warmup** | 3 épocas | `warmup_momentum=0.8`, `warmup_bias_lr=0.1` |

### 1.3 Data Augmentation & Estrategia
Configuración interna de YOLOv11 utilizada:
- **Mosaic:** 0.3 (Probabilidad de usar mosaico de 4 imágenes)
- **Scale:** 0.1 (Escalado aleatorio +/- 10%)
- **Fliplr:** 0.3 (Flip horizontal)
- **Color Jitter:** `hsv_s=0.1`, `hsv_v=0.1`
- **Otros:** `nms=False`, `overlap_mask=True`, `multi_scale=0.0`
- **Dataset:** Formato YOLO (`data.yaml`, archivos `.txt` por imagen)

---

## 2. Modelo Alternativo: SSD V6 (MobileNetV2)

Este modelo representa el mejor esfuerzo en arquitectura ligera personalizada (Keras/TensorFlow), aunque su rendimiento es inferior a YOLO en este dataset específico.

### 2.1 Arquitectura
- **Tipo:** Single Shot MultiBox Detector (SSD) Custom (Lite)
- **Backbone:** Start-of-the-art MobileNetV2
    - `alpha=0.50` (Ancho de red reducido para eficiencia)
    - `weights='imagenet'` (Transfer Learning)
    - `include_top=False`
- **Cabezal (Head):**
    - Capas de predicción personalizadas para regresión de cajas y clasificación.
    - **Max Objects:** Limitado a **2** objetos por imagen para simplificar la salida.
- **Input Size:** 224x224x3

### 2.2 Hiperparámetros de Entrenamiento
- **Generador:** `DualModelGeneratorV6`
- **Batch Size:** 32
- **Input Shape:** `(224, 224, 3)`
- **Targets:**
    1. **Background Class:** Incluida explícitamente (Total clases = 4 objetos + 1 fondo).
    2. **Anchor Boxes:** Configuración simplificada para dataset específico.

### 2.3 Notas de Implementación
- El modelo SSD V6 fue diseñado para ser extremadamente ligero y compatible con TensorFlow Lite Micro.
- Utiliza **Dual Output**: una rama para clasificación (probabilidades) y otra para regresión (coordenadas).
- **Post-Procesamiento:** Requiere decodificación de bounding boxes y Non-Maximum Suppression (NMS) manual en la inferencia (implementado en `ssd_output_to_bboxes`).

---

## 3. Resumen de Trade-offs (Conclusión Fase 4)

| Característica | YOLO V1 (Recomendado) | SSD V6 |
|:---|:---|:---|
| **mAP@50** | **0.52** | 0.08 |
| **Tamaño (FP32)** | 5.17 MB | **2.70 MB** |
| **Recall (promedio)** | ~0.61 | ~0.19 |
| **Consistencia** | Alta (detecta todas las clases) | Baja (muchos Falsos Positivos a 'background') |
| **Despliegue ESP32** | Requiere cuantización (INT8) | Nativo pero baja precisión |

**Decisión Final:** Se prioriza **YOLO V1** cuantizado a INT8 para la siguiente fase de despliegue en el ESP32-S3.
