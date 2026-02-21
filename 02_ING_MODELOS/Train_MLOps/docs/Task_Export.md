# Task Export — ONNX → Formato de Despliegue

> **Entry-point:** `trainer.task_export`  
> **Contenedor:** `pytorch-gpu.2-4.py310:latest` (puede usar CPU-only)

---

## Propósito

Separar la conversión/cuantización del entrenamiento permite:

1. **Reducir coste:** El export puede correr en CPU ($0 GPU)
2. **Iteración rápida:** Re-cuantizar sin reentrenar
3. **Flexibilidad:** Probar diferentes niveles de cuantización (FP32, FP16, INT8)

---

## Pipeline

```
Step 1 — Download     : Descarga modelo ONNX desde GCS
Step 2 — Verify       : onnx.checker + inference test
Step 3 — Convert      : Simplificación + cuantización (INT8/FP16)
Step 4 — Verify       : Verificación del modelo convertido
Step 5 — Upload       : Subida de artefactos a GCS
```

---

## Uso

### Desde CLI

```bash
python -m trainer.task_export \
  --onnx-uri gs://bucket/output/run/export/model.onnx \
  --job-dir gs://bucket/output/run/export_output \
  --project-id project-18f58341-12cf-47bc-861 \
  --family FCOS \
  --imgsz 224 \
  --quantize int8
```

### Como Custom Job de Vertex AI

```bash
python vertex_ai/launch_job.py \
  --family EXPORT \
  --config-name fcos_v3s_v1.yaml \
  --run-name fcos-export-int8 \
  --machine-type n1-standard-4 \
  --accelerator-count 0
```

> Sin GPU (`accelerator-count 0`) para minimizar costes.

---

## Opciones de Cuantización

| Modo | Método | Ventajas | Desventajas |
|---|---|---|---|
| **None** (FP32) | — | Máxima precisión | Modelo grande |
| **FP16** | onnxconverter-common | 50% menos tamaño | Requiere soporte FP16 |
| **INT8** (dynamic) | onnxruntime.quantization | ~75% menos tamaño | Posible degradación |
| **INT8** (static) | Requiere calibration data | Mejor que dynamic | Más complejo |

---

## Integración con ESP32-S3

La conversión final ONNX → ESPDL para el ESP32-S3 se realiza con el
script `convert_onnx_to_espdl.py` de `03_ING_DESPLIEGUE/`. Este task_export
prepara el ONNX optimizado/cuantizado que sirve como input para esa conversión.

```
Training Job  →  ONNX (FP32)
                    │
                    ▼
Export Job    →  ONNX (INT8/FP16)
                    │
                    ▼
03_ING_DESPLIEGUE → convert_onnx_to_espdl.py → ESPDL
                    │
                    ▼
                 ESP32-S3 firmware
```

---

## Artefactos de Salida

```
export_output/
├── model.onnx              # Original descargado
├── converted/
│   ├── model_optimized.onnx  # Simplificado
│   └── model_int8.onnx       # Cuantizado (si --quantize int8)
└── export_metadata.json      # Metadatos de la conversión
```
