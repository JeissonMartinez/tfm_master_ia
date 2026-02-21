# Task YOLO26 Custom — Ultralytics 2-Phase Training

> **Entry-point:** `trainer.task_yolo26_custom`  
> **YAML:** `vertex_ai/configs/yolo26n_custom_v1.yaml`  
> **Contenedor:** `pytorch-gpu.2-4.py310:latest`

---

## Arquitectura del Modelo

YOLO26n (nano) cargado via Ultralytics con pesos preentrenados `yolo11n.pt`.
A diferencia del Ciclo 1 (1 fase), Ciclo 2 usa **entrenamiento custom de 2 fases**
con control explícito de freeze/unfreeze.

### Parámetros del Modelo

| Parámetro | Valor |
|---|---|
| Base model | YOLO11n (via Ultralytics) |
| Pretrained | COCO (yolo11n.pt) |
| Params | ~2.6M |
| Input size | 640 (training), 224 (export) |
| Tamaño ONNX est. | ~6 MB |

---

## Estrategia de Entrenamiento

### Fase 1 — Backbone Congelado (Ultralytics API)

- **Epochs:** 30
- **Freeze layers:** 10
- **LR:** lr0=0.01, lrf=0.01 (cosine)
- **Objetivo:** Entrenar head sin alterar features COCO

### Fase 2 — Todo Descongelado

- **Epochs:** 70
- **Freeze layers:** 0
- **LR:** lr0=0.001, lrf=0.001 (cosine)
- **Objetivo:** Fine-tune completo

> **Total:** 100 epochs (30 + 70), patience=30

### DDP Cleanup (Lección Ciclo 1)

Vertex AI inyecta variables de entorno DDP que rompen Ultralytics en single-GPU.
El entry-point limpia estas variables antes de inicializar:

```python
for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
    os.environ.pop(var, None)
```

---

## Augmentation (Ultralytics Built-in)

| Param | Valor | Descripción |
|---|---|---|
| mosaic | 1.0 | Mosaic augmentation |
| mixup | 0.1 | MixUp alpha |
| close_mosaic | 10 | Disable mosaic last N epochs |
| hsv_h | 0.015 | Hue shift |
| hsv_s | 0.7 | Saturation shift |
| hsv_v | 0.4 | Value shift |
| fliplr | 0.5 | Horizontal flip |
| scale | 0.5 | Scale factor |
| translate | 0.1 | Translation |

---

## Loss Weights

| Componente | Peso |
|---|---|
| Box (CIoU) | 7.5 |
| Cls (BCE) | 0.5 |

---

## Pipeline (8 Bloques)

```
Bloque 1 — Setup         : DDP cleanup + descarga config + dataset + pip install ultralytics
Bloque 2 — Verify        : Verificación dataset YOLO
Bloque 3 — Build Model   : Carga YOLO11n preentrenado
Bloque 4 — Train         : 2 fases via train_yolo26_custom()
Bloque 5 — Curves        : Parse results.csv de Ultralytics → gráficas
Bloque 6 — Val Eval      : model.val() → EvaluationResults
Bloque 7 — Test Eval     : model.val(split="test") → EvaluationResults
Bloque 8 — Save+Upload   : ONNX export (Ultralytics) + GCS upload
```

---

## Lanzamiento

```bash
python vertex_ai/launch_job.py \
  --family YOLO26_CUSTOM \
  --config-name yolo26n_custom_v1.yaml \
  --run-name yolo26n_custom_v1-run1
```

---

## Métricas Esperadas

| Métrica | Target |
|---|---|
| Val mAP@50 | > 0.50 (baseline, YOLO likely strongest) |
| Test mAP@50 | > 0.45 |
| ONNX size | < 8 MB |
| Training time (T4) | ~1h |

---

## Diferencias con Ciclo 1 (YOLO family)

| Aspecto | Ciclo 1 | Ciclo 2 |
|---|---|---|
| Fases | 1 (30-50 epochs) | 2 (30 + 70 = 100) |
| Freeze | No freeze | Phase 1: freeze 10 layers |
| Entry-point | `task_yolo.py` | `task_yolo26_custom.py` |
| Export | Dentro del job (ONNX + TFLite) | Solo ONNX (TFLite en job separado) |
| DDP fix | Sí | Sí (mantenido) |
