# Instructivo: Actualización Ultralytics y Reentrenamiento YOLO26 sin DFL

> **Objetivo:** Actualizar Ultralytics a una versión que implemente la eliminación oficial de DFL en YOLO26, reentrenar el modelo YOLO26n con las mismas condiciones del Train 2, y ejecutar el pipeline completo de cuantización INT8 + exportación ESPDL para ESP32-S3.

---

## Índice

1. [Contexto y Motivación](#1-contexto-y-motivación)
2. [Análisis de Impacto: DFL vs sin DFL](#2-análisis-de-impacto-dfl-vs-sin-dfl)
3. [Prerequisitos y Verificaciones Previas](#3-prerequisitos-y-verificaciones-previas)
4. [Fase 1 — Actualización de Ultralytics](#4-fase-1--actualización-de-ultralytics)
5. [Fase 2 — Validación de la Arquitectura Actualizada](#5-fase-2--validación-de-la-arquitectura-actualizada)
6. [Fase 3 — Preparación del Entorno Vertex AI](#6-fase-3--preparación-del-entorno-vertex-ai)
7. [Fase 4 — Configuración del Entrenamiento](#7-fase-4--configuración-del-entrenamiento)
8. [Fase 5 — Lanzamiento en Vertex AI](#8-fase-5--lanzamiento-en-vertex-ai)
9. [Fase 6 — Descarga y Verificación de Artefactos](#9-fase-6--descarga-y-verificación-de-artefactos)
10. [Fase 7 — Exportación ONNX ESP-compatible](#10-fase-7--exportación-onnx-esp-compatible)
11. [Fase 8 — Cuantización INT8 y Exportación ESPDL](#11-fase-8--cuantización-int8-y-exportación-espdl)
12. [Fase 9 — Evaluación FP32 vs INT8](#12-fase-9--evaluación-fp32-vs-int8)
13. [Fase 10 — Integración en Firmware ESP32-S3](#13-fase-10--integración-en-firmware-esp32-s3)
14. [Resumen de Cambios Esperados](#14-resumen-de-cambios-esperados)
15. [Checklist de Ejecución](#15-checklist-de-ejecución)
16. [Riesgos y Mitigaciones](#16-riesgos-y-mitigaciones)
17. [Referencia: Arquitectura Actual vs Actualizada](#17-referencia-arquitectura-actual-vs-actualizada)

---

## 1. Contexto y Motivación

### Situación actual

El modelo YOLO26n Train 2 fue entrenado con **Ultralytics v8.4.9**. En esta versión, YOLO26 todavía utiliza **DFL (Distribution Focal Loss)** con `reg_max=16`, lo que implica:

- La cabeza `Detect` predice **64 canales** de regresión de cajas (16 bins × 4 coordenadas)
- Se requiere una integral DFL (`softmax → weighted_sum`) en post-procesamiento
- Las salidas ESP del modelo actual son `box{i}: [1, 64, H, W]`

### Qué cambia con la actualización

La versión oficial actualizada de YOLO26 (documentada en https://docs.ultralytics.com/models/yolo26/) introduce **DFL Removal** como característica clave:

> *"The Distribution Focal Loss (DFL) module, while effective, often complicated export and limited hardware compatibility. YOLO26 removes DFL entirely, replacing it with direct bounding box coordinate prediction."*

Beneficios esperados:

| Aspecto | Con DFL (actual) | Sin DFL (actualizado) |
|---|---|---|
| Canales de regresión | 64 (16 × 4) | **4** (directo) |
| Post-procesamiento | softmax + integral DFL | **Ninguno** (coords directas) |
| Parámetros en Detect head | ~16K adicionales (DFL Conv1d) | **Menos** |
| Compatibilidad hardware | Requiere softmax en MCU | **Multiplicación simple** |
| Tamaño ONNX estimado | ~9.97 MB | **Ligeramente menor** |
| Inferencia CPU | Baseline | **~43% más rápido** (claim oficial) |

### Motivación para el TFM

1. **Simplificación del firmware ESP32-S3**: el post-processing en C se reduce drásticamente al eliminar la integral DFL
2. **Mejor compatibilidad con INT8**: 4 canales de box con rango uniforme se cuantizan mejor que 64 canales de distribución probabilística
3. **Reducción de latencia**: menos operaciones flotantes en el MCU
4. **Alineación con la especificación oficial**: nuestro modelo adopta la arquitectura canónica de YOLO26

---

## 2. Análisis de Impacto: DFL vs sin DFL

### 2.1 Cabeza Detect — Comparación estructural

```
ACTUAL (Ultralytics 8.4.9, reg_max=16):
  cv2[i]: Conv → Conv → Conv2d(out=64)    # 16 bins × 4 coords
  cv3[i]: DWConv → Conv → DWConv → Conv → Conv2d(out=5)
  dfl:    DFL(Conv2d(16, 1, 1, 1))         # integral ponderada

ACTUALIZADO (sin DFL, reg_max=1):
  cv2[i]: Conv → Conv → Conv2d(out=4)     # 4 coords directas (l, t, r, b)
  cv3[i]: DWConv → Conv → DWConv → Conv → Conv2d(out=5)
  dfl:    nn.Identity()                     # no-op
```

### 2.2 Impacto en salidas ONNX ESP

| Salida | Actual (con DFL) | Actualizado (sin DFL) |
|---|---|---|
| `box0` (P3, stride 8) | `[1, 64, 28, 28]` | **`[1, 4, 28, 28]`** |
| `score0` (P3) | `[1, 5, 28, 28]` | `[1, 5, 28, 28]` (sin cambio) |
| `box1` (P4, stride 16) | `[1, 64, 14, 14]` | **`[1, 4, 14, 14]`** |
| `score1` (P4) | `[1, 5, 14, 14]` | `[1, 5, 14, 14]` |
| `box2` (P5, stride 32) | `[1, 64, 7, 7]` | **`[1, 4, 7, 7]`** |
| `score2` (P5) | `[1, 5, 7, 7]` | `[1, 5, 7, 7]` |

**Reducción de datos en salida box**: 64 → 4 canales = **16× menos datos** para decodificar.

### 2.3 Impacto en post-procesamiento

```python
# ACTUAL — con DFL (decode_yolo26_esp):
raw_box = output[f"box{level}"]               # [1, 64, H, W]
box = raw_box.reshape(N, 4, 16)               # reshape a bins
box = F.softmax(box, dim=2)                   # softmax por bin
box = (box * torch.arange(16)).sum(dim=2)     # integral ponderada → [N, 4]
# luego dist2bbox con stride

# ACTUALIZADO — sin DFL (decode_yolo26_esp_v3):
raw_box = output[f"box{level}"]               # [1, 4, H, W]
box = raw_box.reshape(N, 4)                   # ya son 4 distancias directas
# luego dist2bbox con stride (igual)
```

---

## 3. Prerequisitos y Verificaciones Previas

### 3.1 Identificar la versión mínima de Ultralytics con DFL Removal

Antes de proceder, verificar qué versión de Ultralytics implementa la eliminación de DFL para YOLO26:

```bash
# Opción A: Consultar el changelog de Ultralytics
pip index versions ultralytics 2>/dev/null | head -5

# Opción B: Inspeccionar releases en GitHub
# https://github.com/ultralytics/ultralytics/releases
# Buscar la release que introduce "DFL Removal" o "reg_max=1 for YOLO26"

# Opción C: Instalar la última versión en un env temporal y verificar
conda create -n test_ultra python=3.10 -y
conda activate test_ultra
pip install ultralytics
python -c "
from ultralytics import YOLO, __version__
print(f'Ultralytics version: {__version__}')
m = YOLO('yolo26n.pt')
for name, mod in m.model.named_modules():
    if hasattr(mod, 'reg_max'):
        print(f'{name}: reg_max={mod.reg_max}')
        if hasattr(mod, 'dfl'):
            print(f'  dfl type: {type(mod.dfl).__name__}')
        break
"
conda deactivate
conda remove -n test_ultra --all -y
```

> **Criterio de aceptación**: `reg_max=1` y `dfl` es `nn.Identity()` (no `DFL(Conv2d(16,1,...))`).

### 3.2 Verificar compatibilidad con PyTorch 2.4

El contenedor Vertex AI usa `pytorch-gpu.2-4.py310`. Verificar que la nueva versión de Ultralytics es compatible:

```python
# En el env temporal:
import torch
print(f"PyTorch: {torch.__version__}")
# Debe funcionar con PyTorch 2.4.x
```

### 3.3 Verificar que los pesos preentrenados `yolo26n.pt` incluyen sin DFL

```python
from ultralytics import YOLO
m = YOLO("yolo26n.pt")
detect = None
for _, mod in m.model.named_modules():
    if hasattr(mod, 'reg_max'):
        detect = mod
        break

assert detect.reg_max == 1, f"ERROR: reg_max={detect.reg_max}, esperado 1"
assert type(detect.dfl).__name__ in ("Identity",), \
    f"ERROR: dfl es {type(detect.dfl).__name__}, esperado Identity"
print("✅ Pesos yolo26n.pt sin DFL confirmado")
```

### 3.4 Verificar disponibilidad del optimizador MuSGD

MuSGD (SGD + Muon hybrid) es nativo de YOLO26. Verificar que sigue disponible:

```python
from ultralytics import YOLO
model = YOLO("yolo26n.yaml")
# Si la versión es compatible, MuSGD seguirá como optimizador por defecto
```

> **Si MuSGD fue renombrado o removido**, ajustar `optimizer` en la configuración YAML.

---

## 4. Fase 1 — Actualización de Ultralytics

### 4.1 Actualizar el entorno local

```bash
cd /Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/Train_MLOps
conda activate ../../env  # o el env que corresponda

# Guardar versión actual por si necesitamos rollback
pip show ultralytics | grep Version > /tmp/ultralytics_version_backup.txt

# Actualizar
pip install --upgrade ultralytics

# Verificar
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
```

### 4.2 Verificar que el modelo YOLO26n carga correctamente

```python
from ultralytics import YOLO

# Desde YAML (arquitectura vacía)
m_yaml = YOLO("yolo26n.yaml")
print(f"Params: {sum(p.numel() for p in m_yaml.model.parameters()):,}")

# Desde .pt (pretrained COCO)
m_pt = YOLO("yolo26n.pt")
# Verificar DFL removal
for _, mod in m_pt.model.named_modules():
    if hasattr(mod, 'reg_max'):
        print(f"reg_max: {mod.reg_max}")
        print(f"dfl: {type(mod.dfl).__name__}")
        if hasattr(mod, 'cv2'):
            print(f"cv2[0] out_channels: {mod.cv2[0][-1].out_channels}")
        break
```

**Resultado esperado**:
```
reg_max: 1
dfl: Identity
cv2[0] out_channels: 4
```

### 4.3 Verificar API de entrenamiento

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")

# Verificar que la API de entrenamiento sigue igual
print(dir(model))  # Buscar: train, val, export, predict
print(model.task)   # Debe ser 'detect'

# Verificar freeze con la nueva versión
model.train(
    data="coco8.yaml",  # dataset toy de Ultralytics
    epochs=1,
    imgsz=224,
    batch=4,
    freeze=10,
    device="cpu",
    verbose=False,
)
print("✅ API de entrenamiento funcional")
```

---

## 5. Fase 2 — Validación de la Arquitectura Actualizada

### 5.1 Inspección de la arquitectura YOLO26n actualizada

```python
from ultralytics import YOLO

model = YOLO("yolo26n.yaml")

# Imprimir la arquitectura completa
print(model.model)

# Contar parámetros
total = sum(p.numel() for p in model.model.parameters())
trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
print(f"\nTotal params:     {total:,}")
print(f"Trainable params: {trainable:,}")
```

### 5.2 Verificar salidas del forward pass

```python
import torch
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.model.eval()

dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model.model(dummy)

# Dependiendo de la API, out puede ser un tensor o tuple
print(f"Output type: {type(out)}")
if isinstance(out, torch.Tensor):
    print(f"Output shape: {out.shape}")
elif isinstance(out, (tuple, list)):
    for i, o in enumerate(out):
        print(f"Output[{i}] shape: {o.shape}")
```

### 5.3 Comparar número de parámetros con el modelo actual

| Versión | Parámetros | Diferencia |
|---|---|---|
| Ultralytics 8.4.9 (con DFL) | ~2,572,280 | Baseline |
| Ultralytics actualizada (sin DFL) | *Por verificar* | *Esperar reducción* |

> La reducción debería ser modesta (~10-15K parámetros menos en la capa DFL y las últimas conv de cv2).

---

## 6. Fase 3 — Preparación del Entorno Vertex AI

### 6.1 Actualizar `task_yolo26_custom.py` — Versión de Ultralytics

**Archivo**: `trainer/task_yolo26_custom.py`

**Cambio**: Actualizar la línea de instalación dinámica de Ultralytics.

```python
# ANTES (línea ~32):
subprocess.check_call(
    [_sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4"],
)

# DESPUÉS:
subprocess.check_call(
    [_sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4,>=X.Y.Z"],
)
# Reemplazar X.Y.Z con la versión mínima que tiene DFL Removal
```

> **Alternativa más segura** (versión fija para reproducibilidad):
> ```python
> subprocess.check_call(
>     [_sys.executable, "-m", "pip", "install", "-q", "ultralytics==X.Y.Z"],
> )
> ```

### 6.2 Actualizar `setup.py` — Version bump

**Archivo**: `setup.py`

```python
# Incrementar versión para forzar cache invalidation en Vertex AI
setup(
    name="tfm-trainer",
    version="2.7.0",  # era 2.6.3
    ...
)
```

Agregar entry en el docstring de cambios:

```python
"""
Changes from v2.6.3:
    - YOLO26 v3: Ultralytics upgrade para DFL Removal
    - task_yolo26_custom: ultralytics>=X.Y.Z (YOLO26 sin DFL)
    - reg_max=1 → cv2 outputs 4 channels (antes: reg_max=16 → 64 channels)
    - Nuevo config: yolo26n_custom_v3.yaml
    - Version bump forces pip cache invalidation on Vertex AI
"""
```

### 6.3 Actualizar `requirements_yolo.txt`

**Archivo**: `requirements_yolo.txt` (en `GoogleCloudAI/`)

```
ultralytics>=X.Y.Z    # Mínima versión con DFL Removal para YOLO26
numpy>=1.26,<2.0
pandas>=2.0
matplotlib>=3.8
opencv-python-headless>=4.9
pyyaml>=6.0
google-cloud-storage>=2.14
google-cloud-aiplatform>=1.40
```

### 6.4 Verificar compatibilidad de `src_colab/` con nueva versión

Los siguientes módulos importan de Ultralytics y podrían requerir ajustes:

```bash
# Buscar imports de ultralytics en src_colab/
grep -rn "ultralytics" src_colab/ trainer/
```

**Módulos críticos a revisar**:

| Archivo | Imports | Riesgo |
|---|---|---|
| `src_colab/utils_model.py` | `build_yolo26_custom_model()` | Bajo — usa `YOLO('yolo11n.pt')` |
| `src_colab/utils_train.py` | `Yolo26CustomConfig`, `train_yolo26_custom()` | Bajo — wrapper sobre `model.train()` |
| `trainer/task_yolo26_custom.py` | `Detect` (solo en DEPLOY VERIFICATION) | Bajo — solo inspección |

**Acción**: Ejecutar un test local rápido antes de subir a Vertex AI:

```bash
cd /Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/Train_MLOps

python -c "
from src_colab import build_yolo26_custom_model, Yolo26CustomConfig
model = build_yolo26_custom_model(variant='yolo11n')
print(f'✅ build_yolo26_custom_model funciona')

# Verificar que reg_max es 1
from ultralytics.nn.modules.head import Detect
for _, mod in model.model.named_modules():
    if isinstance(mod, Detect):
        assert mod.reg_max == 1, f'reg_max={mod.reg_max}'
        print(f'✅ reg_max=1 confirmado')
        print(f'  cv2[0] out: {mod.cv2[0][-1].out_channels}')
        break
"
```

### 6.5 Verificar workarounds existentes

Nuestro `task_yolo26_custom.py` tiene dos workarounds que podrían necesitar revisión:

1. **DDP cleanup** — *Probablemente sigue necesario* (es un tema de Vertex AI, no de Ultralytics)
2. **RandomSampler.set_epoch monkey-patch** — *Verificar si la nueva versión de Ultralytics lo resuelve*

```python
# Test del monkey-patch:
import torch.utils.data as tud
print(f"RandomSampler has set_epoch: {hasattr(tud.RandomSampler, 'set_epoch')}")
# Si es True en la nueva versión, el monkey-patch ya no es necesario (pero tampoco es dañino)
```

---

## 7. Fase 4 — Configuración del Entrenamiento

### 7.1 Crear nuevo config YAML

**Archivo a crear**: `vertex_ai/configs/yolo26n_custom_v3.yaml`

Copiar la configuración del Train 2 exitoso (`yolo26n_custom_v2.yaml`) y ajustar:

```yaml
model:
  family: YOLO26_CUSTOM
  variant: yolo26n_custom
  version: v3                          # ← Nuevo: v3 = sin DFL

dataset:
  name: iodc_yolo
  gcs_uri: gs://project-18f58341-12cf-47bc-861-tfm-data/datasets/iodc_yolo.zip
  class_names: [dog, door, obstacle, person, stair]
  img_size: 640

common:
  batch_size: 16
  patience: 30
  seed: 42
  conf_threshold: 0.15
  iou_threshold: 0.45

yolo26_custom:
  pretrained_weights: yolo26n.pt       # ← CAMBIO: usar yolo26n.pt directamente
                                       #   (asegura pesos sin DFL de la nueva versión)
                                       #   Antes: yolo11n.pt (transferencia cross-model)

  # ── Estrategia de 2 fases (igual que Train 2) ──
  phase1_epochs: 30
  phase1_freeze_layers: 10
  phase1_lr0: 0.01
  phase1_lrf: 0.01

  phase2_epochs: 70
  phase2_lr0: 0.0005                   # ← Mantener: probado exitoso en T2
  phase2_lrf: 0.01                     # ← Mantener: cosine decay con floor
  
  # ── Optimización (igual que Train 2) ──
  optimizer: MuSGD                     # ← Verificar disponibilidad (ver §3.4)
  momentum: 0.9
  weight_decay: 0.0005
  warmup_epochs: 3.0
  warmup_bias_lr: 0.0                  # ← CRÍTICO con MuSGD explícito
  
  # ── Aumentación (igual que Train 2) ──
  mosaic: 1.0
  mixup: 0.1
  close_mosaic: 10
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  fliplr: 0.5
  scale: 0.5
  translate: 0.1
  
  # ── Loss (igual que Train 2) ──
  box: 7.5
  cls: 0.5
  
  # ── Export ──
  export_imgsz: 224
  export_opset: 13
```

### 7.2 Decisión sobre pesos preentrenados

Hay dos opciones para la inicialización:

| Opción | Pretrained | Pros | Contras |
|---|---|---|---|
| **A** | `yolo26n.pt` (COCO, sin DFL) | Arquitectura nativa, transfer directo | Diferente punto de partida que T2 |
| **B** | `yolo11n.pt` (COCO, original) | Misma semilla que T2 exitoso | Cross-model transfer (YOLO11 → YOLO26) |

**Recomendación: Opción A** (`yolo26n.pt`) — Porque:
- Los pesos preentrenados ahora coinciden exactamente con la arquitectura YOLO26 actualizada
- No hay necesidad de cross-model transfer si YOLO26 tiene pesos nativos COCO
- Evita inconsistencias entre backbone de YOLO11 y head de YOLO26 actualizada

> **Nota**: Verificar si `build_yolo26_custom_model()` en `src_colab/utils_model.py` necesita ajuste para aceptar `yolo26n.pt` como variant en lugar de `yolo11n.pt`.

### 7.3 Ajustes potenciales

#### Si MuSGD no está disponible en la nueva versión:

```yaml
# Alternativa 1: SGD puro
optimizer: SGD

# Alternativa 2: auto (Ultralytics elige AdamW para datasets pequeños)
# NO RECOMENDADO — ignora lr0/momentum configurados
optimizer: auto
```

#### Si hay nuevos hiperparámetros en la versión actualizada:

Consultar `model.train()` signature de la nueva versión y agregar si aplica:

```python
from ultralytics import YOLO
help(YOLO.train)  # Revisar nuevos parámetros
```

---

## 8. Fase 5 — Lanzamiento en Vertex AI

### 8.1 Rebuild del paquete

```bash
cd /Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/Train_MLOps

# 1. Limpiar builds anteriores
rm -rf dist/ build/ *.egg-info

# 2. Verificar que el YAML nuevo está en su lugar
cat vertex_ai/configs/yolo26n_custom_v3.yaml

# 3. Verificar la versión en setup.py
grep "version=" setup.py
# Debe mostrar: version="2.7.0"
```

### 8.2 Subir configuración a GCS

```bash
# Subir el nuevo config a GCS para el entry-point
gsutil cp vertex_ai/configs/yolo26n_custom_v3.yaml \
    gs://project-18f58341-12cf-47bc-861-tfm-data/configs/yolo26n_custom_v3.yaml
```

### 8.3 Lanzar el Job

```bash
# Dry-run primero para verificar
./vertex_ai/build_and_launch.sh yolo26n_custom_v3 --dry-run

# Si todo está bien, lanzar
./vertex_ai/build_and_launch.sh yolo26n_custom_v3 --run-name yolo26n_custom_v3-run1
```

**Infraestructura** (sin cambios):

| Parámetro | Valor |
|---|---|
| Máquina | `n1-standard-8` (8 vCPU, 30 GB RAM) |
| GPU | NVIDIA Tesla T4 × 1 |
| Contenedor | `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest` |
| Región | `us-central1` |
| Duración estimada | ~2-3 horas (100 epochs, 1670 images @ 640×640) |

### 8.4 Monitoreo

```bash
# Seguir logs en tiempo real
gcloud ai custom-jobs stream-logs <JOB_ID> \
    --project=project-18f58341-12cf-47bc-861 \
    --region=us-central1
```

**Indicadores de éxito durante el entrenamiento**:
- Phase 1: mAP@50 debe alcanzar ~0.4-0.5 en 30 epochs (similar a T2)
- Phase 2: mAP@50 debe subir a ~0.7+ y estabilizar
- DEPLOY VERIFICATION debe mostrar `reg_max: 1` (no 16)

---

## 9. Fase 6 — Descarga y Verificación de Artefactos

### 9.1 Descargar artefactos de GCS

```bash
# Crear directorio local
mkdir -p outputs/yolo26n_custom_v3-run1

# Descargar todo
gsutil -m rsync -r \
    gs://project-18f58341-12cf-47bc-861-tfm-data/output/yolo26n_custom_v3-run1/ \
    outputs/yolo26n_custom_v3-run1/
```

### 9.2 Estructura esperada

```
outputs/yolo26n_custom_v3-run1/
├── yolo_project/
│   ├── phase1/
│   │   ├── weights/best.pt
│   │   └── results.csv
│   └── phase2/
│       ├── weights/best.pt          ← Modelo final
│       └── results.csv
├── export/
│   └── best.onnx                    ← ONNX estándar (auto-export de Ultralytics)
├── results_combined.csv
├── curves_combined.png
├── val_results.json
├── test_results.json
└── experiment.json
```

### 9.3 Verificación de métricas

```python
import json

with open("outputs/yolo26n_custom_v3-run1/test_results.json") as f:
    results = json.load(f)

print(f"mAP@50:    {results['map50']:.4f}")
print(f"mAP@50-95: {results['map']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall:    {results['recall']:.4f}")
```

**Criterio de aceptación**: mAP@50 ≥ 0.75 (comparable al T2: 0.7747).

> **Nota**: Una pequeña variación (±3%) es aceptable dado que los pesos iniciales son diferentes. Si hay una caída significativa (>10%), revisar la configuración de hiperparámetros.

### 9.4 Verificación de la arquitectura del modelo entrenado

```python
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

model = YOLO("outputs/yolo26n_custom_v3-run1/yolo_project/phase2/weights/best.pt")

for _, mod in model.model.named_modules():
    if isinstance(mod, Detect):
        print(f"reg_max:          {mod.reg_max}")
        print(f"dfl type:         {type(mod.dfl).__name__}")
        print(f"cv2[0] out_ch:    {mod.cv2[0][-1].out_channels}")
        
        assert mod.reg_max == 1, "❌ El modelo todavía tiene DFL"
        assert mod.cv2[0][-1].out_channels == 4, "❌ cv2 no tiene 4 canales"
        print("✅ Modelo sin DFL confirmado")
        break
```

---

## 10. Fase 7 — Exportación ONNX ESP-compatible

### 10.1 Actualizar `export_yolo26_esp.py`

El script de exportación ESP necesita ajustes significativos porque las salidas de `cv2` ahora serán `[1, 4, H, W]` en lugar de `[1, 64, H, W]`.

**Archivo**: `scripts/export_yolo26_esp.py`

**Cambios necesarios**:

```python
# ============================================================================
# CAMBIO 1: Actualizar configuración
# ============================================================================

PT_FILE = TRAIN_MLOPS_DIR / "outputs/yolo26n_custom_v3-run1/yolo_project/phase2/weights/best.pt"
OUTPUT_DIR = TRAIN_MLOPS_DIR / "outputs/yolo26n_custom_v3-run1/export"
OUTPUT_FILE = OUTPUT_DIR / "best_esp.onnx"


# ============================================================================
# CAMBIO 2: El forward patch de ESP_Detect_Forward NO CAMBIA
# ============================================================================
# El monkey-patch sigue siendo el mismo:
#   - self.cv2[i](x[i]) → ahora produce [1, 4, H, W] en vez de [1, 64, H, W]
#   - self.cv3[i](x[i]) → sigue produciendo [1, 5, H, W]
# No hay cambio en el código, solo cambian las shapes automáticamente.


# ============================================================================
# CAMBIO 3: Actualizar shapes esperados en verificación
# ============================================================================

# ANTES:
expected_shapes = {
    "box0": [1, reg_max * 4, 28, 28],  # [1, 64, 28, 28]
    ...
}

# DESPUÉS (reg_max será 1 → reg_max * 4 = 4):
expected_shapes = {
    "box0": [1, reg_max * 4, 28, 28],  # [1, 4, 28, 28] ← automático
    ...
}
# NOTA: Como usa reg_max dinámicamente, NO requiere cambio manual.
# Solo verificar que el printout muestra 4 en vez de 64.
```

> **Nota importante**: dado que `export_yolo26_esp.py` ya usa `reg_max` del modelo dinámicamente (`detect_module.reg_max`), **el script funciona sin cambios en la lógica**. Solo hay que actualizar los paths.

### 10.2 Ejecutar la exportación

```bash
cd /Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/Train_MLOps

# Asegurarse de que el env tiene la nueva versión de Ultralytics
conda run -p ../env python scripts/export_yolo26_esp.py
```

**Salida esperada**:

```
  nc=5, reg_max=1, nl=3              ← Antes: reg_max=16
  Box output channels: 4 (reg_max × 4)   ← Antes: 64
  Score output channels: 5
  
  Output: box0 [1, 4, 28, 28]        ← Antes: [1, 64, 28, 28]
  Output: score0 [1, 5, 28, 28]
  Output: box1 [1, 4, 14, 14]        ← Antes: [1, 64, 14, 14]
  Output: score1 [1, 5, 14, 14]
  Output: box2 [1, 4, 7, 7]          ← Antes: [1, 64, 7, 7]
  Output: score2 [1, 5, 7, 7]
  
  ✅ Exportación exitosa — 6 salidas separadas
```

### 10.3 Verificar tamaño ONNX

```bash
ls -lh outputs/yolo26n_custom_v3-run1/export/best_esp.onnx
# Esperar: ligeramente menor que 9.92 MB (menos parámetros en cv2)
```

---

## 11. Fase 8 — Cuantización INT8 y Exportación ESPDL

### 11.1 Actualizar `convert_onnx_to_espdl.py`

**Archivo**: `scripts/convert_onnx_to_espdl.py`

Agregar una nueva entrada de modelo para la versión v3:

```python
# En la sección de MODEL_CONFIGS:
"yolo26n_t3_esp": {
    "onnx": "outputs/yolo26n_custom_v3-run1/export/best_esp.onnx",
    "input_name": "images",
    "target": "esp32s3",
    "num_calib": 500,
},
```

### 11.2 Ejecutar la cuantización

```bash
cd /Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/Train_MLOps

conda run -p ../env python scripts/convert_onnx_to_espdl.py \
    --calib-dir ../datasets/IODC/coco/train/images \
    --models yolo26n_t3_esp
```

**Proceso**:
1. `fix_negative_axes()` — Convierte atributos de eje negativo a positivo
2. Carga 500 imágenes de calibración (224×224, normalización [0,1])
3. `espdl_quantize_onnx()` — Cuantización INT8 simétrica POWER_OF_2
4. Genera: `best_esp.espdl` + `best_esp.info` + `best_esp.json`

### 11.3 Verificar artefactos ESPDL

```bash
# Verificar que se generaron los archivos
ls -lh outputs/yolo26n_custom_v3-run1/export/best_esp.espdl
ls -lh outputs/yolo26n_custom_v3-run1/export/best_esp.info
ls -lh outputs/yolo26n_custom_v3-run1/export/best_esp.json
```

### 11.4 Inspeccionar exponentes de cuantización

```bash
cat outputs/yolo26n_custom_v3-run1/export/best_esp.info
```

**Verificar**:
- Input: `exponent: -7` (estándar)
- Outputs box{0,1,2}: Esperar exponentes diferentes a T2 (porque las distribuciones cambian sin DFL)
- Outputs score{0,1,2}: Esperar similares a T2 (exponentes -2 a -3)

> **Expectativa de mejora en cuantización**: Las salidas box ahora son 4 valores directos de distancia (rango 0-28 stride-scaled), en vez de 64 valores de distribución probabilística (rango 0-1 softmax). Esto debería dar **mejor resolución INT8** para las boxes porque el rango dinámico es más uniforme.

---

## 12. Fase 9 — Evaluación FP32 vs INT8

### 12.1 Actualizar `eval_fp32_vs_int8.py`

**Archivo**: `scripts/eval_fp32_vs_int8.py`

#### a) Agregar config del nuevo modelo:

```python
"yolo26n_t3_esp": {
    "onnx": "outputs/yolo26n_custom_v3-run1/export/best_esp.onnx",
    "input_name": "images",
    "family": "yolo26_esp_v3",         # ← Nueva familia sin DFL
    "conf_threshold": 0.25,
    "nms_threshold": 0.45,
},
```

#### b) Crear nueva función de decode sin DFL:

```python
def decode_yolo26_esp_v3(outputs, input_size=224, conf_threshold=0.25,
                          nms_threshold=0.45):
    """Decode YOLO26 v3 ESP outputs — SIN DFL.
    
    A diferencia de decode_yolo26_esp(), los boxes ya son 4 canales
    de distancia directa (l, t, r, b) sin necesidad de integral DFL.
    """
    all_boxes = []
    all_scores = []
    all_classes = []
    
    strides = [8, 16, 32]
    
    for level in range(3):
        box_key = f"box{level}"
        score_key = f"score{level}"
        stride = strides[level]
        
        raw_box = outputs[box_key]       # [1, 4, H, W] ← era [1, 64, H, W]
        raw_score = outputs[score_key]   # [1, 5, H, W]
        
        _, nc, H, W = raw_score.shape
        N = H * W
        
        # ── Scores: sigmoid + threshold ──
        scores = raw_score.reshape(nc, N).T         # [N, nc]
        scores = 1 / (1 + np.exp(-scores))           # sigmoid
        
        max_scores = scores.max(axis=1)
        mask = max_scores > conf_threshold
        if not mask.any():
            continue
        
        scores_filt = scores[mask]
        class_ids = scores_filt.argmax(axis=1)
        max_scores_filt = scores_filt[np.arange(len(scores_filt)), class_ids]
        
        # ── Boxes: distancias directas (SIN DFL) ──
        boxes = raw_box.reshape(4, N).T              # [N, 4]
        boxes_filt = boxes[mask]                      # [M, 4] = (l, t, r, b)
        
        # ── Grid centers ──
        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cx = (grid_x.flatten() + 0.5) * stride       # [N]
        cy = (grid_y.flatten() + 0.5) * stride
        cx_filt = cx[mask]
        cy_filt = cy[mask]
        
        # ── dist2bbox ──
        x1 = (cx_filt - boxes_filt[:, 0] * stride) / input_size
        y1 = (cy_filt - boxes_filt[:, 1] * stride) / input_size
        x2 = (cx_filt + boxes_filt[:, 2] * stride) / input_size
        y2 = (cy_filt + boxes_filt[:, 3] * stride) / input_size
        
        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        xyxy = np.clip(xyxy, 0.0, 1.0)
        
        all_boxes.append(xyxy)
        all_scores.append(max_scores_filt)
        all_classes.append(class_ids)
    
    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)
    
    # NMS por clase
    keep = batched_nms_numpy(boxes, scores, classes, nms_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

**Diferencia clave**: Se elimina completamente el bloque de DFL integral:
```python
# ELIMINADO (ya no necesario):
# boxes = raw_box.reshape(N, 4, 16)           # ← No existe (era reg_max=16)
# boxes = softmax(boxes, axis=2)               # ← No existe
# boxes = (boxes * np.arange(16)).sum(axis=2)  # ← No existe
```

### 12.2 Ejecutar la evaluación

```bash
cd /Users/admin/Documents/TFM_UNIR/02_ING_MODELOS/Train_MLOps

conda run -p ../env python scripts/eval_fp32_vs_int8.py \
    --models yolo26n_t3_esp \
    --n-viz 8
```

### 12.3 Criterios de aceptación

| Métrica | Umbral | Notas |
|---|---|---|
| mAP@50 FP32 | ≥ 0.50 | ONNX a 224×224 (comparable a T2: 0.5297) |
| **Degradación INT8** | **≤ 20%** | T2 logró 18.0% — esperar ≤15% sin DFL |
| Visualizaciones | Coherentes | Cajas bien posicionadas, clases correctas |

> **Expectativa**: La degradación INT8 debería ser **menor** que en T2 (18%) porque:
> 1. Sin DFL, los 4 canales de box tienen distribución más uniforme
> 2. Menos operaciones post-cuantización susceptibles a error acumulado
> 3. El rango dinámico de distancias directas es más predecible que distribuciones softmax

---

## 13. Fase 10 — Integración en Firmware ESP32-S3

### 13.1 Actualizar el decode en C/C++ del firmware

El post-processing en el firmware (`03_ING_DESPLIEGUE/main/`) se simplifica drásticamente.

**Código actual (con DFL)**:
```c
// Para cada nivel de detección:
// 1. Leer 64 valores de raw_box por candidato
// 2. Aplicar softmax sobre 16 bins
// 3. Integral ponderada: sum(softmax[i] * i) para cada coordenada
// 4. dist2bbox con stride
```

**Código actualizado (sin DFL)**:
```c
// Para cada nivel de detección:
// 1. Leer 4 valores directos de raw_box por candidato ← MUCHO MÁS SIMPLE
// 2. dist2bbox con stride (igual)
```

**Pseudocódigo C simplificado**:

```c
void decode_yolo26_v3_level(
    const int8_t* box_data,     // [4, H, W] cuantizado
    const int8_t* score_data,   // [5, H, W] cuantizado
    int box_exponent,
    int score_exponent,
    int H, int W, int stride,
    float conf_threshold,
    Detection* detections,
    int* num_detections
) {
    float box_scale = pow(2.0f, box_exponent);
    float score_scale = pow(2.0f, score_exponent);
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;
            
            // ── Scores: dequant + sigmoid ──
            float max_score = -1.0f;
            int best_class = -1;
            for (int c = 0; c < 5; c++) {
                float raw = score_data[c * H * W + idx] * score_scale;
                float score = 1.0f / (1.0f + expf(-raw));
                if (score > max_score) {
                    max_score = score;
                    best_class = c;
                }
            }
            
            if (max_score < conf_threshold) continue;
            
            // ── Boxes: dequant directo (SIN DFL) ──
            float l = box_data[0 * H * W + idx] * box_scale;  // distancia left
            float t = box_data[1 * H * W + idx] * box_scale;  // distancia top
            float r = box_data[2 * H * W + idx] * box_scale;  // distancia right
            float b = box_data[3 * H * W + idx] * box_scale;  // distancia bottom
            
            // ── dist2bbox ──
            float cx = (x + 0.5f) * stride;
            float cy = (y + 0.5f) * stride;
            
            float x1 = cx - l * stride;
            float y1 = cy - t * stride;
            float x2 = cx + r * stride;
            float y2 = cy + b * stride;
            
            // Guardar detección
            detections[*num_detections] = (Detection){
                .x1 = x1, .y1 = y1, .x2 = x2, .y2 = y2,
                .score = max_score,
                .class_id = best_class,
            };
            (*num_detections)++;
        }
    }
}
```

### 13.2 Beneficios en firmware

| Aspecto | Con DFL (actual) | Sin DFL (v3) |
|---|---|---|
| Lecturas de memoria por candidato (box) | 64 bytes | **4 bytes** |
| Operaciones softmax por candidato | 4 × 16 = 64 exp() + norm | **0** |
| Multiplicaciones DFL por candidato | 64 mul + 4 sum | **0** |
| Total FLOPs box decode (1029 cands) | ~132K | **~8K** |
| Código C++ (líneas) | ~80-100 | **~30-40** |

### 13.3 Actualizar `Instructivo_Despliegue_ESPDL.md`

Tras completar la validación, actualizar la sección de YOLO26 ESP en el instructivo de despliegue para reflejar la nueva arquitectura sin DFL.

---

## 14. Resumen de Cambios Esperados

### Archivos a modificar

| Archivo | Cambio | Prioridad |
|---|---|---|
| `setup.py` | version → `2.7.0`, docstring | Alta |
| `trainer/task_yolo26_custom.py` | `ultralytics>=X.Y.Z` | Alta |
| `requirements_yolo.txt` | `ultralytics>=X.Y.Z` | Alta |
| `scripts/export_yolo26_esp.py` | Actualizar paths (v3) | Media |
| `scripts/convert_onnx_to_espdl.py` | Agregar `yolo26n_t3_esp` config | Media |
| `scripts/eval_fp32_vs_int8.py` | Agregar decode sin DFL + config | Media |
| `docs/Instructivo_Despliegue_ESPDL.md` | Actualizar post-processing YOLO26 | Baja (post-validación) |
| `docs/Registro_Cuantizacion_Modelos.md` | Agregar sección T3 v3 | Baja (post-validación) |

### Archivos a crear

| Archivo | Contenido |
|---|---|
| `vertex_ai/configs/yolo26n_custom_v3.yaml` | Config de entrenamiento v3 |

### Pipeline completo (orden de ejecución)

```
1. Verificar versión Ultralytics con DFL Removal
2. Actualizar env local
3. Validar arquitectura YOLO26n sin DFL
4. Crear yolo26n_custom_v3.yaml
5. Actualizar task_yolo26_custom.py (versión ultralytics)
6. Actualizar setup.py (version bump)
7. Test local rápido (build + API check)
8. Lanzar en Vertex AI
9. Monitorear entrenamiento
10. Descargar artefactos
11. Verificar métricas + arquitectura
12. Exportar ONNX ESP (6 salidas)
13. Cuantizar INT8 → ESPDL
14. Evaluar FP32 vs INT8
15. Integrar en firmware ESP32-S3
16. Actualizar documentación
```

---

## 15. Checklist de Ejecución

### Pre-entrenamiento
- [ ] Identificar versión mínima de Ultralytics con DFL Removal
- [ ] Verificar compatibilidad con PyTorch 2.4
- [ ] Verificar pesos `yolo26n.pt` sin DFL (`reg_max=1`, `dfl=Identity`)
- [ ] Verificar MuSGD disponible
- [ ] Actualizar `ultralytics>=X.Y.Z` en `task_yolo26_custom.py`
- [ ] Actualizar `setup.py` → `version="2.7.0"`
- [ ] Crear `vertex_ai/configs/yolo26n_custom_v3.yaml`
- [ ] Test local: `build_yolo26_custom_model()` funciona con nueva versión
- [ ] Test local: API de entrenamiento funcional (1 epoch coco8)
- [ ] Subir config a GCS

### Entrenamiento
- [ ] `./vertex_ai/build_and_launch.sh yolo26n_custom_v3 --dry-run`
- [ ] `./vertex_ai/build_and_launch.sh yolo26n_custom_v3 --run-name yolo26n_custom_v3-run1`
- [ ] Monitorear logs — DEPLOY VERIFICATION muestra `reg_max: 1`
- [ ] Entrenamiento completado sin errores
- [ ] mAP@50 test ≥ 0.75

### Post-entrenamiento (local)
- [ ] Descargar artefactos de GCS
- [ ] Verificar `reg_max=1` y `cv2 out_channels=4` en best.pt
- [ ] Actualizar paths en `export_yolo26_esp.py`
- [ ] Exportar ONNX ESP → verificar shapes `[1, 4, H, W]` para boxes
- [ ] Agregar `yolo26n_t3_esp` en `convert_onnx_to_espdl.py`
- [ ] Cuantizar → `best_esp.espdl` generado
- [ ] Agregar `decode_yolo26_esp_v3()` en `eval_fp32_vs_int8.py`
- [ ] Evaluar FP32 vs INT8 → degradación ≤ 20%
- [ ] Generar visualización comparativa

### Deployment
- [ ] Actualizar decode en firmware C/C++ (eliminar DFL integral)
- [ ] Copiar `.espdl` a `03_ING_DESPLIEGUE/models/`
- [ ] Flash y testear en ESP32-S3
- [ ] Actualizar `Instructivo_Despliegue_ESPDL.md`
- [ ] Actualizar `Registro_Cuantizacion_Modelos.md`

---

## 16. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| MuSGD removido/renombrado en nueva versión | Media | Alto | Probar con SGD puro como fallback; verificar con `--dry-run` en local |
| API de entrenamiento cambió (nuevos params obligatorios) | Baja | Alto | Revisar changelog antes de actualizar; test local con coco8 |
| Degradación de mAP (>10% vs T2) | Baja | Medio | Los pesos `yolo26n.pt` deberían ser tan buenos como `yolo11n.pt` para transfer; re-tuning de LR si necesario |
| esp-ppq no soporta las nuevas operaciones ONNX | Muy baja | Alto | YOLO26 sin DFL tiene **menos** ops, no más; verificar con `fix_negative_axes` |
| Incompatibilidad PyTorch 2.4 + nueva Ultralytics | Baja | Alto | Verificar en env local antes de lanzar en Vertex AI |
| Pesos `yolo26n.pt` no disponibles en nueva versión | Muy baja | Alto | Ultralytics siempre publica pesos preentrenados para sus modelos oficiales |
| `RandomSampler.set_epoch` ya no necesario | N/A | Nulo | El monkey-patch es idempotente, no causa problemas |
| Exponentes ESPDL cambian significativamente | Media | Bajo | Revisar `.info` y ajustar si la degradación INT8 es alta |

### Plan de Rollback

Si la actualización no produce resultados satisfactorios:

```bash
# Revertir a Ultralytics 8.4.9
pip install ultralytics==8.4.9

# El modelo T2 (con DFL) sigue disponible y funcional:
#   outputs/yolo26n_custom_v2-run1/  ← intocado
```

---

## 17. Referencia: Arquitectura Actual vs Actualizada

### YAML de referencia: `yolo26.yaml` en Ultralytics 8.4.9

```yaml
nc: 80
end2end: True
reg_max: 1          # En 8.4.9, se ignora → Detect calcula reg_max=ch[0]//16=16

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]
  - [-1, 2, C2PSA, [1024]]

head:
  ... (idéntica)
  - [[16, 19, 22], 1, Detect, [nc]]
```

### YAML esperado: `yolo26.yaml` post-actualización

```yaml
nc: 80
end2end: True
reg_max: 1          # Ahora se RESPETA → Detect usa reg_max=1 → DFL=Identity

backbone:
  ... (posiblemente idéntica, o con mejoras menores)

head:
  ... (posiblemente idéntica)
  - [[16, 19, 22], 1, Detect, [nc]]
```

La diferencia clave estará en cómo `Detect.__init__()` interpreta `reg_max`:

```python
# 8.4.9 (actual):
self.reg_max = ch[0] // 16    # Ignora el YAML → siempre 16

# Actualizado (esperado):
self.reg_max = reg_max         # Respeta el YAML → 1
# ó
self.reg_max = 1               # Hardcoded sin DFL
```

### Comparación de parámetros del Detect head

| Componente | Con DFL | Sin DFL | Δ Params |
|---|---|---|---|
| `cv2[0]` última conv | Conv2d(?, 64, 1) | Conv2d(?, 4, 1) | -60 × in_ch |
| `cv2[1]` última conv | Conv2d(?, 64, 1) | Conv2d(?, 4, 1) | -60 × in_ch |
| `cv2[2]` última conv | Conv2d(?, 64, 1) | Conv2d(?, 4, 1) | -60 × in_ch |
| `dfl` | Conv2d(16, 1, 1) = 16 params | Identity = 0 params | -16 |
| **Total estimado** | ~2,572,280 | ~2,556,000 (est.) | **~-16K** |

---

> **Documento creado**: 2025-02-24  
> **Autor**: GitHub Copilot  
> **Contexto**: TFM UNIR — Detección de Objetos para ESP32-S3  
> **Versión actual del modelo**: YOLO26n T2 (Ultralytics 8.4.9, con DFL, mAP@50=0.7747)  
> **Objetivo**: YOLO26n T3 (Ultralytics actualizada, sin DFL)
