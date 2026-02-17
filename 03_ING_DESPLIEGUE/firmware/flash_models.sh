#!/usr/bin/env bash
# =============================================================================
# flash_models.sh — Construir imagen compuesta y flashear modelos a partición
#
# Uso:
#   ./flash_models.sh [--port /dev/ttyUSB0]
#
# Los 3 modelos .espdl se concatenan (con padding a 4 KB) en una imagen
# binaria que se escribe a la partición "models" de la ESP32-S3.
#
# Offsets (deben coincidir con app_config.h):
#   MBNTv3S @ 0x000000  (681,088 bytes)
#   YOLO11n @ 0x0A7000  (2,800,272 bytes) — INT8 percentile calib
#   YOLO26n @ 0x353000  (2,639,168 bytes) — re-quantized sin detect head
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/../models"
BUILD_DIR="${SCRIPT_DIR}/build"

# Modelos fuente
MODEL_MBNTV3S="${MODELS_DIR}/MBNTv3S_ssdlite_v1_p2_best.espdl"
MODEL_YOLO11N="${MODELS_DIR}/yolo11n_v1_best.espdl"
MODEL_YOLO26N="${MODELS_DIR}/yolo26n_v1_best.espdl"

# Offsets dentro de la partición (deben coincidir con app_config.h)
OFFSET_MBNTV3S=0
OFFSET_YOLO11N=$((0x0A7000))   # 684032
OFFSET_YOLO26N=$((0x353000))   # 3485696

# Tamaño total de la partición (debe coincidir con partitions.csv)
PARTITION_SIZE=$((0x700000))    # 7 MB

# Puerto serial
PORT="${1:-/dev/ttyUSB0}"
if [[ "${1:-}" == "--port" ]]; then
    PORT="${2:-/dev/ttyUSB0}"
fi

# Imagen compuesta de salida
COMPOSITE="${BUILD_DIR}/models_partition.bin"

echo "=== TFM TinyML — Flash Models ==="
echo ""

# Verificar que los modelos existen
for model in "$MODEL_MBNTV3S" "$MODEL_YOLO11N" "$MODEL_YOLO26N"; do
    if [[ ! -f "$model" ]]; then
        echo "ERROR: No se encuentra: $model"
        exit 1
    fi
    echo "  Found: $(basename "$model") ($(wc -c < "$model") bytes)"
done

# Verificar que IDF_PATH está configurado
if [[ -z "${IDF_PATH:-}" ]]; then
    echo "ERROR: IDF_PATH no está definido. Ejecuta: source \$IDF_PATH/export.sh"
    exit 1
fi

PARTTOOL="python ${IDF_PATH}/components/partition_table/parttool.py"

# Crear directorio build si no existe
mkdir -p "$BUILD_DIR"

echo ""
echo "--- Construyendo imagen compuesta ---"

# 1. Crear archivo vacío del tamaño de la partición (relleno con 0xFF como flash)
python3 -c "
import sys
size = $PARTITION_SIZE
with open('$COMPOSITE', 'wb') as f:
    f.write(b'\xff' * size)
print(f'Created {size} byte partition image (0xFF filled)')
"

# 2. Escribir cada modelo en su offset
python3 -c "
import sys

models = [
    ('MBNTv3S', '$MODEL_MBNTV3S', $OFFSET_MBNTV3S),
    ('YOLO11n', '$MODEL_YOLO11N', $OFFSET_YOLO11N),
    ('YOLO26n', '$MODEL_YOLO26N', $OFFSET_YOLO26N),
]

with open('$COMPOSITE', 'r+b') as out:
    for name, path, offset in models:
        with open(path, 'rb') as src:
            data = src.read()
        out.seek(offset)
        out.write(data)
        end = offset + len(data)
        print(f'  {name}: offset=0x{offset:06X}, size={len(data):,} bytes, end=0x{end:06X}')

print('Composite image ready:', '$COMPOSITE')
"

echo ""
echo "--- Flasheando a partición 'models' ---"
echo "Puerto: $PORT"
echo ""

# 3. Flashear usando parttool.py
${PARTTOOL} \
    --port "$PORT" \
    --baud 921600 \
    write_partition \
    --partition-name models \
    --input "$COMPOSITE"

echo ""
echo "=== Modelos flasheados correctamente ==="
echo ""
echo "Offsets en la partición 'models':"
echo "  MBNTv3S @ 0x000000"
echo "  YOLO11n @ 0x0A7000"
echo "  YOLO26n @ 0x354000"
echo ""
echo "Estos offsets deben coincidir con firmware/main/app_config.h"
