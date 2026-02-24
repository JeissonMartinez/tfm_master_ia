#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# TFM TinyML Detector — Flash modelos ESPDL a particiones independientes
#
# Uso:
#   bash scripts/flash_models.sh [PUERTO]
#
# Si no se indica PUERTO, se intenta autodetectar.
# Los offsets coinciden con partitions.csv (Ciclo 3):
#   model_espdet → 0xA10000  (1 MB)
#   model_yolo26 → 0xB10000  (3 MB)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Modelos
ESPDET_FILE="$PROJECT_DIR/models/espdl/espdet_pico_t4.espdl"
YOLO26_FILE="$PROJECT_DIR/models/espdl/yolo26n_t2_esp.espdl"

# Offsets (deben coincidir con partitions.csv)
ESPDET_OFFSET=0xA10000
YOLO26_OFFSET=0xB10000

# Puerto serial
if [[ $# -ge 1 ]]; then
    PORT="$1"
else
    # Autodetectar en macOS / Linux
    PORT=$(ls /dev/tty.usbmodem* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -1 || true)
    if [[ -z "$PORT" ]]; then
        echo "❌ No se detectó puerto serial. Uso: $0 /dev/ttyXXX"
        exit 1
    fi
    echo "📡 Puerto autodetectado: $PORT"
fi

# Validar archivos
for f in "$ESPDET_FILE" "$YOLO26_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "❌ Archivo no encontrado: $f"
        exit 1
    fi
done

echo "═══════════════════════════════════════════════════"
echo " Flasheando modelos ESPDL a ESP32-S3"
echo " Puerto: $PORT"
echo "═══════════════════════════════════════════════════"

# Tamaños
ESPDET_SIZE=$(stat -f%z "$ESPDET_FILE" 2>/dev/null || stat -c%s "$ESPDET_FILE")
YOLO26_SIZE=$(stat -f%z "$YOLO26_FILE" 2>/dev/null || stat -c%s "$YOLO26_FILE")
echo "  ESPDet Pico T4: $(( ESPDET_SIZE / 1024 )) KB → offset $ESPDET_OFFSET"
echo "  YOLO26n T2 ESP: $(( YOLO26_SIZE / 1024 )) KB → offset $YOLO26_OFFSET"
echo ""

# Flash ambos modelos en un solo comando (más rápido)
python -m esptool --chip esp32s3 --port "$PORT" --baud 921600 \
    write_flash --flash_mode dio --flash_size 16MB \
    "$ESPDET_OFFSET" "$ESPDET_FILE" \
    "$YOLO26_OFFSET" "$YOLO26_FILE"

echo ""
echo "✅ Modelos flasheados correctamente"
echo "   Ejecuta 'idf.py -p $PORT monitor' para verificar la carga"
