#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# download_results.sh — Descarga artefactos de un run desde GCS
# ─────────────────────────────────────────────────────────────
# Uso:
#   bash scripts/download_results.sh <run-name>
#   bash scripts/download_results.sh fcos_v3s_v1-run1
#
# Los artefactos se descargan a: outputs/<run-name>/
# ─────────────────────────────────────────────────────────────

set -euo pipefail

BUCKET="gs://project-18f58341-12cf-47bc-861-tfm-data"
OUTPUT_PREFIX="output"

if [[ $# -lt 1 ]]; then
    echo "Uso: $0 <run-name>"
    echo "Ejemplo: $0 fcos_v3s_v1-run1"
    echo ""
    echo "Runs disponibles:"
    gsutil ls "${BUCKET}/${OUTPUT_PREFIX}/" 2>/dev/null | sed 's|.*/||' | grep -v '^$'
    exit 1
fi

RUN_NAME="$1"
GCS_PATH="${BUCKET}/${OUTPUT_PREFIX}/${RUN_NAME}/"
LOCAL_DIR="outputs/${RUN_NAME}"

echo "═══════════════════════════════════════════════════════════"
echo "  Descargando artefactos: ${RUN_NAME}"
echo "  Desde: ${GCS_PATH}"
echo "  Hacia: ${LOCAL_DIR}/"
echo "═══════════════════════════════════════════════════════════"

# Verificar que el run existe en GCS
if ! gsutil ls "${GCS_PATH}" &>/dev/null; then
    echo "❌ No se encontró el run '${RUN_NAME}' en GCS."
    echo "   Path verificado: ${GCS_PATH}"
    echo ""
    echo "Runs disponibles:"
    gsutil ls "${BUCKET}/${OUTPUT_PREFIX}/" 2>/dev/null | sed 's|.*/||' | grep -v '^$'
    exit 1
fi

# Crear directorio local
mkdir -p "${LOCAL_DIR}"

# Descargar todos los artefactos
echo ""
echo "📥 Descargando..."
gsutil -m cp -r "${GCS_PATH}*" "${LOCAL_DIR}/"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Descarga completada"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📂 Contenido descargado:"
find "${LOCAL_DIR}" -type f | sort | while read -r f; do
    size=$(du -h "$f" | awk '{print $1}')
    echo "   ${size}  $(basename "$f")"
done

echo ""
echo "📊 Resumen:"
n_files=$(find "${LOCAL_DIR}" -type f | wc -l | tr -d ' ')
total_size=$(du -sh "${LOCAL_DIR}" | awk '{print $1}')
echo "   Archivos: ${n_files}"
echo "   Tamaño total: ${total_size}"

# Mostrar métricas del experiment.json si existe
EXP_JSON="${LOCAL_DIR}/experiment.json"
if [[ -f "${EXP_JSON}" ]]; then
    echo ""
    echo "📋 Métricas del experimento:"
    python3 -c "
import json
with open('${EXP_JSON}') as f:
    exp = json.load(f)
print(f\"   Family:      {exp.get('config', {}).get('family', '?')}\")
print(f\"   Status:      {exp.get('status', '?')}\")
print(f\"   Val mAP@50:  {exp.get('val_map50', 0):.4f}\")
print(f\"   Test mAP@50: {exp.get('test_map50', 0):.4f}\")
print(f\"   ONNX size:   {exp.get('onnx_size_mb', 0):.2f} MB\")
print(f\"   Duration:    {exp.get('duration_s', 0)/60:.1f} min\")
" 2>/dev/null || echo "   (no se pudo parsear experiment.json)"
fi
