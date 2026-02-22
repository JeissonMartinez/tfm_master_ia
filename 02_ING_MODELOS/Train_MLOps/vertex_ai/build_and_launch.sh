#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# build_and_launch.sh — Empaqueta, sube y lanza un Custom Job
# ─────────────────────────────────────────────────────────────
#
# Ciclo 2 — PyTorch only (FCOS, YOLO26_CUSTOM, ESPDet, EXPORT)
#
# Uso:
#   ./vertex_ai/build_and_launch.sh fcos_v3s_v1
#   ./vertex_ai/build_and_launch.sh yolo26n_custom_v1 --dry-run
#   ./vertex_ai/build_and_launch.sh espdet_pico_v1 --run-name mi-exp
#   ./vertex_ai/build_and_launch.sh export_fcos_v1
#
# Prerequisitos:
#   - gcloud auth application-default login
#   - pip install google-cloud-aiplatform google-cloud-storage pyyaml
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuración ────────────────────────────────────────────
BUCKET="gs://project-18f58341-12cf-47bc-861-tfm-data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"   # Train_MLOps/

# ── Validar argumento ────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Uso: $0 <nombre_config> [--dry-run] [--run-name <name>]"
    echo ""
    echo "Configs disponibles:"
    ls -1 "$SCRIPT_DIR/configs/"*.yaml 2>/dev/null | xargs -I{} basename {} .yaml
    exit 1
fi

CONFIG_NAME="$1"
shift  # Los args restantes se pasan a launch_job.py
CONFIG_FILE="$SCRIPT_DIR/configs/${CONFIG_NAME}.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config no encontrado: $CONFIG_FILE"
    echo ""
    echo "Configs disponibles:"
    ls -1 "$SCRIPT_DIR/configs/"*.yaml 2>/dev/null | xargs -I{} basename {} .yaml
    exit 1
fi

# ── 1. Empaquetar (sdist) ───────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "📦 Paso 1/3 — Empaquetando código fuente"
echo "═══════════════════════════════════════════════════════════"

cd "$PROJECT_DIR"

# Limpiar builds anteriores
rm -rf dist/ build/ *.egg-info

python setup.py sdist --formats=gztar 2>&1 | tail -3

SDIST=$(ls dist/tfm_trainer-*.tar.gz 2>/dev/null | head -1)
if [[ -z "$SDIST" ]]; then
    echo "❌ Error: no se generó el sdist"
    exit 1
fi
echo "  ✅ Paquete: $SDIST"

# ── 2. Subir paquete a GCS ──────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "☁️  Paso 2/3 — Subiendo paquete a GCS"
echo "═══════════════════════════════════════════════════════════"

GCS_PACKAGE="${BUCKET}/packages/$(basename "$SDIST")"
gsutil cp "$SDIST" "$GCS_PACKAGE"
echo "  ✅ Subido: $GCS_PACKAGE"

# ── 3. Lanzar Custom Job ────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🚀 Paso 3/3 — Lanzando Custom Job en Vertex AI"
echo "═══════════════════════════════════════════════════════════"

python "$SCRIPT_DIR/launch_job.py" \
    --config "$CONFIG_FILE" \
    --package-uri "$GCS_PACKAGE" \
    "$@"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Proceso completado"
echo "═══════════════════════════════════════════════════════════"
