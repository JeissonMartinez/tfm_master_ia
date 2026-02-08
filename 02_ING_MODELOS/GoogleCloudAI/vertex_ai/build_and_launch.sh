#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# build_and_launch.sh — Empaqueta, sube y lanza un Custom Job
# ─────────────────────────────────────────────────────────────
#
# Uso:
#   ./vertex_ai/build_and_launch.sh mobilenet_v3s_ssdlite_v1
#   ./vertex_ai/build_and_launch.sh yolo26n_v1
#   ./vertex_ai/build_and_launch.sh yolo26n_v1 --dry-run
#
# Prerequisitos:
#   - gcloud auth application-default login
#   - pip install google-cloud-aiplatform google-cloud-storage pyyaml
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuración ────────────────────────────────────────────
BUCKET="gs://project-18f58341-12cf-47bc-861-tfm-data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"   # GoogleCloudAI/

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

SDIST=$(ls dist/tfm-trainer-*.tar.gz 2>/dev/null | head -1)
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
    "$@"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Proceso completado"
echo "═══════════════════════════════════════════════════════════"
