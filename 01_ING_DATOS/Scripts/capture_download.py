#!/usr/bin/env python3
"""
capture_download.py — Descarga masiva de fotos desde ESP32-S3 Image Capture
============================================================================

Descarga todas las fotos capturadas en la SD card de la ESP32-S3 a una carpeta
local, organizadas en sesiones por fecha/hora.

Uso:
    # Descargar todas las fotos
    python capture_download.py --ip 192.168.1.X download

    # Descargar y eliminar de la SD después
    python capture_download.py --ip 192.168.1.X download --delete-after

    # Ver estado de la ESP32
    python capture_download.py --ip 192.168.1.X status

    # Eliminar todas las fotos de la SD
    python capture_download.py --ip 192.168.1.X clear

    # Auto-detectar IP via mDNS (requiere zeroconf)
    python capture_download.py download

Destino: 01_ING_DATOS/Datasets/captured_custom/session_YYYYMMDD_HHMMSS/
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Intentar importar tqdm (opcional, fallback a print)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Configuración ───────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # 01_ING_DATOS/
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Datasets" / "captured_custom"
MDNS_HOSTNAME = "esp32-capture.local"
API_TIMEOUT = 30  # segundos por request
PAGE_LIMIT = 50   # fotos por página al listar


def resolve_ip(args_ip: str | None) -> str:
    """Resolver IP de la ESP32: argumento directo, o mDNS."""
    if args_ip:
        return args_ip

    # Intentar mDNS
    print(f"🔍 Buscando ESP32 via mDNS ({MDNS_HOSTNAME})...")
    try:
        import socket
        ip = socket.gethostbyname(MDNS_HOSTNAME)
        print(f"✓ Encontrada en {ip}")
        return ip
    except Exception:
        pass

    # Intentar zeroconf
    try:
        from zeroconf import Zeroconf, ServiceBrowser
        zc = Zeroconf()
        info = zc.get_service_info("_http._tcp.local.", f"ESP32-S3 Image Capture._http._tcp.local.")
        if info:
            ip = ".".join(str(b) for b in info.addresses[0])
            zc.close()
            print(f"✓ Encontrada via zeroconf: {ip}")
            return ip
        zc.close()
    except ImportError:
        pass

    print("✗ No se pudo detectar la ESP32 automáticamente.")
    print("  Usa: --ip <IP_DE_TU_ESP32>")
    print("  O instala zeroconf: pip install zeroconf")
    sys.exit(1)


def api_get(base_url: str, endpoint: str) -> dict | None:
    """GET request a la API de la ESP32."""
    url = f"{base_url}{endpoint}"
    try:
        req = Request(url)
        with urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (URLError, HTTPError, json.JSONDecodeError) as e:
        print(f"✗ Error en {endpoint}: {e}")
        return None


def api_post(base_url: str, endpoint: str) -> dict | None:
    """POST request a la API de la ESP32."""
    url = f"{base_url}{endpoint}"
    try:
        req = Request(url, data=b'', method='POST')
        with urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except (URLError, HTTPError, json.JSONDecodeError) as e:
        print(f"✗ Error en {endpoint}: {e}")
        return None


def download_file(base_url: str, filename: str, dest_path: Path) -> bool:
    """Descargar un archivo individual."""
    url = f"{base_url}/api/photos/{filename}"
    try:
        with urlopen(url, timeout=API_TIMEOUT) as resp:
            data = resp.read()
        dest_path.write_bytes(data)
        return True
    except (URLError, HTTPError) as e:
        print(f"  ✗ Error descargando {filename}: {e}")
        return False


# ── Comandos ────────────────────────────────────────────────────────────────

def cmd_status(base_url: str):
    """Mostrar estado de la ESP32."""
    d = api_get(base_url, "/api/status")
    if not d:
        print("✗ No se pudo conectar a la ESP32.")
        return

    print("\n╔══════════════════════════════════════╗")
    print("║   ESP32-S3 Image Capture — Status    ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  IP:          {d.get('ip', '?'):>20s}  ║")
    print(f"║  WiFi RSSI:   {d.get('wifi_rssi', '?'):>17s} dBm  ║")
    print(f"║  Uptime:      {d.get('uptime_s', 0):>18d}s  ║")
    print(f"║  Photos:      {d.get('photo_count', 0):>20d}  ║")
    print(f"║  SD Total:    {d.get('sd_total_mb', 0):>17d} MB  ║")
    print(f"║  SD Free:     {d.get('sd_free_mb', 0):>17d} MB  ║")
    print(f"║  JPEG native: {str(d.get('jpeg_native', '?')):>20s}  ║")
    print(f"║  PSRAM free:  {d.get('free_psram_kb', 0):>17d} KB  ║")
    print("╚══════════════════════════════════════╝\n")


def cmd_download(base_url: str, delete_after: bool, output_dir: Path):
    """Descargar todas las fotos."""
    # Obtener estado
    status = api_get(base_url, "/api/status")
    if not status:
        print("✗ No se pudo conectar a la ESP32.")
        return

    total = status.get("photo_count", 0)
    if total == 0:
        print("ℹ No hay fotos para descargar.")
        return

    # Crear directorio de sesión
    session = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    dest_dir = output_dir / session
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Descargando {total} fotos → {dest_dir}")
    print(f"   SD: {status.get('sd_free_mb', '?')} MB libre")

    # Recopilar todos los nombres de archivo
    all_files = []
    page = 1
    while True:
        data = api_get(base_url, f"/api/photos?page={page}&limit={PAGE_LIMIT}")
        if not data or not data.get("files"):
            break
        all_files.extend(data["files"])
        if len(data["files"]) < PAGE_LIMIT:
            break
        page += 1

    if not all_files:
        print("✗ No se pudieron listar los archivos.")
        return

    print(f"   Encontrados: {len(all_files)} archivos\n")

    # Descargar
    t0 = time.time()
    downloaded = 0
    total_bytes = 0

    if HAS_TQDM:
        iterator = tqdm(all_files, desc="Downloading", unit="img")
    else:
        iterator = all_files

    for f in iterator:
        name = f["name"]
        dest_path = dest_dir / name

        if download_file(base_url, name, dest_path):
            downloaded += 1
            total_bytes += dest_path.stat().st_size
        else:
            if not HAS_TQDM:
                print(f"  [{downloaded}/{len(all_files)}] ✗ {name}")

        if not HAS_TQDM and downloaded % 10 == 0:
            print(f"  [{downloaded}/{len(all_files)}] descargadas...")

    elapsed = time.time() - t0
    mb = total_bytes / (1024 * 1024)

    print(f"\n✓ Descargadas {downloaded}/{len(all_files)} fotos")
    print(f"  Tamaño: {mb:.1f} MB en {elapsed:.1f}s ({mb/max(elapsed,0.1):.1f} MB/s)")
    print(f"  Destino: {dest_dir}")

    # Eliminar de la SD si se pidió
    if delete_after and downloaded == len(all_files):
        print("\n🗑 Eliminando fotos de la SD card...")
        result = api_post(base_url, "/api/delete_all")
        if result and result.get("ok"):
            print("  ✓ SD card limpiada")
        else:
            print("  ✗ Error al limpiar SD card")
    elif delete_after:
        print("\n⚠ No se eliminaron fotos (descarga incompleta)")


def cmd_clear(base_url: str):
    """Eliminar todas las fotos de la SD."""
    status = api_get(base_url, "/api/status")
    if not status:
        print("✗ No se pudo conectar a la ESP32.")
        return

    total = status.get("photo_count", 0)
    if total == 0:
        print("ℹ No hay fotos que eliminar.")
        return

    confirm = input(f"¿Eliminar {total} fotos de la SD card? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelado.")
        return

    result = api_post(base_url, "/api/delete_all")
    if result and result.get("ok"):
        print(f"✓ {total} fotos eliminadas de la SD card")
    else:
        print("✗ Error al eliminar fotos")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Descarga masiva de fotos desde ESP32-S3 Image Capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s --ip 192.168.1.100 status
  %(prog)s --ip 192.168.1.100 download
  %(prog)s --ip 192.168.1.100 download --delete-after
  %(prog)s download                     # auto-detect via mDNS
  %(prog)s --ip 192.168.1.100 clear
        """
    )
    parser.add_argument("--ip", help="IP de la ESP32 (o auto-detect via mDNS)")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Mostrar estado de la ESP32")

    dl = sub.add_parser("download", help="Descargar todas las fotos")
    dl.add_argument("--delete-after", action="store_true",
                    help="Eliminar fotos de la SD después de descargar")

    sub.add_parser("clear", help="Eliminar todas las fotos de la SD")

    args = parser.parse_args()

    # Resolver IP
    ip = resolve_ip(args.ip)
    base_url = f"http://{ip}"

    if args.command == "status":
        cmd_status(base_url)
    elif args.command == "download":
        cmd_download(base_url, args.delete_after, args.output)
    elif args.command == "clear":
        cmd_clear(base_url)


if __name__ == "__main__":
    main()
