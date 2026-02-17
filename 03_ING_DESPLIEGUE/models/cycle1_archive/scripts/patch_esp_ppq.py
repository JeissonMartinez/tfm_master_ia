"""
patch_esp_ppq.py
================
Corrige un bug en esp-ppq (v1.2.4) donde el handler de Concat no normaliza
ejes negativos (axis=-1) durante la conversión de layout NCHW→NHWC al exportar
a ESPDL, causando: ValueError: -1 is not in list.

Uso:
    python models/patch_esp_ppq.py

Solo necesita ejecutarse una vez por instalación de esp-ppq.
Si se reinstala o actualiza el paquete, volver a ejecutar.
"""

import importlib
import os
import sys

def find_layout_patterns():
    """Localiza el archivo layout_patterns.py de esp-ppq."""
    try:
        import esp_ppq
    except ImportError:
        print("[ERROR] esp-ppq no está instalado.")
        sys.exit(1)

    pkg_dir = os.path.dirname(esp_ppq.__file__)
    target = os.path.join(pkg_dir, "parser", "espdl", "layout_patterns.py")
    if not os.path.isfile(target):
        print(f"[ERROR] No se encontró: {target}")
        sys.exit(1)
    return target


def patch_file(filepath):
    """Aplica el parche de normalización de eje en ResetConcatPattern."""
    with open(filepath, "r") as f:
        content = f.read()

    # Línea buggy (sin normalización de eje negativo en Concat)
    old_code = (
        '                    axis = op.attributes["axis"]\n'
        '                    new_axis = var_perm.index(int(axis))'
    )

    # Línea corregida (normaliza eje negativo igual que el handler de Softmax)
    new_code = (
        '                    axis = (int(op.attributes["axis"]) + len(var_perm)) % len(var_perm)\n'
        '                    new_axis = var_perm.index(axis)'
    )

    if new_code in content:
        print("[OK] El parche ya está aplicado.")
        return False

    if old_code not in content:
        print("[WARN] No se encontró el código a parchear.")
        print("       Puede que la versión de esp-ppq sea diferente a 1.2.4.")
        return False

    content = content.replace(old_code, new_code, 1)
    with open(filepath, "w") as f:
        f.write(content)

    print("[OK] Parche aplicado correctamente.")
    return True


if __name__ == "__main__":
    filepath = find_layout_patterns()
    print(f"Archivo: {filepath}")
    patch_file(filepath)
