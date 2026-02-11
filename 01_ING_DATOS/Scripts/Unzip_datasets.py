"""
Script para descomprimir archivos .zip de datasets
Descomprime todos los archivos .zip en carpetas con el mismo nombre
"""

import zipfile
import os
from pathlib import Path


def unzip_datasets(source_dir=None):
    """
    Descomprime todos los archivos .zip en el directorio especificado.
    
    Args:
        source_dir: Directorio donde se encuentran los archivos .zip.
                   Si es None, usa el directorio actual del script.
    """
    # Si no se especifica directorio, usar el directorio del script
    if source_dir is None:
        source_dir = Path(__file__).parent
    else:
        source_dir = Path(source_dir)
    
    # Buscar todos los archivos .zip
    zip_files = list(source_dir.glob("*.zip"))
    
    if not zip_files:
        print(f"No se encontraron archivos .zip en {source_dir}")
        return
    
    print(f"Se encontraron {len(zip_files)} archivos .zip")
    print("-" * 50)
    
    # Descomprimir cada archivo
    for zip_path in zip_files:
        # Crear nombre de carpeta de destino (sin la extensión .zip)
        output_dir = source_dir / zip_path.stem
        
        print(f"\nDescomprimiendo: {zip_path.name}")
        print(f"Destino: {output_dir.name}/")
        
        try:
            # Crear carpeta si no existe
            output_dir.mkdir(exist_ok=True)
            
            # Descomprimir
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Obtener número de archivos extraídos
            num_files = sum(1 for _ in output_dir.rglob('*') if _.is_file())
            print(f"✓ Completado - {num_files} archivos extraídos")
            
        except zipfile.BadZipFile:
            print(f"✗ Error: {zip_path.name} no es un archivo .zip válido")
        except PermissionError:
            print(f"✗ Error: Sin permisos para escribir en {output_dir}")
        except Exception as e:
            print(f"✗ Error inesperado: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Proceso completado")


if __name__ == "__main__":
    # Ejecutar la descompresión
    unzip_datasets()
