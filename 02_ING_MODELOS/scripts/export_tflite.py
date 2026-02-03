#!/usr/bin/env python3
"""
Script para exportar modelo Keras a TFLite INT8 para ESP32-S3.

Este script se ejecuta en un proceso separado para evitar problemas de memoria
después del entrenamiento del modelo.

Uso:
    python scripts/export_tflite.py --model models/final_export/mobilenetv3_ssdlite_v1_final.keras
    
    # O con todos los parámetros:
    python scripts/export_tflite.py \
        --model models/final_export/mobilenetv3_ssdlite_v1_final.keras \
        --output models/final_export/mobilenetv3_ssdlite_v1_int8.tflite \
        --calibration-dir ../01_ING_DATOS/Dataset/train/augmented2_images \
        --num-samples 100
"""

import os
import sys
import argparse
import gc
from pathlib import Path

# Suprimir warnings de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf


def load_calibration_images(image_dir: str, num_samples: int = 100, img_size: int = 224) -> np.ndarray:
    """Cargar imágenes de calibración desde un directorio."""
    import cv2
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if len(image_files) == 0:
        raise ValueError(f"No se encontraron imágenes en {image_dir}")
    
    # Limitar al número de samples
    image_files = image_files[:num_samples]
    
    images = []
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0
            images.append(img)
        except Exception as e:
            print(f"   ⚠️ Error cargando {img_path.name}: {e}")
            continue
        
        if len(images) >= num_samples:
            break
    
    return np.array(images, dtype=np.float32)


def export_tflite_int8(
    model_path: str,
    output_path: str,
    calibration_images: np.ndarray,
) -> str:
    """Exportar modelo a TFLite INT8."""
    
    print(f"\n📂 Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✅ Modelo cargado: {model.name}")
    
    # Crear función de calibración
    def representative_dataset():
        for i in range(len(calibration_images)):
            sample = calibration_images[i:i+1].astype(np.float32)
            yield [sample]
    
    print(f"\n🔧 Iniciando conversión a TFLite INT8...")
    print(f"   Imágenes de calibración: {len(calibration_images)}")
    
    # ESTRATEGIA 1: Primero intentar via SavedModel (más estable)
    import tempfile
    import shutil
    
    saved_model_dir = tempfile.mkdtemp()
    try:
        print("   Paso 1: Guardando como SavedModel...")
        model.export(saved_model_dir, format='tf_saved_model')
        
        print("   Paso 2: Creando conversor desde SavedModel...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    except Exception as e:
        print(f"   ⚠️ SavedModel falló ({e}), intentando desde Keras...")
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Configurar para INT8 con opciones de compatibilidad
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Deshabilitar el nuevo conversor MLIR que causa problemas
    converter.experimental_new_converter = False
    
    # Permitir operaciones híbridas (más compatible)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Operaciones estándar
    ]
    
    # Intentar conversión con diferentes configuraciones
    tflite_model = None
    
    # Intento 1: Cuantización dinámica (más simple)
    print("   Intento 1: Cuantización dinámica...")
    try:
        tflite_model = converter.convert()
        print("   ✅ Conversión exitosa (dinámica)")
    except Exception as e1:
        print(f"   ❌ Falló: {e1}")
        
        # Intento 2: Con nuevo conversor habilitado
        print("   Intento 2: Con nuevo conversor MLIR...")
        try:
            converter.experimental_new_converter = True
            converter.experimental_new_quantizer = True
            tflite_model = converter.convert()
            print("   ✅ Conversión exitosa (MLIR)")
        except Exception as e2:
            print(f"   ❌ Falló: {e2}")
            
            # Intento 3: INT8 con fallback a float
            print("   Intento 3: INT8 con fallback a operaciones float...")
            try:
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.float32
                tflite_model = converter.convert()
                print("   ✅ Conversión exitosa (INT8 híbrido)")
            except Exception as e3:
                print(f"   ❌ Falló: {e3}")
                raise RuntimeError(f"No se pudo convertir el modelo. Errores: {e1}, {e2}, {e3}")
    
    # Limpiar SavedModel temporal
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    
    # Guardar modelo
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    # Reportar tamaño
    size_bytes = os.path.getsize(output_path)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    
    print(f"\n✅ Modelo TFLite guardado: {output_path}")
    print(f"   Tamaño: {size_kb:.1f} KB ({size_mb:.2f} MB)")
    
    if size_kb < 1024:
        print(f"   ✅ El modelo CABE en ESP32-S3 (límite ~1 MB)")
    else:
        print(f"   ⚠️ El modelo podría ser muy grande para ESP32-S3")
    
    return output_path


def verify_tflite_model(tflite_path: str, input_shape=(1, 224, 224, 3)):
    """Verificar que el modelo TFLite funciona correctamente."""
    print(f"\n🔍 Verificando modelo TFLite...")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input: {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"   Outputs: {len(output_details)}")
    for i, out in enumerate(output_details):
        print(f"      [{i}] {out['name']}: {out['shape']} dtype={out['dtype']}")
    
    # Test de inferencia
    input_data = np.random.randint(0, 255, size=input_shape).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    print(f"   ✅ Inferencia de prueba exitosa")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Exportar modelo Keras a TFLite INT8")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Ruta al modelo Keras (.keras)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Ruta de salida para el modelo TFLite (por defecto: mismo nombre con _int8.tflite)"
    )
    parser.add_argument(
        "--calibration-dir", "-c",
        type=str,
        default=None,
        help="Directorio con imágenes de calibración"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=100,
        help="Número de imágenes de calibración (default: 100)"
    )
    parser.add_argument(
        "--img-size", "-s",
        type=int,
        default=224,
        help="Tamaño de imagen (default: 224)"
    )
    
    args = parser.parse_args()
    
    # Determinar rutas
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: No se encontró el modelo: {model_path}")
        sys.exit(1)
    
    if args.output:
        output_path = args.output
    else:
        output_path = str(model_path.parent / f"{model_path.stem.replace('_final', '')}_int8.tflite")
    
    # Directorio de calibración
    if args.calibration_dir:
        calibration_dir = args.calibration_dir
    else:
        # Por defecto, usar el directorio de train
        calibration_dir = model_path.parent.parent.parent.parent / "01_ING_DATOS/Dataset/train/augmented2_images"
    
    print("="*60)
    print("📦 EXPORTACIÓN TFLite INT8 PARA ESP32-S3")
    print("="*60)
    print(f"\n📁 Modelo entrada: {model_path}")
    print(f"📁 Modelo salida:  {output_path}")
    print(f"📁 Calibración:    {calibration_dir}")
    print(f"📊 Samples:        {args.num_samples}")
    
    # Cargar imágenes de calibración
    print(f"\n📷 Cargando imágenes de calibración...")
    calibration_images = load_calibration_images(
        calibration_dir,
        num_samples=args.num_samples,
        img_size=args.img_size,
    )
    print(f"   Imágenes cargadas: {calibration_images.shape}")
    print(f"   Memoria usada: {calibration_images.nbytes / 1024 / 1024:.1f} MB")
    
    # Exportar
    try:
        tflite_path = export_tflite_int8(
            model_path=str(model_path),
            output_path=output_path,
            calibration_images=calibration_images,
        )
        
        # Verificar
        verify_tflite_model(tflite_path)
        
        print("\n" + "="*60)
        print("✅ EXPORTACIÓN COMPLETADA EXITOSAMENTE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error durante exportación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
