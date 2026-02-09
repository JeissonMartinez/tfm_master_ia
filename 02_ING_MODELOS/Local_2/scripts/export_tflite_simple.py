#!/usr/bin/env python3
"""
Script simplificado para exportar modelo Keras a TFLite para ESP32-S3.

Uso:
    python scripts/export_tflite_simple.py
"""

import os
import sys
from pathlib import Path

# Configurar paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# Suprimir warnings de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2

# Importar TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

print("="*60)
print("📦 EXPORTACIÓN TFLite PARA ESP32-S3")
print("="*60)

# Configuración
MODEL_NAME = "mobilenetv3_ssdlite_v1"
MODEL_PATH = PROJECT_DIR / "models/final_export" / f"{MODEL_NAME}_final.keras"
OUTPUT_PATH = PROJECT_DIR / "models/final_export" / f"{MODEL_NAME}_int8.tflite"
CALIBRATION_DIR = PROJECT_DIR.parent / "01_ING_DATOS/Dataset/train/augmented2_images"
NUM_CALIBRATION_SAMPLES = 50
IMG_SIZE = 224

print(f"\n📁 Modelo entrada: {MODEL_PATH}")
print(f"📁 Modelo salida:  {OUTPUT_PATH}")
print(f"📁 Calibración:    {CALIBRATION_DIR}")

# Verificar que existe el modelo
if not MODEL_PATH.exists():
    print(f"\n❌ Error: No se encontró el modelo: {MODEL_PATH}")
    sys.exit(1)

# Cargar imágenes de calibración
print(f"\n📷 Cargando imágenes de calibración...")
image_files = list(CALIBRATION_DIR.glob("*.jpg"))[:NUM_CALIBRATION_SAMPLES]

calibration_images = []
for img_path in image_files:
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        calibration_images.append(img)
    except:
        continue

calibration_images = np.array(calibration_images, dtype=np.float32)
print(f"   Imágenes cargadas: {calibration_images.shape}")

# Cargar modelo
print(f"\n📂 Cargando modelo...")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
print(f"✅ Modelo cargado: {model.name}")

# Función de calibración
def representative_dataset():
    for i in range(len(calibration_images)):
        sample = calibration_images[i:i+1].astype(np.float32)
        yield [sample]

# Crear conversor directamente desde Keras
print(f"\n🔧 Creando conversor TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Configuración para cuantización dinámica (más compatible)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# IMPORTANTE: Deshabilitar el nuevo conversor MLIR que tiene bugs
converter.experimental_new_converter = False

# Permitir operaciones flexibles para máxima compatibilidad
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Operaciones TFLite estándar
]

print(f"   Convirtiendo modelo...")
print(f"   (Esto puede tomar 1-3 minutos)")

try:
    tflite_model = converter.convert()
    print(f"   ✅ Conversión exitosa")
except Exception as e:
    print(f"\n❌ Error en conversión: {e}")
    
    # Intentar sin dataset representativo
    print(f"\n🔄 Intentando conversión sin cuantización INT8...")
    converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.experimental_new_converter = True
    
    try:
        tflite_model = converter2.convert()
        print(f"   ✅ Conversión exitosa (cuantización dinámica)")
    except Exception as e2:
        print(f"\n❌ Error fatal: {e2}")
        sys.exit(1)

# Guardar modelo
print(f"\n💾 Guardando modelo TFLite...")
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

# Reportar tamaño
size_bytes = os.path.getsize(OUTPUT_PATH)
size_kb = size_bytes / 1024
size_mb = size_kb / 1024

print(f"\n✅ Modelo TFLite guardado: {OUTPUT_PATH}")
print(f"   Tamaño: {size_kb:.1f} KB ({size_mb:.2f} MB)")

if size_kb < 1024:
    print(f"\n🎉 ¡El modelo CABE en ESP32-S3! (límite ~1 MB)")
else:
    print(f"\n⚠️ El modelo podría ser muy grande para ESP32-S3")
    print(f"   Considera usar MobileNetV2 alpha=0.35")

# Verificar modelo
print(f"\n🔍 Verificando modelo TFLite...")
interpreter = tf.lite.Interpreter(model_path=str(OUTPUT_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"   Input: {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
print(f"   Outputs: {len(output_details)}")

# Test de inferencia
test_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(input_details[0]['dtype'])
if input_details[0]['dtype'] == np.uint8:
    test_input = (test_input * 255).astype(np.uint8)
    
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
print(f"   ✅ Inferencia de prueba exitosa")

print("\n" + "="*60)
print("✅ EXPORTACIÓN COMPLETADA")
print("="*60)
