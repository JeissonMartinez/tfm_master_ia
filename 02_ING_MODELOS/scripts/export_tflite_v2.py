#!/usr/bin/env python3
"""
Script para convertir modelo Keras a TFLite usando diferentes estrategias.
Intenta múltiples métodos para máxima compatibilidad.
"""

import os
import sys
from pathlib import Path

# Suprimir todos los warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2

print("="*60)
print("📦 EXPORTACIÓN TFLite PARA ESP32-S3")
print("="*60)

# Configuración
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CALIBRATION_DIR = PROJECT_DIR.parent / "01_ING_DATOS/Dataset/train/augmented2_images"
NUM_SAMPLES = 50
IMG_SIZE = 224

# Obtener nombre del modelo desde argumentos o buscar el más reciente
if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]
else:
    # Buscar el modelo .keras más reciente en final_export
    export_dir = PROJECT_DIR / "models/final_export"
    keras_models = sorted(
        export_dir.glob("*_final.keras"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if keras_models:
        MODEL_NAME = keras_models[0].stem.replace("_final", "")
        print(f"\n🔍 Auto-detectado modelo más reciente: {MODEL_NAME}")
    else:
        print("❌ No se encontraron modelos .keras en final_export")
        sys.exit(1)

MODEL_PATH = PROJECT_DIR / "models/final_export" / f"{MODEL_NAME}_final.keras"
OUTPUT_PATH = PROJECT_DIR / "models/final_export" / f"{MODEL_NAME}_int8.tflite"

print(f"\n📁 Modelo entrada: {MODEL_PATH}")
print(f"📁 Modelo salida:  {OUTPUT_PATH}")

if not MODEL_PATH.exists():
    print(f"\n❌ Error: No se encontró el modelo")
    sys.exit(1)

# Importar TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Cargar imágenes de calibración
print(f"\n📷 Cargando imágenes de calibración...")
image_files = list(CALIBRATION_DIR.glob("*.jpg"))[:NUM_SAMPLES]
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
print(f"   Imágenes: {calibration_images.shape}")

# Función de calibración
def representative_dataset():
    for i in range(len(calibration_images)):
        yield [calibration_images[i:i+1].astype(np.float32)]

# Cargar modelo
print(f"\n📂 Cargando modelo...")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
print(f"✅ Modelo cargado: {model.name}")

# ============================================================
# ESTRATEGIA 1: Concrete function
# ============================================================
print(f"\n🔧 Estrategia 1: Usando Concrete Function...")

try:
    # Crear una función concreta con firma de entrada
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)])
    def serving_fn(x):
        return model(x, training=False)
    
    concrete_func = serving_fn.get_concrete_function()
    
    # Convertir desde concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    tflite_model = converter.convert()
    print(f"   ✅ Conversión exitosa")
    
except Exception as e:
    print(f"   ❌ Falló: {e}")
    
    # ============================================================
    # ESTRATEGIA 2: Sin representative dataset
    # ============================================================
    print(f"\n🔧 Estrategia 2: Cuantización dinámica simple...")
    
    try:
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)])
        def serving_fn2(x):
            return model(x, training=False)
        
        concrete_func = serving_fn2.get_concrete_function()
        
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        tflite_model = converter.convert()
        print(f"   ✅ Conversión exitosa")
        
    except Exception as e2:
        print(f"   ❌ Falló: {e2}")
        
        # ============================================================
        # ESTRATEGIA 3: Float32 sin optimización
        # ============================================================
        print(f"\n🔧 Estrategia 3: Float32 sin cuantización...")
        
        try:
            @tf.function(input_signature=[tf.TensorSpec(shape=[1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)])
            def serving_fn3(x):
                return model(x, training=False)
            
            concrete_func = serving_fn3.get_concrete_function()
            
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            # Sin optimizaciones
            
            tflite_model = converter.convert()
            print(f"   ✅ Conversión exitosa (Float32)")
            
        except Exception as e3:
            print(f"\n❌ Todas las estrategias fallaron")
            print(f"   Error 1: {e}")
            print(f"   Error 2: {e2}")
            print(f"   Error 3: {e3}")
            sys.exit(1)

# Guardar modelo
print(f"\n💾 Guardando modelo TFLite...")
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

# Reportar tamaño
size_kb = os.path.getsize(OUTPUT_PATH) / 1024
size_mb = size_kb / 1024
print(f"\n✅ Modelo guardado: {OUTPUT_PATH}")
print(f"   Tamaño: {size_kb:.1f} KB ({size_mb:.2f} MB)")

if size_kb < 1024:
    print(f"\n🎉 ¡El modelo CABE en ESP32-S3!")
else:
    print(f"\n⚠️ El modelo podría ser muy grande para ESP32-S3")

# Verificar
print(f"\n🔍 Verificando modelo...")
interpreter = tf.lite.Interpreter(model_path=str(OUTPUT_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"   Input: {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
print(f"   Outputs: {len(output_details)}")

test_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
print(f"   ✅ Inferencia OK")

print("\n" + "="*60)
print("✅ EXPORTACIÓN COMPLETADA")
print("="*60)
