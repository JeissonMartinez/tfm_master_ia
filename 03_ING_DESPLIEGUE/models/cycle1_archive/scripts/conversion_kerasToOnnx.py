import tensorflow as tf
import tf2onnx

path_model = "/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.keras"

# 1. Cargar el modelo .keras (sin compilar, solo necesitamos arquitectura + pesos para ONNX)
model = tf.keras.models.load_model(path_model, compile=False)

# 2. Validar el shape de entrada del modelo
print(f"Input shape del modelo: {model.input_shape}")
print(f"Output shape del modelo: {model.output_shape}")

# 3. Convertir directamente a ONNX sin pasar por SavedModel intermedio
spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
output_path = "/Users/admin/Documents/TFM_UNIR/03_ING_DESPLIEGUE/models/MBNTv3S_ssdlite_v1_p2_best.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())