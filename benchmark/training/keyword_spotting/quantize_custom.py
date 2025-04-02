import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('trained_models/kws_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('trained_models/kws_model.tflite', 'wb') as f:
    f.write(tflite_model)

# # Quantization using full integer quantization
# def representative_data_gen():
#     # Provide representative data for calibration
#     for _ in range(100):
#         # Load your sample data and preprocess it
#         yield [sample_data]  # Sample data should match the input shape

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Provide a representative dataset for calibration
# converter.representative_dataset = representative_data_gen

# Set the input and output tensors to be 8-bit integer
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert the model
tflite_quantized_model = converter.convert()

# Save the quantized model
with open('trained_models/kws_model_quant.tflite', 'wb') as f:
    f.write(tflite_quantized_model)
