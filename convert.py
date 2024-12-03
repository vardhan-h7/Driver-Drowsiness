import tensorflow as tf
from keras.models import load_model

# Load the Keras model from the .h5 file
model_path = 'models/cnnCat2.h5'  # Replace with the path to your .h5 file
keras_model = load_model(model_path)

# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the TFLite model to a file
tflite_model_path = 'models/cnnCat2.tflite'  # Replace with the desired path for the .tflite file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TFLite model saved to: {tflite_model_path}')
