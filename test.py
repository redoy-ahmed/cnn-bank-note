import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

import logging
tf.get_logger().setLevel(logging.DEBUG)

# Enable TensorFlow Debugger
tf.debugging.set_log_device_placement(True)

# Define a simple model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Save the model in the native Keras format
model.save('model.keras')

# Load the model without the optimizer state
model = tf.keras.models.load_model('model.keras', compile=False)

# Recompile the model with the desired optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Test the model to ensure it's loaded correctly
print("Model loaded and compiled successfully.")

# Convert the simple model to TensorFlow Lite format
simple_converter = tf.lite.TFLiteConverter.from_keras_model(model)
simple_tflite_model = simple_converter.convert()

# Save the converted model
with open('simple_model.tflite', 'wb') as f:
    f.write(simple_tflite_model)

print("Simple model converted to TensorFlow Lite successfully.")
