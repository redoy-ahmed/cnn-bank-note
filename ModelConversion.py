import tensorflow as tf

# Convert the saved model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("currency_note_classifier.h5")
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("currency_note_classifier.tflite", "wb") as f:
    f.write(tflite_model)
