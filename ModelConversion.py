import tensorflow as tf

try:

    model = tf.keras.models.load_model('fake_currency_detector.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("fake_currency_detector.tflite", "wb").write(tflite_model)

    print("TensorFlow Lite model saved successfully.")

except FileNotFoundError:
    print("Error: Model file not found. Please provide the correct path to the model file.")
except AttributeError as e:
    print("Error:", e)
except Exception as e:
    print("An error occurred during model conversion:", str(e))
