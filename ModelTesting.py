import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Define directories
train_dir = 'path/to/train_data'
validation_dir = 'path/to/validation_data'
test_dir = 'path/to/test_data'

# Load the saved model
model = tf.keras.models.load_model('fake_currency_detector.h5')

# Test a single image
single_image_path = ('C:/Users/redoy/Downloads/Bank Notes Real and '
                     'Fake/test/fake/note_483_3_e61e4641527b462b919c40f656d7603a.jpg')
single_image = tf.keras.preprocessing.image.load_img(single_image_path, target_size=(150, 150))
single_image = tf.keras.preprocessing.image.img_to_array(single_image)
single_image = np.expand_dims(single_image, axis=0) / 255.0

# Predict the class of the single image
prediction = model.predict(single_image)
predicted_class = (prediction[0][0] > 0.5).astype(int)
class_names = ['Fake', 'Real']

# Print and display the prediction
print(f'Predicted class: {class_names[predicted_class]}')

plt.imshow(single_image[0])
plt.title(f'Predicted Class: {class_names[predicted_class]}')
plt.show()
