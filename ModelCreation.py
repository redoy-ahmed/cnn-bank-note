import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Directory paths (replace with your actual paths)
train_dir = 'C:/Users/redoy/Downloads/Bank Notes Real and Fake/train'
test_dir = 'C:/Users/redoy/Downloads/Bank Notes Real and Fake/test'

# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 64  # Updated batch size
epochs = 5  # Reduced number of epochs

# Data preprocessing and augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Data preprocessing for testing set
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Create image generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Convert generators to TensorFlow datasets and repeat them infinitely
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, img_width, img_height, 3], [None])
).repeat()

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, img_width, img_height, 3], [None])
).repeat()

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the updated batch size and epochs
history = model.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_dataset,
    validation_steps=test_generator.samples // batch_size
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the model
model.save("fake_currency_detector.h5")
