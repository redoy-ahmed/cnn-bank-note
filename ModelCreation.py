import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Turn off OneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Directory paths (replace with your actual paths)
train_dir = 'C:/Users/redoy/Downloads/Bank Notes Real and Fake/train'
test_dir = 'C:/Users/redoy/Downloads/Bank Notes Real and Fake/test'

# Image data generator for training data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% of the data will be used for validation
)

# Image data generator for test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Modify the target size for the generators
target_size = (256, 256)

# Load and split the training dataset into training and validation sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='binary',
    subset='training'  # Use 80% of the data for training
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Use 20% of the data for validation
)

# Create the test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='binary'
)

# Adjust the input shape of the first layer in your model
input_shape = target_size + (3,)  # Add the color channels

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Save the model
model.save("currency_note_classifier.h5")
