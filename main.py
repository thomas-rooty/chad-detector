import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, losses
from pathlib import Path
import os.path
import itertools

# Load data
train_dir = Path('data/train')
test_dir = Path('data/test')

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_images = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32,
                                                 class_mode='categorical')
test_images = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=32,
                                               class_mode='categorical')

# Load pre-trained model
vgg_model = Sequential()

pretrained_model = keras.applications.VGG16(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
for layer in pretrained_model.layers:
  layer.trainable = False

vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(256, activation='relu'))
vgg_model.add(Dense(2, activation='softmax'))
vgg_model.summary()
vgg_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=2e-5), metrics=['acc'])

# Train model
history = vgg_model.fit(train_images, epochs=20, steps_per_epoch=len(train_images), validation_data=test_images,
                        validation_steps=int(0.25 * (len(test_images))))

# Save model
vgg_model.save('vgg_model.h5')


# Plot accuracy and loss
def plot_acc_loss(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  # Training accuracy
  plt.plot(epochs, acc, 'r', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend(loc=0)
  plt.figure()
  plt.show()

  # Training loss
  plt.plot(epochs, loss, 'r', label='Training Loss')
  plt.plot(epochs, val_loss, 'b', label='Validation Loss')
  plt.title('Training and validation loss')
  plt.legend(loc=0)
  plt.figure()
  plt.show()


plot_acc_loss(history)
