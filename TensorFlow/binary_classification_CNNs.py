'''
Building a model that binary classifies an image
as a Horse or a Human.
Using ImageDataGenerators, to do the labelling for us.
'''

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile
import os

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
    -O /tmp/horse-or-human.zip

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip \
    -O /tmp/validation-horse-or-human.zip

# Extracting the files (This was written in Google Colab)
file_name = '/tmp/horse-or-human.zip'
zip_ref = ZipFile(file_name, 'r')
zip_ref.extractall('/tmp/horse-or-human')

file_name = '/tmp/validation-horse-or-human.zip'
zip_ref = ZipFile(file_name, 'r')
zip_ref.extractall('/tmp/validation-horse-or-human')

train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
train_human_dir = os.path.join('/tmp/horse-or-human/humans')
validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')

# Setting up ImageDataGenerators. You can implement Augmentation here, to reduce overfitting.
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
                  directory='/tmp/horse-or-human',
                  target_size=(300,300),
                  batch_size=128,
                  class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(
                       directory='/tmp/validation-horse-or-human',
                       class_mode='binary',
                       target_size=(300,300),
                       batch_size=32
)

# Now let's build our model!
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), input_shape=(300,300,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape=(300,300,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), input_shape=(300,300,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])

# Steps per epoch can speed up training. A rule of thumb is to set it equal to number_training_images/batch_size
model.fit(train_generator, validation_data=validation_generator, epochs=15, steps_per_epoch=8, validation_steps=8, verbose=2)


