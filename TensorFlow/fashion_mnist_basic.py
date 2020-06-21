'''
A simple Neural Network to
predict items from the Fashion_MNIST dataset
'''

import numpy as np
from tensorflow import keras

fashion_data = keras.datasets.fashion_mnist
(train_data, train_label), (test_data, test_label) = fashion_data.load_data()

model = keras.Sequential(layers=[keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(units=128, activation=keras.activations.relu),
                                        keras.layers.Dense(units=10, activation=keras.activations.softmax)])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(train_data, train_label, epochs=5)

model.evaluate(train_data, train_label)
