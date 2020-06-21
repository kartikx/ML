'''
A very basic NN to detect a linear function
'''

import numpy as np
from tensorflow import keras

model = keras.Sequential(layers=keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

X = np.array([-2.0, 0.0, 1.0, 3.0 , 4.0], dtype='float32')
Y = np.array([-5.0, -1.0, 1.0, 5.0, 7.0], dtype='float32')

model.fit(X, Y, epochs=500)

model.predict([10.0])
