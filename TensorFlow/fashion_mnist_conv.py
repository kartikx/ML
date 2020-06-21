'''
Training on Fashion_MNIST dataset,
this time using Convolutional Layers.
The main advantage of CNNs over traditional NNs
is that you're emphasizing feature detection, over
raw pixel memorization.
Another advantage is a reduction in parameters.
'''

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Resizing because Conv Layers expect inputs in 3-dimensions.
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train, x_test = x_train/255.0 , x_test/255.0

model = keras.Sequential()
# Input Shape must be 28,28,1 and not 28,28. Channel must be specified.
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), input_shape=(28,28,1), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
