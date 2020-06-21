'''
Detecting Vertical Edges in an image,
by implementing a Convolution Operation
'''

import tensorflow as tf
import tensorflow.keras as keras

import matplotlib.pyplot as plt

original_image = scipy.misc.ascent()
plt.axis('off')
plt.gray()
plt.imshow(original_image)
plt.show()

filters = np.array([[-2,0,2],[-1,0,1],[-2,0,2]], dtype='float32')
# To allow normalizing output, in case sum of filter indices != 1, here it is 1.
weight = 1
transformed_image = np.copy(original_image)

for i in range(1, transformed_image.shape[0]-1):
  for j in range(1, transformed_image.shape[1]-1):
    convolution = np.sum(np.multiply(original_image[i-1:i+2,j-1:j+2], filters))
    convolution *= weight
    if (convolution < 0):
      convolution = 0
    if (convolution > 255):
      convolution = 255
    transformed_image[i,j] = convolution

plt.imshow(transformed_image)
plt.show()

