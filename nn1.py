import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

np.random.seed(0)

X_data, Y_data = sklearn.datasets.make_circles(100, noise=0.05)


def displayPlot(x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


X_data = X_data.T
Y_data = Y_data.reshape(1, Y_data.shape[0])

layers_dims = [2, 4, 3, 1]


def initialize_parameters(layers_dims):
    parameters = {}

    for i in range(1, len(layers_dims)):
        parameters["W" + str(i)] = np.random.randn(layers_dims[i],
                                                   layers_dims[i-1]) * np.sqrt(1/layers_dims[i-1])
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))

    return parameters


def relu(z):
    cache = {}
    cache["z"] = z

    y = z > 0
    z = np.multiply(z, y)

    return z, cache


def sigmoid(z):
    cache = {}
    cache["z"] = z

    return 1 / (1 + np.exp(-z)), cache


def linear_forward(A, W, b):
    linear_cache = {}
    linear_cache["W"] = W
    linear_cache["b"] = b
    linear_cache["A"] = A
    return np.dot(W, A) + b, linear_cache


def linear_activation_forward(A, W, b, activation="relu"):
    z, linear_cache = linear_forward(A, W, b)
    if (activation == "relu"):
        a, activation_cache = relu(z)
    elif (activation == "sigmoid"):
        a, activation_cache = sigmoid(z)
    return a, (linear_cache, activation_cache)



def forward_propagation(A, parameters, L):
    caches = []
    a = A
    for i in range(1, L-1):
        aPrev = a
        a, cache = linear_activation_forward(
            aPrev, parameters["W"+str(i)], parameters["b"+str(i)], activation="relu")
        caches.append(cache)

    aL, cache = linear_activation_forward(
        a, parameters["W"+str(L-1)], parameters["b"+str(L-1)], activation="sigmoid")
    caches.append(cache)

    return aL, caches


def compute_cost(Y, A):
    m = Y.shape[1]
    cost = -(np.dot(Y, (np.log(A)).T) + np.dot((1-Y), (np.log(1-A)).T))/m
    return np.squeeze(cost)


def relu_backward(dA, z):
    z = (z > 0)
    return dA * z


def sig_backward(dA, z):
    s = 1 / (1 + np.exp(-z))
    return dA*s*(1-s)


def linear_backward(dZ, cache):
    m = dZ.shape[1]
    dW = np.dot(dZ, cache["A"].T)/m
    dB = np.sum(dZ, axis=1, keepdims=True)/m
    dA = np.dot(cache["W"].T, dZ)
    return dW, dB, dA


def linear_activation_backward(dA, cache, activation="relu"):
    if(activation == "relu"):
        dZ = relu_backward(dA, cache[1]["z"])
        dW, db, dA = linear_backward(dZ, cache[0])
    if(activation == "sigmoid"):
        dZ = sig_backward(dA, cache[1]["z"])
        dW, db, dA = linear_backward(dZ, cache[0])

    return dA, dW, db


def backward_propagation(y, a, caches):
    gradients = {}
    L = len(caches)
    dAL = -np.divide(y, a) + np.divide(1-y, 1-a)
    dA, gradients["dW"+str(L)], gradients["db"+str(L)] = linear_activation_backward(dAL,
                                                                                    caches[L-1], activation="sigmoid")

    for l in reversed(range(L-1)):
        dA_prev = dA
        dA, gradients["dW"+str(l+1)], gradients["db"+str(l+1)] = linear_activation_backward(dA_prev,
                                                                                            caches[l], activation="relu")

    return gradients


def update_parameters(parameters, gradients, L, learning_rate):
    for i in range(1, L):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate*gradients["dW"+str(i)]
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate*gradients["db"+str(i)]
    return parameters


def nn(n_iterations, learning_rate):
    L = len(layers_dims)
    parameters = initialize_parameters(layers_dims)

    for i in range(n_iterations):
        aL, caches = forward_propagation(X_data, parameters, L)
        J = compute_cost(Y_data, aL)
        gradients = backward_propagation(Y_data, aL, caches)
        parameters = update_parameters(parameters, gradients, L, learning_rate)
        if(i % 10 == 0):
            print(J)


nn(2000, 0.6)
