import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_utils import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def two_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layer_dims
    params = initialize_parameters(n_x, n_h, n_y)
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y)

        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        params = update_parameters(params, grads, learning_rate)
        W1 = params["W1"]
        b1 = params["b1"]
        W2 = params["W2"]
        b2 = params["b2"]
        if print_cost and i%100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i%100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("learning rate = "+str(learning_rate))
    plt.show()

    return params


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    np.random.seed(1)
    costs = []
    params = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, params)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        params= update_parameters(params, grads, learning_rate)
        if print_cost and i%100==0:
            print("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i%100==0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate = " +str(learning_rate))
    plt.show()
    return params



train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

#index = 25
#plt.imshow(train_x_orig[index])
#plt.show()
#print("y = "+ str(train_y[0,index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

"""print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))"""

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255
test_x = test_x_flatten/255
print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

"""n_x = 12288
n_h = 7
n_y = 1
layer_dims = (n_x, n_h, n_y)
params = two_layer_model(train_x, train_y, layer_dims, num_iterations= 2500, print_cost = True)
predictions_train = predict(train_x, train_y, params)
predictions_test = predict(test_x, test_y, params)"""

layers_dims = [12288, 20, 7, 5, 1]
params = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
predictions_train = predict(train_x, train_y, params)
predictions_test = predict(test_x, test_y, params)
print_mislabeled_images(classes, test_x, test_y, predictions_test)


