import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_dataset()
#plt.show()
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        params = initialize_params_zeros(layers_dims)
    elif initialization == "random":
        params = initialize_params_random(layers_dims)
    elif initialization == "he":
        params = initialize_params_he(layers_dims)

    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, params)
        cost = compute_loss(a3, Y)
        grads = backward_propagation(X, Y, cache)
        params = update_parameters(params, grads, learning_rate)
        if print_cost and i%1000 == 0:
            print("cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("learning_rate =" + str(learning_rate))
    plt.show()
    return params


def initialize_params_zeros(layers_dims):
    params = {}
    L = len(layers_dims)
    for l in range(1, L):
        params['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return params


"""params = initialize_params_zeros([3,2,1])
print("W1 = " + str(params["W1"]))
print("b1 = " + str(params["b1"]))
print("W2 = " + str(params["W2"]))
print("b2 = " + str(params["b2"]))
params = model(train_X, train_Y, initialization="zeros")
print("On the train set:")
predictions_train = predict(train_X, train_Y, params)
print("On the test set:")
predictions_test = predict(test_X, test_Y, params)
print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))
plt.title("model with zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)"""


def initialize_params_random(layers_dims):
    np.random.seed(3)
    params = {}
    L = len(layers_dims)
    for i in range(1,L):
        params['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1]) * 10
        params['b' + str(i)] = np.zeros((layers_dims[i], 1))
    return params


"""params = initialize_params_random([3,2,1])
print("W1 = " + str(params["W1"]))
print("b1 = " + str(params["b1"]))
print("W2 = " + str(params["W2"]))
print("b2 = " + str(params["b2"]))

params = model(train_X, train_Y, initialization="random")
print("on the train set: ")
predictions_train = predict(train_X, train_Y, params)
print("on the test set: ")
predictions_test = predict(test_X, test_Y, params)
print(predictions_train)
print(predictions_test)
plt.title("model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)"""

#Xavier vs. He initialization: xavier works better with tanh, sigmoid(symmetric outputs)
#he works better with relu and its variants
def initialize_params_he(layers_dims):
    np.random.seed(3)
    params = {}
    L = len(layers_dims)
    import math
    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * math.sqrt(2./layers_dims[l-1])
        params['b' + str(l)] = np.zeros((layers_dims[l], 1)) * math.sqrt(2./layers_dims[l-1])
    return params


"""params = initialize_params_he([2,4,1])
print("W1 = " + str(params["W1"]))
print("b1 = " + str(params["b1"]))
print("W2 = " + str(params["W2"]))
print("b2 = " + str(params["b2"]))

params = model(train_X, train_Y, initialization= "he")
print("on the train set: ")
predictions_train = predict(train_X, train_Y, params)
print("on the test set: ")
predictions_test = predict(test_X, test_Y, params)
#print(predictions_train)
#print(predictions_test)
plt.title("model with HE initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)"""
