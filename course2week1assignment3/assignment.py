import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()
def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd= .0, keep_prob = 1):
    grads = {}
    costs = []
    m = X.shape[1]
    layer_dims = [X.shape[0], 20, 3, 1]
    params = initialize_parameters(layer_dims)
    for i in range(0, num_iterations):
        if keep_prob == 1:
            a3, cache = forward_propagation(X, params)
        elif keep_prob <1:
            a3, cache = forward_propagation_with_dropout(X, params, keep_prob)
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, params, lambd)
        assert(lambd == 0 or keep_prob == 1)
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd !=0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob<1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        params = update_parameters(params, grads, learning_rate)
        if print_cost  and i%10000 == 0:
            print("cost after iteration {}: {}".format(i, cost))
        if print_cost and i%1000 == 0:
            costs.append(cost)
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iterations (x1,000)")
    plt.title("learning rate = " + str(learning_rate))
    plt.show()
    return params

"""params = model(train_X, train_Y)
print("on the training set: ")
predictions_train = predict(train_X, train_Y, params)
print("on the testing set: ")
predictions_test = predict(test_X, test_Y, params)

plt.title("model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)"""


def compute_cost_with_regularization(A3, Y, params, lambd):
    m = Y.shape[1]
    W1 = params["W1"]
    W2 = params["W2"]
    W3 = params["W3"]
    cross_entropy_cost = compute_cost(A3, Y)
    L2_regularization_cost = lambd / (2*m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    cost = cross_entropy_cost + L2_regularization_cost
    return cost


"""A3, Y_assess, params = compute_cost_with_regularization_test_case()
print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, params, lambd = 0.1)))"""


def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = Y.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m *np.dot(dZ3, A2.T) + (lambd/m) * W3
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2>0))
    dW2 = 1./m *np.dot(dZ2, A1.T) + (lambd/m) * W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1>0))
    dW1 = 1./m *np.dot(dZ1, X.T) + (lambd/m) * W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
            "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
            "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return grads


"""X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd=0.7)
print("dW1 = \n" + str(grads["dW1"]))
print("dW2 = \n" + str(grads["dW2"]))
print("dW3 = \n" + str(grads["dW3"]))"""


"""params = model(train_X, train_Y, lambd=0.7)
print("on the training set:")
predictions_train = predict(train_X, train_Y, params)
print("on the testing set:")
predictions_test = predict(test_X, test_Y, params)

plt.title("model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)"""


def forward_propagation_with_dropout(X, params, keep_prob = 0.5):
    np.random.seed(1)
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = (D1 < keep_prob).astype(int)
    A1 = A1*D1
    A1 = A1 /keep_prob
    ##
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = (D2 < keep_prob).astype(int)
    A2 = A2 * D2
    A2 = A2 / keep_prob
    ##
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache


"""X_assess, params = forward_propagation_with_dropout_test_case()
A3, cache = forward_propagation_with_dropout(X_assess, params, keep_prob=0.7)
print("A3 = " + str(A3))"""


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2*D2
    dA2 = dA2/keep_prob
    dZ2 = np.multiply(dA2, np.int64(A2>0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1
    dA1 = dA1/keep_prob
    dZ1 = np.multiply(dA1, np.int64(A1>0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
             "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
             "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return grads

"""X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()
grads = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob=0.8)
print("dA1 = \n" + str(grads["dA1"]))
print("dA2 = \n" + str(grads["dA2"]))"""


"""params = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, params)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, params)

plt.title("model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)"""