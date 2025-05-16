import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from six import print_
from sympy.physics.vector.printing import params

from testCases_v2 import *
from planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets, sigmoid

np.random.seed(1)

def layer_sizes(X, Y):
    n_x= X.shape[0]
    n_h=4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y,1))

    assert (W1.shape == (n_h, n_x))
    assert (W2.shape == (n_y, n_h))
    assert (b1.shape == (n_h, 1))
    assert (b2.shape == (n_y, 1))

    params = {"W1": W1,
              "W2": W2,
              "b1": b1,
              "b2": b2}
    return params


def forward_propagation(X, params):
    W1=params["W1"]
    b1=params["b1"]
    W2=params["W2"]
    b2=params["b2"]

    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))
    cache = {"Z1":Z1,
             "A1":A1,
             "Z2":Z2,
             "A2":A2}

    return A2, cache


def compute_cost(A2, Y, params):
    m = Y.shape[1]
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log(1-A2))
    cost = (-1/m) * np.sum(logprobs)
    cost = float(np.squeeze(cost))
    assert (isinstance(cost, float))
    return cost


def backward_propagation(params, cache, X, Y):
    m = X.shape[1]
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads


def update_parameters(params, grads, learning_rate):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}

    return params


def nn_model(X, Y, n_h, learning_rate, num_iterations = 10000, print_cost = False):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X,Y)[2]

    params = initialize_parameters(n_x, n_h, n_y)
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, params)
        cost = compute_cost(A2, Y, params)
        grads = backward_propagation(params, cache, X, Y)
        params = update_parameters(params, grads, learning_rate)
        if print_cost and i%1000 ==0:
            print("Cost after iteration %i: %f" %(i, cost))

    return params


def predict(params, X):
    A2, cache = forward_propagation(X, params)
    predictions = (A2 > 0.5)
    return predictions


"""X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the input layer is: n_h = " + str(n_h))
print("The size of the input layer is: n_y = " + str(n_y))

n_x, n_h, n_y = initialize_parameters_test_case()
params = initialize_parameters(n_x, n_h, n_y)
print("W1" + str(params["W1"]))
print("B1" + str(params["b1"]))
print("W2" + str(params["W2"]))
print("B2" + str(params["b2"]))

X_assess, params = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, params)
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

A2, Y_assess, params= compute_cost_test_case()
print("cost = " + str(compute_cost(A2, Y_assess, params)))

params, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_propagation(params, cache, X_assess, Y_assess)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))

params, grads = update_parameters_test_case()
params = update_parameters(params, grads, 1.2)
print("W1" + str(params["W1"]))
print("b1" + str(params["b1"]))
print("W2" + str(params["W2"]))
print("b2" + str(params["b2"])) 

X_assess, Y_assess = nn_model_test_case()
params = nn_model(X_assess, Y_assess, 4, 1.02, num_iterations=10000, print_cost=True)
print("W1 = " + str(params["W1"]))
print("b1 = " + str(params["b1"]))
print("W2 = " + str(params["W2"]))
print("b2 = " + str(params["b2"]))

params,  X_assess = predict_test_case()
predictions = predict(params, X_assess)
print("predictions mean = " + str(np.mean(predictions)))"""

X, Y = load_planar_dataset()

#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
#plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = (X.size)/shape_X[0]
print('the shape of X is ' + str(shape_X))
print('the shape of Y is ' + str(shape_Y))
print('m = %d training examples ' %(m))

"""clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T)

plot_decision_boundary(lambda x: clf.predict(x), X,Y)
plt.title("Logistic Regression")
#plt.show()
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")"""

params = nn_model(X, Y, 4, 1.2, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x: predict(params, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

predictions = predict(params, X)
print ("Accuracy: %d" % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + "%")


plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h,1.2, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    plt.show()

