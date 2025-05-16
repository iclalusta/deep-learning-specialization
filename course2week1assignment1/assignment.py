import numpy as np
from test_cases import *
from gc_utils import sigmoid, relu, vector_to_dictionary, dictionary_to_vector, gradients_to_vector

def forward_propagation(x, theta):
    J = np.dot(theta, x)
    return J


def backward_propagation(x, theta):
    dtheta = x
    return dtheta


def gradient_check(x, theta, epsilon = 1e-7):
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = np.dot(thetaplus, x)
    J_minus = np.dot(thetaminus, x)
    gradapprox = (J_plus - J_minus)/(2 * epsilon)

    grad = x
    numerator = np.linalg.norm(gradapprox-grad)
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
    difference = numerator/denominator

    if difference < 1e-7:
        print("the gradient is correct!")
    else:
        print("the gradient is wrong!!")
    return difference


def forward_propagation_n(X, Y, params):
    m = X.shape[1]
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1-A3), 1 - Y)
    cost = 1./m * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return cost, cache


def backward_propagation_n(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
             "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
             "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return grads


def gradient_check_n(params, grads, X, Y, epsilon= 1e-7):
    params_values, _ = dictionary_to_vector(params)
    grad = gradients_to_vector(grads)
    num_params = params_values.shape[0]
    J_plus = np.zeros((num_params, 1))
    J_minus = np.zeros((num_params, 1))
    gradapprox = np.zeros((num_params, 1))
    for i in range(num_params):
        thetaplus = np.copy(params_values)
        thetaplus[i][0] += epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))
        thetaminus = np.copy(params_values)
        thetaminus -= epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2*epsilon)

    numerator = np.linalg.norm(grad-gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator/denominator

    if difference > 2e-7:
        print("\033[93m"+"There is a mistake!!! difference = " + str(difference))
    else:
        print("\033[92m" + "Works perfectly!!!1 difference = " + str(difference))
    return difference


##x, theta = 2, 4
###difference = gradient_check(x, theta)
###print("difference = " + str(difference))

X, Y, params = gradient_check_n_test_case()
cost, cache = forward_propagation_n(X, Y, params)
grads = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(params, grads, X, Y)