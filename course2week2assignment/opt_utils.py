import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import sklearn
import sklearn.datasets


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def relu(x):
    s = np.maximum(0, x)
    return s


def load_params_and_grads(seed=1):
    np.random.seed(seed)
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)
    dW1 = np.random.randn(2, 3)
    db1 = np.random.randn(2, 1)
    dW2 = np.random.randn(3, 3)
    db2 = np.random.randn(3, 1)
    return W1, b1, W2, b2, dW1, db1, dW2, db2


def initialize_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))
        assert params["W" + str(l)].shape[0] == layer_dims[l], layer_dims[l-1]
        assert params["b" + str(l)].shape[0] == layer_dims[l], 1
    return params


def compute_cost(a3, Y):
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1-a3), 1-Y)
    cost_total = np.sum(logprobs)
    return cost_total


def forward_propagation(X, params):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    return a3, cache


def backward_propagation(X, Y, cache):
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
    dz3 = 1./m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)
    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)
    grads = {"dz3": dz3, "dW3": dW3, "db3": db3,
             "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
             "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    """    for a in grads :
        print(str(grads[a]) + "\n")"""
    return grads


def predict(X, y, params):
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int64)
    a3, caches = forward_propagation(X, params)
    for i in range(0, a3.shape[1]):
        if a3[0, i] >0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("accuracy: " + str(np.mean((p[0, :] == y [0, :]))))
    return p


def load_2D_dataset():
    data = scipy.io.loadmat('data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    #plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral)
    return train_X, train_Y, test_X, test_Y


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min()-1, X[0, :].max() +1
    y_min, y_max = X[1, :].min()-1, X[1, :].max() +1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap= plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(params, X):
    a3, cache = forward_propagation(X, params)
    predictions = (a3 > 0.5)
    return predictions


def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples = 300, noise=0.2)
    #plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y

