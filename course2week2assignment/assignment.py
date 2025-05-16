import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
from opt_utils import load_params_and_grads, initialize_params, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset, load_2D_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_params_with_gd(params, grads, learning_rate):
    L = len(params) // 2
    for l in range(L):
        params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return params


"""params, grads, learning_rate = update_parameters_with_gd_test_case()
params = update_params_with_gd(params,grads, learning_rate)
print("W1 =\n" + str(params["W1"]))
print("b1 =\n" + str(params["b1"]))
print("W2 =\n" + str(params["W2"]))
print("b2 =\n" + str(params["b2"]))"""


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m%mini_batch_size !=0:
        mini_batch_X = shuffled_X[:, int(m/mini_batch_size)*mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, int(m/mini_batch_size)*mini_batch_size : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


"""X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print("shape of the 2st mini_batch_X: " + str(mini_batches[1][0].shape))
print("shape of the 3st mini_batch_X: " + str(mini_batches[2][0].shape))
print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print("shape of the 2st mini_batch_Y: " + str(mini_batches[1][1].shape))
print("shape of the 3st mini_batch_Y: " + str(mini_batches[2][1].shape))
print("mini batch sanity chech: " + str(mini_batches[0][0][0][0:3]))"""


def initialize_velocity(params):
    L = len(params)//2
    v = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((params["W" + str(l+1)].shape[0], params["W" + str(l+1)].shape[1]))
        v["db" + str(l+1)] = np.zeros((params["b" + str(l+1)].shape[0], params["b" + str(l+1)].shape[1]))
    return v


"""params= initialize_velocity_test_case()
v = initialize_velocity(params)
print("v[\"dW1\"] =\n" + str(v["dW1"]))
print("v[\"db1\"] =\n" + str(v["db1"]))
print("v[\"dW2\"] =\n" + str(v["dW2"]))
print("v[\"db2\"] =\n" + str(v["db2"]))"""


def update_params_with_momentum(params, grads, v, beta, learning_rate):
    L = len(params)//2
    for l in range(L):
        v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1-beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1-beta) * grads["db" + str(l+1)]
        params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate* v["dW" + str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate* v["db" + str(l+1)]
    return params, v


"""params, grads, v = update_parameters_with_momentum_test_case()
params, v = update_params_with_momentum(params, grads, v, beta = 0.9, learning_rate=0.01)
print("W1 = \n" + str(params["W1"]))
print("b1 = \n" + str(params["b1"]))
print("W2 = \n" + str(params["W2"]))
print("b2 = \n" + str(params["b2"]))
print("v[\"dW1\" = \n" + str(v["dW1"]))
print("v[\"db1\" = \n" + str(v["db1"]))
print("v[\"dW2\" = \n" + str(v["dW2"]))
print("v[\"db2\" = \n" + str(v["db2"]))"""


def initialize_adam(params):
    L = len(params) // 2
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((params["W" + str(l+1)].shape[0], params["W" + str(l+1)].shape[1]))
        v["db" + str(l+1)] = np.zeros((params["b" + str(l+1)].shape[0], params["b" + str(l+1)].shape[1]))
        s["dW" + str(l+1)] = np.zeros((params["W" + str(l+1)].shape[0], params["W" + str(l+1)].shape[1]))
        s["db" + str(l+1)] = np.zeros((params["b" + str(l+1)].shape[0], params["b" + str(l+1)].shape[1]))
    return v, s


"""params = initialize_adam_test_case()
v, s = initialize_adam(params)
print("v[\"dW1\"] = \n" + str(v["dW1"]))
print("v[\"db1\"] = \n" + str(v["db1"]))
print("v[\"dW2\"] = \n" + str(v["dW2"]))
print("v[\"db2\"] = \n" + str(v["db2"]))
print("s[\"dW1\"] = \n" + str(s["dW1"]))
print("s[\"db1\"] = \n" + str(s["db1"]))
print("s[\"dW2\"] = \n" + str(s["dW2"]))
print("s[\"db2\"] = \n" + str(s["db2"]))"""


def update_params_with_adam(params, grads, v, s, t, learning_rate = 0.01,
                            beta1=0.9, beta2=0.999, epsilon = 1e-8):
    L = len(params) // 2
    v_corrected = {}
    s_corrected = {}
    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * grads["db" + str(l+1)]
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * np.square(grads["dW" + str(l+1)])
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * np.square(grads["db" + str(l+1)])
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)
        params["W" +str(l+1)] = params["W" + str(l+1)] - learning_rate * v_corrected["dW" +str(l+1)] / np.sqrt(s_corrected["dW" + str(l+1)] + epsilon)
        params["b" +str(l+1)] = params["b" + str(l+1)] - learning_rate * v_corrected["db" +str(l+1)] / np.sqrt(s_corrected["db" + str(l+1)] + epsilon)
    return params, v, s

"""params, grads, v, s = update_parameters_with_adam_test_case()
params, v, s = update_params_with_adam(params, grads, v, s, t=2)
print("W1 = \n" + str(params["W1"]))
print("b1 = \n" + str(params["b1"]))
print("W2 = \n" + str(params["W2"]))
print("b2 = \n" + str(params["b2"]))
print("v[\"dW1\"] = \n" + str(v["dW1"]))
print("v[\"db1\"] = \n" + str(v["db1"]))
print("v[\"dW2\"] = \n" + str(v["dW2"]))
print("v[\"db2\"] = \n" + str(v["db2"]))
print("s[\"dW1\"] = \n" + str(s["dW1"]))
print("s[\"db1\"] = \n" + str(s["db1"]))
print("s[\"dW2\"] = \n" + str(s["dW2"]))
print("s[\"db2\"] = \n" + str(s["db2"]))"""

train_X, train_Y = load_dataset()

def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64,
          beta = 0.9, beta1=0.9, beta2=0.999, epsilon =1e-8, num_epochs = 10000, print_cost =True):
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10
    m = X.shape[1]
    print("\nThe number of training examples is: %i\n"%m)
    print("The mini batch size: %i\n"%mini_batch_size)
    params = initialize_params(layers_dims)

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(params)
    elif optimizer == "adam":
        v, s = initialize_adam(params)

    for i in range(num_epochs):
        seed = seed+1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a3, caches = forward_propagation(minibatch_X, params)
            cost_total += compute_cost(a3, minibatch_Y)
            #print(str(minibatch_Y))
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            if optimizer == "gd":
                params = update_params_with_gd(params, grads, learning_rate)
            elif optimizer == "momentum":
                params, v = update_params_with_momentum(params, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t+=1
                params, v, s = update_params_with_adam(params, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
        cost_avg = cost_total / m
        if print_cost and i%1000 == 0:
            print("Cost after epoch %i: %f"%(i, cost_avg))
        if print_cost and i%100 == 0:
            costs.append(cost_avg)
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("epochs")
    plt.title("learningrate = " + str(learning_rate))
    plt.show()
    return params

## gd -> 79, momentum -> 79, adam -> 94
layers_dim = [train_X.shape[0], 5, 2, 1]
params = model(train_X, train_Y, layers_dim, optimizer="adam")
predictions = predict(train_X, train_Y, params)
plt.title("model with gradient descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)
plt.show()