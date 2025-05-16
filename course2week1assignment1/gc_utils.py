import numpy as np

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def relu(x):
    s = np.maximum(0, x)
    return s


def dictionary_to_vector(params):
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        new_vector = np.reshape(params[key], (-1, 1))
        keys += [key]*new_vector.shape[0]
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count +=1
    return  theta, keys


def vector_to_dictionary(theta):
    params = {}
    params["W1"] = theta[:20].reshape((5,4))
    params["b1"] = theta[20:25].reshape((5,1))
    params["W2"] = theta[25:40].reshape((3,5))
    params["b2"] = theta[40:43].reshape((3,1))
    params["W3"] = theta[43:46].reshape((1,3))
    params["b3"] = theta[46:47].reshape((1,1))
    return params


def gradients_to_vector(grads):
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        new_vector = np.reshape(grads[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count += 1
    return theta

