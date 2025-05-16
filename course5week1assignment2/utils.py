import numpy as np

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum(axis = 0)

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]
    print('%s' % (txt, ), end='')

def get_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]
    return txt

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size) * seq_length

def initialize_params(n_a, n_x, n_y):
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x) * 0.01
    Waa = np.random.randn(n_a, n_a) * 0.01
    Wya = np.random.randn(n_y, n_a) * 0.01
    b = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1))
    params = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    return params

def rnn_step_forward(params, a_prev, x):
    Waa, Wax, Wya, by, b = params["Waa"], params["Wax"], params["Wya"], params["by"], params["b"]
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
    p_t = softmax(np.dot(Wya, a_next) + by)
    return a_next, p_t

def rnn_step_backward(dy, grads, params, x, a, a_prev):
    grads["dWya"] += np.dot(dy, a.T)
    grads["dby"] += dy
    da = np.dot(params["Wya"].T, dy) + grads['da_next']
    daraw = (1 - a * a) * da
    grads["db"] += daraw
    grads["dWax"] += np.dot(daraw, x.T)
    grads["dWaa"] += np.dot(daraw, a_prev.T)
    grads["da_next"] += np.dot(params["Waa"].T, daraw)
    return grads

def update_params(params, grads, lr):
    params["Wax"] += -lr * grads["dWax"]
    params["Waa"] += -lr * grads["dWaa"]
    params["Wya"] += -lr * grads["dWya"]
    params["b"] += -lr * grads["db"]
    params["by"] += -lr * grads["dby"]
    return params

def rnn_forward(X, Y, a0, params, vocab_size = 27):
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0
    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        if (X[t] != None):
            x[t][X[t]] = 1
        a[t], y_hat[t] = rnn_step_forward(params, a[t-1], x[t])
        loss -= np.log(y_hat[t][Y[t], 0])
    cache = (y_hat, a, x)
    return loss, cache

def rnn_backward(X, Y, params, cache):
    grads = {}
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = params["Waa"], params["Wax"], params["Wya"], params["by"], params["b"]
    grads["dWax"], grads["dWaa"], grads["dWya"] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    grads["dby"], grads["db"] = np.zeros_like(by), np.zeros_like(b)
    grads["da_next"] = np.zeros_like(a[0])
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        grads = rnn_step_backward(dy, grads, params, x[t], a[t], a[t-1])
    return grads, a


