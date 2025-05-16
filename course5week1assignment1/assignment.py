import numpy as np
from rnn_utils import *
from public_tests import *

def rnn_cell_forward(xt, a_prev, params):
    Wax = params["Wax"]
    Waa = params["Waa"]
    Wya = params["Wya"]
    ba = params["ba"]
    by = params["by"]
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    cache = (a_next, a_prev, xt, params)
    return a_next, yt_pred, cache

"""np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5, 10)
params_tmp = {}
params_tmp['Waa'] = np.random.randn(5,5)
params_tmp['Wax'] = np.random.randn(5,3)
params_tmp['Wya'] = np.random.randn(2,5)
params_tmp['ba'] = np.random.randn(5,1)
params_tmp['by'] = np.random.randn(2,1)
a_next_tmp, yt_pred_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, params_tmp)
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = \n", a_next_tmp.shape)
print("yt_pred[1] =\n", yt_pred_tmp[1])
print("yt_pred.shape = \n", yt_pred_tmp.shape)
rnn_cell_forward_tests(rnn_cell_forward)"""

def rnn_forward(x, a0, params): # Tx = Ty
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = params["Wya"].shape
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, params)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)
    caches = (caches, x)
    return a, y_pred, caches

"""np.random.seed(1)
x_tmp = np.random.randn(3,10,4)
a0_tmp = np.random.randn(5, 10)
params_tmp = {}
params_tmp["Waa"] = np.random.randn(5,5)
params_tmp["Wax"] = np.random.randn(5,3)
params_tmp["Wya"] = np.random.randn(2,5)
params_tmp["ba"] = np.random.randn(5,1)
params_tmp["by"] = np.random.randn(2,1)
a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, params_tmp)
print("a[4][1] = \n", a_tmp[4][1])
print("a.shape = \n", a_tmp.shape)
print("y_pred[1][3] =\n", y_pred_tmp[1][3])
print("y_pred.shape = \n", y_pred_tmp.shape)
print("caches[1][1][3] =\n", caches_tmp[1][1][3])
print("len(caches) = \n", len(caches_tmp))
rnn_forward_test(rnn_forward)"""

### rnn suffers from VANISHING GRADIENTS

def lstm_cell_forward(xt, a_prev, c_prev, params):
    Wf = params["Wf"]
    bf = params["bf"]
    Wi = params["Wi"]
    bi = params["bi"]
    Wc = params["Wc"]
    bc = params["bc"]
    Wo = params["Wo"]
    bo = params["bo"]
    Wy = params["Wy"]
    by = params["by"]
    n_x, m = xt.shape
    n_y, n_a = Wy.shape
    concat = np.concatenate([a_prev,xt])
    cct = np.tanh(np.dot(Wc, concat) + bc)
    it = sigmoid(np.dot(Wi, concat) + bi)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    c_next = it * cct + ft * c_prev
    a_next = ot * (np.tanh(c_next))
    yt_pred = softmax(np.dot(Wy, a_next) + by)
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, params)
    return a_next, c_next, yt_pred, cache

"""np. random.seed(1)
xt_tmp = np.random.randn(3, 10)
a_prev_tmp = np.random.randn(5, 10)
c_prev_tmp = np.random.randn(5, 10)
params_tmp ={}
params_tmp["Wf"] = np.random.randn(5, 5 + 3)
params_tmp["bf"] = np.random.randn(5, 1)
params_tmp["Wi"] = np.random.randn(5, 5 + 3)
params_tmp["bi"] = np.random.randn(5, 1)
params_tmp["Wo"] = np.random.randn(5, 5 + 3)
params_tmp["bo"] = np.random.randn(5, 1)
params_tmp["Wc"] = np.random.randn(5, 5 + 3)
params_tmp["bc"] = np.random.randn(5, 1)
params_tmp["Wy"] = np.random.randn(2, 5)
params_tmp["by"] = np.random.randn(2, 1)
a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, params_tmp)
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = ", a_next_tmp.shape)
print("c_next[2] = \n", c_next_tmp[2])
print("c_next.shape = ", c_next_tmp.shape)
print("yt[1] =", yt_tmp[1])
print("yt.shape = ", yt_tmp.shape)
print("cache[1][3] =\n", cache_tmp[1][3])
print("len(cache) = ", len(cache_tmp))
lstm_cell_forward_test(lstm_cell_forward)"""

def lstm_forward(x, a0, params):
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = params["Wy"].shape
    a = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    c = np.zeros((n_a, m, T_x))
    a_next = a0
    c_next = np.zeros((n_a, m))
    for t in range(T_x):
        xt = x[:, :, t]
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, params)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt
        caches.append(cache)
    caches = (caches, x)
    return a, y, c, caches

"""np. random.seed(1)
x_tmp = np.random.randn(3, 10, 7)
a0_tmp = np.random.randn(5, 10)
params_tmp ={}
params_tmp["Wf"] = np.random.randn(5, 5 + 3)
params_tmp["bf"] = np.random.randn(5, 1)
params_tmp["Wi"] = np.random.randn(5, 5 + 3)
params_tmp["bi"] = np.random.randn(5, 1)
params_tmp["Wo"] = np.random.randn(5, 5 + 3)
params_tmp["bo"] = np.random.randn(5, 1)
params_tmp["Wc"] = np.random.randn(5, 5 + 3)
params_tmp["bc"] = np.random.randn(5, 1)
params_tmp["Wy"] = np.random.randn(2, 5)
params_tmp["by"] = np.random.randn(2, 1)
a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, params_tmp)
print("a[4][3][6] = ", a_tmp[4][3][6])
print("a.shape = ", a_tmp.shape)
print("y[1][4][3] =", y_tmp[1][4][3])
print("y.shape = ", y_tmp.shape)
print("caches[1][1][1] =\n", caches_tmp[1][1][1])
print("c[1][2][1]", c_tmp[1][2][1])
print("len(caches) = ", len(caches_tmp))
lstm_forward_test(lstm_forward)"""





