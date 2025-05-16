import numpy as np
from utils import *
import random
import pprint
import copy

data = open("dinos.txt", "r").read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
#print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
chars = sorted(chars)
#print(chars)
char_to_ix = { ch: i for i, ch in enumerate(chars)}
ix_to_char = { i: ch for i, ch in enumerate(chars)}
pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(char_to_ix)

def clip(grads, maxValue):
    grads2 = copy.deepcopy(grads)
    dWaa, dWax, dWya, db, dby = grads["dWaa"], grads["dWax"], grads["dWya"], grads["db"], grads["dby"]
    for grad in grads:
        np.clip(grads[grad], -maxValue, maxValue, out = grads2[grad])
    grads = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return grads2

def clip_test(target, mValue):
    print(f"\nGradients for mValue={mValue}")
    np.random.seed(3)
    dWax = np.random.randn(5, 3) * 10
    dWaa = np.random.randn(5, 5) * 10
    dWya = np.random.randn(2, 5) * 10
    db = np.random.randn(5, 1) * 10
    dby = np.random.randn(2, 1) * 10
    grads = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
    grads2 = target(grads, mValue)
    print("gradients[\"dWaa\"][1][2] =", grads2["dWaa"][1][2])
    print("gradients[\"dWax\"][3][1] =", grads2["dWax"][3][1])
    print("gradients[\"dWya\"][1][2] =", grads2["dWya"][1][2])
    print("gradients[\"db\"][4] =", grads2["db"][4])
    print("gradients[\"dby\"][1] =", grads2["dby"][1])
    for grad in grads2.keys():
        valuei = grads[grad]
        valuef = grads2[grad]
        mink = np.min(valuef)
        maxk = np.max(valuef)
        assert mink >= -abs(mValue), f"problem with {grad}. set a_min to -mValue."
        assert maxk <= abs(mValue), f"problem with {grad}. set a_max to mValue."
        index_not_clipped = np.logical_and(valuei <= mValue, valuei >= -mValue)
        assert  np.all (valuei[index_not_clipped] == valuef[index_not_clipped]), f"problem with {grad}, "
    print("\033[92m ALL TESTS PASSED!\x1b[0m")

def sample(params, char_to_ix, seed):
    Waa, Wax, Wya, by, b = params["Waa"], params["Wax"], params["Wya"], params["by"], params["b"]
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']
    while (idx != newline_character and counter != 50):
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        y = softmax(np.dot(Wya, a) + by)
        np.random.seed(counter + seed)
        idx = np.random.choice(range(len(y)), p = np.squeeze(y))
        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a
        seed += 1
        counter += 1
    if counter == 50:
        indices.append(char_to_ix['\n'])
    return indices

def sample_test(target):
    np.random.seed(24)
    _, n_a = 20, 100
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    params = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    indices = target(params, char_to_ix, 0)
    print("Sampling:")
    print("list of sampled indices:\n", indices)
    print("list of sampled characters:\n", [ix_to_char[i] for i in indices])
    assert len(indices) < 52, "indices len must be smaller than 52"
    assert indices[-1] == char_to_ix['\n'], "all samples must end with \\n"
    assert min(indices) >= 0 and max(indices) <= len(char_to_ix), f"sampled indexes must be between 0 and len(char_to_ix) = {len(char_to_ix)}"
    assert np.allclose(indices[0:6], [23, 16, 26, 26, 24, 3]), "wrong!!!"
    print("\033[92m All tests passed!!!")

def optimize(X, Y, a_prev, params, learning_rate = 0.01):
    loss, cache = rnn_forward(X, Y, a_prev, params)
    grads, a = rnn_backward(X, Y, params, cache)
    grads = clip(grads, 5)
    params = update_params(params, grads, learning_rate)
    return loss, grads, a[len(X)-1]

def optimize_test(target):
    np.random.seed(1)
    vocab_size, n_a = 27, 100
    a_prev = np.random.randn(n_a, 1)
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    params = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    X = [12, 3, 5, 11, 22, 3]
    Y = [4, 14, 11, 22, 25, 26]
    old_params = copy.deepcopy(params)
    loss, grads, a_last = target(X, Y, a_prev, params, learning_rate=0.01)
    print("Loss =", loss)
    print("gradients[\"dWaa\"][1][2] =", grads["dWaa"][1][2])
    print("np.argmax(gradients[\"dWax\"]) =", np.argmax(grads["dWax"]))
    print("gradients[\"dWya\"][1][2] =", grads["dWya"][1][2])
    print("gradients[\"db\"][4] =", grads["db"][4])
    print("gradients[\"dby\"][1] =", grads["dby"][1])
    print("a_last[4] =", a_last[4])
    assert np.isclose(loss, 126.5039757), "problem with the call of bişi bişi"
    for grad in grads.values():
        assert np.min(grad) >= -5, "clip function problem"
        assert np.max(grad) <= 5, "clip function problem"
    assert np.allclose(grads['dWaa'][1,2], 0.1947093), "unexprected gradient"
    assert np.allclose(grads['dWya'][1, 2], -0.007773876), "Unexpected gradients. Check the rnn_backward call"
    assert not np.allclose(params['Wya'], old_params['Wya']), "params bişib bişi"
    print("\033[92mAll tests passed!")

def model(data_x, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):
    n_x, n_y = vocab_size, vocab_size
    params =  initialize_params(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, dino_names)
    examples = [x.strip() for x in data_x]
    np.random.seed(0)
    np.random.shuffle(examples)
    a_prev = np.zeros((n_a, 1))
    last_dino_name = "abc"
    for j in range(num_iterations):
        idx = j % len(examples)
        single_example_chars = examples[idx]
        single_example_ix = [char_to_ix[c] for c in single_example_chars]
        X = [None] + single_example_ix
        Y = X[1:] + [char_to_ix["\n"]]
        curr_loss, grads, a_prev = optimize(X, Y, a_prev, params, learning_rate=0.001)
        if verbose and j in [0, len(examples)-1, len(examples)]:
            print("j = ", j, "idx = ", idx)
        if verbose and j in [0]:
            print("single_example_chars", single_example_chars)
            print("single_example_ix", single_example_ix)
            print(" X = ", X, "\n", "Y =       ", Y, "\n")
        loss = smooth(loss, curr_loss)
        if j % 2000 == 0:
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            seed = 0
            for name in range(dino_names):
                sample_indices = sample(params, char_to_ix, seed)
                last_dino_name = get_sample(sample_indices, ix_to_char)
                print(last_dino_name.replace('\n', ''))
                seed += 1
            print('\n')
    return params, last_dino_name

params, last_name = model(data.split("\n"), ix_to_char, char_to_ix, verbose = True)
assert last_name == 'Trodonosaurus\n', "Wrong expected output"
print("\033[92mAll tests passed!")