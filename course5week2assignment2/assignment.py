import numpy as np
import emoji
import matplotlib.pyplot as plt
from emo_utils import *
from test_utils import *

X_train, Y_train = read_csv('train_emoji.csv')
X_test, Y_test = read_csv('tesss.csv')
maxLen = len(max(X_train, key = len).split())

"""for idx in range(10):
    print(X_train[idx], label_to_emoji(Y_train[idx]))"""

Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

"""
idx = 50
print(f"sentence '{X_train[50]}' has label index {Y_train[idx]}, which is emoji {label_to_emoji(Y_train[idx])}")
print(f"label index {Y_train[idx]} in one-hot encoding format is {Y_oh_train[idx]}")
"""

word_to_index, index_to_word, word_to_vector_map = read_glove_vecs('glove.6B.50d.txt')

"""word = "cucumber"
idx = 289846
print("the index of", word, "in the vocabulary is", word_to_index[word])
print("the", str(idx) + "th word in the vocabulary is", index_to_word[idx])
"""

def sentence_to_avg(sentence, word_to_vector_map):
    any_word = list(word_to_vector_map.keys())[0]
    words = sentence.lower().split()
    avg = np.zeros(word_to_vector_map[any_word].shape)
    count = 0
    for word in words:
        if word in list(word_to_vector_map.keys()):
            avg += word_to_vector_map[word]
            count += 1
    if count > 0:
        avg /= count
    return avg

#avg = sentence_to_avg("morrocan is my fav dish", word_to_vector_map)
#print("avg = \n", avg)

def sentence_to_avg_test(target):
    word_to_vector_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2],
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0],
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]}
    for key in word_to_vector_map.keys():
        word_to_vector_map[key] = np.array(word_to_vector_map[key])
    avg = target("a a_nw c_w a_s", word_to_vector_map)
    assert tuple(avg.shape) == tuple(word_to_vector_map['a'].shape), "check the shape of ur avg shape"
    assert np.allclose(avg, [1.25, 2.5]), "check that u are finding the 4 words"
    avg = target("love a a_nw c_w a_s", word_to_vector_map)
    assert np.allclose(avg, [1.25, 2.5]), "divide by count not len(words)"
    avg = target("love", word_to_vector_map)
    assert np.allclose(avg, [0, 0]), "avg of no words must give an array of zeros"
    avg = target("c_se foo a a_nw c_w a_s deeplearning c_nw", word_to_vector_map)
    assert np.allclose(avg, [0.1666667, 2.0]), "debug last one"
    print("\033[92mAll tests passed!")

#sentence_to_avg_test(sentence_to_avg)

def model(X, Y, word_to_vector_map, learning_rate=0.01, num_iterations = 200):
    any_word= list(word_to_vector_map.keys())[0]
    cost = 0
    m = Y.shape[0]
    n_y = len(np.unique(Y))
    n_h = word_to_vector_map[any_word].shape[0]
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y, ))
    Y_oh = convert_to_one_hot(Y, C = n_y)
    for t in range(num_iterations):
        for i in range(m):
            avg = sentence_to_avg(X[i], word_to_vector_map)
            z = np.add(np.dot(W, avg), b)
            a = softmax(z)
            cost = -np.sum(np.dot(Y_oh[i], np.log(a)))
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz
            W = W - learning_rate * dW
            b = b - learning_rate * db
        if t%10 == 0:
            print("epoch: " + str(t) + "---csot = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vector_map)
    return pred, W, b

def model_test(target):
    word_to_vector_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 'a_n': [3, 4],
                       'c': [-2, 1], 'c_n': [-2, 2], 'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0],
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]}
    for key in word_to_vector_map.keys():
        word_to_vector_map[key] = np.array(word_to_vector_map[key])
    X = np.asarray(
        ['a a_s synonym_of_a a_n c_sw', 'a a_s a_n c_sw', 'a_s  a a_n', 'synonym_of_a a a_s a_n c_sw', " a_s a_n",
         " a a_s a_n c ", " a_n  a c c c_e",
         'c c_nw c_n c c_ne', 'c_e c c_se c_s', 'c_nw c a_s c_e c_e', 'c_e a_nw c_sw', 'c_sw c c_ne c_ne'])
    Y = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    np.random.seed(10)
    pred, W, b = target(X, Y, word_to_vector_map, 0.0025, 110)
    assert W.shape == (2,2), "w must be (2,2)"
    assert np.allclose(pred.transpose(), Y), "model must give a perfect accuracy"
    assert np.allclose(b[0], -1 * b[1]), "b should be symmetric in this ex"
    print("\033[92mall tests passed!!!")

#model_test(model)

"""print(X_train.shape)
print(Y_train.shape)
print(np.eye(5)[Y_train.reshape(-1)].shape)
print(X_train[0])
print(type(X_train))
Y = np.asarray([5, 0, 0, 5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
print(Y.shape)
X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
 'Lets go party and have drinks','Congrats on the new job','Congratulations',
 'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
 'You totally deserve this prize', 'Let us go play football',
 'Are you down for football this afternoon', 'Work hard play harder',
 'It is surprising how people can be dumb sometimes',
 'I am very disappointed','It is the best day in my life',
 'I think I will end up alone','My life is so boring','Good job',
 'Great so awesome'])
print(X.shape)"""
np.random.seed(1)
pred, W, b = model(X_train, Y_train, word_to_vector_map)
#print(pred)

"""print("training set: ")
pred_train = predict(X_train, Y_train, W, b, word_to_vector_map)
print(pred_train)
print("test set: ")
pred_test = predict(X_test, Y_test, W, b, word_to_vector_map)
print(pred_test)
"""
def predict_single(sentence, W=W, b=b, word_to_vector_map = word_to_vector_map):
    any_word = list(word_to_vector_map.keys())[0]
    n_h = word_to_vector_map[any_word].shape[0]
    words = sentence.lower().split()
    avg = np.zeros((n_h, ))
    count = 0
    for w in words:
        if w in word_to_vector_map:
            avg += word_to_vector_map[w]
            count += 1
    if count > 0:
        avg /= count
    Z = np.dot(W, avg) + b
    A = softmax(Z)
    pred = np.argmax(A)
    return pred

print(label_to_emoji(int(predict_single("I love you"))))
print(label_to_emoji(int(predict_single("I hate you"))))
print(label_to_emoji(int(predict_single("I love baseball"))))
print(label_to_emoji(int(predict_single("I didn't want to have this"))))
