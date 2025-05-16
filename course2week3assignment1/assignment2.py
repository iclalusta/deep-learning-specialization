import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from more_itertools.more import seekable
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)
tf.compat.v1.disable_eager_execution()
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')
loss=tf.Variable((y-y_hat)**2, name='loss')
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as session:
    session.run(init)
    #print(session.run(loss))
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)
#print(c)
sess = tf.compat.v1.Session()
#print(sess.run(c))
x = tf.compat.v1.placeholder(tf.int64, name='x')
#print(sess.run(2*x, feed_dict= {x: 3}))
sess.close()
def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name = 'X')
    W = tf.constant(np.random.randn(4,3), name = 'W')
    b = tf.constant(np.random.randn(4,1), name = 'b')
    Y = tf.constant(np.random.randn(4,1), name = 'Y')
    sess = tf.compat.v1.Session()
    result = sess.run(tf.add(tf.linalg.matmul(W, X), b))
    sess.close()
    return result
#print("result = \n" + str(linear_function()))
def sigmoid(z):
    x = tf.compat.v1.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)
    with tf.compat.v1.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: z})
    return result
#print("sigmoid(0) = " + str(sigmoid(0)))
#print("sigmoid(12) = " + str(sigmoid(12)))
def cost(logits, labels):
    z = tf.compat.v1.placeholder(tf.float32, name='z')
    y = tf.compat.v1.placeholder(tf.float32, name='y')
    cost= tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels=y)
    sess = tf.compat.v1.Session()
    cost = sess.run(cost, feed_dict= {z: logits, y:labels})
    sess.close()
    return cost
logits = np.array([0.2, 0.4, 0.7, 0.9])
cost = cost(logits, np.array([0, 0, 1, 1]))
#print("cost = " + str(cost))
def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(labels, depth = C, axis=0)
    sess = tf.compat.v1.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C=4)
#print("one_hot = \n" + str(one_hot))
def ones(shape):
    ones = tf.ones(shape)
    sess = tf.compat.v1.Session()
    ones = sess.run(ones)
    sess.close()
    return ones
#print("ones = " + str(ones([3])))

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#index = 0
#plt.imshow(X_train_orig[index])
#plt.show()
#print("y = " + str(np.squeeze(Y_train_orig[:, index])))
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
"""print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))"""
def create_placeholders(n_x, n_y):
    X = tf.compat.v1.placeholder(shape=[n_x, None], dtype = tf.float32, name='X')
    Y = tf.compat.v1.placeholder(shape=[n_y, None], dtype = tf.float32, name='Y')
    return X, Y
#X, Y = create_placeholders(12288, 6)
#print(X)
#print(Y)
def initialize_params():
    tf.compat.v1.set_random_seed(1)
    W1 = tf.compat.v1.get_variable("W1", [25, 12288], initializer=tf.keras.initializers.glorot_normal(seed=1))
    b1 = tf.compat.v1.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.compat.v1.get_variable("W2", [12, 25], initializer=tf.keras.initializers.glorot_normal(seed=1))
    b2 = tf.compat.v1.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.compat.v1.get_variable("W3", [6, 12], initializer=tf.keras.initializers.glorot_normal(seed=1))
    b3 = tf.compat.v1.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return params
tf.compat.v1.reset_default_graph()
"""with tf.compat.v1.Session() as sess:
    params = initialize_params()
    print("W1 = " + str(params["W1"]))
    print("b1 = " + str(params["b1"]))
    print("W2 = " + str(params["W2"]))
    print("b2 = " + str(params["b2"]))"""
def forward_propagation(X, params):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3
tf.compat.v1.reset_default_graph()
"""with tf.compat.v1.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    params = initialize_params()
    Z3 = forward_propagation(X, params)
    print("Z3 = " + str(Z3))"""
def compute_cost(Z3, Y):
    logits=tf.transpose(Z3)
    labels=tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost
tf.compat.v1.reset_default_graph()
"""with tf.compat.v1.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    params = initialize_params()
    Z3 = forward_propagation(X, params)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))"""
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()
    tf.compat.v1.set_random_seed(1)
    seed=3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    params = initialize_params()
    Z3 = forward_propagation(X, params)
    cost = compute_cost(Z3, Y)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost/num_minibatches
            if print_cost == True and epoch % 100 ==0:
                print("cost after epoch %i: %f"%(epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        params = sess.run(params)
        print("params have been saved")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("train accuracy: ", accuracy.eval({X: X_train, Y:Y_train}))
        print("test accuracy: ", accuracy.eval({X: X_test, Y: Y_test}))
        return params
params= model(X_train, Y_train, X_test, Y_test)
