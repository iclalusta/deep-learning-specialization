import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time


##print(tf.__version__) #2.17

train_dataset = h5py.File('train_signs.h5', "r")
test_dataset = h5py.File('test_signs.h5', "r")
x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])
x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

#print(type(x_train))
#print(x_train.element_spec)
#print(next(iter(x_train)))
"""unique_labels = set()
for element in y_train:
    unique_labels.add(element.numpy())
print(unique_labels)
images_iter = iter(x_train)
labels_iter = iter(y_train)
plt.figure(figsize=(10,10))
for i in range(25):
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(next(images_iter).numpy().astype("uint8"))
    plt.title(next(labels_iter).numpy().astype("uint8"))
    plt.axis("off")
plt.show()"""

def normalize(image):
    image = tf.cast(image, tf.float32) /255.0
    image = tf.reshape(image, [-1,])
    return image

new_train = x_train.map(normalize)
new_test = x_test.map(normalize)
print(new_train.element_spec)
#print(next(iter(new_train)))

def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name = 'X')
    W = tf.constant(np.random.randn(4,3), name = 'W')
    b = tf.constant(np.random.randn(4,1), name = 'b')
    Y = tf.add(tf.matmul(W, X), b)
    return Y


"""result = linear_function()
print(result)
assert type(result) == EagerTensor, "use the tensorflow api"
assert np.allclose(result,[[-2.15657382], [ 2.95891446], [-1.08926781], [-0.84538042]]), "Error"
print("\033[92mAll test passed")"""

def sigmoid(z):
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a
"""result= sigmoid(-1)
print("type: " + str(type(result)))
print("dtype: " + str(result.dtype))
print("sigmoid(-1): " + str(result))
print("sigmoid(0): " + str(sigmoid(0.0)))
print("sigmoid(12): " + str(sigmoid(12)))"""
def sigmoid_test(target):
    result = target(0)
    assert (type(result) == EagerTensor)
    assert (result.dtype == tf.float32)
    assert sigmoid(0) == 0.5, "error"
    assert sigmoid(-1) == 0.26894143, "error"
    assert sigmoid(12) == 0.99999386, "error"
    print("\033[92mAll test passed")
#sigmoid_test(sigmoid)


def one_hot_matrix(label, depth=6):
    one_hot = tf.reshape(tf.one_hot(label, depth, axis=0), (depth,))
    return one_hot
def one_hot_matrix_test(target):
    label = tf.constant(1)
    depth = 4
    result = target(label, depth)
    print("test 1:", result)
    assert result.shape[0] == depth, "use parameter depth"
    assert np.allclose(result, [0., 1., 0., 0.]), "wrong output use tf.onehot"
    label_2 = [2]
    result = target(label_2, depth)
    print("test2: ", result)
    assert result.shape[0] == depth, "use parameter depth"
    assert np.allclose(result, [0., 0., 1., 0.]), "wrong output, use tf.reshape"
    print("\033[92m All test passed")
#one_hot_matrix_test(one_hot_matrix)

new_y_test = y_test.map(one_hot_matrix)
new_y_train = y_train.map(one_hot_matrix)
#print(next(iter(new_y_test)))

def initialize_params():
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    W1 = tf.Variable(initializer(shape=(25, 12288)))
    b1 = tf.Variable(initializer(shape=(25, 1)))
    W2 = tf.Variable(initializer(shape=(12, 25)))
    b2 = tf.Variable(initializer(shape=(12, 1)))
    W3 = tf.Variable(initializer(shape=(6, 12)))
    b3 = tf.Variable(initializer(shape=(6, 1)))
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return params
def initialize_params_test(target):
    params = target()
    values = {"W1": (25, 12288),
              "b1": (25, 1),
              "W2": (12, 25),
              "b2": (12, 1),
              "W3": (6, 12),
              "b3": (6, 1)}
    for key in params:
        print(f"{key} shape: {tuple(params[key].shape)}")
        assert type(params[key]) == ResourceVariable, "all params must bla bla"
        assert tuple(params[key].shape) == values[key], f"{key}: wrong shape"
        assert np.abs(np.mean(params[key].numpy())) < 0.5, f"{key}: use glorotnormal"
        assert np.std(params[key].numpy()) > 0 and np.std(params[key].numpy()) < 1, f"{key}: use glorotnormal"

    print("\033[92mAlllll tests passed")
#initialize_params_test(initialize_params)


def forward_propagation(X, params):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]
    Z1 = tf.math.add(tf.linalg.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 =tf.math.add(tf.linalg.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.math.add(tf.linalg.matmul(W3, A2), b3)
    return Z3
#!!!!!!!
"""def forward_propagation_test(target, examples):
    minibatches = examples.batch(2)
    for minibatch in minibatches:
        forward_pass = target(tf.transpose(minibatch), params)
        print(forward_pass)
        assert type(forward_pass) == EagerTensor, "your putput not a tenspr"
        assert forward_pass.shape == (6,2), "last layer must use w3 and b3"
        assert np.allclose(forward_pass, [[-0.13430887,  0.14086473],
                                          [ 0.21588647, -0.02582335],
                                          [ 0.7059658,   0.6484556 ],
                                          [-1.1260961,  -0.9329492 ],
                                          [-0.20181894, -0.3382722 ],
                                          [ 0.9558965,   0.94167566]]), "output does not match"
        break
    print("\033[92malll test passed")"""
#forward_propagation_test(forward_propagation, new_train)


def compute_cost(logits, labels):
    cost = tf.reduce_sum(tf.keras.metrics.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits = True))
    return cost
def compute_cost_set(target, Y):
    pred = tf.constant([[ 2.4048107,   5.0334096 ],
             [-0.7921977,  -4.1523376 ],
             [ 0.9447198,  -0.46802214],
             [ 1.158121,    3.9810789 ],
             [ 4.768706,    2.3220146 ],
             [ 6.1481323,   3.909829  ]])
    minibatches = Y.batch(2)
    for minibatch in minibatches:
        result = target(pred, tf.transpose(minibatch))
        break
    print(result)
    assert (type(result) == EagerTensor), "use tensorflow api"
    assert (np.abs(result - (0.25361037 + 0.5566767) / 2.0) < 1e-7), "test does not match"
    print("\033[92m all tests passed")
#compute_cost_set(compute_cost, new_y_train)


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    costs = []
    train_acc = []
    test_acc = []
    params = initialize_params()
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip(X_test, Y_test)
    m = dataset.cardinality().numpy()
    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)
    for epoch in range(num_epochs):
        epoch_cost = 0.
        train_accuracy.reset_state()
        for (minibatch_X, minibatch_Y) in minibatches:
            with tf.GradientTape() as tape:
                Z3 = forward_propagation(tf.transpose(minibatch_X), params)
                minibatch_cost = compute_cost(Z3, tf.transpose(minibatch_Y))
            train_accuracy.update_state(tf.transpose(Z3), minibatch_Y)
            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost
        epoch_cost /= m
        if print_cost == True and epoch%10 ==0:
            print("cost after epoch %i: %f" % (epoch, epoch_cost))
            print("train accuracy: ", train_accuracy.result())
            for (minibatch_X, minibatch_Y) in test_minibatches:
                Z3 = forward_propagation(tf.transpose(minibatch_X), params)
                test_accuracy.update_state(tf.transpose(Z3), minibatch_Y)
            print("test_accuracy: ", test_accuracy.result())
            costs.append(epoch_cost)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_state()
    return params, costs, train_acc, test_acc

params, costs, train_acc, test_acc = model(new_train, new_y_train, new_test, new_y_test, num_epochs=100)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per fives)')
plt.title("learning rate = " + str(0.0001))
plt.show()

plt.plot(np.squeeze(train_acc))
plt.ylabel('train accuracy')
plt.xlabel('iterations (per fives)')
plt.title("learning rate = " + str(0.0001))
plt.show()

plt.plot(np.squeeze(test_acc))
plt.ylabel('testing accuracy')
plt.xlabel('iterations (per fives)')
plt.title("learning rate = " + str(0.0001))
plt.show()