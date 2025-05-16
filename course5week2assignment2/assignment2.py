import numpy as np
import tensorflow
from debugpy.server.cli import in_range
from keras.src.models import Model
from keras.src.layers import Dense, Input, Dropout, LSTM, Activation, Embedding
from keras.api.preprocessing import sequence
from keras.api.initializers import glorot_uniform
from test_utils import comparator, summary
from emo_utils import *
np.random.seed(1)

X_train, Y_train = read_csv('train_emoji.csv')
X_test, Y_test = read_csv('tesss.csv')
maxLen = len(max(X_train, key = len).split())
Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)
word_to_index, index_to_word, word_to_vector_map = read_glove_vecs('glove.6B.50d.txt')


"""for idx, val in enumerate(["I", "like", "learning"]):
    print(idx, val)
"""
def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
                j +=1
    return X_indices

def sentences_to_indices_test(target):
    word_to_index = {}
    for idx, val in enumerate(["i", "like", "learning", "deep", "machine", "love", "smile", '´0.=']):
        word_to_index[val] = idx
    max_len = 4
    sentences = np.array(["I like deep learning", "deep ´0.= love machine", "machine learning smile"])
    indexes = target(sentences, word_to_index, max_len)
    print(indexes)
    assert type(indexes) == np.ndarray, "wrong type. use np arrays in the function"
    assert indexes.shape == (sentences.shape[0], max_len), "wrong shape of output matrix"
    assert np.allclose(indexes, [[0, 1, 3, 2],
                                 [3, 7, 5, 4],
                                 [4, 2, 6, 0]]), "Wrong values. Debug with the given examples"
    print("\033[92mAll tests passed")

#sentences_to_indices_test(sentences_to_indices)

"""X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1, word_to_index, max_len=5)
print("X1 = ", X1)
print("X1_indices = \n", X1_indices)"""

def pretrained_embedding_layer(word_to_vector_map, word_to_index):
    vocab_size = len(word_to_index) + 1
    any_word = list(word_to_vector_map.keys())[0]
    emb_dim = word_to_vector_map[any_word].shape[0]
    embedding_matrix = np.zeros([vocab_size, emb_dim])
    for word, idx in word_to_index.items():
        embedding_matrix[idx, :] = word_to_vector_map[word]
    emb_layer = Embedding(vocab_size, emb_dim, trainable=False)
    emb_layer.build((None, ))
    emb_layer.set_weights([embedding_matrix])
    return emb_layer

def pretrained_embedding_layer_test(target):
    word_to_vector_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 'a_n': [3, 4],
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0],
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]}
    for key in word_to_vector_map.keys():
        word_to_vector_map[key] = np.array(word_to_vector_map[key])

    word_to_index = {}
    for idx, val in enumerate(list(word_to_vector_map.keys())):
        word_to_index[val] = idx
    np.random.seed(1)
    embed_layer = target(word_to_vector_map, word_to_index)
    assert type(embed_layer) == Embedding, "Wrong type"
    assert embed_layer.input_dim == len(list(word_to_vector_map.keys())) + 1, "wrong input shape"
    assert embed_layer.output_dim == len(word_to_vector_map['a']), "wrong output shape"
    assert np.allclose(embed_layer.get_weights(), [[[ 3, 3], [ 3, 3], [ 2, 4], [ 3, 2], [ 3, 4],
                       [-2, 1], [-2, 2], [-1, 2], [-1, 1], [-1, 0],
                       [-2, 0], [-3, 0], [-3, 1], [-3, 2], [ 0, 0]]]), "wrong values"
    print("\033[92mALL TESTS PASSED!")

#pretrained_embedding_layer_test(pretrained_embedding_layer)

def emojify_V2(input_shape, word_to_vector_map, word_to_index):
    sentence_indices = Input(shape = input_shape, dtype='int32')
    embed_layer = pretrained_embedding_layer(word_to_vector_map, word_to_index)
    embeddings = embed_layer(sentence_indices)
    X = LSTM(units= 128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(units = 128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(5)(X)
    X = Activation('softmax')(X)

    model = Model(inputs = sentence_indices, outputs = X)
    return model

def emojify_V2_test(target):
    word_to_vector_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 'a_n': [3, 4],
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0],
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]}
    for key in word_to_vector_map.keys():
        word_to_vector_map[key] = np.array(word_to_vector_map[key])
    word_to_index = {}
    for idx, val in enumerate(list(word_to_vector_map.keys())):
        word_to_index[val] = idx
    maxLen = 4
    model = target((maxLen, ), word_to_vector_map, word_to_index)
    expectedModel = [['InputLayer', [(None, 4)], 0], ['Embedding', (None, 4, 2), 30], ['LSTM', (None, 4, 128), 67072, (None, 4, 2), 'tanh', True], ['Dropout', (None, 4, 128), 0, 0.5], ['LSTM', (None, 128), 131584, (None, 4, 128), 'tanh', False], ['Dropout', (None, 128), 0, 0.5], ['Dense', (None, 5), 645, 'linear'], ['Activation', (None, 5), 0]]
    comparator(summary(model), expectedModel)

#emojify_V2_test(emojify_V2)

model = emojify_V2((maxLen, ), word_to_vector_map, word_to_index)
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
model.fit(X_train_indices, Y_oh_train, epochs=100, batch_size= 32, shuffle=True)
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
loss, acc = model.evaluate(X_test_indices, Y_oh_test)
print()
print("test accuracy = ", str(acc))
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num!=Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())

x_test = np.array(["raise your hands motherfucker"])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))