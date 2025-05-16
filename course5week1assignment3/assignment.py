import IPython
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from music21 import *
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from outputs import *
from test_utils import *

from keras.src.layers import Dense, Activation, Input, LSTM, Reshape, Lambda, RepeatVector, Dropout
from keras.src.models import Model
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical

#IPython.display.Audio('30s_seq.wav')
X, Y, n_values, indices_values, chords = load_music_utils('original_metheny.mid')
print('number of training examples:', X.shape[0])
print('Tx (length) of the sequence:', X.shape[1])
print('total number of unique values:', n_values)
print('shape of X:', X.shape)
print('shape of Y:', Y.shape)
print('number of chords', len(chords))
n_a = 64
n_values = 90
reshaper = Reshape((1, n_values))
LSTM_cell = LSTM(n_a, return_state=True)
densor = Dense(n_values, activation='softmax')

def djmodel(Tx, LSTM_cell, densor, reshaper):
    n_values = densor.units
    n_a = LSTM_cell.units
    X = Input(shape=(Tx, n_values))
    a0 = Input(shape = (n_a, ), name='a0')
    c0 = Input(shape=(n_a, ), name='c0')
    a = a0
    c = c0
    outputs = []
    for t in range(Tx):
        x = X[:, t, :]
        x = reshaper(x)
        a, _, c = LSTM_cell(x, initial_state = [a, c])
        out = densor(a)
        outputs.append(out)
    model = Model(inputs = [X, a0, c0], outputs= outputs)
    return model

model = djmodel(Tx = 30, LSTM_cell=LSTM_cell, densor=densor, reshaper=reshaper)
output = summary(model)
comparator(output, djmodel_out)
model.summary()
