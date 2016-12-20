from __future__ import division, print_function, absolute_import

print('Start')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

from datetime import datetime
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
from tflearn.data_utils import samplewise_zero_center

from PIL import Image

def fix_columns(X, H, W, Colors):
    # input is (H, W, Colors, N)
    # output is (N, H, W, Colors)
    N = X.shape[-1]
    out = np.zeros((N, H, W, Colors), dtype=np.float32)
    for i in xrange(N):
        for j in xrange(Colors):
            out[i, :, :, j] = X[:, :, j, i]
    return out / 255

train_orig = loadmat('train_32x32.mat')  # cropped single digits
test_orig  = loadmat('test_32x32.mat')   # labels.  1-10, with 10=0

# fix the column ordering, and subtract the mean
Xtrain = samplewise_zero_center(fix_columns(train_orig['X'], 32, 32, 3))
Ytrain = to_categorical(train_orig['y'] - 1, 10)  # shift from 1-10 to 0-9

Xtest = samplewise_zero_center(fix_columns(test_orig['X'], 32, 32, 3))
Ytest = to_categorical(test_orig['y'] - 1, 10)  # shift from 1-10 to 0-9

print(tf.__version__)
print('Xtrain.shape: {}, Ytrain.shape: {}'.format(Xtrain.shape, Ytrain.shape))
print('Xtest.shape: {}, Ytest.shape: {}'.format(Xtest.shape, Ytest.shape))

network = input_data(shape=[None, 32, 32, 3]) # orig input 227x227
network = conv_2d(network, 48, 8, strides=3, activation='relu')     # orig output (55x55x96), filters 11x11, stride 4
network = max_pool_2d(network, 2, strides=1)                        # orig output (27x27x96), 3x3 stride 2
network = local_response_normalization(network)                     # orig output same as input
network = conv_2d(network, 126, 2, strides=1, activation='relu')    # orig output (23x23x256), filters 5x5, stride 1
network = max_pool_2d(network, 2, strides=1)                        # orig output (11x11x256), 3x3 stride 2
network = local_response_normalization(network)                     # orig output same as input
network = conv_2d(network, 192, 3, strides=1, activation='relu')    # orig output (9x9x384), filters 3x3 stride 1
network = conv_2d(network, 192, 2, strides=1, activation='relu')    # orig output (7x7x384), filters 3x3 stride 1
network = conv_2d(network, 128, 2, strides=1, activation='relu')    # orig output (5x5x256), filters x3 stride 1
network = max_pool_2d(network, 2, strides=1)                        # orig output (2x2x256), 3x3 stride 2
network = local_response_normalization(network)                     # orig output same as input
network = fully_connected(network, 1024, activation='tanh')         # orig input unrolled to 1024x1, orig output 4096x1
network = dropout(network, 0.5)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network,
                      optimizer='momentum',
                      loss='categorical_crossentropy',
                      learning_rate=0.01)

model = tflearn.DNN(network, checkpoint_path=None, tensorboard_verbose=0)
print("Defined model")

# Check to see if we can reload weights from a previous session
if os.path.isfile('SVHN.tfl'):
    print('Found previous weights, loading')
    model.load('SVHN.tfl')
else:
    print('No previous weights found')

max_save = 100
saved = 0
for i in xrange(0, len(Xtest)-1):
    prediction = (np.argmax(model.predict([Xtest[i]])[0]) + 1) % 10
    label = (np.argmax(Ytest[i]) + 1) % 10

    if prediction != label:
        saved += 1
        print('Prediction: {}, Label: {}'.format(prediction, label))
        X = test_orig['X'][:,:,:,i]
        img = Image.fromarray(X, 'RGB')
        fname = str(prediction) + '.' + str(label) + '.png'
        img.save('./misclassed/' + fname)
    
    if saved >= max_save:
        break

print('Done')