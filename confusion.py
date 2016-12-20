from __future__ import division, print_function, absolute_import

print('Starting imports')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

from scipy.io import loadmat
import itertools

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
from tflearn.data_utils import samplewise_zero_center

from sklearn.metrics import confusion_matrix

print('Imports done')

def fix_columns(X, H, W, Colors):
    # input is (H, W, Colors, N)
    # output is (N, H, W, Colors)
    N = X.shape[-1]
    out = np.zeros((N, H, W, Colors), dtype=np.float32)
    for i in xrange(N):
        for j in xrange(Colors):
            out[i, :, :, j] = X[:, :, j, i]
    return out / 255

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

print('Defining network')

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
print("Defined network")

# Check to see if we can reload weights from a previous session
if os.path.isfile('SVHN.tfl'):
    print('Found previous weights, loading')
    model.load('SVHN.tfl')
else:
    print('No previous weights found')

print('Predicting Xtest')
y_pred = model.predict(Xtest)
print('Predictions complete')

print('Mapping predictions and labels to argmax')
y_pred_classes = list(map((lambda y: np.argmax(y)), y_pred))
y_true_classes = list(map((lambda y: np.argmax(y)), Ytest))

print('Creating confusion')
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
np.set_printoptions(precision=2)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
print('Confusion complete')

plt.show()

print('Done')