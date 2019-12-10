from __future__ import absolute_import, division, print_function, unicode_literals

import pdb
import time

import numpy as np
import tensorflow as tf
from skmultiflow.drift_detection import ADWIN

from benchmark.onn import ONN

filepath = '/Users/AnhVu/Study/PhD/mypaper/online_deep_forest/data/csv/mnist_5_5_abrupt.csv'
dataset = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
print(dataset.shape)
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

print(X.shape)
print(y.shape)

n_classes = np.unique(y).shape[0]
n_features = X.shape[1]
print('n_classes = ', n_classes)
print('n_features = ', n_features)

n_layers = 20
learning_rate = 0.01


def get_model():
    return ONN(n_features, n_classes, n_layers=n_layers, n_hidden_units=100, learning_rate=learning_rate)


model = get_model()
adwin = ADWIN(delta=1e-10)

epochs = range(1)
cnt = 0
res = [None] * (n_layers + 1)
tres = []

s1 = time.time()
tf.config.experimental_run_functions_eagerly(False)

for i in range(X.shape[0]):

    inputs = tf.constant(X[i, :], shape=[1, n_features])
    pred = model.predict(inputs)
    pred = pred[0]
    if pred == y[i]:
        cnt += 1

    if i % 1000 == 0:
        print('#{}'.format(i))
        print(cnt / (i + 1))
        print('time for 1000: ', time.time() - s1)
        s1 = time.time()
        print(model.alphas)
        # pdb.set_trace()

    targets = tf.constant(y[i], shape=[1])

    old = adwin.estimation
    adwin.add_element(0.0 if pred == y[i] else 1.0)
    if adwin.detected_change():
        if adwin.estimation > old:
            print('Change detected')
            del model
            model = get_model()

    try:
        model.partial_fit(inputs, targets)
    except:
        pdb.set_trace()

# terr = np.sum(y == tres) / y.shape[0]
