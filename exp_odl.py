from __future__ import absolute_import, division, print_function, unicode_literals

import pdb
import time

import numpy as np
import tensorflow as tf
from data_helper import data_folder, file_list
import os
import pdb

from benchmark.onn import ONN

tf.random.set_seed(0)
freq = 100


def get_model(n_features_, n_classes_, n_layers_, learning_rate_):
    return ONN(n_features_, n_classes_, n_layers=n_layers_, learning_rate=learning_rate_)


# dataset = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
# print(dataset.shape)
# X = dataset[:, :-1]
# y = dataset[:, -1].astype(int)
file_name = 'susy'
data_file = np.load('/Users/AnhVu/Study/PhD/mypaper/online_self_supervised/data/{}.npz'.format(file_name))
X = data_file['x_train']
y = data_file['y_train']
y = np.argmax(y, axis=1)

print(X.shape)
print(y.shape)

# --------- params for model ------------ #
n_classes = np.unique(y).shape[0]
n_features = X.shape[1]
n_layers = 20
learning_rate = 0.01

print('n_classes = ', n_classes)
print('n_features = ', n_features)
# --------- params for model ------------ #

model = get_model(n_features, n_classes, n_layers, learning_rate)

epochs = range(1)
cnt = 0
res = [None] * (n_layers + 1)
tres = []

output_path = "result/{}/".format(type(model).__name__)
if not os.path.exists(output_path):
    os.makedirs(output_path)

f = open('{}/{}.csv'.format(output_path, file_name), "w+")
f.write(
    "learning evaluation instances,evaluation time (cpu seconds),model cost (RAM-Hours),classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),Kappa M Statistic (percent),model training instances,model serialized size (bytes)\n")

time_mark = time.time()
stime = 0
for i in range(X.shape[0]):
    print('instance {}'.format(i))

    inputs = tf.constant(X[i, :], shape=[1, n_features])
    pred = model.predict(inputs)
    pred = pred[0]
    if pred == y[i]:
        cnt += 1

    if (i + 1) % freq == 0:
        print('#{}'.format(i + 1))
        print('accuracy = {}'.format(cnt / (i + 1)))
        curtime = time.time() - time_mark
        print('time for 1000: ', curtime)
        time_mark = time.time()
        stime += curtime

        f.write(
            "{instances},{time},{ram_hours},{classified_instances},{accuracy},{a},{b},{c},{d},{e}\n".format(
                instances=i + 1,
                time=stime,
                ram_hours=-1,
                classified_instances=-1,
                accuracy=(cnt / (i + 1)), a=-1, b=-1, c=-1, d=-1, e=-1)
        )

        print(model.alphas)

    targets = tf.constant(y[i], shape=[1])
    # pdb.set_trace()

    model.partial_fit(inputs, targets)

print('#{}'.format(i + 1))
curtime = time.time() - time_mark
print('time for {}: '.format(freq), curtime)
stime += curtime

f.write(
    "{instances},{time},{ram_hours},{classified_instances},{accuracy},{a},{b},{c},{d},{e}\n".format(
        instances=i + 1,
        time=stime,
        ram_hours=-1,
        classified_instances=-1,
        accuracy=(cnt / (i + 1)), a=-1, b=-1, c=-1, d=-1, e=-1)
)

f.close()