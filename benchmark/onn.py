from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import pdb


class ONN(tf.keras.Model):
    def __init__(self, n_features, n_classes,
                 n_hidden_units=10, beta=0.99,
                 learning_rate=0.01, s=0.2,
                 n_layers=20):
        super(ONN, self).__init__(name='ONN')
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.beta = beta
        self.s = s
        self.learning_rate = learning_rate
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.temp = tf.Variable(np.array([1., 2., 3., np.nan]))

        self.hidden_layers = []
        self.output_layers = []

        for i in range(n_layers):
            self.hidden_layers.append(
                Dense(n_hidden_units)
            )

        for i in range(n_layers + 1):
            self.output_layers.append(
                Dense(n_classes)
            )

        self.alphas = tf.Variable(
            np.ones(n_layers + 1) * (1. / (n_layers + 1)),
            dtype=tf.float32
        )

    def __call__(self, inputs):
        hidden_connections = []
        x = inputs
        hidden_connections.append(x)

        for i in range(self.n_layers):
            hidden_connections.append(
                tf.nn.relu(
                    self.hidden_layers[i](
                        hidden_connections[i]
                    )
                )
            )

        output_class = []
        for i in range(self.n_layers + 1):
            output_class.append(
                self.output_layers[i](
                    hidden_connections[i]
                )
            )

        return output_class

    @tf.function
    def loss(self, x, y):
        output_class = self(x)
        losses = []
        for i in range(len(output_class)):
            losses.append(
                # tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output_class[i])[0]
                self.loss_obj(y_true=y, y_pred=output_class[i])
                # self.temp
            )

            tf.debugging.check_numerics(losses[i],
                                        "loss {} contains nan \n {} \n {}".format(i, y, output_class[i]))

            # try:
            #     tf.debugging.check_numerics(self.temp, "loss {} contains nan".format(i))
            #     # raise Exception('hehe')
            # except Exception:
            #     # print('den day roi')
            #     # pdb.set_trace()
            #     # pass
            #     # raise Exception('quit')
            #     tf.print("loss {} contains nan".format(i))
            #     tf.print("y = ", y)
            #     tf.print("pred = ", output_class[i])
            #     raise Exception('quit')

        return losses

    def predict_proba(self, inputs):
        outputs = self(inputs)

        probs = np.zeros(self.n_classes)
        for k in range(len(outputs)):
            t = self.alphas[k].numpy() * tf.nn.softmax(outputs[k])
            cur_prob = t.numpy()
            probs = probs + cur_prob

        probs = probs / np.sum(probs)
        return probs

    def predict(self, inputs):
        prob = self.predict_proba(inputs)

        preds = np.argmax(prob, axis=1)
        return preds

    @tf.function
    def partial_fit(self, inputs, targets):
        with tf.GradientTape(persistent=True) as tape:
            losses = self.loss(inputs, targets)

        # layer 0
        dV0_weight, dV0_bias = tape.gradient(
            losses[0], self.output_layers[0].variables
        )
        # update weight
        self.output_layers[0].variables[0].assign_sub(
            self.alphas[0] * self.learning_rate * dV0_weight
        )
        # update bias
        self.output_layers[0].variables[1].assign_sub(
            self.alphas[0] * self.learning_rate * dV0_bias
        )

        w = []
        b = []

        # layer 1,2,...,L
        for i in range(1, len(losses)):
            variables = []
            for j in range(i):
                variables.append(self.hidden_layers[j].variables)

            variables.append(self.output_layers[i].variables)

            grads = tape.gradient(
                losses[i], variables
            )

            # 1. For output layer
            dVi_weight, dVi_bias = grads[-1]

            # update weight
            self.output_layers[i].variables[0].assign_sub(
                self.alphas[i] * self.learning_rate * dVi_weight
            )
            # update bias
            self.output_layers[i].variables[1].assign_sub(
                self.alphas[i] * self.learning_rate * dVi_bias
            )

            # 2. For hidden layer
            for j in range(i):
                if len(w) < j + 1:
                    w.append(self.alphas[i] * grads[j][0])
                    b.append(self.alphas[i] * grads[j][1])
                else:
                    w[j] = w[j] + self.alphas[i] * grads[j][0]
                    b[j] = b[j] + self.alphas[i] * grads[j][1]

        for i in range(len(losses) - 1):
            self.hidden_layers[i].variables[0].assign_sub(
                self.learning_rate * w[i]
            )

            self.hidden_layers[i].variables[1].assign_sub(
                self.learning_rate * b[i]
            )

        for i in range(len(losses)):
            # pdb.set_trace()
            self.alphas[i].assign(
                self.alphas[i] *
                tf.math.pow(
                    self.beta, losses[i]
                )
            )

            self.alphas[i].assign(
                tf.math.maximum(
                    self.alphas[i],
                    #                 self.s / self.n_layers
                    0.01
                )
            )

        z_t = tf.math.reduce_sum(self.alphas)
        tf.debugging.check_numerics(z_t, "z_t is nan")
        self.alphas.assign(self.alphas / z_t)
        tf.debugging.check_numerics(self.alphas, "alphas contains nan")
