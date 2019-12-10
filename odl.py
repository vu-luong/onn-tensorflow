from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
import numpy as np
from skmultiflow.drift_detection import ADWIN

from benchmark.onn import ONN
import tensorflow as tf


class OnlineDeepLearner(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, seed=0):
        """ OnlineDeepLearner constructor
        """
        self.seed = seed
        self.classes = None
        self.n_classes = None
        self.n_features = None
        self.adwin = ADWIN(delta=0.0001)
        self.learner = None

    def init_learner(self):
        """ Initialize onn
        """
        if self.n_classes is None or self.n_features is None:
            raise ValueError("Cannot initialize classifier with n_classes=None or n_features=None")

        self.learner = ONN(n_features=self.n_features, n_classes=self.n_classes,
                           n_hidden_units=10, beta=0.99,
                           learning_rate=1, s=0.2,
                           n_layers=0)

    def partial_fit(self, X_numpy, y_numpy, classes=None, sample_weight=None):

        if self.classes is None:
            if classes is None:
                raise ValueError("The first partial_fit call should pass all the classes.")
            else:
                self.classes = classes
                self.n_classes = len(classes)
                self.n_features = X_numpy.shape[1]
                self.init_learner()

        if self.classes is not None and classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError("The classes passed to the partial_fit function differ from those passed earlier.")

        X = tf.constant(X_numpy, shape=[X_numpy.shape[0], X_numpy.shape[1]])
        y = tf.constant(y_numpy, shape=[1])

        y_preds = self.predict(X)
        for i in range(y_preds.shape[0]):
            old = self.adwin.estimation
            self.adwin.add_element(0.0 if y_preds[i] == y[i] else 1.0)
            if self.adwin.detected_change():
                if self.adwin.estimation > old:
                    print('Change detected')
                    self.init_learner()

        self.learner.partial_fit(X, y)

    def predict(self, X):
        if self.classes is None:
            return np.zeros(X.shape[0])

        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def predict_proba(self, X_):
        if isinstance(X_, np.ndarray):
            X = tf.constant(X_, shape=[X_.shape[0], X_.shape[1]])
        else:
            X = X_

        prob = self.learner.predict_proba(X)
        return prob
