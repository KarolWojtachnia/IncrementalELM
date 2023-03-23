import numpy as np
from numpy.linalg import inv as inv
from numpy.linalg import multi_dot as multi_dot
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def relu(x):
    my_relu = np.maximum(x, 0)
    return my_relu


def sigmoid(x):
    my_sigmoid = 1 / (1 + np.exp(-x))
    return my_sigmoid


class ExtremeLearningMachine(BaseEstimator, ClassifierMixin):
    """
        Extreme Learning Machine for imbalanced learning
        Klasyfikator Extreme Learning Machine na potrzeby klasyfikacji niezbalansowanej
    """

    def __init__(self, C=pow(2, 5), weighted=False, hidden_units=1000, delta=0.9):
        self.output_weights = None
        self.W = None
        self.biases_ = None
        self.input_weights = None
        self.hidden_units = int(hidden_units)
        self.C = C
        self.weighted = weighted
        self.delta = delta

    def fit(self, X, y):
        new_y = y
        new_y[new_y == 0] = -1
        new_y[new_y > 0] = 1

        X_, y_ = check_X_y(X, new_y)
        training_samples = X_.shape[0]
        input_units = X_.shape[1]

        self.input_weights = np.random.normal(size=[self.hidden_units, input_units])
        self.biases_ = np.random.normal(size=[self.hidden_units])

        H = sigmoid(np.dot(X_, self.input_weights.T) + self.biases_)

        if self.weighted:
            self.__calculate_weights_matrix(training_samples, y_)
            self.output_weights = self.__compute_output_weights_weighted(H, training_samples, y_)

        else:
            self.output_weights = self.__compute_output_weights_NOT_weighted(H, training_samples, y_)
        return self

    def partial_fit(self, X, y, classes=None):
        y[y == 0] = -1
        if self.biases_ is None:
            self.fit(X, y)
        else:
            input_to_hidden_nodes = sigmoid(np.dot(X, self.input_weights.T) + self.biases_)
            if self.weighted:
                new_output_weights = self.__compute_output_weights_weighted(input_to_hidden_nodes, X.shape[0], y)
                self.output_weights = self.delta * self.output_weights + (1 - self.delta) * new_output_weights
            else:
                new_output_weights = self.__compute_output_weights_NOT_weighted(input_to_hidden_nodes, X.shape[0], y)
                self.output_weights = self.delta * self.output_weights + (1 - self.delta) * new_output_weights

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        prediction_H = sigmoid(np.dot(X, self.input_weights.T) + self.biases_)
        output = np.sign(np.dot(prediction_H, self.output_weights))
        output[output == -1] = 0
        return output

    def __compute_output_weights_NOT_weighted(self, H, training_samples, y_):
        if training_samples < self.hidden_units:
            inverse_complicated_part = inv(np.eye(training_samples) / self.C + multi_dot([H, H.T]))
            return multi_dot([H.T, inverse_complicated_part, y_])
        else:
            inverse_complicated_part = inv(np.eye(self.hidden_units) / self.C + multi_dot([H.T, H]))
            return multi_dot([inverse_complicated_part, H.T, y_])

    def __compute_output_weights_weighted(self, H, training_samples, y_):
        if training_samples < self.hidden_units:
            inverse_complicated_part = inv(np.eye(training_samples) / self.C + multi_dot([self.W, H, H.T]))
            return multi_dot([H.T, inverse_complicated_part, self.W, y_])
        else:
            inverse_complicated_part = inv(np.eye(self.hidden_units) / self.C + multi_dot([H.T, self.W, H]))
            return multi_dot([inverse_complicated_part, H.T, self.W, y_])

    def __calculate_weights_matrix(self, training_samples, y_):
        self.W = np.zeros((training_samples, training_samples))
        positive_samples = np.count_nonzero(y_ == 1)
        negative_samples = training_samples - positive_samples
        positive_ratio = 1 / np.count_nonzero
        negative_ratio = 1 / negative_samples
        for i in range(training_samples):
            if y_[i] == 1:
                self.W[i, i] = positive_ratio
            else:
                self.W[i, i] = negative_ratio
