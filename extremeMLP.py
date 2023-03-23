import numpy as np
import sklearn.utils
import strlearn.metrics as metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier as mlp

from elm import ExtremeLearningMachine as elm


class ExtremeMLP(BaseEstimator, ClassifierMixin):
    """
        [ENG] A combination of ExtremeLearningMachine and MLP for data stream classification
        [PL]  Klasyfikator dla zadania klasyfikacji strumieni danych będący kombinacją ELM oraz MLP
    """

    def __init__(self, weighted=False, hidden_units=1000, C=pow(2, 5), delta=0.9, omikron=0.5, metric=metrics.geometric_mean_score_1):
        self.weighted = weighted
        self.hidden_units = hidden_units
        self.elm = elm(C, weighted, hidden_units)
        self.mlp = mlp(hidden_units)
        self.mlp_used = False
        self.delta = delta
        self.omikron = omikron
        self.metric = metric
        self.first_chunk = True

    def check_is_MLP_better(self, X, y):
        y_mlp = self.mlp.predict(X)
        y_elm = self.elm.predict(X)
        mlp_score = self.metric(y, y_mlp)
        elm_score = self.metric(y, y_elm)

        return mlp_score > elm_score

    def fit(self, X, y):
        self.elm.fit(X, y)
        self.mlp.fit(X, y)

    def partial_fit(self, X, y, classes=None):
        sklearn.utils.check_X_y(X, y)
        if not self.first_chunk:
            self.mlp_used = self.check_is_MLP_better(X, y)

        self.mlp.partial_fit(X, y, np.unique(y))
        self.elm.partial_fit(X, y)

        if self.first_chunk:
            self.first_chunk = False

    def predict(self, X):
        if self.mlp_used:
            return mlp.predict(X)
        else:
            return elm.predict(X)
