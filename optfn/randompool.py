import numpy as np
import numpy.random as rng
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression


class RandomPool(BaseEstimator, ClassifierMixin):
    def __init__(self, splits_count, clf_factory=None):
        self.splits_count = splits_count
        self.splits = None
        self.indexes = None
        self.regressor = None
        self.mean = None
        self.std = None
        self.clf_factory = clf_factory if clf_factory is not None else (lambda: LinearRegression())

    def fit(self, x, y):
        x, y = self._fix_xy(x, y, True)

        dim = x.shape[0]
        self.splits = rng.randn(3, dim * dim * self.splits_count, 1)
        i1 = np.repeat(np.arange(dim), dim * self.splits_count)
        i2 = np.tile(np.arange(dim), dim * self.splits_count)
        self.indexes = np.vstack((i1, i2))
        fit_x = self.get_fit_x(x)
        self.regressor = self.clf_factory()
        return self.regressor.fit(fit_x, y)

    def predict(self, x):
        x, y = self._fix_xy(x, None, False)
        fit_x = self.get_fit_x(x)
        return self.regressor.predict(fit_x)

    def predict_proba(self, x):
        x, y = self._fix_xy(x, None, False)
        fit_x = self.get_fit_x(x)
        return self.regressor.predict_proba(fit_x)

    def get_fit_x(self, x):
        split_res = self.splits[0]*x[self.indexes[0]] + self.splits[2] - self.splits[1]*x[self.indexes[1]]
        #split_res = (split_res - split_res.mean(1).reshape(-1, 1)) / split_res.std(1).reshape(-1, 1)
        return np.concatenate((x, split_res), axis=0).T

    def _fix_xy(self, x, y, recalc_norm):
        if x is not None:
            x = np.atleast_2d(x)
            x = x.T
            if recalc_norm:
                self.mean = x.mean(1).reshape(-1, 1)
                self.std = np.maximum(1e-8, x.std(1)).reshape(-1, 1)
            x = (x - self.mean) / self.std
        if y is not None:
            y = np.atleast_2d(y)
            y = y.reshape(-1)
        return x, y