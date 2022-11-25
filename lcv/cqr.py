import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from lcv.scores import QuantileScore


class ConformalizedQuantileRegression(BaseEstimator):
    def __init__(self, base_model, alpha=0.05):
        self.base_model = base_model
        self.alpha = alpha

    def fit(self, X, y):
        self.base_model.fit(X, y)
        self.q_ = 0.0
        return self

    def calib(self, X, y):
        check_is_fitted(self)
        n = X.shape[0]
        pred = self.predict(X)
        scores = np.max(np.c_[pred[:, 0] - y, y - pred[:, 1]], axis=1)
        self.q_ = np.quantile(scores, q=np.ceil((n + 1) * (1 - self.alpha)) / n)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return np.c_[
            self.base_model.predict(X)[:, 0] - self.q_,
            self.base_model.predict(X)[:, 1] + self.q_,
        ]
