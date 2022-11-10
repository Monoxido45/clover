import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted


class QuantileGradientBoosting(BaseEstimator):
    def __init__(self, alpha=0.05, random_state=None, **kwargs):
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        self.lower_ = GradientBoostingRegressor(
            loss="quantile", alpha=self.alpha / 2, random_state=self.random_state
        )
        self.upper_ = GradientBoostingRegressor(
            loss="quantile", alpha=1 - self.alpha / 2, random_state=self.random_state
        )

        self.lower_.fit(X, y)
        self.upper_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return np.c_[self.lower_.predict(X), self.upper_.predict(X)]
