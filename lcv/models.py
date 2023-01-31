import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.utils.validation import check_is_fitted
from lcv.scores import QuantileScore


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


# random forest quantile regression wrapper for conformal prediction
class QuantileForest(BaseEstimator):
    def __init__(self, alpha=0.05, **kwargs):
        self.alpha = alpha
        self.model = RandomForestQuantileRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        quantiles = [self.alpha / 2, 1 - self.alpha / 2]
        return self.model.predict(X, quantiles=quantiles)
