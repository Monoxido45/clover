from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split


# defining score basic class
class Scores(ABC):
    def __init__(self, base_model, **kwargs):
        if base_model is not None:
            self.base_model = base_model(**kwargs)

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def compute(self, X_calib, y_calib):
        pass

    @abstractmethod
    def predict(self, X_test, cutoff):
        pass


# two main non conformity scores
# regression score
class RegressionScore(Scores):
    """
    Non conformity regression score
    --------------------------------------------------------
    base_model: point prediction model object
        Base-model that will be used to compute non-conformity scores
    """

    def fit(self, X, y):
        if self.base_model is not None:
            self.base_model.fit(X, y)
            return self
        else:
            return self

    def compute(self, X_calib, y_calib):
        if self.base_model is not None:
            pred = self.base_model.predict(X_calib)
            res = np.abs(pred - y_calib)
            return res
        else:
            return np.abs(y_calib)

    def predict(self, X_test, cutoff):
        pred_mu = self.base_model.predict(X_test)
        pred = np.vstack((pred_mu - cutoff, pred_mu + cutoff)).T
        return pred


# local rgression score
class LocalRegressionScore(Scores):
    """
    Non conformity regression local-variant score
    --------------------------------------------------------
    base_model: point prediction model object
        Base-model that will be used to compute non-conformity scores
    """

    def fit(self, X, y, mad_model_cte=False):
        # fitting regression to all training set
        self.base_model.fit(X, y)

        # training mad model
        if mad_model_cte:
            self.mad_model_cte = True
        else:
            self.mad_model_cte = False
            res_model = np.abs(y - self.base_model.predict(X))
            self.mad_model = clone(self.base_model).fit(X, res_model)

        return self

    def compute(self, X_calib, y_calib):
        pred_reg = self.base_model.predict(X_calib)
        if self.mad_model_cte:
            res = np.abs(pred_reg - y_calib)
            return res

        else:
            res_model = np.abs(y_calib - pred_reg)
            pred_mad = self.mad_model.predict(X_calib)

            # saving mad and vanilla res to objects if needed
            self.pred_mad = pred_mad
            self.vanilla_res = np.abs(pred_reg - y_calib)

            res = res_model / pred_mad
            return res

    def predict(self, X_test, cutoff):
        pred_mu = self.base_model.predict(X_test)
        if not self.mad_model_cte:
            pred_mad = self.mad_model.predict(X_test)
            pred = np.vstack(
                (pred_mu - (pred_mad * cutoff), pred_mu + (pred_mad * cutoff))
            ).T
        else:
            pred = np.vstack((pred_mu - cutoff, pred_mu + cutoff)).T
        return pred


# quantile score
# need to specify only base-model
class QuantileScore(Scores):
    """
    Non conformity quantile score
    --------------------------------------------------------
    base_model: Quantilic model object
        Base-model that will be used to compute non-conformity scores
    """

    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self

    def compute(self, X_calib, y_calib):
        pred = self.base_model.predict(X_calib)
        scores = np.column_stack((pred[:, 0] - y_calib, y_calib - pred[:, 1]))
        res = np.max(scores, axis=1)
        return res

    def predict(self, X_test, cutoff):
        quantiles = self.nc_score.base_model.predict(X_test)
        pred = np.vstack((quantiles[:, 0] - cutoff, quantiles[:, 1] + cutoff)).T
        return pred
