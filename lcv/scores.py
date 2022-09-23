import numpy as np
from sklearn.base import BaseEstimator

# abstract classes
from abc import ABC, abstractmethod

# defining score basic class
class Scores(ABC):
    def __init__(self, base_model, **kwargs):
        self.base_model = base_model(**kwargs)

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def compute(self, X_calib, y_calib):
        pass    

# two main non conformity scores
# regression score
class RegressionScore(Scores):
    ''' 
    Non conformity regression score
    --------------------------------------------------------
    base_model: point prediction model object
        Base-model that will be used to compute non-conformity scores
    '''
    
    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self
    
    def compute(self, X_calib, y_calib):
        pred = self.base_model.predict(X_calib)
        res = np.abs(pred - y_calib)
        return res
    

# quantile score
# need to specify only base-model
class QuantileScore(Scores):
    ''' 
    Non conformity quantile score
    --------------------------------------------------------
    base_model: Quantilic model object
        Base-model that will be used to compute non-conformity scores
    '''
        
    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self
    
    def compute(self, X_calib, y_calib):
        pred = self.base_model.predict(X_calib)
        scores = np.column_stack((pred[:, 0] - y_calib, y_calib - pred[:, 1]))
        res = np.max(scores, axis = 1)
        return res