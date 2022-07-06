#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:17:56 2022

@author: kuben
"""
from __future__ import division

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor

class Valid_pred_sets(BaseEstimator):
    '''
    Validation of conditional coverage. Here we test $H0: P(Y  \in R(X)|X) = 1 - \alpha$
    (This class is in Scikit-Learn style)
    '''
    def __init__(self,
                 conf,
                 alpha):
        '''
        Input:  (i)  conf_model: already calibrated Conformal prediction model from nonconformist package
                (ii) alpha: significance level for testing
        '''
        
        self.conf = conf
        self.alpha = alpha
        
    
    def fit(self, X_calib, y_calib, random_seed = 1250, test_size = 0.2, **kwargs):
        # predicting each interval
        preds = self.conf.predict(X_calib, significance = self.alpha)
        np.random.seed(random_seed)
        # obtaining each w
        w = np.zeros(y_calib.shape[0])
        for i in range(y_calib.shape[0]):
            w[i] = int(y_calib[i] >= preds[i, 0] and y_calib[i] <= preds[i, 1])
        # splitting training and testing sets
        self.X_train, self.X_test, self.w_train, self.w_test = train_test_split(X_calib,w,
                                                                                test_size = test_size)
        
        # regressing w on x using random forest model
        self.model = RandomForestClassifier(**kwargs).fit(self.X_train, self.w_train)
        return self
    
    def check(self, i):
        return "Iteracao %d" %i
    
    def r_prob(self, X_grid):
        # predicting for each x in X_grid
        r = self.model.predict_proba(X_grid)[:, 1]
        return r
    
    def monte_carlo_test(self, B = 1000, random_seed = 1250, **kwargs):      
        # observed statistic
        r = self.model.predict_proba(self.X_test)[:, 1]
        t_obs = np.mean(np.abs(r  - (1 - self.alpha)))
        
        # computing monte-carlo samples
        np.random.seed(random_seed)
        t_b = np.zeros(B)
        
        # generating new weights from bernoulli
        for i in range(B):
            self.new_w = stats.binom.rvs(n = 1, p = 1 - self.alpha, size = self.w_train.shape[0])
            model_temp = RandomForestClassifier(**kwargs).fit(self.X_train, self.new_w)
            self.pred = model_temp.predict_proba(self.X_test)
            if len(self.pred[0]) == 1:
                new_r = self.pred
            else:
                new_r = self.pred[:, 1]
            t_b[i] = np.mean(np.abs(new_r - (1 - self.alpha)))
        
        # computing p-value from the proportion of generated t's larger than the observed t
        p_value = (t_b > t_obs).mean()
        return {"p-value":p_value, "Observed statistic":t_obs}


# creating a adapter class to quantile regression in python which outputs a matrix of predictions
 
class LinearQuantileRegression(BaseEstimator):
    def __init__(self, 
                 coverage = 0.05, 
                 alpha = 0, 
                  fit_intercept = True, 
                  solver = 'highs', 
                  solver_options = None):
        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])
        self.lower = QuantileRegressor()
        self.upper = clone(QuantileRegressor())
    
    def fit(self, X, y):
        # creating upper and lower fit according to specified coverage
        quantiles = [self.coverage/2, 1 - self.coverage/2]
        self.lower.set_params(quantile = quantiles[0],
                              alpha = self.alpha,
                              fit_intercept = self.fit_intercept,
                              solver = self.solver,
                              solver_options = self.solver_options)
        self.lower.fit(X, y)
        
        self.upper.set_params(quantile = quantiles[1],
                              alpha = self.alpha,
                              fit_intercept = self.fit_intercept,
                              solver = self.solver,
                              solver_options = self.solver_options)
        
        self.upper.fit(X, y)
        return self
    
    def predict(self, X, **kwargs):
        # predicting for both lower and upper and then concatenating in numpy matrix
        lower = self.lower.predict(X)
        upper = self.upper.predict(X)
        interval = np.vstack((lower, upper)).T
        return interval
        
# gradient boosting quantile regression for conformal prediction
class GradientBoostingQuantileRegression(BaseEstimator):
    def __init__(self, 
                 coverage = 0.05,
                 loss = "quantile",
                 **kwargs):
        self.coverage = coverage
        self.loss = loss
        quantiles = [self.coverage/2, 1 - self.coverage/2]       
        self.lower = GradientBoostingRegressor(loss = self.loss,
                                               alpha = quantiles[0], 
                                               **kwargs)
        self.upper = clone(GradientBoostingRegressor(loss = self.loss,
                                                     alpha = quantiles[1], 
                                                     **kwargs))
    
    def fit(self, X, y):
        self.lower.fit(X, y)
        self.upper.fit(X, y)
        return self
    
    def predict(self, X, **kwargs):
        lower = self.lower.predict(X)
        upper = self.upper.predict(X)
        interval = np.vstack((lower, upper)).T
        return interval

# random forest boosting quantile regression form conformal prediction
class RandomForestQuantileRegression(BaseEstimator):
    def __init__(self,
                 coverage = 0.05,
                 **kwargs):
        self.coverage = coverage
        self.model = RandomForestRegressor(**kwargs)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        cart_data = np.zeros((X.shape[0], len(self.model.estimators_)))
        i = 0
        
        for cart in self.model.estimators_:
            cart_data[:, i] = cart.predict(X)
            i += 1
            
        quantiles = [self.coverage/2, 1 - self.coverage/2]  
        lower = np.quantile(cart_data, quantiles[0], axis = 1)
        upper = np.quantile(cart_data, quantiles[1], axis = 1)
        intervals = np.vstack((lower, upper)).T
        
        return intervals




    
    