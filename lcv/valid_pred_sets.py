#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow import keras
from tensorflow.random import set_seed
from pygam import LogisticGAM

class Valid_pred_sets(BaseEstimator):
    '''
    Validation of conditional coverage. Here we test $H0: P(Y  \in R(X)|X) = 1 - \alpha$
    (This class is in Scikit-Learn style)
    '''
    def __init__(self,
                 conf,
                 alpha,
                 coverage_evaluator = "RF"):
        '''
        Input:  (i)  conf: already calibrated Conformal prediction model from nonconformist package or any model with predict method
                (ii) alpha: significance level for testing
                (iii) Which coverage evalutor to use: random forest (RF) or neural networks (nnet)
        '''
        
        self.conf = conf
        self.alpha = alpha
        self.coverage_evaluator = coverage_evaluator
    
    def fit(self, X_calib, y_calib, random_seed = 1250, test_size = 0.2, **kwargs):
        # predicting each interval
        preds = self.conf.predict(X_calib, significance = self.alpha)
        np.random.seed(random_seed)
        set_seed(random_seed + 2)
        # obtaining each w
        w = np.zeros(y_calib.shape[0])
        for i in range(y_calib.shape[0]):
            w[i] = int(y_calib[i] >= preds[i, 0] and y_calib[i] <= preds[i, 1])
        # splitting training and testing sets
        self.X_train, self.X_test, self.w_train, self.w_test = train_test_split(X_calib,w,
                                                                                test_size = test_size)
        
        # regressing w on x using select coverage evaluator model
        return self._init_coverage_evaluator(random_seed = random_seed, **kwargs)
    
    def _init_coverage_evaluator(self, random_seed, **kwargs):
        if self.coverage_evaluator == "RF":
            self.model = RandomForestClassifier(**kwargs).fit(self.X_train, self.w_train)
        elif self.coverage_evaluator == "GAM":
            self.model = LogisticGAM().gridsearch(self.X_train, self.w_train).fit(self.X_train, self.w_train)
        else:
            # setting tensorflow and numpy seeds to guarantee reproducibility
            np.random.seed(random_seed)
            set_seed(random_seed + 2)
            
            # defining nnet model
            self.model = keras.models.Sequential([
                keras.layers.Dense(units = 64, 
                                   input_dim = self.X_train.shape[1],
                                   activation = "selu",
                                   kernel_initializer = keras.initializers.LecunNormal(
                                       seed = random_seed),
                       bias_initializer='zeros'),
                keras.layers.Dropout(0.65),
                keras.layers.Dense(units = 32,
                                   activation = "selu",
                                   kernel_initializer = keras.initializers.LecunNormal(
                                       seed = random_seed),
                                   bias_initializer='zeros'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(units = 16,
                                   activation = "selu",
                                   kernel_initializer = keras.initializers.LecunNormal(
                                       seed = random_seed),
                                   bias_initializer='zeros'),
                keras.layers.Dropout(0.35),
                keras.layers.Dense(units = 1, activation = "sigmoid",
                      kernel_initializer = keras.initializers.LecunNormal(
                                       seed = random_seed),
                       bias_initializer='zeros')])
            
            # compiling
            self.model.compile(loss = "binary_crossentropy",
                               optimizer = keras.optimizers.Adam(learning_rate = 0.0025),
                               metrics=['accuracy'])
            # obtaining initial weights
            self.model_init_weights = self.model.get_weights()
            
            # fitting and adding early stopping
            self.es = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20)
            self.history = self.model.fit(self.X_train, self.w_train, validation_split = 0.33, 
                           epochs = 100, batch_size = 50, callbacks = [self.es], verbose = 0)
            
        return self
    
    
    def predict(self, X_test):
        if self.coverage_evaluator == "RF":
            pred = self.model.predict_proba(X_test)
            if len(pred[0]) == 1:
                return pred
            else:
                return pred[:, 1]
        elif self.coverage_evaluator == "GAM":
            pred = self.model.predict_proba(X_test)
            return pred
        else:
            pred = self.model.predict(X_test, verbose = 0).flatten(order = "C")
            return pred
    
    def retrain(self, X_train, new_w, X_test):
        if self.coverage_evaluator == "RF":
            model_temp = clone(self.model).fit(X_train, new_w)
            pred = model_temp.predict_proba(self.X_test)
            if len(pred[0]) == 1:
                new_r = pred
            else:
                new_r = pred[:, 1]
            return new_r
        elif self.coverage_evaluator == "GAM":
            model_temp = LogisticGAM().grid_search(X_train, new_w).fit(X_train, new_w)
            new_r = model_temp.predict_proba(self.X_test)
            return new_r
        else:
            model_temp = keras.models.clone_model(self.model)
            model_temp.set_weights(self.model_init_weights)
            model_temp.compile(loss = "binary_crossentropy",
                               optimizer = keras.optimizers.Adam(learning_rate = 0.0025),
                               metrics=['accuracy'])
            model_temp.fit(X_train, new_w, validation_split = 0.33, 
                           epochs = 100, batch_size = 50, 
                           callbacks = [self.es], 
                           verbose = 0)
            new_r = model_temp.predict(X_test, verbose = 0).flatten(order = "C")
            return new_r
        
    
    def check(self, i):
        return "Iteracao %d" %i
    
    def r_prob(self, X_grid):
        # predicting for each x in X_grid
        r = self.predict(X_grid)
        return r
    
    def monte_carlo_test(self, B = 1000, random_seed = 1250):      
        # observed statistic
        r = self.predict(self.X_test)
        t_obs = np.mean(np.abs(r  - (1 - self.alpha)))
                # computing monte-carlo samples
        np.random.seed(random_seed)
        t_b = np.zeros(B)
        
        # generating new weights from bernoulli
        for i in range(B):
            new_w = stats.binom.rvs(n = 1, p = 1 - self.alpha, size = self.w_train.shape[0])
            new_r = self.retrain(self.X_train, new_w, self.X_test)
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




    
    