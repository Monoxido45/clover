#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:17:56 2022

@author: kuben
"""
from __future__ import division

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.base import BaseEstimator

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
        
    
    def fit(self, X_calib, y_calib, random_seed = None, test_size = 0.4, **kwargs):
        # predicting each interval
        preds = self.conf.predict(X_calib, significance = self.alpha)
        
        # obtaining each w
        w = np.zeros(y_calib.shape[0])
        for i in range(y_calib.shape[0]):
            w[i] = int(y_calib[i] >= preds[i, 0] and y_calib[i] <= preds[i, 1])
        # regressing w on x using random forest model
        self.X_train, self.X_test, self.w_train, self.w_test = train_test_split(X_calib,w,
                                                                                test_size = test_size,
                                                                                random_state = random_seed)
        self.model = RandomForestClassifier(**kwargs).fit(self.X_train, self.w_train)
        return self
    
    def test(self, signif):
        r = self.model.predict_proba(self.X_test)[:, 1]
        t = np.mean((r  - (1 - self.alpha))**(2))
        return t


        
    
    
    
