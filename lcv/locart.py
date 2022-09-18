from __future__ import division

# basic libraries to use
import numpy as np
from scipy import stats

# sklearn modules
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

import multiprocessing as mp


class LocartSplit(BaseEstimator):
    '''
    Local CART split
    ----------------------------------------------------------------
    nc_score: class or function
      Non conformity score of choosing with base model embeded on it. It can be specified by a class or function
    base_model: Base model to be fitted to non conformity score
    alpha: float between 0 and 1
      Miscoverage level of resulting prediction region
    base_model_type: Bool
      Boolean indicating wether interval prediction base_model is being used or not. Default is None
    cart_type: string
      Which CART algorithm should be fitted. For now it is between "CART" and "RFCDE"
    split_calib: bool
      Boolean indicating whether we split the calibration set into training and test set. Default is True
    **kwargs: keyword arguments passed to fit base_model
    '''
    def __init__(self, nc_score, base_model, alpha, base_model_type = None, cart_type = "CART", split_calib = True, **kwargs):
        self.base_model = base_model
        self.base_model_type = base_model_type
        if ("Quantile" in str(nc_score)) or (base_model_type == True):
            self.nc_score = nc_score(self.base_model, coverage = alpha, **kwargs)
        else:
            self.nc_score = nc_score(self.base_model, **kwargs)
        self.alpha = alpha
        self.cart_type = cart_type
        self.split_calib = split_calib

    def fit(self, X, y):
        '''
        Fit non conformity score to training samples
        --------------------------------------------------------
        X: Feature matrix
        y: label for training samples
        '''
        self.nc_score.fit(X, y)
        return self
    
    def calib(self, X_calib, y_calib, random_seed = 1250, **kwargs):
        '''
        Calibrate non conformity score using LocartSplit
        --------------------------------------------------------
        X_calib: Feature matrix
        y_calib: label for training samples
        random_seed: int
            Random seed for CART or RFCDE 
        '''
        res = self.nc_score.compute(X_calib, y_calib)
        # splitting calibration data into a training half and a prediction half
        if self.split_calib:
            X_calib_train, X_calib_test, res_calib_train, res_calib_test = train_test_split(X_calib, res, test_size = 0.5,  random_state = random_seed)
        else:
            X_calib_train, X_calib_test, res_calib_train, res_calib_test = X_calib, X_calib, res, res

        if self.cart_type == "CART":
            # training decision tree
            self.cart = DecisionTreeRegressor(random_state = random_seed,
            min_samples_leaf = 100).set_params(**kwargs).fit(X_calib_train, res_calib_train)
            # predicting leafs index
            leafs_idx = self.cart.apply(X_calib_test)
            unique_leafs = np.unique(leafs_idx)
            n_leafs = unique_leafs.shape[0]

            self.cutoffs = np.zeros(n_leafs)
            for i in range(n_leafs):
                self.cutoffs[i] = np.quantile(res_calib_test[leafs_idx == unique_leafs[i]], q = 1 - self.alpha)

        # TODO: implement RFCDE version

        return self.cutoffs
    
    def plot_locart(self):
        if self.cart_type == "CART":
            plot_tree(self.cart, filled=True)
            plt.title("Decision Tree fitted to non-conformity score")
            plt.show()


    def predict_coverage(self, X_test, y_test):
        '''
        Predict local coverage for each X acording to partitions obtained in LOCART
        --------------------------------------------------------
        X_test: feature matrix
        y_test: label for test samples
        '''
        res = self.nc_score.compute(X_test, y_test)
        if self.cart_type == "CART":
            leafs_idx = self.cart.apply(X_test)
            unique_leafs = np.unique(leafs_idx)
            coverage = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            coverage[i] = np.mean(res[leafs_idx == leafs_idx[i]] <= self.cutoffs[np.where(unique_leafs == leafs_idx[i])])
        return coverage       

    def predict(self, X):
        '''
        Predict $1 - \alpha$ non conformity score for each test sample using LocartSplit
        '''


