from __future__ import division

# basic libraries to use
import numpy as np
from scipy import stats

# sklearn modules
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from .scores import QuantileScore

# uniform binning
import itertools as it

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
    
    def calib(self, X_calib, y_calib, random_seed = 1250, prune_tree = True, prune_seed = 780, **kwargs):
        '''
        Calibrate non conformity score using LocartSplit
        --------------------------------------------------------
        X_calib: Feature matrix
        y_calib: label for training samples
        random_seed: int
            Random seed for CART or RFCDE 
        prune_tree: boolean
            Wether the tree should be pruned or not.
        prune_seed: int
            If prune_tree = True, random seed for data splitting to prune    
        '''
        res = self.nc_score.compute(X_calib, y_calib)
        # splitting calibration data into a training half and a validation half to prune the tree
        X_calib_train, X_calib_test, res_calib_train, res_calib_test = train_test_split(X_calib, res, test_size = 0.5,  random_state = random_seed)

        if self.cart_type == "CART":
            # declaring decision tree
            self.cart = DecisionTreeRegressor(random_state = random_seed,
            min_samples_leaf = 100).set_params(**kwargs)
            # obtaining optimum alpha to prune decision tree
            if prune_tree:
                X_train_prune, X_test_prune, res_train_prune, res_test_prune = train_test_split(X_calib, res, test_size = 0.5, 
                random_state = prune_seed)
                optim_ccp = self.prune_tree(X_train_prune, X_test_prune, res_train_prune, res_test_prune, random_state = 1250)
                # pruning decision tree
                self.cart.set_params(ccp_alpha=optim_ccp)

            # fitting and predicting leaf labels
            if self.split_calib:
                self.cart.fit(X_calib_train, res_calib_train)
                leafs_idx = self.cart.apply(X_calib_test)
            else:
                self.cart.fit(X_calib, res)
                leafs_idx = self.cart.apply(X_calib)

            unique_leafs = np.unique(leafs_idx)
            n_leafs = unique_leafs.shape[0]

            self.cutoffs = np.zeros(n_leafs)
            for i in range(n_leafs):
                if self.split_calib:
                    self.cutoffs[i] = np.quantile(res_calib_test[leafs_idx == unique_leafs[i]], q = 1 - self.alpha)
                else:
                    self.cutoffs[i] = np.quantile(res[leafs_idx == unique_leafs[i]], q = 1 - self.alpha)

        # TODO: implement RFCDE version

        return self.cutoffs
    
    def prune_tree(self, X_train, X_valid, res_train, res_valid, **kwargs):
        prune_path = self.cart.cost_complexity_pruning_path(X_train, res_train)
        ccp_alphas = prune_path.ccp_alphas
        current_loss = 1000
        # cross validation by data splitting to choose alphas
        for ccp_alpha in ccp_alphas:
            preds_ccp = DecisionTreeRegressor(min_samples_leaf = 100,  ccp_alpha=ccp_alpha, 
            **kwargs).fit(X_train, res_train).predict(X_valid)
            loss_ccp = mean_squared_error(res_valid, preds_ccp)
            if loss_ccp < current_loss:
                current_loss = loss_ccp
                optim_ccp = ccp_alpha

        return optim_ccp       

        

            

    # uniform binning methods
    def uniform_binning(self, X_calib, y_calib):
        # obtaining the residuals
        res = self.nc_score.compute(X_calib, y_calib)
        # generating uniform binning of the feature space based on the locart size
        num_partitions = int(np.ceil(len(self.cutoffs)**(1/X_calib.shape[1])))

        # partitioning each x using quantiles
        alphas = np.arange(1, (num_partitions + 1))/num_partitions
        quantiles = np.quantile(X_calib, q = alphas, axis = 0)
        # increasing 0.5 in the maximum to avoid X's above the maximum in testing
        quantiles[(num_partitions -1), :] = quantiles[(num_partitions -1), :] + 0.5

        # splitting the quantile array into k num_partitions slices
        q_split = np.split(quantiles.T, X_calib.shape[1], axis = 0)
        q_split = [i.reshape(-1) for i in q_split]

        # obtaining all possible iterations of quantiles and grouping them into rows
        self.unif_intervals = np.array(np.meshgrid(*q_split)).T.reshape(-1, X_calib.shape[1])

        # obtaining each partition index for calibration data
        int_idx = np.zeros(X_calib.shape[0])
        for i in range(X_calib.shape[0]):
            int_idx[i] = np.where(np.all(X_calib[i, :] <= self.unif_intervals, axis = 1))[0][0] + 1
        unique_int = np.unique(int_idx)

        # after splitting, obtaining uniform cutoffs
        self.unif_cutoffs = np.zeros(int(num_partitions))
        for i in range(int(num_partitions)):
            self.unif_cutoffs[i] = np.quantile(res[int_idx == unique_int[i]], q = 1 - self.alpha)
        return self.unif_cutoffs
    
    def predict_coverage_uniform(self, X_test, y_test, marginal = False):
        res = self.nc_score.compute(X_test, y_test)

        int_idx = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            int_idx[i] = np.where(np.all(X_test[i, :] <= self.unif_intervals, axis = 1))[0][0] + 1
        unique_int = np.unique(int_idx)
        coverage = np.zeros(X_test.shape[0])

        if not marginal:
            for i in range(X_test.shape[0]):
                coverage[i] = np.mean(res[int_idx == int_idx[i]] <= self.cutoffs[np.where(unique_int == int_idx[i])])
        else:
            for i in range(X_test.shape[0]):
                coverage[i] = (res[i] <= self.cutoffs[np.where(unique_int == int_idx[i])])
        return coverage

    
    def plot_locart(self):
        if self.cart_type == "CART":
            plot_tree(self.cart, filled=True)
            plt.title("Decision Tree fitted to non-conformity score")
            plt.show()


    def predict_coverage(self, X_test, y_test, marginal = False):
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
        if not marginal:
            coverage = np.zeros(X_test.shape[0])
            for i in range(X_test.shape[0]):
                coverage[i] = np.mean(res[leafs_idx == leafs_idx[i]] <= self.cutoffs[np.where(unique_leafs == leafs_idx[i])])
        else:
            coverage  = np.zeros(X_test.shape[0])
            for i in range(X_test.shape[0]):
                coverage[i] = res[i] <= self.cutoffs[np.where(unique_leafs == leafs_idx[i])]
        return coverage 


    def predict_mean_coverage(self, X_test,  y_test):
        coverage = self.predict_coverage(X_test, y_test)
        return np.mean(np.abs(coverage - (1 - self.alpha)))    

    def predict(self, X):
        '''
        Predict $1 - \alpha$ prediction region for each test sample using LocartSplit local cutoff points
        '''
        


class QuantileSplit(BaseEstimator):
    def __init__(self, base_model, alpha,**kwargs):
        self.base_model = base_model
        self.nc_score = QuantileScore(self.base_model, coverage = alpha, **kwargs)
        self.alpha = alpha
    
    def fit(self, X_train, y_train):
        self.nc_score.fit(X_train, y_train)
        return self
    
    def calibrate(self, X_calib, y_calib):
        res = self.nc_score.compute(X_calib, y_calib)
        self.cutoff = np.quantile(res, q = 1 - self.alpha)
        return None
    
    def predict(self, X_test):
        quantiles = self.nc_score.base_model.predict(X_test)
        pred = np.vstack((quantiles[:, 0] - self.cutoff, quantiles[:, 1] + self.cutoff)).T
        return pred




        

