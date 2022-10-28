from __future__ import division

# basic libraries we are using
import numpy as np
from scipy import stats

# sklearn modules
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, clone

class KmeansSplit(BaseEstimator):
    '''
    Local K-means split
    ----------------------------------------------------------------
    nc_score: class or function
      Non conformity score of choosing with base model embeded on it. It can be specified by a class or function
    base_model: Base model to be fitted to non conformity score
    alpha: float between 0 and 1
      Miscoverage level of resulting prediction region
    base_model_type: Bool
      Boolean indicating wether interval prediction base_model is being used or not. Default is None
    quantile_type: string
      Which Quantile prediction algorithm should be fitted. For now it is only"GBQR"
    **kwargs: keyword arguments passed to fit base_model
    '''
    def __init__(self, nc_score, base_model, alpha, base_model_type = None, quantile_type = "GBQR", **kwargs):
        self.base_model = base_model
        self.base_model_type = base_model_type
        if ("Quantile" in str(nc_score)) or (base_model_type == True):
            self.nc_score = nc_score(self.base_model, coverage = alpha, **kwargs)
        else:
            self.nc_score = nc_score(self.base_model, **kwargs)
        self.alpha = alpha
        self.quantile_type = quantile_type

    def fit(self, X, y):
        '''
        Fit non conformity score to training samples
        --------------------------------------------------------
        X: Feature matrix
        y: label for training samples
        '''
        # saving minimum and maximum for y
        self.min_y, self.max_y = (np.min(y) - 3), (np.max(y) + 3)
        self.nc_score.fit(X, y)
        return self
    
    def update_limits(self, value_min, value_max):
        if value_min < self.min_y:
            self.min_y = value_min
        if value_max > self.max_y:
            self.max_y = value_max
    
    def calib(self, X_calib, y_calib, random_seed = 1250, tune_k = True, prop_k = np.arange(2, 11), **kwargs):
        '''
        Calibrate non conformity score using KmeansSplit
        --------------------------------------------------------
        X_calib: Feature matrix
        y_calib: label for training samples
        random_seed: int
            Random seed for K-means
        tune_k: boolean
            Wether the number of clustering should be tuned or not.
        prop_k: array of ints or int
            If tune_k = False, the proposed K to be used for splitting. If tune_k = True, it is the grid of k's to be evaluated
        **kwargs: keyword arguments to be passed to quantile gradient boosting ensemble
        '''
        res = self.nc_score.compute(X_calib, y_calib)
        # update min and maximum in y
        self.update_limits(np.min(y_calib), np.max(y_calib))

        # splitting calibration data into a training half and a validation half
        X_calib_train, X_calib_test, res_calib_train, res_calib_test = train_test_split(X_calib, res, test_size = 0.5,  random_state = random_seed)

        if self.quantile_type == "GBQR":
            # declaring gbqr ensemble
            self.qmodel = GradientBoostingQuantileEnsemble(**kwargs)

            # training gbqr model
            self.qmodel.fit(X_calib_train, res_calib_train)

            # obtaining scaled new X
            self.scaler = StandardScaler()
            new_X = self.scaler.fit_transform(self.qmodel.predict(X_calib_train))
            
            # predicting gbqr constructing test set to train k-means
            new_X_test = self.scaler.transform(self.qmodel.predict(X_calib_test))

            # obtaining best k-means model according to silhouette score
            current_sil = -1
            if tune_k and prop_k.shape[0] > 1:
                for k in prop_k:
                    labels = KMeans(n_clusters = k, random_state = random_seed, n_init = 30).fit(new_X_test).labels_
                    new_sil = silhouette_score(new_X, labels, metric = "euclidean")
                    if new_sil > current_sil:
                        current_sil = new_sil
                        current_k = k
                
                # fitting
                self.kmeans = KMeans(n_clusters = current_k, random_state = random_seed, n_init = 30).fit(new_X_test)
            else:
                self.kmeans = KMeans(n_clusters = prop_k, random_state = random_seed, n_init = 30).fit(new_X_test)


            # prediciting groups in the test set
            groups = self.apply(new_X_test)
            self.groups_idx = np.unique(groups)
            n_groups = self.groups_idx.shape[0]

            self.cutoffs = np.zeros(n_groups)
            for i in range(n_groups):
                self.cutoffs[i] = np.quantile(res_calib_test[groups == self.groups_idx[i]], q = 1 - self.alpha)

        return self.cutoffs
    
    def apply(self, X):
        return(self.kmeans.predict(X))

    def predict(self, X, length = 1500):
        '''
        Predict $1 - \alpha$ prediction region for each test sample using LocartSplit local cutoff points
        '''
        y_grid = np.linspace(self.min_y, self.max_y, length)
        intervals_list = []
        # first obtaining new quantile matrix from X
        new_X = self.scaler.transform(self.qmodel.predict(X))
        for i in range(X.shape[0]):
            # computing residual for all the grid
            res = self.nc_score.compute(X[i, :].reshape(1, -1), y_grid)
            
            # obtaining cutoff indexes and cutoff points according to choosed type of model
            cutoff_idx = np.where(self.groups_idx == self.apply(new_X[i, :].reshape(1, -1)))[0][0]

            # finding interval/region limits
            ident_int = np.diff((res <= self.cutoffs[cutoff_idx]) + 0)
            ident_idx = np.where(ident_int != 0)[0]

            # -1 indicates end of the invervals and 1 the beggining
            # if we start the identifier with -1, that means the first entry is the beggining
            if ident_int[ident_idx[0]] == -1:
                ident_idx = np.insert(ident_idx, 0, -1)
            # if we finish with 0, that means the last entry is the end
            if ident_int[ident_idx[-1]] == 1:
                ident_idx = np.append(ident_idx, y_grid.shape[0] - 1)

            # after turning the array even shaped we add one to the lower limit of intervals
            int_idx = ident_idx + np.tile(np.array([1, 0]), int(ident_idx.shape[0]/2))
            intervals_list.append(y_grid[int_idx])

        return intervals_list


# gradient boosting to compute several quantiles
class GradientBoostingQuantileEnsemble(BaseEstimator):
    def __init__(self, 
                 quantiles = [0.8, 0.85, 0.9, 0.95],
                 random_states = [750, 750, 750, 750],
                 loss = "quantile",
                 **kwargs):
        self.quantiles = quantiles
        self.loss = loss
        # declarando os modelos
        self.model_list = [GradientBoostingRegressor(loss = self.loss,
                                               alpha = quantiles[0],
                                               random_state = random_states[0], 
                                               **kwargs)]
        for i in range(1, len(self.model_list)):
            self.model_list.append(clone(GradientBoostingRegressor(loss = self.loss,
                                                     alpha = quantiles[i],
                                                     random_state = random_states[i],
                                                     **kwargs)))       
    
    def fit(self, X, y):
        for model in self.model_list:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        pred_matrix = np.zeros((X.shape[0], len(self.quantiles)))
        i = 0
        for model in self.model_list:
            pred_matrix[:, i] = model.predict(X)
            i += 1
        return pred_matrix
