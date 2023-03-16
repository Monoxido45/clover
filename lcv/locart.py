from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, clone
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
import inspect

from lcv.scores import LocalRegressionScore, RegressionScore, QuantileScore


class LocartSplit(BaseEstimator):
    """
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
    """

    def __init__(
        self,
        nc_score,
        base_model,
        alpha,
        base_model_type=None,
        cart_type="CART",
        split_calib=True,
        **kwargs
    ):

        self.base_model_type = base_model_type
        if ("Quantile" in str(nc_score)) or (base_model_type == True):
            self.nc_score = nc_score(base_model, alpha=alpha, **kwargs)
        else:
            self.nc_score = nc_score(base_model, **kwargs)

        self.base_model = self.nc_score.base_model
        self.alpha = alpha
        self.cart_type = cart_type
        self.split_calib = split_calib

    def fit(self, X, y, mad_model_cte=False):
        """
        Fit non conformity score to training samples
        --------------------------------------------------------
        X: Feature matrix
        y: label for training samples
        """
        fun_args = inspect.getfullargspec(self.nc_score.fit).args
        if "mad_model_cte" in fun_args:
            self.nc_score.fit(X, y, mad_model_cte=mad_model_cte)
        else:
            self.nc_score.fit(X, y)
        return self

    def calib(
        self,
        X_calib,
        y_calib,
        random_seed=1250,
        prune_tree=True,
        prune_seed=780,
        cart_train_size=0.5,
        random_projections=False,
        m=1000,
        projections_seed=1250,
        **kwargs
    ):
        """
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
        """
        res = self.nc_score.compute(X_calib, y_calib)

        # splitting calibration data into a training half and a validation half
        if self.split_calib:
            (
                X_calib_train,
                X_calib_test,
                res_calib_train,
                res_calib_test,
            ) = train_test_split(
                X_calib, res, test_size=1 - cart_train_size, random_state=random_seed
            )
            if random_projections:
                self.rp = True
                np.random.seed(projections_seed)
                self.S_matrix = np.random.standard_normal(
                    (m, X_calib_train.shape[1])
                ) / np.sqrt(m)
                self.rp_scaler = StandardScaler()
                X_calib_train = self.add_random_projections(
                    self.rp_scaler.fit_transform(X_calib_train)
                )
                X_calib_test = self.add_random_projections(
                    self.rp_scaler.transform(X_calib_test)
                )
            else:
                self.rp = False
        else:
            if random_projections:
                self.rp = True
                np.random.seed(projections_seed)
                self.S_matrix = np.random.standard_normal(
                    (m, X_calib_train.shape[1])
                ) / np.sqrt(m)
                self.rp_args = [m, projections_seed]
                self.rp_scaler = StandardScaler()
                X_calib = self.add_random_projections(
                    self.rp_scaler.fit_transform(X_calib)
                )
            else:
                self.rp = False

        if self.cart_type == "CART":
            # declaring decision tree
            self.cart = DecisionTreeRegressor(
                random_state=random_seed, min_samples_leaf=300
            ).set_params(**kwargs)
            # obtaining optimum alpha to prune decision tree
            if prune_tree:
                if self.split_calib:
                    (
                        X_train_prune,
                        X_test_prune,
                        res_train_prune,
                        res_test_prune,
                    ) = train_test_split(
                        X_calib_train,
                        res_calib_train,
                        test_size=0.5,
                        random_state=prune_seed,
                    )
                else:
                    (
                        X_train_prune,
                        X_test_prune,
                        res_train_prune,
                        res_test_prune,
                    ) = train_test_split(
                        X_calib, res, test_size=0.5, random_state=prune_seed,
                    )

                optim_ccp = self.prune_tree(
                    X_train_prune, X_test_prune, res_train_prune, res_test_prune
                )
                # pruning decision tree
                self.cart.set_params(ccp_alpha=optim_ccp)

            # fitting and predicting leaf labels
            if self.split_calib:
                self.cart.fit(X_calib_train, res_calib_train)
                leafs_idx = self.cart.apply(X_calib_test)
            else:
                self.cart.fit(X_calib, res)
                leafs_idx = self.cart.apply(X_calib)

            self.leaf_idx = np.unique(leafs_idx)
            n_leafs = self.leaf_idx.shape[0]

            self.cutoffs = np.zeros(n_leafs)
            for i in range(n_leafs):
                if self.split_calib:
                    self.cutoffs[i] = np.quantile(
                        res_calib_test[leafs_idx == self.leaf_idx[i]], q=1 - self.alpha
                    )
                else:
                    self.cutoffs[i] = np.quantile(
                        res[leafs_idx == self.leaf_idx[i]], q=1 - self.alpha
                    )

        # TODO: implement RFCDE version

        return self.cutoffs

    def make_random_projections(self, X):
        projections = np.dot(X, self.S_matrix.transpose())
        return np.concatenate((X, projections), axis=1)

    def prune_tree(self, X_train, X_valid, res_train, res_valid):
        prune_path = self.cart.cost_complexity_pruning_path(X_train, res_train)
        ccp_alphas = prune_path.ccp_alphas
        current_loss = float("inf")
        # cross validation by data splitting to choose alphas
        for ccp_alpha in ccp_alphas:
            preds_ccp = (
                clone(self.cart)
                .set_params(ccp_alpha=ccp_alpha)
                .fit(X_train, res_train)
                .predict(X_valid)
            )
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
        num_partitions = int(np.floor(len(self.cutoffs) ** (1 / X_calib.shape[1])))

        # partitioning each x using quantiles
        alphas = np.arange(1, (num_partitions + 1)) / num_partitions
        quantiles = np.quantile(X_calib, q=alphas, axis=0)
        # increasing 0.5 in the maximum to avoid X's above the maximum in testing
        quantiles[(num_partitions - 1), :] = quantiles[(num_partitions - 1), :] + 0.5

        # splitting the quantile array into k num_partitions slices
        q_split = np.split(quantiles.T, X_calib.shape[1], axis=0)
        q_split = [i.reshape(-1) for i in q_split]

        # obtaining all possible iterations of quantiles and grouping them into rows
        self.unif_intervals = np.array(np.meshgrid(*q_split)).T.reshape(
            -1, X_calib.shape[1]
        )

        # obtaining each partition index for calibration data
        int_idx = self.uniform_apply(X_calib)
        self.cartesian_ints = np.unique(int_idx)

        # after splitting, obtaining uniform cutoffs
        self.unif_cutoffs = np.zeros(int(num_partitions))
        for i in range(int(num_partitions)):
            self.unif_cutoffs[i] = np.quantile(
                res[int_idx == self.cartesian_ints[i]], q=1 - self.alpha
            )
        return self.unif_cutoffs

    def uniform_apply(self, X):
        int_idx = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            int_idx[i] = (
                np.where(np.all(X[i, :] <= self.unif_intervals, axis=1))[0][0] + 1
            )
        return int_idx

    def plot_locart(self):
        if self.cart_type == "CART":
            plot_tree(self.cart, filled=True)
            plt.title("Decision Tree fitted to non-conformity score")
            plt.show()

    def predict(self, X, type_model="CART"):
        """
        Predict $1 - \alpha$ prediction region for each test sample using LocartSplit local cutoff points
        """
        # identifying cutoff point
        if type_model == "CART":
            if self.rp:
                X = self.make_random_projections(self.rp_scaler.transform(X))

            leaves_idx = self.cart.apply(X)
            # obtaining order of leaves
            cutoffs = self.cutoffs[
                st.rankdata(leaves_idx.astype(int), method="dense") - 1
            ]
            pred = self.nc_score.predict(X, cutoffs)

        elif type_model == "euclidean":
            idx = self.uniform_apply(X)
            cutoffs = self.unif_cutoffs[
                st.rankdata(idx.astype(int), method="dense") - 1
            ]
            pred = self.nc_score.predict(X, cutoffs)

        return pred


class QuantileSplit(BaseEstimator):
    def __init__(self, base_model, alpha, **kwargs):
        self.base_model = base_model
        self.nc_score = QuantileScore(self.base_model, alpha=alpha, **kwargs)
        self.alpha = alpha

    def fit(self, X_train, y_train):
        self.nc_score.fit(X_train, y_train)
        return self

    def calibrate(self, X_calib, y_calib):
        res = self.nc_score.compute(X_calib, y_calib)
        self.cutoff = np.quantile(res, q=1 - self.alpha)
        return None

    def predict(self, X_test):
        return self.nc_score.predict(X_test, self.cutoff)


# Local regression split proposed by Lei et al
class LocalRegressionSplit(BaseEstimator):
    def __init__(self, base_model, alpha, **kwargs):
        self.base_model = base_model
        self.nc_score = LocalRegressionScore(self.base_model, **kwargs)
        self.alpha = alpha

    def fit(self, X_train, y_train, mad_model_cte=False):
        # fitting the base model
        self.nc_score.fit(X_train, y_train, mad_model_cte=mad_model_cte)
        return self

    def calibrate(self, X_calib, y_calib):
        res = self.nc_score.compute(X_calib, y_calib)
        self.cutoff = np.quantile(res, q=1 - self.alpha)
        return None

    def predict(self, X_test):
        return self.nc_score.predict(X_test, self.cutoff)


# Mondrian split method proposed by Bostrom et al
class MondrianRegressionSplit(BaseEstimator):
    def __init__(self, base_model, alpha, k=10, **kwargs):
        self.base_model = base_model
        self.k = k
        self.nc_score = RegressionScore(self.base_model, **kwargs)
        self.alpha = alpha

    def fit(self, X_train, y_train, **kwargs):
        # fitting the base model
        self.nc_score.fit(X_train, y_train)
        # training RandomForestRegressor for difficulty estimation if base model is not RandomForest
        if not isinstance(self.nc_score.base_model, RandomForestRegressor):
            self.dif_model = RandomForestRegressor(**kwargs).fit(X_train, y_train)
        else:
            self.dif_model = deepcopy(self.nc_score.base_model)

        return self

    def calibrate(self, X_calib, y_calib, random_state=1250):
        # making the split
        X_score, X_final, y_score, y_final = train_test_split(
            X_calib, y_calib, test_size=0.5, random_state=random_state
        )

        # computing the difficulty score for each X_score
        pred_dif = self.compute_difficulty(X_score)

        # computing vanilla score in held out data
        res = self.nc_score.compute(X_final, y_final)

        # now making local partitions based on variance percentile
        # binning into k percentiles
        alphas = np.arange(1, self.k) / self.k
        self.mondrian_quantiles = np.quantile(pred_dif, q=alphas, axis=0)

        # iterating percentiles to obtain local cutoffs
        # first obtaining interval index by apply function
        new_dif = self.compute_difficulty(X_final)
        int_idx = self.apply(new_dif)
        self.mondrian_cutoffs = np.zeros(self.k)

        # obtaing all cutoffs
        for i in range(0, self.k):
            self.mondrian_cutoffs[i] = np.quantile(
                res[np.where(int_idx == i)], q=1 - self.alpha
            )
        return None

    def compute_difficulty(self, X):
        cart_pred = np.zeros((X.shape[0], len(self.dif_model.estimators_)))
        i = 0
        # computing the difficulty score for each X_score
        for cart in self.dif_model.estimators_:
            cart_pred[:, i] = cart.predict(X)
            i += 1
        # computing variance for each line
        return cart_pred.var(1)

    def apply(self, mad):
        int_idx = np.zeros(mad.shape[0])
        for i in range(mad.shape[0]):
            index = np.where(mad[i] <= self.mondrian_quantiles)[0]
            # first testing if mad is in any interval before the last quantile
            if index.shape[0] >= 1:
                int_idx[i] = index[0]
            else:
                int_idx[i] = self.k - 1
        return int_idx

    def predict(self, X_test):
        # prediciting difficulty
        pred_dif = self.compute_difficulty(X_test)

        # assigning different cutoffs based on difficulty
        # first obtaining interval indexes
        int_idx = self.apply(pred_dif)
        cutoffs = self.mondrian_cutoffs[int_idx.astype(int)]

        return self.nc_score.predict(X_test, cutoffs)
