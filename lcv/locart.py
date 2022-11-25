from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

from lcv.scores import LocalRegressionScore, QuantileScore


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

    def fit(self, X, y):
        """
        Fit non conformity score to training samples
        --------------------------------------------------------
        X: Feature matrix
        y: label for training samples
        """
        # saving minimum and maximum for y
        self.min_y, self.max_y = (np.min(y) - 3), (np.max(y) + 3)
        self.nc_score.fit(X, y)
        return self

    def update_limits(self, value_min, value_max):
        if value_min < self.min_y:
            self.min_y = value_min
        if value_max > self.max_y:
            self.max_y = value_max

    def calib(
        self,
        X_calib,
        y_calib,
        random_seed=1250,
        prune_tree=True,
        prune_seed=780,
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
        # update min and maximum in y
        self.update_limits(np.min(y_calib), np.max(y_calib))

        # splitting calibration data into a training half and a validation half
        X_calib_train, X_calib_test, res_calib_train, res_calib_test = train_test_split(
            X_calib, res, test_size=0.5, random_state=random_seed
        )

        if self.cart_type == "CART":
            # declaring decision tree
            self.cart = DecisionTreeRegressor(
                random_state=random_seed, min_samples_leaf=300
            ).set_params(**kwargs)
            # obtaining optimum alpha to prune decision tree
            if prune_tree:
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

    def prune_tree(self, X_train, X_valid, res_train, res_valid):
        prune_path = self.cart.cost_complexity_pruning_path(X_train, res_train)
        ccp_alphas = prune_path.ccp_alphas
        current_loss = 1000
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

    def predict(self, X, length=1500, type_model="CART"):
        """
        Predict $1 - \alpha$ prediction region for each test sample using LocartSplit local cutoff points
        """
        y_grid = np.linspace(self.min_y, self.max_y, length)
        intervals_list = []
        for i in range(X.shape[0]):
            # computing residual for all the grid
            res = self.nc_score.compute(X[i, :].reshape(1, -1), y_grid)
            # obtaining cutoff indexes and cutoff points according to choosed type of model
            if type_model == "CART":
                cutoff_idx = np.where(
                    self.leaf_idx == self.cart.apply(X[i, :].reshape(1, -1))
                )[0][0]
                # finding interval/region limits
                ident_int = np.diff((res <= self.cutoffs[cutoff_idx]) + 0)

            elif type_model == "euclidean":
                # obtaining X cutoff index
                cutoff_idx = np.where(
                    self.cartesian_ints == self.uniform_apply(X[i, :].reshape(1, -1))
                )[0][0]
                # finding interval/region limits
                ident_int = np.diff((res <= self.unif_cutoffs[cutoff_idx]) + 0)

            ident_idx = np.where(ident_int != 0)[0]

            if len(ident_idx) == 0 and self.base_model_type == True:
                intervals_list.append(
                    self.base_model.predict(X[i, :].reshape(1, -1)).flatten()
                )
            elif len(ident_idx) == 0 and self.base_model_type is None:
                intervals_list.append(np.array([self.min_y, self.max_y]))
            else:
                # -1 indicates end of the invervals and 1 the beggining
                # if we start the identifier with -1, that means the first entry is the beggining
                if ident_int[ident_idx[0]] == -1:
                    ident_idx = np.insert(ident_idx, 0, -1)
                # if we finish with 0, that means the last entry is the end
                if ident_int[ident_idx[-1]] == 1:
                    ident_idx = np.append(ident_idx, y_grid.shape[0] - 1)

                # after turning the array even shaped we add one to the lower limit of intervals
                int_idx = ident_idx + np.tile(
                    np.array([1, 0]), int(ident_idx.shape[0] / 2)
                )
                intervals_list.append(y_grid[int_idx])

        return np.array(intervals_list)


class QuantileSplit(BaseEstimator):
    def __init__(self, base_model, alpha, **kwargs):
        self.base_model = base_model
        self.nc_score = QuantileScore(self.base_model, coverage=alpha, **kwargs)
        self.alpha = alpha

    def fit(self, X_train, y_train):
        self.nc_score.fit(X_train, y_train)
        return self

    def calibrate(self, X_calib, y_calib):
        res = self.nc_score.compute(X_calib, y_calib)
        self.cutoff = np.quantile(res, q=1 - self.alpha)
        return None

    def predict(self, X_test):
        quantiles = self.nc_score.base_model.predict(X_test)
        pred = np.vstack(
            (quantiles[:, 0] - self.cutoff, quantiles[:, 1] + self.cutoff)
        ).T
        return pred


# Local regression split proposed by Lei et al
class LocalRegressionSplit(BaseEstimator):
    def __init__(self, base_model, alpha, **kwargs):
        self.base_model = base_model
        self.nc_score = LocalRegressionScore(self.base_model, **kwargs)
        self.alpha = alpha

    def fit(self, X_train, y_train):
        # fitting the base model
        self.nc_score.fit(X_train, y_train)
        return self

    def calibrate(self, X_calib, y_calib):
        res = self.nc_score.compute(X_calib, y_calib)
        self.cutoff = np.quantile(res, q=1 - self.alpha)
        return None

    def predict(self, X_test):
        pred_mu = self.nc_score.base_model.predict(X_test)
        pred_mad = self.nc_score.mad_model.predict(X_test)
        pred = np.vstack(
            (pred_mu - (pred_mad * self.cutoff), pred_mu + (pred_mad * self.cutoff))
        ).T
        return pred
