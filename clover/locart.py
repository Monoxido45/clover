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
from operator import itemgetter
import scipy.stats as st

from clover.scores import LocalRegressionScore, RegressionScore, QuantileScore


class LocartSplit(BaseEstimator):
    """
    Local Regression Tree and Local Forests class.
    Fit LOCART and LOFOREST local calibration methods for any conformity score and base model of interest. The specification of the conformity score
    can be made through the usage of the basic class "Scores". Through the "split_calib" parameter we can decide whether to use all calibration set to
    obtain both the partition and cutoffs or split it into two sets, one specific for partitioning and other for obtaining the local cutoffs. Also, if
    desired, we can fit the augmented version of both our methods (A-LOCART and A-LOFOREST) by the "weighting" parameter, which if True, adds difficulty
    estimates to our feature matrix in the calibration and prediction step.
    ----------------------------------------------------------------
    """

    def __init__(
        self,
        nc_score,
        base_model,
        alpha,
        is_fitted=False,
        base_model_type=None,
        cart_type="CART",
        split_calib=True,
        weighting=False,
        **kwargs
    ):
        """
        Input: (i)    nc_score: Conformity score of choosing. It can be specified by instantiating a conformal score class based on the Scores basic class.
               (ii)   base_model: Base model with fit and predict methods to be embedded in the conformity score class.
               (iii)  alpha: Float between 0 and 1 specifying the miscoverage level of resulting prediction region.
               (iv)   base_model_type: Boolean indicating whether the base model ouputs quantiles or not. Default is False.
               (v)    cart_type: Set "CART" to obtain LOCART prediction intervals and "RF" to obtain LOFOREST prediction intervals. Default is CART.
               (vi)   split_calib: Boolean designating if we should split the calibration set into partitioning and cutoff set. Default is True.
               (vii)  **kwargs: Additional keyword arguments passed to fit base_model.
               (viii) weighting: Set whether we should augment the feature space with conditional variance (difficulty) estimates. Default is False.
        """
        self.base_model_type = base_model_type
        if ("Quantile" in str(nc_score)) or (base_model_type == True):
            self.nc_score = nc_score(
                base_model, is_fitted=is_fitted, alpha=alpha, **kwargs
            )
        else:
            self.nc_score = nc_score(base_model, is_fitted=is_fitted, **kwargs)

        # checking if base model is fitted
        self.base_model = self.nc_score.base_model
        self.alpha = alpha
        self.cart_type = cart_type
        self.split_calib = split_calib
        self.weighting = weighting

    def fit(self, X, y, random_seed_tree=1250, **kwargs):
        """
        Fit base model embeded in the conformal score class to the training set.
        If "weigthing" is True, we fit a Random Forest model to obtain variance estimations as done in Bostrom et.al.(2021).
        --------------------------------------------------------

        Input: (i)    X: Training numpy feature matrix
               (ii)   y: Training label array
               (iii)  random_seed_tree: Random Forest random seed for variance estimation (if weighting parameter is True).
               (iv)   **kwargs: Keyword arguments passed to fit the random forest used for variance estimation.

        Output: LocartSplit object
        """
        self.nc_score.fit(X, y)
        if self.weighting == True:
            if not isinstance(self.nc_score.base_model, RandomForestRegressor):
                self.dif_model = (
                    RandomForestRegressor(random_state=random_seed_tree)
                    .set_params(**kwargs)
                    .fit(X, y)
                )
            else:
                self.dif_model = deepcopy(self.nc_score.base_model)
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
        h=1,
        projections_seed=1250,
        **kwargs
    ):
        """
        Calibrate conformity score using CART or Random Forest
        As default, we fix "min_samples_leaf" as 100 for both the CART and RF algorithms,meaning that each partition element will have at least
        100 samples each, and use the sklearn default for the remaining parameters. To generate other partitioning schemes, all RF and CART parameters
        can be changed through keyword arguments, but we recommend changing only the "min_samples_leaf" argument if needed.
        --------------------------------------------------------

        Input: (i)    X_calib: Calibration numpy feature matrix
               (ii)   y_calib: Calibration label array
               (iii)  random_seed: Random seed for CART or Random Forest fitted to the confomity scores.
               (iv)   prune_tree: Boolean indicating whether CART tree should be pruned or not.
               (v)    prune_seed: Random seed set for data splitting in the prune step.
               (vi)   cart_train_size: Proportion of calibration data used in partitioning.
               (vii)  random_projections: Boolean specifying if we should augment the feature space with random projections or random fourier features.
               Default is False.
               (viii) m: Number of random projections to augment feature space. Default is 1000.
               (ix)   h: Random projections scale. Default is 1.
               (x)    **kwargs: Keyword arguments to be passed to CART or Random Forest.

        Ouput: Vector of cutoffs.
        """
        res = self.nc_score.compute(X_calib, y_calib)

        if self.weighting:
            w = self.compute_difficulty(X_calib)
            X_calib = np.concatenate((X_calib, w.reshape(-1, 1)), axis=1)

        # splitting calibration data into a partitioning set and a cutoff set
        if self.split_calib:
            (
                X_calib_train,
                X_calib_test,
                res_calib_train,
                res_calib_test,
            ) = train_test_split(
                X_calib, res, test_size=1 - cart_train_size, random_state=random_seed
            )
            if random_projections and self.cart_type == "CART":
                self.rp = True
                np.random.seed(projections_seed)
                self.S_matrix = np.random.normal(
                    scale=np.sqrt(h), size=(m, X_calib_train.shape[1])
                )
                self.ratio_factor = np.sqrt(m)
                self.b = np.random.uniform(0, 2 * np.pi, size=(1, m))
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
            if random_projections and self.cart_type == "CART":
                self.rp = True
                np.random.seed(projections_seed)
                self.S_matrix = np.random.normal(
                    scale=np.sqrt(h), size=(m, X_calib_train.shape[1])
                )
                self.ratio_factor = np.sqrt(m)
                self.b = np.random.uniform(0, 2 * np.pi, size=(1, m))
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
                random_state=random_seed, min_samples_leaf=100
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
                        X_calib,
                        res,
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
            self.cutoffs = {}

            for leaf in self.leaf_idx:
                if self.split_calib:
                    current_res = res_calib_test[leafs_idx == leaf]

                    # correcting 1 - alpha
                    n = current_res.shape[0]

                    self.cutoffs[leaf] = np.quantile(
                        current_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
                    )
                else:
                    current_res = res[leafs_idx == leaf]

                    # correcting 1 - alpha
                    n = current_res.shape[0]

                    self.cutoffs[leaf] = np.quantile(
                        current_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
                    )
        # random forest instead of CART
        elif self.cart_type == "RF":
            self.RF = RandomForestRegressor(
                random_state=random_seed, min_samples_leaf=100
            ).set_params(**kwargs)
            if self.split_calib:
                self.RF.fit(X_calib_train, res_calib_train)
                self.cutoffs = self.create_rf_cutoffs(X_calib_test, res_calib_test)
            else:
                self.RF.fit(X_calib, res)
                self.cutoffs = self.create_rf_cutoffs(X_calib, res)

        # TODO: implement RFCDE version

        return self.cutoffs

    def compute_difficulty(self, X):
        """
        Auxiliary function to compute difficulty for each sample.
        --------------------------------------------------------
        input: (i)    X: specified numpy feature matrix

        output: Vector of variance estimates for each sample.
        """
        cart_pred = np.zeros((X.shape[0], len(self.dif_model.estimators_)))
        i = 0
        # computing the difficulty score for each X_score
        for cart in self.dif_model.estimators_:
            cart_pred[:, i] = cart.predict(X)
            i += 1
        # computing variance for each dataset row
        return cart_pred.var(1)

    # creating random forest cutoffs
    def create_rf_cutoffs(self, X, res):
        """
        Auxiliary function to compute loforest cutoffs.
        --------------------------------------------------------
        input: (i)    X: numpy feature matrix
               (ii)   res: conformal scores

        output: Dictionary with each leaf's cutoff.
        """
        # looping through every decision tree in random forest
        cutoffs_list = []
        # getting all leafs
        all_leaves = self.RF.apply(X)
        for i in range(0, all_leaves.shape[1]):
            leaves_idx = all_leaves[:, i]
            leaf_idx = np.unique(leaves_idx)
            cutoffs = {}
            for leaf in leaf_idx:
                current_res = res[leaves_idx == leaf]
                # correcting 1 - alpha
                n = current_res.shape[0]
                cutoffs[leaf] = np.quantile(
                    current_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
                )
            cutoffs_list.append(cutoffs)
        return cutoffs_list

    def add_random_projections(self, X):
        """
        Auxiliary function to add random projections to feature space.
        --------------------------------------------------------
        input: (i)    X: specified numpy feature matrix

        output: Matrix of random projections.
        """
        projections = (
            np.sqrt(2)
            * np.cos(
                np.dot(X, self.S_matrix.transpose())
                + np.repeat(self.b, X.shape[0], axis=0)
            )
            / self.ratio_factor
        )
        return np.concatenate((X, projections), axis=1)

    def prune_tree(self, X_train, X_valid, res_train, res_valid):
        """
        Auxiliary function to conduct decision tree post pruning.
        --------------------------------------------------------
        Input: (i)    X_train: numpy feature matrix used to fit decision trees for each cost complexity alpha values.
               (ii)   X_valid: numpy feature matrix used to validate each cost complexity path.
               (iii)  res_train: conformal scores used to fit decision trees for each cost complexity alpha values.
               (iv)   res_valid: conformal scores used to validate each cost complexity path.

        Output: Optimal cost complexity path to perform pruning.
        """
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
        """
        Optional method used to perform euclidean binning of the feature space. Does not work well in high-dimensional datasets.
        --------------------------------------------------------
        input: (i)    X_calib: calibration numpy feature matrix
               (ii)   y_calib: calibration label vector

        output: Binning cutoffs.
        """
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
        """
        Auxiliary function to retrieve the uniform cutoff index for new observations.
        --------------------------------------------------------
        Input: (i)    X: numpy feature matrix

        Output: Vector of indices.
        """
        int_idx = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            int_idx[i] = (
                np.where(np.all(X[i, :] <= self.unif_intervals, axis=1))[0][0] + 1
            )
        return int_idx

    def plot_locart(self, title=None):
        """
        Plot decision tree feature space partition
        --------------------------------------------------------
        Output: Decision tree plot object.
        """
        if self.cart_type == "CART":
            plot_tree(self.cart, filled=True)
            if title == None:
                plt.title("Decision Tree fitted to non-conformity score")
            else:
                plt.title(title)
            plt.show()

    def predict(self, X, type_model="Tree"):
        """
        Predict 1 - alpha prediction intervals for each test sample using locart/loforest local cutoffs.
        Alternatively, this function can be used to obtain euclidean binning prediction intervals.
        --------------------------------------------------------
        Input: (i)    X: test numpy feature matrix
               (ii)   type_model: String indicating the type of partitioning used to obtain local cutoffs. "Tree" sets the locart/loforest partitioning, while
               "euclidean" sets the euclidean binning. Default is "Tree".

        Output: Prediction intervals for each test sample.
        """
        # identifying cutoff point
        if self.weighting:
            w = self.compute_difficulty(X)
            X_tree = np.concatenate((X, w.reshape(-1, 1)), axis=1)
        else:
            X_tree = X

        if self.cart_type == "CART" and type_model == "Tree":
            if self.rp:
                X_tree = self.add_random_projections(self.rp_scaler.transform(X))
            elif not self.weighting:
                X_tree = X
            leaves_idx = self.cart.apply(X_tree)
            # obtaining order of leaves
            cutoffs = np.array(itemgetter(*leaves_idx)(self.cutoffs))
            pred = self.nc_score.predict(X, cutoffs)

        elif self.cart_type == "RF" and type_model == "Tree":
            all_leaves = self.RF.apply(X_tree)
            # ranking the order of the leaves by row
            cutoffs_matrix = np.zeros((X_tree.shape[0], all_leaves.shape[1]))
            for i in range(0, cutoffs_matrix.shape[1]):
                cutoffs_matrix[:, i] = np.array(
                    itemgetter(*all_leaves[:, i])(self.cutoffs[i])
                )

            # obtaining cutoff means
            final_cutoffs = np.mean(cutoffs_matrix, axis=1)
            pred = self.nc_score.predict(X, final_cutoffs)

        elif type_model == "euclidean":
            idx = self.uniform_apply(X)
            cutoffs = self.unif_cutoffs[st.rankdata(idx.astype(int), method="dense") - 1]
            pred = self.nc_score.predict(X, cutoffs)

        return pred


class RegressionSplit(BaseEstimator):
    """
    Basic Regression Split class
    ----------------------------------------------------------------
    """

    def __init__(self, base_model, alpha=0.1, is_fitted=False, **kwargs) -> None:
        super().__init__()
        self.nc_score = RegressionScore(base_model, is_fitted=is_fitted, **kwargs)
        self.alpha = alpha

    def fit(self, X_train, y_train):
        self.nc_score.fit(X_train, y_train)

    def calibrate(self, X_calib, y_calib):
        res = self.nc_score.compute(X_calib, y_calib)
        n = X_calib.shape[0]
        self.cutoff = np.quantile(res, q=np.ceil((n + 1) * (1 - self.alpha)) / n)
        return None

    def predict(self, X_test):
        return self.nc_score.predict(X_test, self.cutoff)


class QuantileSplit(BaseEstimator):
    """
    Conformalized Quantile Regression class
    ----------------------------------------------------------------
    """

    def __init__(self, base_model, alpha, is_fitted=False, **kwargs):
        """
        Input: (i)    base_model: Quantile regression model with fit and predict methods defined
               (ii)   alpha: Miscalibration level
               (iii)  is_fitted: Boolean informing whether the base model is already fitted.
               (iv)   **kwargs: Additional keyword arguments to be passed to the base model.
        """
        self.nc_score = QuantileScore(
            base_model, is_fitted=is_fitted, alpha=alpha, **kwargs
        )
        self.alpha = alpha

    def fit(self, X_train, y_train):
        """
        Fit base model embedded on quantile conformal score to the training set.
        ----------------------------------------------------------------

        Input: (i)    X_train: Training numpy feature matrix.
               (ii)   y_train: Training labels.

        Output: QuantileSplit object.
        """
        self.nc_score.fit(X_train, y_train)
        return self

    def calibrate(self, X_calib, y_calib):
        """
        Calibrate quantile conformity score
        ----------------------------------------------------------------

        Input: (i)    X_calib: Calibration numpy feature matrix.
               (ii)   y_calib: Calibration labels.

        Output: None
        """
        res = self.nc_score.compute(X_calib, y_calib)
        n = X_calib.shape[0]
        self.cutoff = np.quantile(res, q=np.ceil((n + 1) * (1 - self.alpha)) / n)
        return None

    def predict(self, X_test):
        """
        Predict 1 - alpha prediction intervals for each test samples using CQR cutoff
        ----------------------------------------------------------------

        Input: (i)    X_test: Test numpy feature matrix.

        Output: Prediction intervals for each test sample.
        """
        return self.nc_score.predict(X_test, self.cutoff)


# Local regression split proposed by Lei et al
class LocalRegressionSplit(BaseEstimator):
    """
    Local Regression Split class
    ----------------------------------------------------------------
    """

    def __init__(self, base_model, alpha, is_fitted=False, **kwargs):
        """
        Input: (i)    base_model: Regression base model with fit and predict methods defined.
               (ii)   alpha: Miscalibration level.
               (iii)  is_fitted: Boolean informing whether the base model is already fitted.
               (iv)   **kwargs: Additional keyword arguments to be passed to the base model.
        """
        self.nc_score = LocalRegressionScore(base_model, is_fitted=is_fitted, **kwargs)
        self.alpha = alpha

    def fit(self, X_train, y_train):
        """
        Fit base model embedded on local regression conformal score to the training set.
        ----------------------------------------------------------------

        Input: (i)    X_train: Training numpy feature matrix.
               (ii)   y_train: Training labels.

        Output: QuantileSplit object.
        """
        # fitting the base model
        self.nc_score.fit(X_train, y_train)
        return self

    def calibrate(self, X_calib, y_calib):
        """
        Calibrate local regression score
        ----------------------------------------------------------------

        Input: (i)    X_calib: Calibration numpy feature matrix.
               (ii)   y_calib: Calibration labels.

        Output: None
        """
        res = self.nc_score.compute(X_calib, y_calib)
        n = X_calib.shape[0]
        self.cutoff = np.quantile(res, q=np.ceil((n + 1) * (1 - self.alpha)) / n)
        return None

    def predict(self, X_test):
        """
        Predict 1 - alpha prediction intervals for each test samples using local regression split cutoff
        ----------------------------------------------------------------

        Input: (i)    X_test: Test numpy feature matrix.

        Output: Prediction intervals for each test sample.
        """
        return self.nc_score.predict(X_test, self.cutoff)


# Mondrian split method proposed by Bostrom et al
class MondrianRegressionSplit(BaseEstimator):
    """
    Mondrian regression split class
    ----------------------------------------------------------------
    """

    def __init__(self, base_model, alpha, is_fitted=False, k=10, **kwargs):
        """
        Input: (i)    base_model: Regression base model with fit and predict methods defined.
               (ii)   alpha: Miscalibration level.
               (iii)  is_fitted: Boolean informing whether the base model is already fitted.
               (iv)   k: Number of bins for splitting the conditional variance (diffulty).
               (v)    **kwargs: Additional keyword arguments to be passed to the base model.
        """
        self.base_model = base_model
        self.k = k
        self.nc_score = RegressionScore(self.base_model, is_fitted=is_fitted, **kwargs)
        self.alpha = alpha

    def fit(self, X_train, y_train, random_seed_tree=550, **kwargs):
        """
        Fit both base model embedded on the regression score to the training set and a Random Forest model to estimate variance in the calibration step.
        ----------------------------------------------------------------
        Input: (i)    X_train: Training numpy feature matrix.
               (ii)   y_train: Train numpy labels.
               (iii)  random_seed_tree: Random Forest random seed for variance estimation.
               (iv)   **kwargs: Additional keyword arguments to be passed to the Random Forest model.
        """
        # fitting the base model
        self.nc_score.fit(X_train, y_train)
        # training RandomForestRegressor for difficulty estimation if base model is not RandomForest
        if not isinstance(self.nc_score.base_model, RandomForestRegressor):
            self.dif_model = (
                RandomForestRegressor(random_state=random_seed_tree)
                .set_params(**kwargs)
                .fit(X_train, y_train)
            )
        else:
            self.dif_model = deepcopy(self.nc_score.base_model)

        return self

    def calibrate(
        self, X_calib, y_calib, split=False, binning_size=0.5, random_state=1250
    ):
        """
        Calibrate conformity scores using mondrian binning.
        ----------------------------------------------------------------

        Input: (i)    X_calib: Calibration numpy feature matrix.
               (ii)   y_calib: Calibration labels.
               (iii)  split: Whether to split the dataset into a binning set and a cutoff set in the binning step. Default is False.
               (iv)   binning_size: Proportion of the splitted dataset destined for binning if split is True. Default is 0.5.
               (v)    random_state: Random seed for splitting the dataset if split is True.

        Output: None
        """
        if split:
            # making the split
            X_score, X_final, y_score, y_final = train_test_split(
                X_calib, y_calib, test_size=1 - binning_size, random_state=random_state
            )
        else:
            X_score, X_final, y_score, y_final = X_calib, X_calib, y_calib, y_calib

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
            current_res = res[np.where(int_idx == i)]
            n = current_res.shape[0]
            self.mondrian_cutoffs[i] = np.quantile(
                current_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
            )
        return None

    def compute_difficulty(self, X):
        """
        Auxiliary function to estimate difficulty (variance) for a given sample.
        ----------------------------------------------------------------

        Input: (i)    X: Calibration numpy feature matrix.

        Output: Difficulty estimates for each sample.
        """
        cart_pred = np.zeros((X.shape[0], len(self.dif_model.estimators_)))
        i = 0
        # computing the difficulty score for each X_score
        for cart in self.dif_model.estimators_:
            cart_pred[:, i] = cart.predict(X)
            i += 1
        # computing variance for each line
        return cart_pred.var(1)

    def apply(self, dif):
        """
        Auxiliary function to obtain bin index for each difficulty estimate.
        ----------------------------------------------------------------

        Input: (i)    dif: Difficulty estimate vector.

        Output: Vector of indices.
        """
        int_idx = np.zeros(dif.shape[0])
        for i in range(dif.shape[0]):
            index = np.where(dif[i] <= self.mondrian_quantiles)[0]
            # first testing if dif is in any interval before the last quantile
            if index.shape[0] >= 1:
                int_idx[i] = index[0]
            else:
                int_idx[i] = self.k - 1
        return int_idx

    def predict(self, X_test):
        """
        Predict 1 - alpha prediction intervals for each test samples using mondrian cutoffs.
        ----------------------------------------------------------------

        Input: (i)    X_test: Test numpy feature matrix.

        Output: Prediction intervals for each test sample.
        """
        # prediciting difficulty
        pred_dif = self.compute_difficulty(X_test)

        # assigning different cutoffs based on difficulty
        # first obtaining interval indexes
        int_idx = self.apply(pred_dif)
        cutoffs = self.mondrian_cutoffs[int_idx.astype(int)]

        return self.nc_score.predict(X_test, cutoffs)
