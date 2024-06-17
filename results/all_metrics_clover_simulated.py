import numpy as np
import pandas as pd
import os
from os import path

# base models and graph tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# conformal methods
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from clover.locart import (
    LocalRegressionSplit,
    LocartSplit,
    MondrianRegressionSplit,
)
from clover.locart import LocartSplit, MondrianRegressionSplit
from clover.scores import RegressionScore, LocalRegressionScore
from clover.simulation import make_correlated_design

# importing LCP-RF
from acpi import ACPI

# simulation and performance measures
import time
from clover.simulation import simulation
from clover.utils import (
    compute_interval_length,
    split,
    real_coverage,
    smis,
)

original_path = os.getcwd()


def compute_metrics_sim(
    n_it=100,
    n_train=10000,
    complete=False,
    iter_completing=50,
    p_completing=3,
    kind="homoscedastic",
    p=np.array([1, 3, 5]),
    d=20,
    hetero_value=0.25,
    hetero_exp=0.5,
    asym_value=0.6,
    t_degree=4,
    rho=0.7,
    rate=1.5,
    base_model=RandomForestRegressor,
    sig=0.1,
    save_all=True,
    calib_size=0.5,
    coef=2,
    B_x=5000,
    B_y=1000,
    random_seed=1250,
    random_projections=False,
    h=20,
    m=300,
    split_calib=False,
    split_mondrian=False,
    nbins=30,
    criterion="squared_error",
    max_depth=None,
    max_leaf_nodes=None,
    min_samples_leaf=150,
    prune=True,
    **kwargs
):
    # starting experiment
    print("Starting experiments for {} data with p = {}".format(kind, p))
    start_kind = time.time()

    # managing directories
    # testing if _V2 is inside kind
    # creating paths
    if "_V2" in kind:
        asym_value = 1.5
        kind = "asymmetric"
        folder_path = "/results/pickle_files/locart_all_metrics_experiments/{}_data_eta_{}".format(
            kind, asym_value
        )
    else:
        folder_path = (
            "/results/pickle_files/locart_all_metrics_experiments/{}_data".format(kind)
        )

    # creating directories to each file
    if not (path.exists(original_path + folder_path)):
        os.mkdir(original_path + folder_path)

    # generating two random seeds vector
    np.random.seed(random_seed)
    random_seeds = np.random.randint(0, 10 ** (8), n_it)
    random_seeds_X = np.random.randint(0, 10 ** (8), n_it)

    for n_var in p:
        if n_var == p_completing:
            completing = complete
        else:
            completing = False
        # testing wheter we already have all saved
        # if not, we run all and save all together in the same folder
        var_path = "/{}_score_regression_p_{}_{}_samples_measures".format(
            kind, n_var, n_train
        )
        if not (path.exists(original_path + folder_path + var_path)) or completing:
            if not completing:
                print(
                    "running the experiments for {} significant variables and {} training instances in the {} setting".format(
                        n_var, n_train, kind
                    )
                )
                # measures to be saved at last
                # real measures
                mean_int_length_vector = np.zeros((n_it, 9))
                mean_diff_vector, median_diff_vector, max_diff_vector = (
                    np.zeros((n_it, 9)),
                    np.zeros((n_it, 9)),
                    np.zeros((n_it, 9)),
                )
                mean_coverage_vector = np.zeros((n_it, 9))

                # estimated measures
                smis_vector = np.zeros((n_it, 9))

                # running times
                times = np.zeros((n_it, 9))
                init_it = 0
            else:
                os.chdir(original_path + folder_path + var_path)
                print(
                    "continuing experiments for {} significant variables and {} training instances in the {} setting".format(
                        n_var, n_train, kind
                    )
                )
                # measures to be saved at last
                # real measures
                mean_int_length_vector = np.load(
                    "mean_interval_length_p_{}_{}_data.npy".format(n_var, kind)
                )
                mean_diff_vector, median_diff_vector, max_diff_vector = (
                    np.load("mean_diff_p_{}_{}_data.npy".format(n_var, kind)),
                    np.load("median_diff_p_{}_{}_data.npy".format(n_var, kind)),
                    np.load("max_diff_p_{}_{}_data.npy".format(n_var, kind)),
                )
                mean_coverage_vector = np.load(
                    "mean_coverage_p_{}_{}_data.npy".format(n_var, kind)
                )

                # estimated measures
                smis_vector = np.load("smis_p_{}_{}_data.npy".format(n_var, kind))

                # running times
                times = np.load("run_times_p_{}_{}_data.npy".format(n_var, kind))
                init_it = iter_completing

            noise = n_var == 1
            for it in range(init_it, n_it):
                if (it + 1) % 25 == 0:
                    print(
                        "running {} iteration for {} significant variables".format(
                            it + 1, n_var
                        )
                    )

                seed_X = random_seeds_X[it]
                seed = random_seeds[it]

                # type of simulation
                sim_obj = simulation(
                    dim=d,
                    coef=coef,
                    hetero_value=hetero_value,
                    noise=noise,
                    signif_vars=n_var,
                    asym_value=asym_value,
                    t_degree=t_degree,
                    rho=rho,
                    rate=rate,
                    hetero_exp=hetero_exp,
                )
                r_kind = getattr(sim_obj, kind + "_r")
                sim_kind = getattr(sim_obj, kind)

                # generating testing samples
                if (
                    kind == "correlated_heteroscedastic"
                    or kind == "correlated_homoscedastic"
                ):
                    X_test = make_correlated_design(
                        n_samples=B_x, n_features=d, rho=rho, random_state=seed_X
                    )
                else:
                    np.random.seed(seed_X)
                    X_test = np.random.uniform(low=-1.5, high=1.5, size=(B_x, d))

                if noise:
                    X_grid = X_test[:, 0]
                else:
                    X_grid = X_test

                # generating y_test
                y_test = r_kind(X_grid, B=1).flatten()

                # simulating train and calibration sets
                sim_kind(2 * n_train, random_seed=seed)
                data = split(
                    sim_obj.X,
                    sim_obj.y,
                    test_size=calib_size,
                    calibrate=False,
                    random_seed=seed,
                )

                # matrix of y's associated to each X in test set
                y_mat = r_kind(X_grid, B=B_y)

                # fitting model
                model = base_model(**kwargs).fit(data["X_train"], data["y_train"])

                # fitting all methods, saving running times and each metric
                # fitting normal locart
                start_loc = time.time()
                locart_obj = LocartSplit(
                    nc_score=RegressionScore,
                    cart_type="CART",
                    base_model=model,
                    is_fitted=True,
                    alpha=sig,
                    split_calib=split_calib,
                    **kwargs
                )
                locart_obj.fit(data["X_train"], data["y_train"])
                locart_obj.calib(
                    data["X_test"],
                    data["y_test"],
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    prune_tree=prune,
                    random_projections=random_projections,
                    m=m,
                    h=h,
                )

                end_loc = time.time() - start_loc
                times[it, 0] = end_loc

                # predictions
                locart_pred = np.array(locart_obj.predict(X_test))
                cond_locart_real = real_coverage(locart_pred, y_mat)

                # average, median and max distance
                dif_locart = np.abs(cond_locart_real - (1 - sig))
                (
                    mean_diff_vector[it, 0],
                    median_diff_vector[it, 0],
                    max_diff_vector[it, 0],
                ) = (np.mean(dif_locart), np.median(dif_locart), np.max(dif_locart))

                # marginal coverage
                marg_cover = (
                    np.logical_and(
                        y_test >= locart_pred[:, 0], y_test <= locart_pred[:, 1]
                    )
                    + 0
                )
                mean_coverage_vector[it, 0] = np.mean(marg_cover)

                # smis
                smis_vector[it, 0] = smis(locart_pred, y_test, alpha=sig)

                # mean interval length
                mean_int_length_vector[it, 0] = np.mean(
                    compute_interval_length(locart_pred)
                )

                # fitting normal RF-locart
                start_loc = time.time()
                rf_locart_obj = LocartSplit(
                    nc_score=RegressionScore,
                    cart_type="RF",
                    base_model=model,
                    is_fitted=True,
                    alpha=sig,
                    split_calib=split_calib,
                    **kwargs
                )
                rf_locart_obj.fit(data["X_train"], data["y_train"])
                rf_locart_obj.calib(
                    data["X_test"],
                    data["y_test"],
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    prune_tree=prune,
                    random_projections=random_projections,
                    m=m,
                    h=h,
                )

                end_loc = time.time() - start_loc
                times[it, 1] = end_loc

                # predictions
                rf_locart_pred = np.array(rf_locart_obj.predict(X_test))
                cond_rf_locart_real = real_coverage(rf_locart_pred, y_mat)

                # average, median and max distance
                dif_rf_locart = np.abs(cond_rf_locart_real - (1 - sig))
                (
                    mean_diff_vector[it, 1],
                    median_diff_vector[it, 1],
                    max_diff_vector[it, 1],
                ) = (
                    np.mean(dif_rf_locart),
                    np.median(dif_rf_locart),
                    np.max(dif_rf_locart),
                )

                # smis
                smis_vector[it, 1] = smis(rf_locart_pred, y_test, alpha=sig)

                # mean interval length
                mean_int_length_vector[it, 1] = np.mean(
                    compute_interval_length(rf_locart_pred)
                )

                # marginal coverage
                marg_cover = (
                    np.logical_and(
                        y_test >= rf_locart_pred[:, 0], y_test <= rf_locart_pred[:, 1]
                    )
                    + 0
                )
                mean_coverage_vector[it, 1] = np.mean(marg_cover)

                # fitting normal difficulty locart
                start_loc = time.time()
                dlocart_obj = LocartSplit(
                    nc_score=RegressionScore,
                    cart_type="CART",
                    base_model=model,
                    is_fitted=True,
                    alpha=sig,
                    split_calib=split_calib,
                    weighting=True,
                    **kwargs
                )
                dlocart_obj.fit(data["X_train"], data["y_train"])
                dlocart_obj.calib(
                    data["X_test"],
                    data["y_test"],
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    prune_tree=prune,
                    random_projections=random_projections,
                    m=m,
                    h=h,
                )

                end_loc = time.time() - start_loc
                times[it, 2] = end_loc

                # predictions
                dlocart_pred = np.array(dlocart_obj.predict(X_test))
                cond_dlocart_real = real_coverage(dlocart_pred, y_mat)

                # average, median and max distance
                dif_dlocart = np.abs(cond_dlocart_real - (1 - sig))
                (
                    mean_diff_vector[it, 2],
                    median_diff_vector[it, 2],
                    max_diff_vector[it, 2],
                ) = (np.mean(dif_dlocart), np.median(dif_dlocart), np.max(dif_dlocart))

                # smis
                smis_vector[it, 2] = smis(dlocart_pred, y_test, alpha=sig)

                # mean interval length
                mean_int_length_vector[it, 2] = np.mean(
                    compute_interval_length(dlocart_pred)
                )

                # marginal coverage
                marg_cover = (
                    np.logical_and(
                        y_test >= dlocart_pred[:, 0], y_test <= dlocart_pred[:, 1]
                    )
                    + 0
                )
                mean_coverage_vector[it, 2] = np.mean(marg_cover)

                # fitting RF difficulty locart
                start_loc = time.time()
                rf_dlocart_obj = LocartSplit(
                    nc_score=RegressionScore,
                    cart_type="RF",
                    base_model=model,
                    is_fitted=True,
                    alpha=sig,
                    split_calib=split_calib,
                    weighting=True,
                    **kwargs
                )
                rf_dlocart_obj.fit(data["X_train"], data["y_train"])
                rf_dlocart_obj.calib(
                    data["X_test"],
                    data["y_test"],
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    prune_tree=prune,
                    random_projections=random_projections,
                    m=m,
                    h=h,
                )

                end_loc = time.time() - start_loc
                times[it, 3] = end_loc

                # predictions
                rf_dlocart_pred = np.array(rf_dlocart_obj.predict(X_test))
                cond_rf_dlocart_real = real_coverage(rf_dlocart_pred, y_mat)

                # average, median and max distance
                dif_rf_dlocart = np.abs(cond_rf_dlocart_real - (1 - sig))
                (
                    mean_diff_vector[it, 3],
                    median_diff_vector[it, 3],
                    max_diff_vector[it, 3],
                ) = (
                    np.mean(dif_rf_dlocart),
                    np.median(dif_rf_dlocart),
                    np.max(dif_rf_dlocart),
                )

                # smis
                smis_vector[it, 3] = smis(rf_dlocart_pred, y_test, alpha=sig)

                # mean interval length
                mean_int_length_vector[it, 3] = np.mean(
                    compute_interval_length(rf_dlocart_pred)
                )

                # marginal coverage
                marg_cover = (
                    np.logical_and(
                        y_test >= rf_dlocart_pred[:, 0], y_test <= rf_dlocart_pred[:, 1]
                    )
                    + 0
                )
                mean_coverage_vector[it, 3] = np.mean(marg_cover)

                # fitting ACPI/LCP-RF
                start_loc = time.time()

                acpi = ACPI(model_cali=model, n_estimators=100)
                acpi.fit(data["X_test"], data["y_test"], nonconformity_func=None)
                acpi.fit_calibration(
                    data["X_test"], data["y_test"], quantile=1 - sig, only_qrf=True
                )

                end_loc = time.time() - start_loc
                times[it, 4] = end_loc

                acpi_pred = np.stack((acpi.predict_pi(X_test, method="qrf")), axis=-1)
                cond_acpi_real = real_coverage(acpi_pred, y_mat)

                # average, median and max distance
                dif_acpi = np.abs(cond_acpi_real - (1 - sig))
                (
                    mean_diff_vector[it, 4],
                    median_diff_vector[it, 4],
                    max_diff_vector[it, 4],
                ) = (np.mean(dif_acpi), np.median(dif_acpi), np.max(dif_acpi))

                # smis
                smis_vector[it, 4] = smis(acpi_pred, y_test, alpha=sig)

                # mean interval length
                mean_int_length_vector[it, 4] = np.mean(
                    compute_interval_length(acpi_pred)
                )

                # marginal coverage
                marg_cover = (
                    np.logical_and(y_test >= acpi_pred[:, 0], y_test <= acpi_pred[:, 1])
                    + 0
                )
                mean_coverage_vector[it, 4] = np.mean(marg_cover)

                # fitting wlocart
                start_loc = time.time()

                wlocart_obj = LocartSplit(
                    nc_score=LocalRegressionScore,
                    cart_type="RF",
                    base_model=model,
                    is_fitted=True,
                    alpha=sig,
                    split_calib=split_calib,
                    **kwargs
                )
                wlocart_obj.fit(data["X_train"], data["y_train"])
                wlocart_obj.calib(
                    data["X_test"],
                    data["y_test"],
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    prune_tree=prune,
                    random_projections=random_projections,
                    m=m,
                    h=h,
                )

                end_loc = time.time() - start_loc
                times[it, 5] = end_loc

                # predictions
                wlocart_pred = np.array(wlocart_obj.predict(X_test))
                cond_wlocart_real = real_coverage(wlocart_pred, y_mat)

                # average, median and max distance
                dif_wlocart = np.abs(cond_wlocart_real - (1 - sig))
                (
                    mean_diff_vector[it, 5],
                    median_diff_vector[it, 5],
                    max_diff_vector[it, 5],
                ) = (np.mean(dif_wlocart), np.median(dif_wlocart), np.max(dif_wlocart))

                # smis
                smis_vector[it, 5] = smis(wlocart_pred, y_test, alpha=sig)

                # mean interval length
                mean_int_length_vector[it, 5] = np.mean(
                    compute_interval_length(wlocart_pred)
                )

                # marginal coverage
                marg_cover = (
                    np.logical_and(
                        y_test >= wlocart_pred[:, 0], y_test <= wlocart_pred[:, 1]
                    )
                    + 0
                )
                mean_coverage_vector[it, 5] = np.mean(marg_cover)

                # interval length | coveraqe
                cover_idx = np.where(marg_cover == 1)
                wlocart_interval_len_cover = np.mean(
                    compute_interval_length(wlocart_pred[cover_idx])
                )

                wloc_cutoffs = wlocart_obj.cutoffs

                # fitting default regression split
                start_split = time.time()
                nc = NcFactory.create_nc(model)
                icp = IcpRegressor(nc)
                icp.fit(data["X_train"], data["y_train"])
                icp.calibrate(data["X_test"], data["y_test"])

                end_split = time.time() - start_split
                times[it, 6] = end_split

                # predictions
                icp_pred = icp.predict(X_test, significance=sig)
                icp_pred_cond = icp.predict(X_test, significance=sig)
                cond_icp_real = real_coverage(icp_pred_cond, y_mat)

                # average, median and max distance
                dif_icp = np.abs(cond_icp_real - (1 - sig))
                (
                    mean_diff_vector[it, 6],
                    median_diff_vector[it, 6],
                    max_diff_vector[it, 6],
                ) = (np.mean(dif_icp), np.median(dif_icp), np.max(dif_icp))

                # icp smis
                smis_vector[it, 6] = smis(icp_pred, y_test, alpha=sig)

                # ICP interval length
                mean_int_length_vector[it, 6] = np.mean(
                    compute_interval_length(icp_pred)
                )

                # marginal coverage
                marg_cover = (
                    np.logical_and(y_test >= icp_pred[:, 0], y_test <= icp_pred[:, 1])
                    + 0
                )
                mean_coverage_vector[it, 6] = np.mean(marg_cover)

                # fitting wighted regression split
                start_weighted_split = time.time()
                wicp = LocalRegressionSplit(
                    base_model=model, is_fitted=True, alpha=sig, **kwargs
                )
                wicp.fit(data["X_train"], data["y_train"])
                wicp.calibrate(data["X_test"], data["y_test"])

                end_weighted_split = time.time() - start_weighted_split
                times[it, 7] = end_weighted_split

                # predictions
                wicp_pred = wicp.predict(X_test)
                wicp_pred_cond = wicp.predict(X_test)
                cond_wicp_real = real_coverage(wicp_pred_cond, y_mat)

                wicp_dif = np.abs(cond_wicp_real - (1 - sig))
                (
                    mean_diff_vector[it, 7],
                    median_diff_vector[it, 7],
                    max_diff_vector[it, 7],
                ) = (np.mean(wicp_dif), np.median(wicp_dif), np.max(wicp_dif))

                # smis
                smis_vector[it, 7] = smis(wicp_pred, y_test, alpha=sig)

                # ICP interval length
                mean_int_length_vector[it, 7] = np.mean(
                    compute_interval_length(wicp_pred)
                )

                # marginal coverage
                marg_cover = (
                    np.logical_and(y_test >= wicp_pred[:, 0], y_test <= wicp_pred[:, 1])
                    + 0
                )
                mean_coverage_vector[it, 7] = np.mean(marg_cover)

                start_weighted_split = time.time()
                micp = MondrianRegressionSplit(
                    base_model=model, is_fitted=True, alpha=sig, k=nbins, **kwargs
                )
                micp.fit(data["X_train"], data["y_train"])
                micp.calibrate(data["X_test"], data["y_test"], split=split_mondrian)

                end_weighted_split = time.time() - start_weighted_split
                times[it, 8] = end_weighted_split

                # predictions
                micp_pred = micp.predict(X_test)
                micp_pred_cond = micp.predict(X_test)
                cond_micp_real = real_coverage(micp_pred_cond, y_mat)

                micp_dif = np.abs(cond_micp_real - (1 - sig))
                (
                    mean_diff_vector[it, 8],
                    median_diff_vector[it, 8],
                    max_diff_vector[it, 8],
                ) = (np.mean(micp_dif), np.median(micp_dif), np.max(micp_dif))

                # smis
                smis_vector[it, 8] = smis(micp_pred, y_test, alpha=sig)

                # ICP interval length
                mean_int_length_vector[it, 8] = np.mean(
                    compute_interval_length(micp_pred)
                )

                # marginal coverage
                marg_cover = (
                    np.logical_and(y_test >= micp_pred[:, 0], y_test <= micp_pred[:, 1])
                    + 0
                )
                mean_coverage_vector[it, 8] = np.mean(marg_cover)

                if (it + 1) % 25 == 0 or (it + 1 == 1) or save_all:
                    print("Saving data checkpoint on iteration {}".format(it + 1))
                    # saving checkpoint of metrics
                    saving_metrics(
                        original_path,
                        folder_path,
                        var_path,
                        kind,
                        n_var,
                        mean_int_length_vector,
                        mean_diff_vector,
                        median_diff_vector,
                        max_diff_vector,
                        mean_coverage_vector,
                        smis_vector,
                        times,
                    )

            # saving all metrics again
            saving_metrics(
                original_path,
                folder_path,
                var_path,
                kind,
                n_var,
                mean_int_length_vector,
                mean_diff_vector,
                median_diff_vector,
                max_diff_vector,
                mean_coverage_vector,
                smis_vector,
                times,
            )

        else:
            continue

    print("Experiments finished for {} setting".format(kind))
    end_kind = time.time() - start_kind
    print(
        "Time Elapsed to compute all metrics in the {} setting: {}".format(
            kind, end_kind
        )
    )
    return end_kind


# saving metrics function
def saving_metrics(
    original_path,
    folder_path,
    var_path,
    kind,
    n_var,
    mean_int_length_vector,
    mean_diff_vector,
    median_diff_vector,
    max_diff_vector,
    mean_coverage_vector,
    smis_vector,
    times,
):
    # checking if path exsist
    if not (path.exists(original_path + folder_path + var_path)):
        # creating directory
        os.mkdir(original_path + folder_path + var_path)

    # changing working directory to the current folder
    os.chdir(original_path + folder_path + var_path)

    # saving all matrices into npy files
    # interval length
    np.save(
        "mean_interval_length_p_{}_{}_data.npy".format(n_var, kind),
        mean_int_length_vector,
    )

    # conditional difference
    np.save("mean_diff_p_{}_{}_data.npy".format(n_var, kind), mean_diff_vector)
    np.save("median_diff_p_{}_{}_data.npy".format(n_var, kind), median_diff_vector)
    np.save("max_diff_p_{}_{}_data.npy".format(n_var, kind), max_diff_vector)

    # mean coverage
    np.save("mean_coverage_p_{}_{}_data.npy".format(n_var, kind), mean_coverage_vector)

    # estimated metrics
    np.save("smis_p_{}_{}_data.npy".format(n_var, kind), smis_vector)

    # running times
    np.save("run_times_p_{}_{}_data.npy".format(n_var, kind), times)

    # returning to original path
    os.chdir(original_path)


# method that make all the computations for all kinds of data


def compute_all_conformal_metrics(
    kinds_list=[
        "homoscedastic",
        "heteroscedastic",
        "asymmetric",
        "asymmetric_V2",
        "t_residuals",
        "non_cor_heteroscedastic",
    ],
    base_model=RandomForestRegressor,
    complete=False,
    iter_completing=50,
    p_completing=3,
    save_all=True,
    n_it=100,
    p=np.array([1, 3, 5]),
    d=20,
    **kwargs
):
    print("Starting all experiments")
    start_exp = time.time()
    times_list = list()
    if type(kinds_list) == list:
        for kinds in kinds_list:
            times_list.append(
                compute_metrics_sim(
                    kind=kinds,
                    n_it=n_it,
                    complete=complete,
                    iter_completing=iter_completing,
                    save_all=save_all,
                    p=p,
                    d=d,
                    **kwargs
                )
            )
        end_exp = time.time() - start_exp
        print("Time elapsed to conduct all experiments: {}".format(end_exp))
        np.save(
            "results/pickle_files/locart_all_metrics_experiments/sim_running_times.npy",
            np.array(times_list.append(end_exp)),
        )
    else:
        compute_metrics_sim(
            kind=kinds_list,
            complete=complete,
            iter_completing=iter_completing,
            n_it=n_it,
            p=p,
            d=d,
            **kwargs
        )
        end_exp = time.time() - start_exp
        print("Time elapsed to conduct {} experiments: {}".format(kinds_list, end_exp))
        np.save(
            "results/pickle_files/locart_all_metrics_experiments/sim_{}_running_times.npy".format(
                kinds_list
            ),
            np.array(times_list.append(end_exp)),
        )
    return None


if __name__ == "__main__":
    print("We will now compute all conformal statistics for several simulated examples")
    model = input("Which model would like to use as base model? ")
    separated = input("Would you like to run each setting in separated terminals? ")
    if separated == "yes":
        kind = input("What kind of data would you like to simulate? ")
        interrupted = input("Did you interrupted the program at some point earlier?")
        if interrupted == "yes":
            p = int(input("In which p did you stop?"))
            completing = True
            iter_complete = int(input("In which iteration did you stop?"))

    if model == "Random Forest":
        random_state = 650
        if separated == "yes":
            if interrupted == "yes":
                compute_all_conformal_metrics(
                    kinds_list=kind,
                    complete=completing,
                    p_completing=p,
                    iter_completing=iter_complete,
                    random_state=random_state,
                )
            else:
                compute_all_conformal_metrics(
                    kinds_list=kind, random_state=random_state
                )
        else:
            compute_all_conformal_metrics(random_state=random_state)
    elif model == "KNN":
        if separated == "yes":
            compute_all_conformal_metrics(
                kinds_list=kind, base_model=KNeighborsRegressor, n_neighbors=30
            )
        else:
            compute_all_conformal_metrics(
                base_model=KNeighborsRegressor, n_neighbors=30
            )
