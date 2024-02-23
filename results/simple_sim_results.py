import numpy as np
import pandas as pd
import os
import scipy.stats as st

# base models and graph tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

# conformal methods
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from clover.locart import (
    LocalRegressionSplit,
    LocartSplit,
    MondrianRegressionSplit,
    QuantileSplit,
)
from clover.locart import LocartSplit, MondrianRegressionSplit
from clover.locluster import KmeansSplit
from clover.scores import RegressionScore, QuantileScore, LocalRegressionScore

# importing LCP-RF
from acpi import ACPI

# simulation and performance measures
import time
from clover.simulation import simulation
from clover.utils import (
    compute_interval_length,
    ILS_coverage,
    split,
    real_coverage,
    smis,
    wsc_coverage,
    pearson_correlation,
    HSIC_correlation,
)
from clover.valid_pred_sets import Valid_pred_sets

original_path = os.getcwd()

# figure path
images_dir = "figures"


def testing_metrics_sim(
    n=25000,
    kind="homoscedastic",
    d=20,
    hetero_value=0.25,
    asym_value=0.6,
    t_degree=4,
    base_model=RandomForestRegressor,
    sig=0.1,
    test_size=0.2,
    valid_test_size=0.2,
    valid_split=True,
    valid_min_sample=100,
    valid_prune=False,
    calib_size=0.5,
    coef=2,
    noise=True,
    signif_vars=5,
    B_x=5000,
    B_y=1000,
    random_seed_X=850,
    random_seed=1250,
    random_projections=False,
    h=20,
    m=300,
    split_calib=False,
    nbins=30,
    criterion="squared_error",
    max_depth=None,
    max_leaf_nodes=None,
    min_samples_leaf=150,
    prune=True,
    **kwargs
):
    if "_V2" in kind:
        asym_value = 1.5
        kind = "asymmetric"

    # generating X_test
    np.random.seed(random_seed_X)

    X_test = np.random.uniform(low=-1.5, high=1.5, size=(B_x, d))
    sim_obj = simulation(
        dim=d,
        coef=coef,
        hetero_value=hetero_value,
        noise=noise,
        signif_vars=signif_vars,
        asym_value=asym_value,
        t_degree=t_degree,
    )
    sim_kind = getattr(sim_obj, kind)
    sim_kind(n, random_seed=random_seed)
    data = split(
        sim_obj.X,
        sim_obj.y,
        test_size=test_size,
        calib_size=calib_size,
        calibrate=True,
        random_seed=random_seed,
    )
    r_kind = getattr(sim_obj, kind + "_r")

    model = base_model(**kwargs).fit(data["X_train"], data["y_train"])
    if noise:
        y_mat = r_kind(X_test[:, 0], B=B_y)
    else:
        y_mat = r_kind(X_test, B=B_y)

    # fitting normal locart
    print("Fitting deafult locart to toy example:")
    start_loc = time.time()
    locart_obj = LocartSplit(
        nc_score=RegressionScore,
        cart_type="CART",
        base_model=model,
        alpha=sig,
        split_calib=split_calib,
        is_fitted=True,
    )
    locart_obj.fit(data["X_train"], data["y_train"])
    locart_obj.calib(
        data["X_calib"],
        data["y_calib"],
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
    print("Time Elapsed to fit Locart: ", end_loc)

    print("Computing metrics")
    start_loc = time.time()
    # predictions
    locart_pred = np.array(locart_obj.predict(data["X_test"]))
    locart_pred_cond = np.array(locart_obj.predict(X_test))
    cond_locart_real = real_coverage(locart_pred_cond, y_mat)

    # average, median and max distance
    dif_locart = np.abs(cond_locart_real - (1 - sig))
    locart_ave_dist, locart_med_dist, locart_max_dist = (
        np.mean(dif_locart),
        np.median(dif_locart),
        np.max(dif_locart),
    )

    locart_pcor = pearson_correlation(locart_pred, data["y_test"])
    locart_hsic = HSIC_correlation(locart_pred, data["y_test"])

    # valid pred sets
    locart_valid = Valid_pred_sets(
        conf=locart_obj,
        alpha=sig,
        coverage_evaluator="CART",
        prune=valid_prune,
        split_train=valid_split,
    )
    locart_valid.fit(
        data["X_test"],
        data["y_test"],
        test_size=valid_test_size,
        min_samples_leaf=valid_min_sample,
    )
    pred_set_dif_loc, max_set_dif_loc = locart_valid.compute_dif()

    # smis
    locart_smis = smis(locart_pred, data["y_test"], alpha=sig)

    # mean interval length
    locart_interval_len = np.mean(compute_interval_length(locart_pred))

    # marginal coverage
    marg_cover = (
        np.logical_and(
            data["y_test"] >= locart_pred[:, 0], data["y_test"] <= locart_pred[:, 1]
        )
        + 0
    )
    locart_ave_marginal_cov = np.mean(marg_cover)

    # interval length | coveraqe
    cover_idx = np.where(marg_cover == 1)
    locart_interval_len_cover = np.mean(compute_interval_length(locart_pred[cover_idx]))

    end_loc = time.time() - start_loc
    print("Time Elapsed to compute metrics for Locart: ", end_loc)

    # fitting normal RF-locart
    print("Fitting deafult RF-locart to toy example:")
    start_loc = time.time()
    rf_locart_obj = LocartSplit(
        nc_score=RegressionScore,
        cart_type="RF",
        base_model=model,
        alpha=sig,
        split_calib=split_calib,
        is_fitted=True,
        **kwargs
    )
    rf_locart_obj.fit(data["X_train"], data["y_train"])
    rf_locart_obj.calib(
        data["X_calib"],
        data["y_calib"],
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
    print("Time Elapsed to fit RF-Locart: ", end_loc)

    print("Computing metrics")
    start_loc = time.time()
    # predictions
    rf_locart_pred = np.array(rf_locart_obj.predict(data["X_test"]))
    rf_locart_pred_cond = np.array(rf_locart_obj.predict(X_test))
    cond_rf_locart_real = real_coverage(rf_locart_pred_cond, y_mat)

    # average, median and max distance
    dif_rf_locart = np.abs(cond_rf_locart_real - (1 - sig))
    rf_locart_ave_dist, rf_locart_med_dist, rf_locart_max_dist = (
        np.mean(dif_rf_locart),
        np.median(dif_rf_locart),
        np.max(dif_rf_locart),
    )

    # valid pred sets
    rf_locart_valid = Valid_pred_sets(
        conf=rf_locart_obj,
        alpha=sig,
        coverage_evaluator="CART",
        prune=valid_prune,
        split_train=valid_split,
    )
    rf_locart_valid.fit(
        data["X_test"],
        data["y_test"],
        test_size=valid_test_size,
        min_samples_leaf=valid_min_sample,
    )
    pred_set_dif_rf_loc, max_set_dif_rf_loc = rf_locart_valid.compute_dif()

    # smis
    rf_locart_smis = smis(rf_locart_pred, data["y_test"], alpha=sig)

    # mean interval length
    rf_locart_interval_len = np.mean(compute_interval_length(rf_locart_pred))

    # marginal coverage
    marg_cover = (
        np.logical_and(
            data["y_test"] >= rf_locart_pred[:, 0],
            data["y_test"] <= rf_locart_pred[:, 1],
        )
        + 0
    )
    rf_locart_ave_marginal_cov = np.mean(marg_cover)

    # interval length | coveraqe
    cover_idx = np.where(marg_cover == 1)
    rf_locart_interval_len_cover = np.mean(
        compute_interval_length(rf_locart_pred[cover_idx])
    )

    end_loc = time.time() - start_loc
    print("Time Elapsed to compute metrics for RF-Locart: ", end_loc)

    # fitting normal difficulty locart
    print("Fitting difficulty locart to toy example:")
    start_loc = time.time()
    dlocart_obj = LocartSplit(
        nc_score=RegressionScore,
        cart_type="CART",
        base_model=model,
        alpha=sig,
        split_calib=split_calib,
        is_fitted=True,
        weighting=True,
        **kwargs
    )
    dlocart_obj.fit(data["X_train"], data["y_train"])
    dlocart_obj.calib(
        data["X_calib"],
        data["y_calib"],
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
    print("Time Elapsed to fit Locart: ", end_loc)

    print("Computing metrics")
    start_loc = time.time()
    # predictions
    dlocart_pred = np.array(dlocart_obj.predict(data["X_test"]))
    dlocart_pred_cond = np.array(dlocart_obj.predict(X_test))
    cond_dlocart_real = real_coverage(dlocart_pred_cond, y_mat)

    # average, median and max distance
    dif_dlocart = np.abs(cond_dlocart_real - (1 - sig))
    dlocart_ave_dist, dlocart_med_dist, dlocart_max_dist = (
        np.mean(dif_dlocart),
        np.median(dif_dlocart),
        np.max(dif_dlocart),
    )

    # valid pred sets
    dlocart_valid = Valid_pred_sets(
        conf=dlocart_obj,
        alpha=sig,
        coverage_evaluator="CART",
        prune=valid_prune,
        split_train=valid_split,
    )
    dlocart_valid.fit(
        data["X_test"],
        data["y_test"],
        test_size=valid_test_size,
        min_samples_leaf=valid_min_sample,
    )
    pred_set_dif_dloc, max_set_dif_dloc = dlocart_valid.compute_dif()

    # smis
    dlocart_smis = smis(dlocart_pred, data["y_test"], alpha=sig)

    # mean interval length
    dlocart_interval_len = np.mean(compute_interval_length(dlocart_pred))

    # marginal coverage
    marg_cover = (
        np.logical_and(
            data["y_test"] >= dlocart_pred[:, 0], data["y_test"] <= dlocart_pred[:, 1]
        )
        + 0
    )
    dlocart_ave_marginal_cov = np.mean(marg_cover)

    # interval length | coveraqe
    cover_idx = np.where(marg_cover == 1)
    dlocart_interval_len_cover = np.mean(
        compute_interval_length(dlocart_pred[cover_idx])
    )

    end_loc = time.time() - start_loc
    print("Time Elapsed to compute metrics for Locart: ", end_loc)

    # fitting RF difficulty locart
    print("Fitting difficulty RF-locart to toy example:")
    start_loc = time.time()
    rf_dlocart_obj = LocartSplit(
        nc_score=RegressionScore,
        cart_type="RF",
        base_model=model,
        alpha=sig,
        split_calib=split_calib,
        is_fitted=True,
        weighting=True,
        **kwargs
    )
    rf_dlocart_obj.fit(data["X_train"], data["y_train"])
    rf_dlocart_obj.calib(
        data["X_calib"],
        data["y_calib"],
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
    print("Time Elapsed to fit Locart: ", end_loc)

    print("Computing metrics")
    start_loc = time.time()
    # predictions
    rf_dlocart_pred = np.array(rf_dlocart_obj.predict(data["X_test"]))
    rf_dlocart_pred_cond = np.array(rf_dlocart_obj.predict(X_test))
    cond_rf_dlocart_real = real_coverage(rf_dlocart_pred_cond, y_mat)

    # average, median and max distance
    dif_rf_dlocart = np.abs(cond_rf_dlocart_real - (1 - sig))
    rf_dlocart_ave_dist, rf_dlocart_med_dist, rf_dlocart_max_dist = (
        np.mean(dif_rf_dlocart),
        np.median(dif_rf_dlocart),
        np.max(dif_rf_dlocart),
    )

    # valid pred sets
    rf_dlocart_valid = Valid_pred_sets(
        conf=rf_dlocart_obj,
        alpha=sig,
        coverage_evaluator="CART",
        prune=valid_prune,
        split_train=valid_split,
    )
    rf_dlocart_valid.fit(
        data["X_test"],
        data["y_test"],
        test_size=valid_test_size,
        min_samples_leaf=valid_min_sample,
    )
    pred_set_dif_rf_dloc, max_set_dif_rf_dloc = rf_dlocart_valid.compute_dif()

    # smis
    rf_dlocart_smis = smis(rf_dlocart_pred, data["y_test"], alpha=sig)

    # mean interval length
    rf_dlocart_interval_len = np.mean(compute_interval_length(rf_dlocart_pred))

    # marginal coverage
    marg_cover = (
        np.logical_and(
            data["y_test"] >= rf_dlocart_pred[:, 0],
            data["y_test"] <= rf_dlocart_pred[:, 1],
        )
        + 0
    )
    rf_dlocart_ave_marginal_cov = np.mean(marg_cover)

    # interval length | coveraqe
    cover_idx = np.where(marg_cover == 1)
    rf_dlocart_interval_len_cover = np.mean(
        compute_interval_length(rf_dlocart_pred[cover_idx])
    )

    end_loc = time.time() - start_loc
    print("Time Elapsed to compute metrics for RF difficulty-Locart: ", end_loc)

    # fitting ACPI/LCP-RF
    print("Fitting LCP-RF to toy example:")
    start_loc = time.time()

    acpi = ACPI(model_cali=model, n_estimators=100)
    acpi.fit(data["X_calib"], data["y_calib"], nonconformity_func=None)
    acpi.fit_calibration(
        data["X_calib"], data["y_calib"], quantile=1 - sig, only_qrf=True
    )

    end_loc = time.time() - start_loc
    print("Time Elapsed to fit LCP-RF: ", end_loc)

    print("Computing metrics")
    start_loc = time.time()

    acpi_pred = np.stack((acpi.predict_pi(data["X_test"], method="qrf")), axis=-1)
    acpi_pred_cond = np.stack((acpi.predict_pi(X_test, method="qrf")), axis=-1)
    cond_acpi_real = real_coverage(acpi_pred_cond, y_mat)

    # valid pred sets
    acpi_valid = Valid_pred_sets(
        conf=acpi,
        alpha=sig,
        islcp=True,
        coverage_evaluator="CART",
        prune=valid_prune,
        split_train=valid_split,
    )
    acpi_valid.fit(
        data["X_test"],
        data["y_test"],
        test_size=valid_test_size,
        min_samples_leaf=valid_min_sample,
    )
    pred_set_dif_acpi, max_set_dif_acpi = acpi_valid.compute_dif()

    # average, median and max distance
    dif_acpi = np.abs(cond_acpi_real - (1 - sig))
    acpi_ave_dist, acpi_med_dist, acpi_max_dist = (
        np.mean(dif_acpi),
        np.median(dif_acpi),
        np.max(dif_acpi),
    )

    # smis
    acpi_smis = smis(acpi_pred, data["y_test"], alpha=sig)

    # mean interval length
    acpi_interval_len = np.mean(compute_interval_length(acpi_pred))

    # marginal coverage
    marg_cover = (
        np.logical_and(
            data["y_test"] >= acpi_pred[:, 0], data["y_test"] <= acpi_pred[:, 1]
        )
        + 0
    )
    acpi_ave_marginal_cov = np.mean(marg_cover)

    # interval length | coveraqe
    cover_idx = np.where(marg_cover == 1)
    acpi_interval_len_cover = np.mean(compute_interval_length(acpi_pred[cover_idx]))

    end_loc = time.time() - start_loc
    print("Time Elapsed to compute metrics for LCP-RF: ", end_loc)

    # fitting wlocart
    print("Fitting weighted RF-locart to toy example:")
    start_loc = time.time()

    wlocart_obj = LocartSplit(
        nc_score=LocalRegressionScore,
        cart_type="RF",
        base_model=model,
        alpha=sig,
        split_calib=split_calib,
        is_fitted=True,
        **kwargs
    )
    wlocart_obj.fit(
        data["X_train"],
        data["y_train"],
    )
    wlocart_obj.calib(
        data["X_calib"],
        data["y_calib"],
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
    print("Time Elapsed to fit Locart: ", end_loc)

    print("Computing metrics")
    start_loc = time.time()
    # predictions
    wlocart_pred = np.array(wlocart_obj.predict(data["X_test"]))
    wlocart_pred_cond = np.array(wlocart_obj.predict(X_test))
    cond_wlocart_real = real_coverage(wlocart_pred_cond, y_mat)

    # average, median and max distance
    dif_wlocart = np.abs(cond_wlocart_real - (1 - sig))
    wlocart_ave_dist, wlocart_med_dist, wlocart_max_dist = (
        np.mean(dif_wlocart),
        np.median(dif_wlocart),
        np.max(dif_wlocart),
    )

    # valid pred sets
    wlocart_valid = Valid_pred_sets(
        conf=wlocart_obj,
        alpha=sig,
        coverage_evaluator="CART",
        prune=valid_prune,
        split_train=valid_split,
    )
    wlocart_valid.fit(
        data["X_test"],
        data["y_test"],
        test_size=valid_test_size,
        min_samples_leaf=valid_min_sample,
    )
    pred_set_dif_wloc, max_set_dif_wloc = wlocart_valid.compute_dif()

    # smis
    wlocart_smis = smis(wlocart_pred, data["y_test"], alpha=sig)

    # mean interval length
    wlocart_interval_len = np.mean(compute_interval_length(wlocart_pred))

    # marginal coverage
    marg_cover = (
        np.logical_and(
            data["y_test"] >= wlocart_pred[:, 0], data["y_test"] <= wlocart_pred[:, 1]
        )
        + 0
    )
    wlocart_ave_marginal_cov = np.mean(marg_cover)

    # interval length | coveraqe
    cover_idx = np.where(marg_cover == 1)
    wlocart_interval_len_cover = np.mean(
        compute_interval_length(wlocart_pred[cover_idx])
    )

    wloc_cutoffs = wlocart_obj.cutoffs

    end_loc = time.time() - start_loc
    print("Time Elapsed to compute metrics for Locart: ", end_loc)

    # fitting default regression split
    print("Fitting regression split")
    start_split = time.time()
    new_model = base_model(**kwargs)
    nc = NcFactory.create_nc(new_model)
    icp = IcpRegressor(nc)
    icp.fit(data["X_train"], data["y_train"])
    icp.calibrate(data["X_calib"], data["y_calib"])

    end_split = time.time() - start_split
    print("Time Elapsed to fit regression split: ", end_split)

    print("Computing metrics")
    start_split = time.time()
    # predictions
    icp_pred = icp.predict(data["X_test"], significance=sig)
    icp_pred_cond = icp.predict(X_test, significance=sig)
    cond_icp_real = real_coverage(icp_pred_cond, y_mat)

    # average, median and max distance
    dif_icp = np.abs(cond_icp_real - (1 - sig))
    icp_ave_dist, icp_med_dist, icp_max_dist = (
        np.mean(dif_icp),
        np.median(dif_icp),
        np.max(dif_icp),
    )

    # valid pred sets
    icp_valid = Valid_pred_sets(
        conf=icp,
        alpha=sig,
        isnc=True,
        coverage_evaluator="CART",
        prune=valid_prune,
        split_train=valid_split,
    )
    icp_valid.fit(
        data["X_test"],
        data["y_test"],
        test_size=valid_test_size,
        min_samples_leaf=valid_min_sample,
    )
    pred_set_dif_icp, max_set_dif_icp = icp_valid.compute_dif()

    # icp smis
    icp_smis = smis(icp_pred, data["y_test"], alpha=sig)

    # ICP interval length
    icp_interval_len = np.mean(compute_interval_length(icp_pred))

    # marginal coverage
    marg_cover = (
        np.logical_and(
            data["y_test"] >= icp_pred[:, 0], data["y_test"] <= icp_pred[:, 1]
        )
        + 0
    )
    icp_ave_marginal_cov = np.mean(marg_cover)

    # interval length | coveraqe
    cover_idx = np.where(marg_cover == 1)
    icp_interval_len_cover = np.mean(compute_interval_length(icp_pred[cover_idx]))

    end_split = time.time() - start_split
    print("Time Elapsed to compute statistics for regression split: ", end_split)

    # fitting wighted regression split
    print("Fitting weighted regression split")
    start_weighted_split = time.time()
    wicp = LocalRegressionSplit(model, alpha=sig, is_fitted=True, **kwargs)
    wicp.fit(data["X_train"], data["y_train"])
    wicp.calibrate(data["X_calib"], data["y_calib"])

    end_weighted_split = time.time() - start_weighted_split
    print("Time Elapsed to fit weighted regression split: ", end_weighted_split)

    print("Computing metrics")
    start_weighted_split = time.time()
    # predictions
    wicp_pred = wicp.predict(data["X_test"])
    wicp_pred_cond = wicp.predict(X_test)
    cond_wicp_real = real_coverage(wicp_pred_cond, y_mat)

    wicp_dif = np.abs(cond_wicp_real - (1 - sig))
    wicp_ave_dist, wicp_med_dist, wicp_max_dist = (
        np.mean(wicp_dif),
        np.median(wicp_dif),
        np.max(wicp_dif),
    )

    # valid pred sets
    wicp_valid = Valid_pred_sets(
        conf=wicp,
        alpha=sig,
        coverage_evaluator="CART",
        prune=valid_prune,
        split_train=valid_split,
    )
    wicp_valid.fit(
        data["X_test"],
        data["y_test"],
        test_size=valid_test_size,
        min_samples_leaf=valid_min_sample,
    )
    pred_set_dif_wicp, max_set_dif_wicp = wicp_valid.compute_dif()

    # smis
    wicp_smis = smis(wicp_pred, data["y_test"], alpha=sig)

    # ICP interval length
    wicp_interval_len = np.mean(compute_interval_length(wicp_pred))

    # marginal coverage
    marg_cover = (
        np.logical_and(
            data["y_test"] >= wicp_pred[:, 0], data["y_test"] <= wicp_pred[:, 1]
        )
        + 0
    )
    wicp_ave_marginal_cov = np.mean(marg_cover)

    wicp_cutoff = wicp.cutoff

    # interval length | coveraqe
    cover_idx = np.where(marg_cover == 1)
    wicp_interval_len_cover = np.mean(compute_interval_length(wicp_pred[cover_idx]))
    print(
        "Time Elapsed to compute statistics for weighted regression split: ",
        end_weighted_split,
    )

    # mondrian split
    print("Fitting mondrian regression split")
    start_weighted_split = time.time()
    micp = MondrianRegressionSplit(model, alpha=sig, k=nbins, is_fitted=True, **kwargs)
    micp.fit(data["X_train"], data["y_train"])
    micp.calibrate(data["X_calib"], data["y_calib"])

    end_weighted_split = time.time() - start_weighted_split
    print("Time Elapsed to fit mondrian regression split: ", end_weighted_split)

    print("Computing metrics")
    start_weighted_split = time.time()
    # predictions
    micp_pred = micp.predict(data["X_test"])
    micp_pred_cond = micp.predict(X_test)
    cond_micp_real = real_coverage(micp_pred_cond, y_mat)

    micp_dif = np.abs(cond_micp_real - (1 - sig))
    micp_ave_dist, micp_med_dist, micp_max_dist = (
        np.mean(micp_dif),
        np.median(micp_dif),
        np.max(micp_dif),
    )

    # valid pred sets
    micp_valid = Valid_pred_sets(
        conf=micp,
        alpha=sig,
        coverage_evaluator="CART",
        prune=valid_prune,
        split_train=valid_split,
    )
    micp_valid.fit(
        data["X_test"],
        data["y_test"],
        test_size=valid_test_size,
        min_samples_leaf=valid_min_sample,
    )
    pred_set_dif_micp, max_set_dif_micp = micp_valid.compute_dif()

    # smis
    micp_smis = smis(micp_pred, data["y_test"], alpha=sig)

    # ICP interval length
    micp_interval_len = np.mean(compute_interval_length(micp_pred))

    # marginal coverage
    marg_cover = (
        np.logical_and(
            data["y_test"] >= micp_pred[:, 0], data["y_test"] <= micp_pred[:, 1]
        )
        + 0
    )
    micp_ave_marginal_cov = np.mean(marg_cover)

    # interval length | coveraqe
    cover_idx = np.where(marg_cover == 1)
    micp_interval_len_cover = np.mean(compute_interval_length(micp_pred[cover_idx]))

    all_results = pd.DataFrame(
        data={
            "Methods": [
                "LOCART",
                "RF-LOCART",
                "D-LOCART",
                "RF-D-LOCART",
                "LCP-RF",
                "Weighted LOCART",
                "Regresion split",
                "Weighted regression split",
                "Mondrian regression split",
            ],
            "valid pred set": [
                pred_set_dif_loc,
                pred_set_dif_rf_loc,
                pred_set_dif_dloc,
                pred_set_dif_rf_dloc,
                pred_set_dif_acpi,
                pred_set_dif_wloc,
                pred_set_dif_icp,
                pred_set_dif_wicp,
                pred_set_dif_micp,
            ],
            "smis": [
                locart_smis,
                rf_locart_smis,
                dlocart_smis,
                rf_dlocart_smis,
                acpi_smis,
                wlocart_smis,
                icp_smis,
                wicp_smis,
                micp_smis,
            ],
            "max pred set": [
                max_set_dif_loc,
                max_set_dif_rf_loc,
                max_set_dif_dloc,
                max_set_dif_rf_dloc,
                max_set_dif_acpi,
                max_set_dif_wloc,
                max_set_dif_icp,
                max_set_dif_wicp,
                max_set_dif_micp,
            ],
            "Average marginal coverage": [
                locart_ave_marginal_cov,
                rf_locart_ave_marginal_cov,
                dlocart_ave_marginal_cov,
                rf_dlocart_ave_marginal_cov,
                acpi_ave_marginal_cov,
                wlocart_ave_marginal_cov,
                icp_ave_marginal_cov,
                wicp_ave_marginal_cov,
                micp_ave_marginal_cov,
            ],
            "Average interval length": [
                locart_interval_len,
                rf_locart_interval_len,
                dlocart_interval_len,
                rf_dlocart_interval_len,
                acpi_interval_len,
                wlocart_interval_len,
                icp_interval_len,
                wicp_interval_len,
                micp_interval_len,
            ],
            "Average interval length given coverage": [
                locart_interval_len_cover,
                rf_locart_interval_len_cover,
                dlocart_interval_len_cover,
                rf_dlocart_interval_len_cover,
                acpi_interval_len_cover,
                wlocart_interval_len_cover,
                icp_interval_len_cover,
                wicp_interval_len_cover,
                micp_interval_len_cover,
            ],
            "Average distance": [
                locart_ave_dist,
                rf_locart_ave_dist,
                dlocart_ave_dist,
                rf_dlocart_ave_dist,
                acpi_ave_dist,
                wlocart_ave_dist,
                icp_ave_dist,
                wicp_ave_dist,
                micp_ave_dist,
            ],
            "Median distance": [
                locart_med_dist,
                rf_locart_med_dist,
                dlocart_med_dist,
                rf_dlocart_med_dist,
                acpi_med_dist,
                wlocart_med_dist,
                icp_med_dist,
                wicp_med_dist,
                micp_med_dist,
            ],
            "Max distance": [
                locart_max_dist,
                rf_locart_max_dist,
                dlocart_max_dist,
                rf_dlocart_max_dist,
                acpi_max_dist,
                wlocart_max_dist,
                icp_max_dist,
                wicp_max_dist,
                micp_max_dist,
            ],
        }
    )

    return all_results


kinds_list = [
    "homoscedastic",
    "heteroscedastic",
    "asymmetric",
    "asymmetric_V2",
    "t_residuals",
    "non_cor_heteroscedastic",
]

if __name__ == "__main__":
    folder_path = "/results/pickle_files/fast_experiments/simulated"
    print("Starting simulated data single experiments")
    for kind in kinds_list:
        for p in [1, 3, 5]:
            noise = p == 1
            print("Starting experiments for {} kind and p = {}".format(kind, p))
            stats = testing_metrics_sim(
                kind=kind, signif_vars=p, noise=noise, random_state=650
            )
            stats.to_csv(
                original_path
                + folder_path
                + "/exp_stats_{}_nvar_{}.csv".format(kind, p)
            )
