import numpy as np
import pandas as pd
import os
import scipy.stats as st

from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

from clover.locart import (
    LocalRegressionSplit,
    LocartSplit,
    MondrianRegressionSplit,
    QuantileSplit,
)
from clover.locluster import KmeansSplit
from acpi import ACPI
from clover.scores import RegressionScore, QuantileScore, LocalRegressionScore
import time


from clover.locart import LocartSplit, MondrianRegressionSplit
from clover.scores import RegressionScore
from clover.simulation import simulation
from clover.utils import (
    compute_interval_length,
    HSIC_correlation,
    pearson_correlation,
    ILS_coverage,
    split,
    smis,
)
from clover.valid_pred_sets import Valid_pred_sets
import gc

original_path = os.getcwd()

# figure path
images_dir = "figures"


# running real data experiments for only one repetition
def obtain_main_metrics(
    data_name,
    type_score="regression",
    base_model=RandomForestRegressor,
    test_size=0.2,
    valid_test_size=0.3,
    is_fitted=True,
    valid_sample_leaf=100,
    valid_prune=False,
    valid_split=True,
    calib_size=0.5,
    random_seed=1250,
    sig=0.1,
    split_calib=False,
    nbins=30,
    split_mondrian=False,
    criterion="squared_error",
    max_depth=None,
    max_leaf_nodes=None,
    min_samples_leaf=300,
    prune=True,
    **kwargs
):
    # importing data, selecting some rows and then splitting
    data_path = original_path + "/data/processed/" + data_name + ".csv"

    # reading data using pandas
    data = pd.read_csv(data_path)
    y = data["target"].to_numpy()
    X = data.drop("target", axis=1).to_numpy()

    print(
        "Number of samples used for training and calibration: {}".format(
            (1 - test_size) * X.shape[0]
        )
    )
    print("Number of samples used for testing: {}".format(test_size * X.shape[0]))
    print(
        "Number of samples used for fitting coverage evaluator: {}".format(
            (1 - valid_test_size) * test_size * X.shape[0]
        )
    )
    print(
        "Number of samples used to compute conditional coverage difference {}".format(
            valid_test_size * test_size * X.shape[0]
        )
    )

    # splitting data into train, calibration and test
    data = split(X, y, test_size, calib_size, calibrate=True, random_seed=random_seed)

    # setting seed
    np.random.seed(random_seed)

    if is_fitted:
        model = base_model(**kwargs).fit(data["X_train"], data["y_train"])
    else:
        model = base_model

    if type_score == "regression":
        # fitting mondrian regression split
        print("Fitting mondrian regression split")
        start_mondrian_split = time.time()
        micp = MondrianRegressionSplit(
            model, alpha=sig, is_fitted=is_fitted, k=nbins, **kwargs
        )
        micp.fit(data["X_train"], data["y_train"])
        micp.calibrate(data["X_calib"], data["y_calib"], split=split_mondrian)

        end_mondrian_split = time.time() - start_mondrian_split
        print("Time Elapsed to fit mondrian regression split: ", end_mondrian_split)

        print("Computing metrics")
        start_mondrian_split = time.time()
        # predictions
        micp_pred = micp.predict(data["X_test"])

        # mondrian icp correlations
        micp_pcor = pearson_correlation(micp_pred, data["y_test"])
        micp_hsic = HSIC_correlation(micp_pred, data["y_test"])

        # valid pred sets
        vp = Valid_pred_sets(
            conf=micp,
            alpha=sig,
            coverage_evaluator="CART",
            prune=valid_prune,
            split_train=valid_split,
        )
        vp.fit(
            data["X_test"],
            data["y_test"],
            test_size=valid_test_size,
            min_samples_leaf=valid_sample_leaf,
        )
        pred_set_dif_micp, max_set_dif_micp = vp.compute_dif()

        # smis
        micp_smis = smis(micp_pred, data["y_test"], alpha=sig)

        # mean interval length
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

        # deleting some objects
        del micp
        del vp
        del cover_idx
        gc.collect()

        end_mondrian_split = time.time() - start_mondrian_split
        print(
            "Time Elapsed to compute statistics for mondrian regression split: ",
            end_mondrian_split,
        )

        print("Fitting locart")
        start_loc = time.time()
        # fitting locart
        locart_obj = LocartSplit(
            cart_type="CART",
            nc_score=RegressionScore,
            base_model=model,
            is_fitted=is_fitted,
            alpha=sig,
            split_calib=split_calib,
            **kwargs
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
        )

        end_loc = time.time() - start_loc
        print("Time Elapsed to fit Locart: ", end_loc)

        print("Computing metrics")
        start_loc = time.time()
        # predictions
        locart_pred = np.array(locart_obj.predict(data["X_test"]))

        # mondrian icp correlations
        locart_pcor = pearson_correlation(locart_pred, data["y_test"])
        locart_hsic = HSIC_correlation(locart_pred, data["y_test"])

        vp = Valid_pred_sets(
            conf=locart_obj,
            alpha=sig,
            coverage_evaluator="CART",
            prune=valid_prune,
            split_train=valid_split,
        )
        vp.fit(
            data["X_test"],
            data["y_test"],
            test_size=valid_test_size,
            min_samples_leaf=valid_sample_leaf,
        )
        pred_set_dif_locart, max_set_dif_locart = vp.compute_dif()

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
        locart_interval_len_cover = np.mean(
            compute_interval_length(locart_pred[cover_idx])
        )
        end_loc = time.time() - start_loc
        print("Time Elapsed to compute metrics for Locart: ", end_loc)
        del locart_obj
        del vp
        del cover_idx
        gc.collect()

        print("Fitting LCP-RF:")

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

        vp = Valid_pred_sets(
            conf=acpi,
            alpha=sig,
            islcp=True,
            coverage_evaluator="CART",
            prune=valid_prune,
            split_train=valid_split,
        )
        vp.fit(
            data["X_test"],
            data["y_test"],
            test_size=valid_test_size,
            min_samples_leaf=valid_sample_leaf,
        )
        pred_set_dif_acpi, max_set_dif_acpi = vp.compute_dif()

        # correlations
        acpi_pcor = pearson_correlation(acpi_pred, data["y_test"])
        acpi_hsic = HSIC_correlation(acpi_pred, data["y_test"])

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
        print("Time Elapsed to compute metrics for Locart: ", end_loc)
        del acpi
        del vp
        del cover_idx
        gc.collect()

        print("Fitting locart weighted")
        start_loc = time.time()
        # fitting locart
        wlocart_obj = LocartSplit(
            cart_type="RF",
            nc_score=LocalRegressionScore,
            base_model=model,
            is_fitted=is_fitted,
            alpha=sig,
            split_calib=split_calib,
            **kwargs
        )
        wlocart_obj.fit(data["X_train"], data["y_train"])
        wlocart_obj.calib(
            data["X_calib"],
            data["y_calib"],
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            prune_tree=prune,
        )

        end_loc = time.time() - start_loc
        print("Time Elapsed to fit Locart: ", end_loc)

        print("Computing metrics")
        start_loc = time.time()
        # predictions
        wlocart_pred = np.array(wlocart_obj.predict(data["X_test"]))

        # mondrian icp correlations
        wlocart_pcor = pearson_correlation(wlocart_pred, data["y_test"])
        wlocart_hsic = HSIC_correlation(wlocart_pred, data["y_test"])

        vp = Valid_pred_sets(
            conf=wlocart_obj,
            alpha=sig,
            coverage_evaluator="CART",
            prune=valid_prune,
            split_train=valid_split,
        )
        vp.fit(
            data["X_test"],
            data["y_test"],
            test_size=valid_test_size,
            min_samples_leaf=valid_sample_leaf,
        )
        pred_set_dif_wlocart, max_set_dif_wlocart = vp.compute_dif()

        # smis
        wlocart_smis = smis(wlocart_pred, data["y_test"], alpha=sig)

        # mean interval length
        wlocart_interval_len = np.mean(compute_interval_length(wlocart_pred))

        # marginal coverage
        marg_cover = (
            np.logical_and(
                data["y_test"] >= wlocart_pred[:, 0],
                data["y_test"] <= wlocart_pred[:, 1],
            )
            + 0
        )
        wlocart_ave_marginal_cov = np.mean(marg_cover)

        # interval length | coveraqe
        cover_idx = np.where(marg_cover == 1)
        wlocart_interval_len_cover = np.mean(
            compute_interval_length(wlocart_pred[cover_idx])
        )
        end_loc = time.time() - start_loc
        print("Time Elapsed to compute metrics for Locart: ", end_loc)
        del wlocart_obj
        del vp
        del cover_idx
        gc.collect()

        print("Fitting difficulty locart:")
        start_loc = time.time()
        dlocart_obj = LocartSplit(
            nc_score=RegressionScore,
            cart_type="CART",
            base_model=model,
            is_fitted=is_fitted,
            alpha=sig,
            split_calib=split_calib,
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
        )

        end_loc = time.time() - start_loc
        print("Time Elapsed to fit Locart: ", end_loc)

        print("Computing metrics")
        start_loc = time.time()

        # predictions
        dlocart_pred = np.array(dlocart_obj.predict(data["X_test"]))

        # mondrian icp correlations
        dlocart_pcor = pearson_correlation(dlocart_pred, data["y_test"])
        dlocart_hsic = HSIC_correlation(dlocart_pred, data["y_test"])

        vp = Valid_pred_sets(
            conf=dlocart_obj,
            alpha=sig,
            coverage_evaluator="CART",
            prune=valid_prune,
            split_train=valid_split,
        )
        vp.fit(
            data["X_test"],
            data["y_test"],
            test_size=valid_test_size,
            min_samples_leaf=valid_sample_leaf,
        )
        pred_set_dif_dlocart, max_set_dif_dlocart = vp.compute_dif()

        # smis
        dlocart_smis = smis(dlocart_pred, data["y_test"], alpha=sig)

        # mean interval length
        dlocart_interval_len = np.mean(compute_interval_length(dlocart_pred))

        # marginal coverage
        marg_cover = (
            np.logical_and(
                data["y_test"] >= dlocart_pred[:, 0],
                data["y_test"] <= dlocart_pred[:, 1],
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
        print("Time Elapsed to compute metrics for RF-Locart: ", end_loc)
        del dlocart_obj
        del vp
        del cover_idx
        gc.collect()

        # fitting LOFOREST
        print("Fitting RF-locart")
        start_loc = time.time()
        # fitting locart
        rf_locart_obj = LocartSplit(
            cart_type="RF",
            nc_score=RegressionScore,
            base_model=model,
            is_fitted=is_fitted,
            alpha=sig,
            split_calib=split_calib,
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
        )

        end_loc = time.time() - start_loc
        print("Time Elapsed to fit RF-Locart: ", end_loc)

        print("Computing metrics")
        start_loc = time.time()
        # predictions
        rf_locart_pred = np.array(rf_locart_obj.predict(data["X_test"]))

        # mondrian icp correlations
        rf_locart_pcor = pearson_correlation(rf_locart_pred, data["y_test"])
        rf_locart_hsic = HSIC_correlation(rf_locart_pred, data["y_test"])

        vp = Valid_pred_sets(
            conf=rf_locart_obj,
            alpha=sig,
            coverage_evaluator="CART",
            prune=valid_prune,
            split_train=valid_split,
        )
        vp.fit(
            data["X_test"],
            data["y_test"],
            test_size=valid_test_size,
            min_samples_leaf=valid_sample_leaf,
        )
        pred_set_dif_rf_locart, max_set_dif_rf_locart = vp.compute_dif()

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
        del rf_locart_obj
        del vp
        del cover_idx
        gc.collect()

        # fitting LOFOREST
        print("Fitting RF-Dlocart")
        start_loc = time.time()
        # fitting locart
        rf_dlocart_obj = LocartSplit(
            cart_type="RF",
            nc_score=RegressionScore,
            base_model=model,
            is_fitted=is_fitted,
            alpha=sig,
            split_calib=split_calib,
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
        )

        end_loc = time.time() - start_loc
        print("Time Elapsed to fit RF-dLocart: ", end_loc)

        print("Computing metrics")
        start_loc = time.time()
        # predictions
        rf_dlocart_pred = np.array(rf_dlocart_obj.predict(data["X_test"]))

        # mondrian icp correlations
        rf_dlocart_pcor = pearson_correlation(rf_dlocart_pred, data["y_test"])
        rf_dlocart_hsic = HSIC_correlation(rf_dlocart_pred, data["y_test"])

        vp = Valid_pred_sets(
            conf=rf_dlocart_obj,
            alpha=sig,
            coverage_evaluator="CART",
            prune=valid_prune,
            split_train=valid_split,
        )
        vp.fit(
            data["X_test"],
            data["y_test"],
            test_size=valid_test_size,
            min_samples_leaf=valid_sample_leaf,
        )
        pred_set_dif_rf_dlocart, max_set_dif_rf_dlocart = vp.compute_dif()

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
        print("Time Elapsed to compute metrics for Locart: ", end_loc)
        del rf_dlocart_obj
        del vp
        del cover_idx
        gc.collect()

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

        # ICP icp correlations
        icp_pcor = pearson_correlation(icp_pred, data["y_test"])
        icp_hsic = HSIC_correlation(icp_pred, data["y_test"])

        vp = Valid_pred_sets(
            conf=icp,
            alpha=sig,
            coverage_evaluator="CART",
            isnc=True,
            prune=valid_prune,
            split_train=valid_split,
        )
        vp.fit(
            data["X_test"],
            data["y_test"],
            test_size=valid_test_size,
            min_samples_leaf=valid_sample_leaf,
        )
        pred_set_dif_icp, max_set_dif_icp = vp.compute_dif()

        # smis
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
        del icp
        del new_model
        del vp
        del cover_idx
        gc.collect()

        # fitting wighted regression split
        print("Fitting weighted regression split")
        start_weighted_split = time.time()
        wicp = LocalRegressionSplit(model, alpha=sig, is_fitted=is_fitted, **kwargs)
        wicp.fit(data["X_train"], data["y_train"])
        wicp.calibrate(data["X_calib"], data["y_calib"])

        end_weighted_split = time.time() - start_weighted_split
        print("Time Elapsed to fit weighted regression split: ", end_weighted_split)

        print("Computing metrics")
        start_weighted_split = time.time()
        # predictions
        wicp_pred = wicp.predict(data["X_test"])

        # ICP icp correlations
        wicp_pcor = pearson_correlation(wicp_pred, data["y_test"])
        wicp_hsic = HSIC_correlation(wicp_pred, data["y_test"])

        vp = Valid_pred_sets(
            conf=wicp,
            alpha=sig,
            coverage_evaluator="CART",
            prune=valid_prune,
            split_train=valid_split,
        )
        vp.fit(
            data["X_test"],
            data["y_test"],
            test_size=valid_test_size,
            min_samples_leaf=valid_sample_leaf,
        )
        pred_set_dif_wicp, max_set_dif_wicp = vp.compute_dif()

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

        # interval length | coveraqe
        cover_idx = np.where(marg_cover == 1)
        wicp_interval_len_cover = np.mean(compute_interval_length(wicp_pred[cover_idx]))
        print(
            "Time Elapsed to compute statistics for weighted regression split: ",
            end_weighted_split,
        )
        del wicp
        del vp
        del cover_idx
        gc.collect()

        all_results = pd.DataFrame(
            data={
                "Methods": [
                    "LOCART",
                    "DLOCART",
                    "RF-LOCART",
                    "RF-DLOCART",
                    "LCP-RF",
                    "Weighted LOCART",
                    "Regresion split",
                    "Weighted regression split",
                    "Mondrian regression split",
                ],
                "Pearson correlation": [
                    locart_pcor,
                    dlocart_pcor,
                    rf_locart_pcor,
                    rf_dlocart_pcor,
                    acpi_pcor,
                    wlocart_pcor,
                    icp_pcor,
                    wicp_pcor,
                    micp_pcor,
                ],
                "HSIC correlation": [
                    locart_hsic,
                    dlocart_hsic,
                    rf_locart_hsic,
                    rf_dlocart_hsic,
                    acpi_hsic,
                    wlocart_hsic,
                    icp_hsic,
                    wicp_hsic,
                    micp_hsic,
                ],
                "mean pred dif": [
                    pred_set_dif_locart,
                    pred_set_dif_dlocart,
                    pred_set_dif_rf_locart,
                    pred_set_dif_rf_dlocart,
                    pred_set_dif_acpi,
                    pred_set_dif_wlocart,
                    pred_set_dif_icp,
                    pred_set_dif_wicp,
                    pred_set_dif_micp,
                ],
                "max pred dif": [
                    max_set_dif_locart,
                    max_set_dif_dlocart,
                    max_set_dif_rf_locart,
                    max_set_dif_rf_dlocart,
                    max_set_dif_acpi,
                    max_set_dif_wlocart,
                    max_set_dif_icp,
                    max_set_dif_wicp,
                    max_set_dif_micp,
                ],
                "smis": [
                    locart_smis,
                    dlocart_smis,
                    rf_locart_smis,
                    rf_dlocart_smis,
                    acpi_smis,
                    wlocart_smis,
                    icp_smis,
                    wicp_smis,
                    micp_smis,
                ],
                "Average marginal coverage": [
                    locart_ave_marginal_cov,
                    dlocart_ave_marginal_cov,
                    rf_locart_ave_marginal_cov,
                    rf_dlocart_ave_marginal_cov,
                    acpi_ave_marginal_cov,
                    wlocart_ave_marginal_cov,
                    icp_ave_marginal_cov,
                    wicp_ave_marginal_cov,
                    micp_ave_marginal_cov,
                ],
                "Average interval length": [
                    locart_interval_len,
                    dlocart_interval_len,
                    rf_locart_interval_len,
                    rf_dlocart_interval_len,
                    acpi_interval_len,
                    wlocart_interval_len,
                    icp_interval_len,
                    wicp_interval_len,
                    micp_interval_len,
                ],
                "Average interval length given coverage": [
                    locart_interval_len_cover,
                    dlocart_interval_len_cover,
                    rf_locart_interval_len_cover,
                    rf_dlocart_interval_len_cover,
                    acpi_interval_len_cover,
                    wlocart_interval_len_cover,
                    icp_interval_len_cover,
                    wicp_interval_len_cover,
                    micp_interval_len_cover,
                ],
            }
        )
    return all_results


data_names = [
    "winewhite",
    "winered",
    "concrete",
    "airfoil",
    "electric",
    "superconductivity",
    "cycle",
    "protein",
    "news",
    "bike",
    "star",
    "meps19",
]

if __name__ == "__main__":
    folder_path = "/results/pickle_files/fast_experiments"
    print("Starting all real data single experiments")
    for data_name in data_names:
        print("Starting experiments for {} data".format(data_name))
        stats = obtain_main_metrics(data_name, random_state=650)
        stats.to_csv(
            original_path + folder_path + "/exp_stats_{}.csv".format(data_name)
        )
